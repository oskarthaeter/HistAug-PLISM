"""CycleGAN training on pre-extracted patch embeddings.

Two modes (auto-detected from the .pt file):
  Unconditioned (single pair): Z_A / Z_B only.  Two independent generators.
  Multi-target conditioned:    Z_A / Z_B / src_ids / tgt_ids / scanner_vocab.
      ONE shared generator G(x, tgt_id) and ONE discriminator D(z, tgt_id),
      both conditioned on the TARGET scanner.  src_id is only used as the
      "target" of the cycle reverse pass — not needed at inference.

Usage:
    python -m histaug.cyclegan_features.train \\
        --config src/histaug/config/CycleGAN_h0mini_all_pairs.yaml
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .config import CycleGANConfig
from .data import (
    MmapMultiTargetDataset,
    MultiTargetPairedDataset,
    PairedFeatureDataset,
    load_multitarget_tensors,
    load_paired_tensors,
)
from .losses import (
    adversarial_loss,
    coral_loss,
    cycle_loss,
    identity_loss,
    paired_loss,
    relational_loss,
)
from .models import FeatureDiscriminator, FeatureGenerator
from .utils import NormStats, ReplayBuffer, save_checkpoint

# ---------------------------------------------------------------------------
# Generator warm-initialization from scanner-transfer checkpoints
# ---------------------------------------------------------------------------


def _build_scanner_transfer_remap(state: dict, g: FeatureGenerator) -> dict:
    """Return {source_key: target_key} for scanner-transfer → FeatureGenerator.

    A None target means "discard this key" (used for spectral-norm auxiliary vectors).
    """
    remap: dict = {}
    if not g.use_film and "proj.0.weight" in state:
        # ScannerTransferLinearModel (Sequential): proj.{0,2,4} → mlp.{0,3,6}
        # proj skips non-parametric activations; mlp includes LayerNorm layers between linears
        for pi, mi in {0: 0, 2: 3, 4: 6}.items():
            for suffix in ("weight", "bias"):
                src = f"proj.{pi}.{suffix}"
                if src in state:
                    remap[src] = f"mlp.{mi}.{suffix}"

    # Spectral normalization: weight_orig → weight, discard auxiliary u/v vectors.
    # Applies to both linear and FiLM models stored with spectral_norm_all=True.
    for k in state:
        if k.endswith(".weight_orig"):
            remap[k] = k[: -len("_orig")]  # "layers.0.weight_orig" → "layers.0.weight"
        elif k.endswith(".weight_u") or k.endswith(".weight_v"):
            remap[k] = None  # discard
    return remap


def _init_generator_from_ckpt(
    g: FeatureGenerator, ckpt_path: str, ckpt_key: str = "g_ab"
) -> None:
    """Warm-initialize a FeatureGenerator from a CycleGAN or Lightning checkpoint.

    CycleGAN checkpoints (keys g_ab / g_ba) are loaded directly.
    Lightning scanner-transfer checkpoints have their model.* and _orig_mod.*
    prefixes stripped, then keys are remapped and shape-filtered into g.
    For the first linear layer, columns are copied up to min(src, tgt) width so that
    conditioning-dimension mismatches (e.g. 7-scanner → 4-scanner) are handled
    gracefully: feature columns transfer exactly, conditioning columns default to zero.
    """
    print(f"  Warm init ({ckpt_key}): loading from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt_key in ckpt:
        source_sd = ckpt[ckpt_key]
    elif "state_dict" in ckpt:
        source_sd = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model."):
                k = k[len("model.") :]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod.") :]
            source_sd[k] = v
    else:
        print("    Unrecognized checkpoint format; skipping.")
        return

    remap = _build_scanner_transfer_remap(source_sd, g)
    remapped: dict = {}
    for k, v in source_sd.items():
        tgt = remap.get(k, k)
        if tgt is not None:  # None = discard (spectral-norm auxiliary vectors)
            remapped[tgt] = v

    target_sd = g.state_dict()
    loaded: list = []
    for k, v in remapped.items():
        if k not in target_sd:
            continue
        t = target_sd[k]
        if t.shape == v.shape:
            target_sd[k] = v.to(t.dtype)
            loaded.append(k)
        elif v.dim() == 2 and t.dim() == 2 and t.shape[0] == v.shape[0]:
            # Only partial-copy when the feature dimension (g.dim) is involved.
            # This skips FiLM conditioning weights where cond_dim mismatches
            # (e.g. 7-scanner → 4-scanner) — those start from their zero-init default.
            if g.dim not in (t.shape[0], t.shape[1]):
                continue
            n_feat = min(t.shape[1], v.shape[1], g.dim)
            target_sd[k][:, :n_feat] = v[:, :n_feat].to(t.dtype)
            if n_feat < t.shape[1]:
                target_sd[k][:, n_feat:] = 0.0
            loaded.append(f"{k}[:, :{n_feat}/{t.shape[1]}]")

    g.load_state_dict(target_sd)
    print(f"    Transferred {len(loaded)}/{len(target_sd)} tensors: {loaded}")


def _progress(iterable, **kwargs):
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable


_SCANNER_VENDOR: Dict[str, str] = {
    "AT2": "Leica",
    "GT450": "Leica",
    "P": "Philips",
    "S210": "Hamamatsu",
    "S360": "Hamamatsu",
    "S60": "Hamamatsu",
    "SQ": "Hamamatsu",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CycleGAN feature translation")
    p.add_argument("--config", default=None)
    p.add_argument("--paired_data", default=None)
    p.add_argument("--val_data", default=None)
    p.add_argument("--unpaired_data", default=None)
    p.add_argument("--feature_dim", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=None)
    p.add_argument("--lr_decay_epochs", type=int, default=0)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_idt", type=float, default=5.0)
    p.add_argument("--lambda_paired", type=float, default=10.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument(
        "--loss_alpha",
        type=float,
        default=1.0,
        help="Feature loss blend: 1.0=L1, 0.0=cosine, 0.5=equal",
    )
    p.add_argument("--lambda_coral", type=float, default=0.0)
    p.add_argument("--lambda_rel", type=float, default=0.0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--discriminator_dropout", type=float, default=0.3)
    p.add_argument(
        "--discriminator_use_spectral_norm",
        action="store_true",
        help="Enable spectral normalization on discriminator linear layers",
    )
    p.add_argument(
        "--filter_scanners",
        nargs="*",
        default=None,
        help="Restrict training to this scanner subset (no re-preparation needed)",
    )
    p.add_argument("--replay_buffer_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--output_dir", default="cyclegan_output")
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _is_mmap_format(pt: dict) -> bool:
    return "chunks" in pt


def _filter_chunks(
    chunks: list, vocab: List[str], keep_scanners: List[str]
) -> Tuple[list, List[str]]:
    """Subset chunk list to the given scanner names and re-index src_id/tgt_id."""
    keep_set = set(keep_scanners)
    keep_idx = sorted(i for i, s in enumerate(vocab) if s in keep_set)
    missing = keep_set - {vocab[i] for i in keep_idx}
    if missing:
        raise ValueError(f"filter_scanners: {missing} not in scanner_vocab {vocab}")
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    new_vocab = [vocab[i] for i in keep_idx]
    filtered = [
        {**c, "src_id": old_to_new[c["src_id"]], "tgt_id": old_to_new[c["tgt_id"]]}
        for c in chunks
        if vocab[c["src_id"]] in keep_set and vocab[c["tgt_id"]] in keep_set
    ]
    print(
        f"filter_scanners={keep_scanners}: {len(filtered)}/{len(chunks)} chunks kept, "
        f"vocab={new_vocab}"
    )
    return filtered, new_vocab


def _build_mmap_datasets(cfg: CycleGANConfig, meta_train: dict):
    """Build datasets from the mmap metadata format (chunks + norm stats)."""
    norm_stats = NormStats.from_meta(meta_train)
    vocab = meta_train["scanner_vocab"]
    chunks_train = meta_train["chunks"]

    if cfg.filter_scanners:
        chunks_train, vocab = _filter_chunks(chunks_train, vocab, cfg.filter_scanners)

    train_ds = MmapMultiTargetDataset(
        chunks_train,
        feature_dim=meta_train["feature_dim"],
        n_coord_cols=meta_train["n_coord_cols"],
        norm_stats=norm_stats,
        data_fraction=cfg.data_fraction,
        seed=cfg.seed,
    )

    if cfg.val_data is None:
        raise ValueError("val_data must be set when using the mmap metadata format")
    meta_val = torch.load(Path(cfg.val_data), weights_only=True)
    chunks_val = meta_val["chunks"]
    if cfg.filter_scanners:
        chunks_val, _ = _filter_chunks(
            chunks_val, meta_val["scanner_vocab"], cfg.filter_scanners
        )
    val_ds = MmapMultiTargetDataset(
        chunks_val,
        feature_dim=meta_val["feature_dim"],
        n_coord_cols=meta_val["n_coord_cols"],
        norm_stats=norm_stats,
    )

    cond_dim = len(vocab)
    return train_ds, val_ds, cond_dim, vocab, norm_stats


def _build_tensor_datasets(cfg: CycleGANConfig, train_pt: dict):
    """Build datasets from the legacy tensor format (Z_A / Z_B in memory)."""
    if cfg.filter_scanners and "src_ids" in train_pt:
        # Inline filter for legacy format: mask rows, re-index ids.
        vocab: List[str] = train_pt["scanner_vocab"]
        keep_set = set(cfg.filter_scanners)
        keep_idx = sorted(i for i, s in enumerate(vocab) if s in keep_set)
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        src_ids, tgt_ids = train_pt["src_ids"], train_pt["tgt_ids"]
        src_ok = torch.zeros(len(src_ids), dtype=torch.bool)
        tgt_ok = torch.zeros(len(tgt_ids), dtype=torch.bool)
        for ki in keep_idx:
            src_ok |= src_ids == ki
            tgt_ok |= tgt_ids == ki
        mask = src_ok & tgt_ok
        train_pt = {
            **train_pt,
            "Z_A": train_pt["Z_A"][mask],
            "Z_B": train_pt["Z_B"][mask],
            "src_ids": torch.tensor([old_to_new[s.item()] for s in src_ids[mask]]),
            "tgt_ids": torch.tensor([old_to_new[t.item()] for t in tgt_ids[mask]]),
            "scanner_vocab": [vocab[i] for i in keep_idx],
        }
        print(
            f"filter_scanners={cfg.filter_scanners}: {mask.sum():,}/{len(mask):,} rows kept"
        )

    norm_stats = NormStats.from_data(train_pt["Z_A"].float(), train_pt["Z_B"].float())

    multitarget = "src_ids" in train_pt and "tgt_ids" in train_pt
    if multitarget:
        z_a = norm_stats.normalize_a(train_pt["Z_A"].float()).to(train_pt["Z_A"].dtype)
        z_b = norm_stats.normalize_b(train_pt["Z_B"].float()).to(train_pt["Z_B"].dtype)
        vocab = train_pt["scanner_vocab"]
        cond_dim = len(vocab)
        if cfg.val_data is not None:
            va, vb, vs, vt, _ = load_multitarget_tensors(cfg.val_data)
            train_ds = MultiTargetPairedDataset(
                z_a, z_b, train_pt["src_ids"], train_pt["tgt_ids"]
            )
            val_ds = MultiTargetPairedDataset(
                norm_stats.normalize_a(va.float()).to(va.dtype),
                norm_stats.normalize_b(vb.float()).to(vb.dtype),
                vs,
                vt,
            )
        else:
            n_val = max(1, int(len(z_a) * cfg.val_split))
            full_ds = MultiTargetPairedDataset(
                z_a, z_b, train_pt["src_ids"], train_pt["tgt_ids"]
            )
            train_ds, val_ds = random_split(
                full_ds,
                [len(z_a) - n_val, n_val],
                generator=torch.Generator().manual_seed(cfg.seed),
            )
    else:
        z_a = norm_stats.normalize_a(train_pt["Z_A"].float()).to(train_pt["Z_A"].dtype)
        z_b = norm_stats.normalize_b(train_pt["Z_B"].float()).to(train_pt["Z_B"].dtype)
        vocab, cond_dim = [], 0
        if cfg.val_data is not None:
            va, vb = load_paired_tensors(cfg.val_data)
            train_ds = PairedFeatureDataset(z_a, z_b)
            val_ds = PairedFeatureDataset(
                norm_stats.normalize_a(va.float()).to(va.dtype),
                norm_stats.normalize_b(vb.float()).to(vb.dtype),
            )
        else:
            n_val = max(1, int(len(z_a) * cfg.val_split))
            full_ds = PairedFeatureDataset(z_a, z_b)
            train_ds, val_ds = random_split(
                full_ds,
                [len(z_a) - n_val, n_val],
                generator=torch.Generator().manual_seed(cfg.seed),
            )

    if 0.0 < cfg.data_fraction < 1.0:
        n_total = len(train_ds)
        n_keep = max(1, int(n_total * cfg.data_fraction))
        rng = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(n_total, generator=rng)[:n_keep].tolist()
        train_ds = Subset(train_ds, idx)
        print(
            f"data_fraction={cfg.data_fraction}: using {n_keep:,}/{n_total:,} train samples"
        )

    return train_ds, val_ds, cond_dim, vocab, norm_stats


def _unpack_batch(batch, cond_dim: int, device: torch.device):
    if len(batch) == 4:
        z_src, z_tgt, src_id, tgt_id = batch
        z_src = z_src.to(device=device, dtype=torch.float32)
        z_tgt = z_tgt.to(device=device, dtype=torch.float32)
        cond_fwd = F.one_hot(tgt_id.to(device), num_classes=cond_dim).float()
        cond_bwd = F.one_hot(src_id.to(device), num_classes=cond_dim).float()
        return z_src, z_tgt, cond_fwd, cond_bwd, src_id, tgt_id
    else:
        z_src, z_tgt = batch
        return (
            z_src.to(device=device, dtype=torch.float32),
            z_tgt.to(device=device, dtype=torch.float32),
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: CycleGANConfig) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt = torch.load(Path(cfg.paired_data), weights_only=True)
    if _is_mmap_format(pt):
        print(
            "Detected mmap metadata format — reading features.npy via memory mapping."
        )
        train_ds, val_ds, cond_dim, scanner_vocab, norm_stats = _build_mmap_datasets(
            cfg, pt
        )
    else:
        print("Detected legacy tensor format — loading full feature tensors into RAM.")
        train_ds, val_ds, cond_dim, scanner_vocab, norm_stats = _build_tensor_datasets(
            cfg, pt
        )
        del pt
    norm_stats.save(output_dir / "norm_stats.pt")

    if cond_dim:
        print(f"Multi-target mode: cond_dim={cond_dim}, vocab={scanner_vocab}")
        with open(output_dir / "scanner_vocab.json", "w") as f:
            json.dump(scanner_vocab, f)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    dim = cfg.feature_dim
    hidden_dims = list(cfg.hidden_dims) if cfg.hidden_dims else [dim, dim]
    discriminator_hidden_dim = cfg.discriminator_hidden_dim or dim

    if cond_dim:
        g = FeatureGenerator(dim, hidden_dims, cond_dim, cfg.use_film).to(device)
        d = FeatureDiscriminator(
            dim,
            discriminator_hidden_dim,
            cond_dim,
            dropout=cfg.discriminator_dropout,
            use_spectral_norm=cfg.discriminator_use_spectral_norm,
        ).to(device)
        g_fwd = g_bwd = g
        d_fwd = d_bwd = d
    else:
        g_fwd = FeatureGenerator(dim, hidden_dims).to(device)
        g_bwd = FeatureGenerator(dim, hidden_dims).to(device)
        d_fwd = FeatureDiscriminator(
            dim,
            discriminator_hidden_dim,
            dropout=cfg.discriminator_dropout,
            use_spectral_norm=cfg.discriminator_use_spectral_norm,
        ).to(device)
        d_bwd = FeatureDiscriminator(
            dim,
            discriminator_hidden_dim,
            dropout=cfg.discriminator_dropout,
            use_spectral_norm=cfg.discriminator_use_spectral_norm,
        ).to(device)

    g_params = list(set(g_fwd.parameters()) | set(g_bwd.parameters()))
    d_params = list(set(d_fwd.parameters()) | set(d_bwd.parameters()))
    n_g = sum(p.numel() for p in g_params)
    n_d = sum(p.numel() for p in d_params)
    print(f"Generator params: {n_g:,}   Discriminator params: {n_d:,}")

    if cfg.generator_init_checkpoint:
        print(f"Warm-initializing generator(s) from: {cfg.generator_init_checkpoint}")
        _init_generator_from_ckpt(g_fwd, cfg.generator_init_checkpoint, ckpt_key="g_ab")
        if g_fwd is not g_bwd:
            _init_generator_from_ckpt(
                g_bwd, cfg.generator_init_checkpoint, ckpt_key="g_ba"
            )

    lr_d = cfg.lr_d if cfg.lr_d is not None else cfg.lr
    opt_g = torch.optim.Adam(g_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(d_params, lr=lr_d, betas=(cfg.beta1, cfg.beta2))

    def _lr_lambda(epoch: int) -> float:
        if cfg.lr_decay_epochs <= 0:
            return 1.0
        stable = cfg.epochs - cfg.lr_decay_epochs
        return (
            1.0
            if epoch < stable
            else max(0.0, 1.0 - (epoch - stable) / cfg.lr_decay_epochs)
        )

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, _lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, _lr_lambda)

    buf_fwd = ReplayBuffer(cfg.replay_buffer_size)
    buf_bwd = ReplayBuffer(cfg.replay_buffer_size)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- Wandb ---
    run = _init_wandb(cfg, n_g, n_d, scanner_vocab)

    history: list[dict] = []
    best_val_l1 = float("inf")
    global_step = 0

    epoch_bar = _progress(range(1, cfg.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        g_fwd.train()
        g_bwd.train()
        d_fwd.train()
        d_bwd.train()
        epoch_sums: dict[str, float] = defaultdict(float)
        n_batches = 0

        if cfg.lambda_adv_ramp_epochs > 0:
            effective_lambda_adv = cfg.lambda_adv * min(
                1.0, epoch / cfg.lambda_adv_ramp_epochs
            )
        else:
            effective_lambda_adv = cfg.lambda_adv

        batch_bar = _progress(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            leave=False,
            unit="batch",
        )
        for batch in batch_bar:
            real_src, real_tgt, cond_fwd, cond_bwd, _, _ = _unpack_batch(
                batch, cond_dim, device
            )

            opt_g.zero_grad()
            loss_g, g_components = _generator_step(
                real_src,
                real_tgt,
                g_fwd,
                g_bwd,
                d_fwd,
                d_bwd,
                cfg,
                is_paired=True,
                cond_fwd=cond_fwd,
                cond_bwd=cond_bwd,
                alpha=cfg.loss_alpha,
                lambda_adv_override=effective_lambda_adv,
            )
            loss_g.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(g_params, cfg.grad_clip_norm)
            opt_g.step()

            opt_d.zero_grad()
            with torch.no_grad():
                fake_tgt = g_fwd(real_src, cond_fwd)
                fake_src = g_bwd(real_tgt, cond_bwd)
            loss_d, d_components = _discriminator_step(
                real_src,
                real_tgt,
                fake_src,
                fake_tgt,
                d_fwd,
                d_bwd,
                buf_fwd,
                buf_bwd,
                cond_fwd=cond_fwd,
                cond_bwd=cond_bwd,
                dim=dim,
            )
            loss_d.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(d_params, cfg.grad_clip_norm)
            opt_d.step()

            for k, v in {**g_components, **d_components}.items():
                epoch_sums[k] += v
            n_batches += 1
            global_step += 1

            if _HAS_TQDM:
                batch_bar.set_postfix(
                    G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}"
                )

        sched_g.step()
        sched_d.step()

        # Per-epoch training averages
        train_metrics = {f"train/{k}": v / n_batches for k, v in epoch_sums.items()}

        # Validation
        val_metrics, heatmap_data = _validate(
            val_loader, g_fwd, g_bwd, device, cond_dim
        )
        val_l1 = (val_metrics["val/l1_fwd"] + val_metrics["val/l1_bwd"]) / 2

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        if epoch % cfg.log_interval == 0 or epoch == 1:
            msg = (
                f"Epoch {epoch:4d}/{cfg.epochs}  "
                f"G={epoch_sums['loss_g']/n_batches:.4f}  "
                f"D={epoch_sums['loss_d']/n_batches:.4f}  "
                f"D_acc={epoch_sums['d_acc']/n_batches:.2%}  "
                f"(R={epoch_sums['d_acc_real']/n_batches:.2%} "
                f"F={epoch_sums['d_acc_fake']/n_batches:.2%})  "
                f"val/l1={val_l1:.4f}  "
                f"val/cos={val_metrics['val/cos_fwd']:.4f}  "
                f"val/rel={val_metrics.get('val/relative_improvement', float('nan')):.4f}  "
                f"lr_g={sched_g.get_last_lr()[0]:.2e}"
            )
            if _HAS_TQDM:
                epoch_bar.write(msg)
            else:
                print(msg)

        if run is not None:
            # Single log per epoch at the last global_step — avoids duplicate W&B panels
            # from mixed per-batch / per-epoch commits.
            wandb_row = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr_g": sched_g.get_last_lr()[0],
                "lr_d": sched_d.get_last_lr()[0],
                "lambda_adv_effective": effective_lambda_adv,
            }
            if heatmap_data and scanner_vocab:
                heatmap_imgs = _build_heatmap_images(heatmap_data, scanner_vocab)
                wandb_row.update(heatmap_imgs)
            run.log(wandb_row, step=global_step, commit=True)

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            save_checkpoint(
                output_dir, epoch, g_fwd, g_bwd, d_fwd, d_bwd, opt_g, opt_d, tag="best"
            )

        if cfg.save_interval and epoch % cfg.save_interval == 0:
            save_checkpoint(output_dir, epoch, g_fwd, g_bwd, d_fwd, d_bwd, opt_g, opt_d)

    save_checkpoint(
        output_dir, cfg.epochs, g_fwd, g_bwd, d_fwd, d_bwd, opt_g, opt_d, tag="final"
    )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val paired L1: {best_val_l1:.4f}")

    if run is not None:
        run.finish()


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _relational_loss_grouped(
    x: torch.Tensor, y: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    """Relational loss averaged over per-scanner groups.

    Splits the batch by scanner ID so pairwise similarity comparisons stay
    within a single domain.  Groups with fewer than 2 samples are skipped.
    Falls back to a scalar zero if every group is too small.
    """
    losses = []
    for gid in group_ids.unique():
        mask = group_ids == gid
        if mask.sum() < 2:
            continue
        losses.append(relational_loss(x[mask], y[mask]))
    if not losses:
        return x.new_zeros(1).squeeze()
    return torch.stack(losses).mean()


def _generator_step(
    real_src,
    real_tgt,
    g_fwd: FeatureGenerator,
    g_bwd: FeatureGenerator,
    d_fwd: FeatureDiscriminator,
    d_bwd: FeatureDiscriminator,
    cfg: CycleGANConfig,
    is_paired: bool,
    cond_fwd: Optional[torch.Tensor],
    cond_bwd: Optional[torch.Tensor],
    alpha: float = 1.0,
    lambda_adv_override: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    fake_tgt = g_fwd(real_src, cond_fwd)
    fake_src = g_bwd(real_tgt, cond_bwd)
    rec_src = g_bwd(fake_tgt, cond_bwd)
    rec_tgt = g_fwd(fake_src, cond_fwd)
    idt_tgt = g_fwd(real_tgt, cond_fwd)
    idt_src = g_bwd(real_src, cond_bwd)

    _lambda_adv = (
        lambda_adv_override if lambda_adv_override is not None else cfg.lambda_adv
    )
    loss_adv = _lambda_adv * (
        adversarial_loss(d_fwd(fake_tgt, cond_fwd), target_is_real=True)
        + adversarial_loss(d_bwd(fake_src, cond_bwd), target_is_real=True)
    )
    loss_cyc = cfg.lambda_cyc * (
        cycle_loss(rec_src, real_src, alpha) + cycle_loss(rec_tgt, real_tgt, alpha)
    )
    loss_idt = cfg.lambda_idt * (
        identity_loss(idt_tgt, real_tgt, alpha)
        + identity_loss(idt_src, real_src, alpha)
    )
    loss_paired = (
        cfg.lambda_paired
        * (
            paired_loss(fake_tgt, real_tgt, alpha)
            + paired_loss(fake_src, real_src, alpha)
        )
        if is_paired
        else real_src.new_zeros(1).squeeze()
    )
    loss_coral = (
        cfg.lambda_coral
        * (coral_loss(fake_tgt, real_tgt) + coral_loss(fake_src, real_src))
        if cfg.lambda_coral > 0
        else real_src.new_zeros(1).squeeze()
    )
    if cfg.lambda_rel > 0:
        if cond_fwd is not None:
            loss_rel = cfg.lambda_rel * (
                _relational_loss_grouped(real_src, fake_src, cond_bwd.argmax(-1))
                + _relational_loss_grouped(real_tgt, fake_tgt, cond_fwd.argmax(-1))
            )
        else:
            loss_rel = cfg.lambda_rel * (
                relational_loss(real_src, fake_src)
                + relational_loss(real_tgt, fake_tgt)
            )
    else:
        loss_rel = real_src.new_zeros(1).squeeze()

    total = loss_adv + loss_cyc + loss_idt + loss_paired + loss_coral + loss_rel
    return total, {
        "loss_g": total.item(),
        "loss_adv_g": loss_adv.item(),
        "loss_cyc": loss_cyc.item(),
        "loss_idt": loss_idt.item(),
        "loss_paired": loss_paired.item(),
        "loss_coral": loss_coral.item(),
        "loss_rel": loss_rel.item(),
    }


def _discriminator_step(
    real_src,
    real_tgt,
    fake_src,
    fake_tgt,
    d_fwd: FeatureDiscriminator,
    d_bwd: FeatureDiscriminator,
    buf_fwd: ReplayBuffer,
    buf_bwd: ReplayBuffer,
    cond_fwd: Optional[torch.Tensor],
    cond_bwd: Optional[torch.Tensor],
    dim: int,
) -> Tuple[torch.Tensor, dict]:
    def replay(buf, fake, cond):
        if cond is not None:
            stored = torch.cat([fake.detach(), cond], dim=-1)
            popped = buf.push_and_pop(stored)
            return popped[:, :dim], popped[:, dim:]
        return buf.push_and_pop(fake.detach()), None

    fake_tgt_r, cond_fwd_r = replay(buf_fwd, fake_tgt, cond_fwd)
    fake_src_r, cond_bwd_r = replay(buf_bwd, fake_src, cond_bwd)

    d_real_tgt = d_fwd(real_tgt, cond_fwd)
    d_fake_tgt = d_fwd(fake_tgt_r, cond_fwd_r)
    d_real_src = d_bwd(real_src, cond_bwd)
    d_fake_src = d_bwd(fake_src_r, cond_bwd_r)

    loss_fwd = 0.5 * (
        adversarial_loss(d_real_tgt, target_is_real=True)
        + adversarial_loss(d_fake_tgt, target_is_real=False)
    )
    loss_bwd = 0.5 * (
        adversarial_loss(d_real_src, target_is_real=True)
        + adversarial_loss(d_fake_src, target_is_real=False)
    )
    total = loss_fwd + loss_bwd

    # Discriminator accuracy: LSGAN threshold at 0.5 (real > 0.5, fake < 0.5)
    with torch.no_grad():
        acc_real = torch.cat([d_real_tgt, d_real_src]).gt(0.5).float().mean().item()
        acc_fake = torch.cat([d_fake_tgt, d_fake_src]).lt(0.5).float().mean().item()

    return total, {
        "loss_d": total.item(),
        "d_acc_real": acc_real,
        "d_acc_fake": acc_fake,
        "d_acc": (acc_real + acc_fake) / 2,
    }


# ---------------------------------------------------------------------------
# Validation + per-pair heatmap accumulation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _validate(
    loader: DataLoader,
    g_fwd: FeatureGenerator,
    g_bwd: FeatureGenerator,
    device: torch.device,
    cond_dim: int,
) -> Tuple[dict, Optional[dict]]:
    g_fwd.eval()
    g_bwd.eval()

    l1_fwd = l1_bwd = cyc = 0.0
    cos_fwd_sum = cos_bwd_sum = cos_baseline_sum = cos_idt_sum = 0.0
    n = 0

    # Per-pair accumulators for heatmaps (multi-target only)
    pair_cos_fwd: dict = {}  # (src, tgt) → [cos]
    pair_cos_orig: dict = {}  # (src, tgt) → [cos baseline]
    id_cos_tgt: dict = {}  # tgt → [identity cos]

    for batch in loader:
        real_src, real_tgt, cond_fwd, cond_bwd, src_ids_t, tgt_ids_t = _unpack_batch(
            batch, cond_dim, device
        )

        fake_tgt = g_fwd(real_src, cond_fwd)
        fake_src = g_bwd(real_tgt, cond_bwd)
        rec_src = g_bwd(fake_tgt, cond_bwd)
        rec_tgt = g_fwd(fake_src, cond_fwd)
        idt_tgt = g_fwd(real_tgt, cond_fwd)

        cos_fwd_s = F.cosine_similarity(fake_tgt, real_tgt)
        cos_bwd_s = F.cosine_similarity(fake_src, real_src)
        cos_base_s = F.cosine_similarity(real_src, real_tgt)
        cos_idt_s = F.cosine_similarity(idt_tgt, real_tgt)

        l1_fwd += F.l1_loss(fake_tgt, real_tgt).item()
        l1_bwd += F.l1_loss(fake_src, real_src).item()
        cyc += (F.l1_loss(rec_src, real_src) + F.l1_loss(rec_tgt, real_tgt)).item() / 2
        cos_fwd_sum += cos_fwd_s.mean().item()
        cos_bwd_sum += cos_bwd_s.mean().item()
        cos_baseline_sum += cos_base_s.mean().item()
        cos_idt_sum += cos_idt_s.mean().item()
        n += 1

        if cond_dim > 0 and src_ids_t is not None:
            cf = cos_fwd_s.cpu().tolist()
            cb = cos_base_s.cpu().tolist()
            ci = cos_idt_s.cpu().tolist()
            for i, (si, ti) in enumerate(zip(src_ids_t.tolist(), tgt_ids_t.tolist())):
                pair_cos_fwd.setdefault((si, ti), []).append(cf[i])
                pair_cos_orig.setdefault((si, ti), []).append(cb[i])
                id_cos_tgt.setdefault(ti, []).append(ci[i])

    g_fwd.train()
    g_bwd.train()

    cos_fwd_mean = cos_fwd_sum / n
    cos_orig_mean = cos_baseline_sum / n
    metrics = {
        "val/l1_fwd": l1_fwd / n,
        "val/l1_bwd": l1_bwd / n,
        "val/cyc": cyc / n,
        "val/cos_fwd": cos_fwd_mean,
        "val/cos_bwd": cos_bwd_sum / n,
        "val/cos_baseline": cos_orig_mean,
        "val/cos_identity": cos_idt_sum / n,
    }
    if cos_orig_mean < 1.0 - 1e-6:
        metrics["val/relative_improvement"] = (cos_fwd_mean - cos_orig_mean) / (
            1.0 - cos_orig_mean
        )

    heatmap_data = (
        {
            "pair_cos_fwd": pair_cos_fwd,
            "pair_cos_orig": pair_cos_orig,
            "id_cos_tgt": id_cos_tgt,
        }
        if cond_dim > 0
        else None
    )
    return metrics, heatmap_data


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def _init_wandb(cfg: CycleGANConfig, n_g: int, n_d: int, scanner_vocab: List[str]):
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — logging disabled. pip install wandb")
        return None
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        tags=cfg.wandb_tags or [],
        dir=cfg.output_dir,
        config={
            **asdict(cfg),
            "generator_params": n_g,
            "discriminator_params": n_d,
            "scanner_vocab": scanner_vocab,
        },
    )
    return run


def _build_heatmap_images(heatmap_data: dict, scanner_vocab: List[str]) -> dict:
    """Build scanner-pair cosine heatmaps and return as wandb.Image objects."""
    try:
        import wandb
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    pair_cos_fwd = heatmap_data["pair_cos_fwd"]
    pair_cos_orig = heatmap_data["pair_cos_orig"]
    id_cos_tgt = heatmap_data["id_cos_tgt"]
    n = len(scanner_vocab)

    # pred matrix: off-diagonal = G(src, tgt) cosine; diagonal = identity pass
    pred_mat = np.full((n, n), np.nan)
    for (si, ti), vals in pair_cos_fwd.items():
        pred_mat[si, ti] = float(np.mean(vals))
    for ti, vals in id_cos_tgt.items():
        pred_mat[ti, ti] = float(np.mean(vals))

    # baseline matrix: cosine(src_feat, tgt_feat) without model; diagonal = 1
    orig_mat = np.full((n, n), np.nan)
    for i in range(n):
        orig_mat[i, i] = 1.0
    for (si, ti), vals in pair_cos_orig.items():
        orig_mat[si, ti] = float(np.mean(vals))

    diff_mat = np.full((n, n), np.nan)
    rel_mat = np.full((n, n), np.nan)
    for si in range(n):
        for ti in range(n):
            if si == ti:
                continue
            p, o = pred_mat[si, ti], orig_mat[si, ti]
            if not (np.isnan(p) or np.isnan(o)):
                diff_mat[si, ti] = p - o
                gap = 1.0 - o
                if gap > 1e-6:
                    rel_mat[si, ti] = (p - o) / gap

    heatmaps = [
        ("val/scanner_pred_heatmap", pred_mat, "pred cosine (diag = identity)", False),
        ("val/scanner_baseline_heatmap", orig_mat, "baseline cosine (no model)", False),
        ("val/scanner_diff_heatmap", diff_mat, "improvement (pred − baseline)", True),
        ("val/scanner_rel_heatmap", rel_mat, "relative improvement", True),
    ]

    result = {}
    for key, mat, title, diverging in heatmaps:
        finite = mat[~np.isnan(mat)]
        if diverging:
            abs_max = max(float(np.abs(finite).max()), 1e-6) if finite.size > 0 else 0.1
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = max(float(finite.min()) - 0.02, 0.0) if finite.size > 0 else 0.7
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(max(4, n + 1), max(3, n)))
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(scanner_vocab, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(scanner_vocab, fontsize=8)
        ax.set_xlabel("Target scanner")
        ax.set_ylabel("Source scanner")
        ax.set_title(title)

        span = max(vmax - vmin, 1e-6)
        for si in range(n):
            for ti in range(n):
                v = mat[si, ti]
                if not np.isnan(v):
                    brightness = (v - vmin) / span
                    txt_color = "black" if 0.3 < brightness < 0.75 else "white"
                    ax.text(
                        ti,
                        si,
                        f"{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=txt_color,
                    )
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        result[key] = wandb.Image(fig)
        plt.close(fig)

    # Per-pair bucket breakdown: same_scanner / same_vendor / diff_vendor
    bucket_cos: dict[str, list[float]] = {}
    for (si, ti), vals in pair_cos_fwd.items():
        if si == ti:
            bucket = "same_scanner"
        else:
            sv = _SCANNER_VENDOR.get(scanner_vocab[si], "")
            tv = _SCANNER_VENDOR.get(scanner_vocab[ti], "")
            bucket = "same_vendor" if sv == tv else "diff_vendor"
        bucket_cos.setdefault(bucket, []).extend(vals)
    for bucket, vals in bucket_cos.items():
        result[f"val/cos_{bucket}"] = float(np.mean(vals))

    return result


# ---------------------------------------------------------------------------
# Config loading & main
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --config: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = build_arg_parser().parse_args()

    yaml_vals: dict = {}
    if args.config is not None:
        yaml_vals = _load_yaml_config(args.config)

    field_names = {f.name for f in fields(CycleGANConfig)}
    cfg_kwargs: dict = {k: v for k, v in yaml_vals.items() if k in field_names}

    cli_defaults = build_arg_parser().parse_args([])
    for f in fields(CycleGANConfig):
        cli_val = getattr(args, f.name, None)
        default_val = getattr(cli_defaults, f.name, None)
        if cli_val is not None and cli_val != default_val:
            cfg_kwargs[f.name] = cli_val

    cfg = CycleGANConfig(**cfg_kwargs)
    if not cfg.paired_data:
        raise ValueError("paired_data must be set via --paired_data or the YAML config")

    train(cfg)


if __name__ == "__main__":
    main()
