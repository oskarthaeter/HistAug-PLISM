"""
Generate cosine-similarity heatmaps for a single example tile.

Loads pre-extracted features for a specific tile (by row index from the filename prefix),
applies the per-(src, tgt) best augmentation parameters via the HistAug model, and
produces cosine-similarity and improvement heatmaps for that one tile.

Usage:
    python src/histaug/scripts/tile_heatmap.py \
        --log-dir  src/histaug/logs/aug_sweep_virchow2_v2 \
        --tile-token  tile_16_104_129 \
        --staining  GV \
        --out  src/histaug/logs/aug_sweep_virchow2_v2/staining_example_grids/GV_pair_best_heatmaps.png
"""

from __future__ import annotations

import argparse
import csv
import math
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


# ── helpers ──────────────────────────────────────────────────────────────────

DISCRETE_BINARY = {"h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"}
DISCRETE_MULTI  = {"rotation", "crop"}
CONTINUOUS      = {"brightness", "contrast", "saturation", "hue", "gamma", "hed"}
N_COORD_COLS    = 3


def resolve_histaug_module(model):
    if hasattr(model, "transform_embeddings"):
        return model
    nested = getattr(model, "histaug", None)
    if nested is not None and hasattr(nested, "transform_embeddings"):
        return nested
    raise ValueError("Model has no .transform_embeddings")


def get_transform_names(histaug_module):
    return list(getattr(histaug_module, "transform_embeddings").keys())


def build_aug_params(
    transform_names, continuous_values: Dict[str, float], device, histaug_module=None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    te = None
    if histaug_module is not None:
        te = getattr(histaug_module, "transform_embeddings", None)

    params = {}
    for pos, name in enumerate(transform_names):
        if name in DISCRETE_MULTI:
            val = torch.zeros(1, dtype=torch.int64, device=device)
        elif name in DISCRETE_BINARY:
            val = torch.zeros(1, dtype=torch.int32, device=device)
        elif name in CONTINUOUS:
            val = torch.tensor([float(continuous_values.get(name, 0.0))],
                               dtype=torch.float32, device=device)
        else:
            emb = te[name] if (te is not None and name in te) else None
            if isinstance(emb, nn.Embedding):
                val = torch.zeros(1, dtype=torch.int64, device=device)
            elif emb is not None:
                val = torch.tensor([float(continuous_values.get(name, 0.0))],
                                   dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unknown transform: {name}")
        position = torch.tensor([pos], dtype=torch.long, device=device)
        params[name] = (val, position)
    return params


def make_heatmaps(
    mat_cos: np.ndarray,
    mat_base: np.ndarray,
    scanner_names,
    title_prefix: str,
    out_path: Path,
) -> None:
    mat_imp = mat_cos - mat_base

    panels = [
        (mat_base, f"{title_prefix}\nBaseline cosine (no aug)", "RdYlGn", False),
        (mat_cos,  f"{title_prefix}\nBest cosine (HistAug)",    "RdYlGn", False),
        (mat_imp,  f"{title_prefix}\nImprovement over baseline", "RdYlGn", True),
    ]

    n = len(scanner_names)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    for ax, (mat, title, cmap, diverging) in zip(axes, panels):
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        if math.isnan(vmin) or math.isnan(vmax):
            vmin, vmax = 0.0, 1.0
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-3
        if diverging:
            abs_max = max(abs(vmin), abs(vmax), 1e-4)
            vmin, vmax = -abs_max, abs_max

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Target scanner", fontsize=9)
        ax.set_ylabel("Source scanner", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(scanner_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(scanner_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                            fontsize=6.5, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir",    type=Path, required=True)
    p.add_argument("--tile-token", type=str,  default="tile_16_104_129",
                   help="Coordinate token from the tile filename, e.g. tile_16_104_129")
    p.add_argument("--staining",   type=str,  default="GV")
    p.add_argument("--out",        type=Path, default=None,
                   help="Output PNG path (default: <log_dir>/staining_example_grids/"
                        "<staining>_pair_best_heatmaps.png)")
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir

    # Load run config
    cfg = json.loads((log_dir / "run_config_and_best.json").read_text())
    features_root = Path(cfg["features_root"])
    model_id      = cfg["model_id"]
    emb_dim       = int(cfg["embedding_dim"])
    sweep_aug_names = cfg["sweep_aug_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = args.out or (
        log_dir / "staining_example_grids" /
        f"{args.staining}_pair_best_heatmaps.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load per-(staining, src, tgt) best aug params
    best_aug: Dict[Tuple[str, str], Dict[str, float]] = {}
    with open(log_dir / "staining_pair_best_aug.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["staining_group"] != args.staining:
                continue
            key = (row["src_scanner"], row["tgt_scanner"])
            best_aug[key] = {n: float(row[n]) for n in sweep_aug_names}

    if not best_aug:
        raise ValueError(f"No rows for staining '{args.staining}' in staining_pair_best_aug.csv")

    scanner_names = sorted({s for pair in best_aug for s in pair})
    print(f"Scanners: {scanner_names}")

    # Discover tile row index from features dir: filename prefix == row index
    tile_token = args.tile_token
    ref_slide_dir = next(
        (features_root / f"{args.staining}_{sc}_to_GMH_S60.tif"
         for sc in scanner_names
         if (features_root / f"{args.staining}_{sc}_to_GMH_S60.tif").exists()),
        None,
    )
    if ref_slide_dir is None:
        raise FileNotFoundError(f"No feature dir found for staining {args.staining}")

    # Find tile index from jpeg directory (same structure: prefix == row index)
    jpeg_root = Path("/workspaces/HistAug-PLISM/plism_jpeg")
    ref_jpeg_dir = next(
        (jpeg_root / f"{args.staining}_{sc}_to_GMH_S60.tif"
         for sc in scanner_names
         if (jpeg_root / f"{args.staining}_{sc}_to_GMH_S60.tif").exists()),
        None,
    )
    if ref_jpeg_dir is None:
        raise FileNotFoundError(f"No jpeg dir found for staining {args.staining}")

    matches = sorted(ref_jpeg_dir.glob(f"*__{tile_token}.jpg"))
    if not matches:
        raise FileNotFoundError(f"Tile '{tile_token}' not found in {ref_jpeg_dir}")
    tile_row_idx = int(matches[0].name.split("__")[0])
    print(f"Tile '{tile_token}' -> row index {tile_row_idx}")

    # Load one embedding per scanner
    embeddings: Dict[str, torch.Tensor] = {}
    for sc in scanner_names:
        feat_path = features_root / f"{args.staining}_{sc}_to_GMH_S60.tif" / "features.npy"
        arr = np.load(feat_path, mmap_mode="r")
        emb = arr[tile_row_idx, N_COORD_COLS : N_COORD_COLS + emb_dim].astype(np.float32)
        embeddings[sc] = torch.from_numpy(emb.copy()).unsqueeze(0).to(device)
    print(f"Loaded embeddings for {len(embeddings)} scanners, dim={emb_dim}")

    # Load model
    print(f"Loading model: {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    histaug_module  = resolve_histaug_module(model)
    transform_names = get_transform_names(histaug_module)

    # Compute heatmap entries
    n = len(scanner_names)
    name_to_idx = {sc: i for i, sc in enumerate(scanner_names)}
    mat_cos  = np.full((n, n), np.nan, dtype=np.float32)
    mat_base = np.full((n, n), np.nan, dtype=np.float32)

    with torch.inference_mode():
        for (src, tgt), aug_values in best_aug.items():
            si, ti = name_to_idx[src], name_to_idx[tgt]
            feat_src = embeddings[src]
            feat_tgt = embeddings[tgt]

            # Baseline
            base_cos = float(F.cosine_similarity(feat_src, feat_tgt, dim=-1).item())
            mat_base[si, ti] = base_cos

            # HistAug prediction
            aug_params = build_aug_params(
                transform_names, aug_values, device, histaug_module
            )
            pred = model(feat_src, aug_params)
            pred_cos = float(F.cosine_similarity(pred, feat_tgt, dim=-1).item())
            mat_cos[si, ti] = pred_cos

    title_prefix = (
        f"Staining: {args.staining} | Tile: {tile_token}\n"
        f"(row {tile_row_idx}, best aug per pair from sweep)"
    )
    make_heatmaps(mat_cos, mat_base, scanner_names, title_prefix, out_path)


if __name__ == "__main__":
    main()
