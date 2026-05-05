"""
Config-driven augmentation sweep for scanner transfer in latent space using a
pretrained HistAug model from Hugging Face.

This script:
1. Loads paired pre-extracted features from a PLISM-style feature root.
2. Builds source->target slide pairs (scanner/staining constrained).
3. Sweeps configured continuous augmentation values (e.g., HED, saturation,
   brightness) while forcing all non-swept transforms to identity.
4. Reports global and per-(src_scanner, tgt_scanner) best augmentation combo by
   cosine to target.
5. Optionally runs separate summaries per staining group and renders per-staining
   7x7 scanner example grids.

Expected feature layout:
    <features_root>/<staining>_<scanner>_to_<ref>.tif/features.npy

Each features.npy is expected to have shape (N, 3 + D), where first 3 columns are
coords and remaining D columns are embeddings.

Example:
    python src/histaug/scripts/sweep_histaug_hed_transfer.py \
        --config src/histaug/config/sweep_histaug_multiaug.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
from PIL import Image
from torch import nn
from transformers import AutoModel

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DISCRETE_BINARY = {"h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"}
DISCRETE_MULTI = {"rotation", "crop"}
CONTINUOUS = {"brightness", "contrast", "saturation", "hue", "gamma", "hed"}
SUPPORTED_SWEEP_AUGS = ("hed", "saturation", "brightness", "hue", "gamma", "contrast")
N_COORD_COLS = 3


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    staining: str
    scanner: str
    features_path: Path


@dataclass(frozen=True)
class PairRecord:
    features_path_a: Path
    features_path_b: Path
    scanner_a: str
    scanner_b: str
    staining_a: str
    staining_b: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep HistAug continuous augmentations for scanner-transfer "
            "approximation from a YAML/JSON config."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument(
        "--max-patches-per-pair",
        type=int,
        default=None,
        help=(
            "Optional CLI override for config.max_patches_per_pair. "
            "Use very small values (e.g., 1-5) for quick sensitivity checks."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Optional cap on number of slide pairs to evaluate after filtering/pairing. "
            "Useful with tiny tile counts for fast exploratory sweeps."
        ),
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        cfg = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML config. Install pyyaml or use JSON config."
            )
        cfg = yaml.safe_load(text)
    else:
        raise ValueError("Config extension must be .json, .yaml, or .yml")

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    return cfg


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def _parse_staining_scanner(slide_id: str) -> Tuple[str, str]:
    tokens = slide_id.split("_")
    if len(tokens) < 2:
        raise ValueError(
            f"Invalid slide id '{slide_id}'. Expected '<staining>_<scanner>[_...]'."
        )
    return tokens[0], tokens[1]


def discover_slides(root: Path) -> List[SlideRecord]:
    slides: List[SlideRecord] = []
    for slide_dir in sorted(root.iterdir()):
        if not slide_dir.is_dir():
            continue
        features_path = slide_dir / "features.npy"
        if not features_path.exists():
            continue
        dir_name = slide_dir.name
        slide_id = dir_name.split("_to_")[0] if "_to_" in dir_name else dir_name
        staining, scanner = _parse_staining_scanner(slide_id)
        slides.append(
            SlideRecord(
                slide_id=slide_id,
                staining=staining,
                scanner=scanner,
                features_path=features_path,
            )
        )
    if not slides:
        raise ValueError(f"No slide folders with features.npy found under {root}")
    return slides


def filter_slides(
    slides: Sequence[SlideRecord],
    scanners_allowlist: Sequence[str],
    stainings_allowlist: Sequence[str],
) -> List[SlideRecord]:
    scanners = {str(s) for s in scanners_allowlist if str(s).strip()}
    stainings = {str(s) for s in stainings_allowlist if str(s).strip()}
    out: List[SlideRecord] = []
    for s in slides:
        scanner_ok = (not scanners) or (s.scanner in scanners)
        staining_ok = (not stainings) or (s.staining in stainings)
        if scanner_ok and staining_ok:
            out.append(s)
    return out


def build_pairs(
    slides: Sequence[SlideRecord],
    allow_cross_staining: bool,
    allow_same_scanner: bool,
    symmetric: bool,
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    n = len(slides)
    for i in range(n):
        a = slides[i]
        j_range: Iterable[int] = range(n) if symmetric else range(i + 1, n)
        for j in j_range:
            if i == j:
                continue
            b = slides[j]
            same_staining = a.staining == b.staining
            same_scanner = a.scanner == b.scanner
            if not same_staining and not allow_cross_staining:
                continue
            if same_scanner and not allow_same_scanner:
                continue
            pairs.append(
                PairRecord(
                    features_path_a=a.features_path,
                    features_path_b=b.features_path,
                    scanner_a=a.scanner,
                    scanner_b=b.scanner,
                    staining_a=a.staining,
                    staining_b=b.staining,
                )
            )
    return pairs


def infer_embedding_dim(sample_features: Path) -> int:
    arr = np.load(sample_features, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] <= N_COORD_COLS:
        raise ValueError(f"Unexpected feature shape in {sample_features}: {arr.shape}")
    return int(arr.shape[1] - N_COORD_COLS)


def resolve_histaug_module(model):
    if hasattr(model, "transform_embeddings"):
        return model
    nested = getattr(model, "histaug", None)
    if nested is not None and hasattr(nested, "transform_embeddings"):
        return nested
    raise ValueError(
        "Loaded model does not expose transform embeddings. "
        "Expected a HistAug-compatible model or wrapper with .histaug."
    )


def get_transform_names(histaug_module) -> List[str]:
    if not hasattr(histaug_module, "transform_embeddings"):
        raise ValueError(
            "Loaded model does not expose transform embeddings. "
            "Expected a HistAug-compatible model."
        )
    return list(getattr(histaug_module, "transform_embeddings").keys())


def build_sweep_grid(
    config: Dict[str, Any],
    histaug_module,
) -> Tuple[
    List[str],
    List[Tuple[float, ...]],
    Dict[str, List[float]],
    Dict[str, Dict[str, float]],
]:
    sweep_cfg = config.get("sweep", {})
    if not isinstance(sweep_cfg, dict):
        raise ValueError("config.sweep must be a mapping")

    transforms_cfg = sweep_cfg.get("transforms", {})
    if not isinstance(transforms_cfg, dict) or not transforms_cfg:
        raise ValueError(
            "config.sweep.transforms must be a non-empty mapping. "
            "Expected entries like hed/saturation/brightness."
        )

    inferred_bounds: Dict[str, Tuple[float, float]] = {}
    tp = getattr(histaug_module, "transforms_parameters", None)
    if isinstance(tp, dict):
        for name in SUPPORTED_SWEEP_AUGS:
            val = tp.get(name, None)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                inferred_bounds[name] = (float(val[0]), float(val[1]))

    aug_names: List[str] = []
    values_by_aug: Dict[str, List[float]] = {}
    bounds_used: Dict[str, Dict[str, float]] = {}

    for name, aug_cfg in transforms_cfg.items():
        if name not in SUPPORTED_SWEEP_AUGS:
            raise ValueError(
                f"Unsupported sweep transform '{name}'. Supported: {SUPPORTED_SWEEP_AUGS}"
            )
        if not isinstance(aug_cfg, dict):
            raise ValueError(f"config.sweep.transforms.{name} must be a mapping")

        if "values" in aug_cfg:
            vals = [float(v) for v in aug_cfg["values"]]
            if not vals:
                raise ValueError(f"config.sweep.transforms.{name}.values is empty")
            values_by_aug[name] = vals
            bounds_used[name] = {"min": float(min(vals)), "max": float(max(vals))}
        else:
            min_v = aug_cfg.get("min", None)
            max_v = aug_cfg.get("max", None)
            steps = int(aug_cfg.get("steps", 0))

            if min_v is None or max_v is None:
                inferred = inferred_bounds.get(name, None)
                if inferred is None:
                    raise ValueError(
                        f"Missing min/max for {name}, and model has no inferred bounds"
                    )
                if min_v is None:
                    min_v = inferred[0]
                if max_v is None:
                    max_v = inferred[1]

            min_v = float(min_v)
            max_v = float(max_v)
            if min_v >= max_v:
                raise ValueError(
                    f"Invalid bounds for {name}: min ({min_v}) must be < max ({max_v})"
                )
            if steps < 2:
                raise ValueError(
                    f"config.sweep.transforms.{name}.steps must be >= 2 when using min/max"
                )

            vals = np.linspace(min_v, max_v, steps, dtype=np.float32).tolist()
            values_by_aug[name] = [float(v) for v in vals]
            bounds_used[name] = {"min": min_v, "max": max_v}

        aug_names.append(name)

    value_lists = [values_by_aug[n] for n in aug_names]
    combos = [tuple(float(v) for v in c) for c in itertools.product(*value_lists)]
    if not combos:
        raise ValueError("Sweep produced no parameter combinations")
    return aug_names, combos, values_by_aug, bounds_used


def build_aug_params(
    transform_names: Sequence[str],
    batch_size: int,
    continuous_values: Dict[str, float],
    device: torch.device,
    histaug_module=None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    aug_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    transform_embeddings = None
    if histaug_module is not None and hasattr(histaug_module, "transform_embeddings"):
        transform_embeddings = getattr(histaug_module, "transform_embeddings")

    for pos, name in enumerate(transform_names):
        if name in DISCRETE_MULTI:
            value = torch.zeros(batch_size, dtype=torch.int64, device=device)
        elif name in DISCRETE_BINARY:
            value = torch.zeros(batch_size, dtype=torch.int32, device=device)
        elif name in CONTINUOUS:
            base = float(continuous_values.get(name, 0.0))
            value = torch.full(
                (batch_size,),
                fill_value=base,
                dtype=torch.float32,
                device=device,
            )
        else:
            emb = None
            if transform_embeddings is not None and name in transform_embeddings:
                emb = transform_embeddings[name]

            if isinstance(emb, nn.Embedding):
                value = torch.zeros(batch_size, dtype=torch.int64, device=device)
            elif emb is not None:
                base = float(continuous_values.get(name, 0.0))
                value = torch.full(
                    (batch_size,),
                    fill_value=base,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown transform name: {name}")

        position = torch.full((batch_size,), pos, dtype=torch.long, device=device)
        aug_params[name] = (value, position)
    return aug_params


def cosine_sum_count(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, int]:
    vals = F.cosine_similarity(x, y, dim=-1)
    return float(vals.sum().item()), int(vals.numel())


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_nan(sum_val: float, count: int) -> float:
    return sum_val / count if count > 0 else float("nan")


def save_heatmaps(
    output_dir: Path,
    scanner_names: List[str],
    best_cos_by_pair: Dict[Tuple[str, str], float],
    baseline_cos_by_pair: Dict[Tuple[str, str], float],
) -> None:
    name_to_idx = {n: i for i, n in enumerate(scanner_names)}
    n = len(scanner_names)

    mat_cos = np.full((n, n), np.nan, dtype=np.float32)
    mat_imp = np.full((n, n), np.nan, dtype=np.float32)

    for pair, best_cos in best_cos_by_pair.items():
        si = name_to_idx[pair[0]]
        ti = name_to_idx[pair[1]]
        mat_cos[si, ti] = best_cos
        base = baseline_cos_by_pair.get(pair, float("nan"))
        if not math.isnan(base):
            mat_imp[si, ti] = best_cos - base

    panels = [
        (
            mat_cos,
            "Best cosine per scanner pair (max mean cosine over all sweep combos)",
            "RdYlGn",
            np.nanmin(mat_cos),
            np.nanmax(mat_cos),
        ),
        (
            mat_imp,
            "Improvement at best combo (best mean cosine - baseline mean cosine)",
            "RdYlGn",
            np.nanmin(mat_imp),
            np.nanmax(mat_imp),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        if math.isnan(float(vmin)) or math.isnan(float(vmax)):
            vmin, vmax = 0.0, 1.0
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-3
        im = ax.imshow(
            mat, cmap=cmap, vmin=float(vmin), vmax=float(vmax), aspect="auto"
        )
        ax.set_title(title)
        ax.set_xlabel("Target scanner")
        ax.set_ylabel("Source scanner")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(scanner_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(scanner_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_dir / "scanner_pair_heatmaps.png", dpi=180)
    plt.close(fig)


def staining_group_name(pair: PairRecord) -> str:
    if pair.staining_a == pair.staining_b:
        return pair.staining_a
    return f"{pair.staining_a}_to_{pair.staining_b}"


def _apply_hed_to_pil(img: Image.Image, hed_value: float) -> Image.Image:
    if abs(float(hed_value)) < 1e-12:
        return img.copy()

    x = torch.from_numpy(np.array(img.convert("RGB"), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).contiguous()

    hed2rgb = torch.tensor(
        [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
        dtype=torch.float32,
    )
    rgb2hed = torch.linalg.inv(hed2rgb)

    epsilon = 3.14159
    c, h, w = x.shape
    flat_rgb = x.view(c, -1).permute(1, 0)
    hed = -torch.log(flat_rgb + epsilon) @ rgb2hed

    sigma = 0.03
    factors = torch.tensor([-2.0, 2.0, -3.0], dtype=torch.float32)
    alpha = 1 + float(hed_value) * factors * sigma
    beta = float(hed_value) * factors * sigma
    hed_shifted = hed * alpha + beta

    rgb_shifted = torch.exp(-hed_shifted @ hed2rgb) - epsilon
    rgb_clipped = rgb_shifted.clamp(0.0, 1.0).permute(1, 0).view(c, h, w)
    out = (rgb_clipped.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def _apply_aug_combo_to_pil(
    img: Image.Image,
    aug_values: Dict[str, float],
    apply_order: Sequence[str],
) -> Image.Image:
    out = img.copy()
    for name in apply_order:
        val = float(aug_values.get(name, 0.0))
        if abs(val) < 1e-12:
            continue
        if name == "hed":
            out = _apply_hed_to_pil(out, val)
        elif name == "brightness":
            out = TF.adjust_brightness(out, brightness_factor=1.0 + val)
        elif name == "saturation":
            out = TF.adjust_saturation(out, saturation_factor=1.0 + val)
        elif name == "contrast":
            out = TF.adjust_contrast(out, contrast_factor=1.0 + val)
        elif name == "hue":
            out = TF.adjust_hue(out, hue_factor=val)
        elif name == "gamma":
            out = TF.adjust_gamma(out, gamma=1.0 + val)
        else:
            raise ValueError(f"Unsupported visualization transform: {name}")
    return out


def save_staining_example_grids(
    output_dir: Path,
    scanner_names: List[str],
    staining_pair_best_rows: List[Dict],
    example_tile_path: Path,
    grid_size: int,
    sweep_aug_names: Sequence[str],
    num_example_tiles: int = 1,
) -> None:
    if not example_tile_path.exists():
        raise FileNotFoundError(
            f"example_tile_path does not exist: {example_tile_path}"
        )
    if num_example_tiles <= 0:
        raise ValueError("num_example_tiles must be > 0")

    scanners = scanner_names[: grid_size if grid_size > 0 else len(scanner_names)]
    if len(scanners) < 2:
        return

    jpeg_root = example_tile_path.parent.parent

    def _save_grid_jpeg(fig: plt.Figure, out_path: Path) -> None:
        # Use Pillow to save JPEG quality consistently across matplotlib versions.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        Image.fromarray(rgba, mode="RGBA").convert("RGB").save(
            out_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

    all_tokens = []
    seed_token = example_tile_path.name.split("__")[-1]
    all_tokens.append(seed_token)
    for cand in sorted(example_tile_path.parent.glob("*__tile_*.jpg")):
        tok = cand.name.split("__")[-1]
        if tok not in all_tokens:
            all_tokens.append(tok)
        if len(all_tokens) >= num_example_tiles:
            break
    coord_tokens = all_tokens[:num_example_tiles]

    parent_name = example_tile_path.parent.name
    anchor_src = parent_name.split("_to_")[0] if "_to_" in parent_name else parent_name
    anchor_staining, anchor_scanner = _parse_staining_scanner(anchor_src)

    by_staining: Dict[str, List[Dict]] = {}
    for row in staining_pair_best_rows:
        by_staining.setdefault(str(row["staining_group"]), []).append(row)

    grids_dir = output_dir / "staining_example_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    for st in sorted(by_staining.keys()):
        pair_to_aug: Dict[Tuple[str, str], Dict[str, float]] = {}
        for row in by_staining[st]:
            pair_to_aug[(str(row["src_scanner"]), str(row["tgt_scanner"]))] = {
                n: float(row[n]) for n in sweep_aug_names
            }

        for coord_token in coord_tokens:
            source_tiles: Dict[str, Image.Image] = {}
            for src in scanners:
                # Find scanner-matched tile with same coordinates token for this staining.
                # Filename format is typically '<idx>__<hash>__tile_16_x_y.jpg'.
                pattern = f"{st}_{src}_to_*.tif/*__{coord_token}"
                matches = sorted(jpeg_root.glob(pattern))
                if not matches:
                    continue
                source_tiles[src] = Image.open(matches[0]).convert("RGB")

            fig, axes = plt.subplots(
                len(scanners),
                len(scanners),
                figsize=(2.2 * len(scanners), 2.2 * len(scanners)),
                constrained_layout=True,
            )
            if len(scanners) == 1:
                axes = np.array([[axes]])

            for i, src in enumerate(scanners):
                for j, tgt in enumerate(scanners):
                    ax = axes[i, j]
                    src_img = source_tiles.get(src, None)
                    out_img = None
                    if src_img is not None:
                        if src == tgt:
                            out_img = src_img
                        else:
                            pair_aug = pair_to_aug.get((src, tgt), None)
                            out_img = (
                                _apply_aug_combo_to_pil(
                                    src_img, pair_aug, apply_order=sweep_aug_names
                                )
                                if pair_aug is not None
                                else None
                            )

                    if out_img is None:
                        ax.imshow(np.full((256, 256, 3), 240, dtype=np.uint8))
                        ax.text(
                            0.5,
                            0.5,
                            "N/A",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="black",
                        )
                    else:
                        ax.imshow(out_img)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(tgt, fontsize=9)
                    if j == 0:
                        ax.set_ylabel(src, fontsize=9)

            fig.suptitle(
                (
                    f"Staining: {st} | Coord token: {coord_token} | "
                    f"Reference: {anchor_staining}_{anchor_scanner}"
                ),
                fontsize=11,
            )
            coord_stem = Path(coord_token).stem
            out_path = grids_dir / f"{st}_scanner_grid_{coord_stem}.jpg"
            _save_grid_jpeg(fig, out_path)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(_cfg_get(cfg, "seed", 2026))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("medium")

    features_root = Path(_cfg_get(cfg, "features_root", ""))
    model_id = str(_cfg_get(cfg, "model_id", ""))
    output_dir = Path(_cfg_get(cfg, "output_dir", ""))
    device = torch.device(
        str(_cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    )
    batch_size = int(_cfg_get(cfg, "batch_size", 2048))
    max_patches_per_pair = _cfg_get(cfg, "max_patches_per_pair", 50000)
    if args.max_patches_per_pair is not None:
        max_patches_per_pair = args.max_patches_per_pair
    embedding_dim = _cfg_get(cfg, "embedding_dim", None)
    allow_cross_staining = bool(_cfg_get(cfg, "allow_cross_staining", False))
    allow_same_scanner = bool(_cfg_get(cfg, "allow_same_scanner", False))
    symmetric = bool(_cfg_get(cfg, "symmetric", False))
    scanners = list(_cfg_get(cfg, "scanners", []))
    stainings = list(_cfg_get(cfg, "stainings", []))
    separate_by_staining = bool(_cfg_get(cfg, "separate_by_staining", False))
    example_tile_path_str = str(
        _cfg_get(
            cfg,
            "example_tile_path",
            "/workspaces/HistAug-PLISM/plism_jpeg/GIV_P_to_GMH_S60.tif/"
            "00000066__e1af80860b__tile_16_104_129.jpg",
        )
    )
    example_grid_size = int(_cfg_get(cfg, "example_grid_size", 7))
    compile_model = bool(_cfg_get(cfg, "compile_model", False))
    example_num_tiles = int(_cfg_get(cfg, "example_num_tiles", 1))

    if not str(features_root).strip():
        raise ValueError("config.features_root is required")
    if not model_id.strip():
        raise ValueError("config.model_id is required")
    if not str(output_dir).strip():
        raise ValueError("config.output_dir is required")
    if batch_size <= 0:
        raise ValueError("config.batch_size must be > 0")
    if example_grid_size <= 0:
        raise ValueError("config.example_grid_size must be > 0")
    if example_num_tiles <= 0:
        raise ValueError("config.example_num_tiles must be > 0")

    example_tile_enabled = example_tile_path_str.strip() != ""
    example_tile_path = Path(example_tile_path_str)

    # For tiny sweeps (less than 10 patches/pair), emit a grid per candidate tile
    # so content sensitivity can be inspected quickly.
    if (
        max_patches_per_pair is not None
        and int(max_patches_per_pair) < 10
        and example_tile_enabled
    ):
        example_num_tiles = max(example_num_tiles, 10)

    need_staining_breakdown = bool(separate_by_staining or example_tile_enabled)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_root}")
    slides = discover_slides(features_root)
    slides = filter_slides(slides, scanners, stainings)
    if not slides:
        raise ValueError("No slides left after scanner/staining filtering.")

    pairs = build_pairs(
        slides,
        allow_cross_staining=allow_cross_staining,
        allow_same_scanner=allow_same_scanner,
        symmetric=symmetric,
    )
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise ValueError("--max-pairs must be > 0 when provided")
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(
            "No slide pairs formed. Consider allow_same_scanner/allow_cross_staining "
            "or broader scanner/staining filters."
        )

    if max_patches_per_pair is not None and int(max_patches_per_pair) <= 0:
        raise ValueError("max_patches_per_pair must be > 0 when provided")

    emb_dim = embedding_dim
    if emb_dim is None:
        emb_dim = infer_embedding_dim(slides[0].features_path)
    emb_dim = int(emb_dim)

    print(
        f"Discovered {len(slides)} slides and {len(pairs)} pairs. Embedding dim={emb_dim}"
    )
    if max_patches_per_pair is not None:
        print(f"Max patches per pair: {int(max_patches_per_pair)}")

    print(f"Loading HistAug model: {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    if compile_model:
        model.compile()

    histaug_module = resolve_histaug_module(model)
    transform_names = get_transform_names(histaug_module)
    sweep_aug_names, sweep_combos, values_by_aug, bounds_used = build_sweep_grid(
        cfg, histaug_module
    )

    missing_in_model = [n for n in sweep_aug_names if n not in transform_names]
    if missing_in_model:
        raise ValueError(
            f"Sweep transforms missing from model checkpoint: {missing_in_model}. "
            f"Model transforms: {transform_names}"
        )

    print(f"Transform names in checkpoint: {transform_names}")
    print(f"Sweeping transforms: {sweep_aug_names}")
    for name in sweep_aug_names:
        print(
            f"  {name}: {len(values_by_aug[name])} values in "
            f"[{bounds_used[name]['min']:.4f}, {bounds_used[name]['max']:.4f}]"
        )
    print(f"Total parameter combinations: {len(sweep_combos)}")

    mmap_cache: Dict[str, np.ndarray] = {}

    def load_arr(path: Path) -> np.ndarray:
        key = str(path)
        if key not in mmap_cache:
            mmap_cache[key] = np.load(path, mmap_mode="r")
        return mmap_cache[key]

    global_pred_sum = {combo: 0.0 for combo in sweep_combos}
    global_pred_count = {combo: 0 for combo in sweep_combos}
    global_baseline_sum = 0.0
    global_baseline_count = 0

    pair_pred_sum: Dict[Tuple[str, str, Tuple[float, ...]], float] = {}
    pair_pred_count: Dict[Tuple[str, str, Tuple[float, ...]], int] = {}
    pair_baseline_sum: Dict[Tuple[str, str], float] = {}
    pair_baseline_count: Dict[Tuple[str, str], int] = {}

    staining_pred_sum: Dict[Tuple[str, Tuple[float, ...]], float] = {}
    staining_pred_count: Dict[Tuple[str, Tuple[float, ...]], int] = {}
    staining_baseline_sum: Dict[str, float] = {}
    staining_baseline_count: Dict[str, int] = {}
    staining_pair_pred_sum: Dict[Tuple[str, str, str, Tuple[float, ...]], float] = {}
    staining_pair_pred_count: Dict[Tuple[str, str, str, Tuple[float, ...]], int] = {}
    staining_pair_baseline_sum: Dict[Tuple[str, str, str], float] = {}
    staining_pair_baseline_count: Dict[Tuple[str, str, str], int] = {}

    rng = np.random.default_rng(seed)

    with torch.inference_mode():
        for pair in tqdm.tqdm(pairs, desc="Pairs", disable=False):
            arr_a = load_arr(pair.features_path_a)
            arr_b = load_arr(pair.features_path_b)
            n_rows = min(arr_a.shape[0], arr_b.shape[0])
            if n_rows <= 0:
                continue

            if max_patches_per_pair is not None and max_patches_per_pair > 0:
                n_take = min(n_rows, int(max_patches_per_pair))
            else:
                n_take = n_rows

            if n_take < n_rows:
                row_idx = rng.choice(n_rows, size=n_take, replace=False)
                row_idx.sort()
            else:
                row_idx = np.arange(n_rows)

            pair_key = (pair.scanner_a, pair.scanner_b)
            st_group = staining_group_name(pair)

            for start in range(0, n_take, batch_size):
                ridx = row_idx[start : start + batch_size]
                a_np = arr_a[ridx, N_COORD_COLS : N_COORD_COLS + emb_dim].astype(
                    np.float32
                )
                b_np = arr_b[ridx, N_COORD_COLS : N_COORD_COLS + emb_dim].astype(
                    np.float32
                )

                feats_a = torch.from_numpy(a_np).to(device, non_blocking=True)
                feats_b = torch.from_numpy(b_np).to(device, non_blocking=True)
                bsz = feats_a.shape[0]

                base_sum, base_count = cosine_sum_count(feats_a, feats_b)
                global_baseline_sum += base_sum
                global_baseline_count += base_count
                pair_baseline_sum[pair_key] = (
                    pair_baseline_sum.get(pair_key, 0.0) + base_sum
                )
                pair_baseline_count[pair_key] = (
                    pair_baseline_count.get(pair_key, 0) + base_count
                )

                if need_staining_breakdown:
                    staining_baseline_sum[st_group] = (
                        staining_baseline_sum.get(st_group, 0.0) + base_sum
                    )
                    staining_baseline_count[st_group] = (
                        staining_baseline_count.get(st_group, 0) + base_count
                    )
                    st_pair_key = (st_group, pair.scanner_a, pair.scanner_b)
                    staining_pair_baseline_sum[st_pair_key] = (
                        staining_pair_baseline_sum.get(st_pair_key, 0.0) + base_sum
                    )
                    staining_pair_baseline_count[st_pair_key] = (
                        staining_pair_baseline_count.get(st_pair_key, 0) + base_count
                    )

                for combo in tqdm.tqdm(sweep_combos, desc="  Combos", disable=True):
                    combo_dict = {
                        name: combo[idx] for idx, name in enumerate(sweep_aug_names)
                    }
                    aug_params = build_aug_params(
                        transform_names=transform_names,
                        batch_size=bsz,
                        continuous_values=combo_dict,
                        device=device,
                        histaug_module=histaug_module,
                    )
                    pred = model(feats_a, aug_params)
                    pred_sum, pred_count = cosine_sum_count(pred, feats_b)

                    global_pred_sum[combo] += pred_sum
                    global_pred_count[combo] += pred_count

                    k = (pair.scanner_a, pair.scanner_b, combo)
                    pair_pred_sum[k] = pair_pred_sum.get(k, 0.0) + pred_sum
                    pair_pred_count[k] = pair_pred_count.get(k, 0) + pred_count

                    if need_staining_breakdown:
                        st_key = (st_group, combo)
                        staining_pred_sum[st_key] = (
                            staining_pred_sum.get(st_key, 0.0) + pred_sum
                        )
                        staining_pred_count[st_key] = (
                            staining_pred_count.get(st_key, 0) + pred_count
                        )

                        st_pair_h_key = (
                            st_group,
                            pair.scanner_a,
                            pair.scanner_b,
                            combo,
                        )
                        staining_pair_pred_sum[st_pair_h_key] = (
                            staining_pair_pred_sum.get(st_pair_h_key, 0.0) + pred_sum
                        )
                        staining_pair_pred_count[st_pair_h_key] = (
                            staining_pair_pred_count.get(st_pair_h_key, 0) + pred_count
                        )

    baseline_global = _mean_or_nan(global_baseline_sum, global_baseline_count)

    global_rows: List[Dict] = []
    for combo in sweep_combos:
        mean_pred = _mean_or_nan(global_pred_sum[combo], global_pred_count[combo])
        row: Dict[str, Any] = {
            "mean_pred_cos": mean_pred,
            "mean_baseline_cos": baseline_global,
            "improvement": mean_pred - baseline_global,
            "n_samples": global_pred_count[combo],
        }
        for idx, name in enumerate(sweep_aug_names):
            row[name] = combo[idx]
        global_rows.append(row)
    global_rows.sort(key=lambda r: r["mean_pred_cos"], reverse=True)
    best_global = global_rows[0]

    pair_rows: List[Dict] = []
    for pair in sorted({(p.scanner_a, p.scanner_b) for p in pairs}):
        base_mean = _mean_or_nan(
            pair_baseline_sum.get(pair, 0.0), pair_baseline_count.get(pair, 0)
        )
        for combo in sweep_combos:
            k = (pair[0], pair[1], combo)
            mean_pred = _mean_or_nan(
                pair_pred_sum.get(k, 0.0), pair_pred_count.get(k, 0)
            )
            row: Dict[str, Any] = {
                "src_scanner": pair[0],
                "tgt_scanner": pair[1],
                "mean_pred_cos": mean_pred,
                "mean_baseline_cos": base_mean,
                "improvement": mean_pred - base_mean,
                "n_samples": pair_pred_count.get(k, 0),
            }
            for idx, name in enumerate(sweep_aug_names):
                row[name] = combo[idx]
            pair_rows.append(row)

    best_pair_rows: List[Dict] = []
    best_cos_by_pair: Dict[Tuple[str, str], float] = {}
    baseline_by_pair: Dict[Tuple[str, str], float] = {}
    pair_index: Dict[Tuple[str, str], List[Dict]] = {}
    for row in pair_rows:
        key = (row["src_scanner"], row["tgt_scanner"])
        pair_index.setdefault(key, []).append(row)

    for key, rows in sorted(pair_index.items()):
        rows_sorted = sorted(rows, key=lambda r: r["mean_pred_cos"], reverse=True)
        best = rows_sorted[0]
        best_pair_rows.append(best)
        best_cos_by_pair[key] = float(best["mean_pred_cos"])
        baseline_by_pair[key] = float(best["mean_baseline_cos"])

    write_csv(
        output_dir / "global_aug_sweep.csv",
        rows=global_rows,
        fieldnames=list(sweep_aug_names)
        + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
    )
    write_csv(
        output_dir / "pair_aug_sweep.csv",
        rows=pair_rows,
        fieldnames=["src_scanner", "tgt_scanner"]
        + list(sweep_aug_names)
        + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
    )
    write_csv(
        output_dir / "pair_best_aug.csv",
        rows=best_pair_rows,
        fieldnames=["src_scanner", "tgt_scanner"]
        + list(sweep_aug_names)
        + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
    )

    scanner_names = sorted({p.scanner for p in slides})
    if "hed" in sweep_aug_names:
        save_heatmaps(
            output_dir=output_dir,
            scanner_names=scanner_names,
            best_cos_by_pair=best_cos_by_pair,
            baseline_cos_by_pair=baseline_by_pair,
        )

    staining_rows: List[Dict] = []
    staining_best_rows: List[Dict] = []
    staining_pair_rows: List[Dict] = []
    staining_pair_best_rows: List[Dict] = []

    if need_staining_breakdown:
        staining_groups = sorted({staining_group_name(p) for p in pairs})
        for st in staining_groups:
            st_base = _mean_or_nan(
                staining_baseline_sum.get(st, 0.0),
                staining_baseline_count.get(st, 0),
            )
            st_candidates: List[Dict] = []
            for combo in sweep_combos:
                key = (st, combo)
                st_mean = _mean_or_nan(
                    staining_pred_sum.get(key, 0.0),
                    staining_pred_count.get(key, 0),
                )
                row: Dict[str, Any] = {
                    "staining_group": st,
                    "mean_pred_cos": st_mean,
                    "mean_baseline_cos": st_base,
                    "improvement": st_mean - st_base,
                    "n_samples": staining_pred_count.get(key, 0),
                }
                for idx, name in enumerate(sweep_aug_names):
                    row[name] = combo[idx]
                staining_rows.append(row)
                st_candidates.append(row)

            st_candidates.sort(key=lambda r: r["mean_pred_cos"], reverse=True)
            if st_candidates:
                staining_best_rows.append(st_candidates[0])

        st_pair_index: Dict[Tuple[str, str, str], List[Dict]] = {}
        for st in sorted({staining_group_name(p) for p in pairs}):
            for src, tgt in sorted(
                {
                    (p.scanner_a, p.scanner_b)
                    for p in pairs
                    if staining_group_name(p) == st
                }
            ):
                st_pair_base_key = (st, src, tgt)
                st_pair_base = _mean_or_nan(
                    staining_pair_baseline_sum.get(st_pair_base_key, 0.0),
                    staining_pair_baseline_count.get(st_pair_base_key, 0),
                )
                for combo in sweep_combos:
                    st_pair_h_key = (st, src, tgt, combo)
                    st_pair_mean = _mean_or_nan(
                        staining_pair_pred_sum.get(st_pair_h_key, 0.0),
                        staining_pair_pred_count.get(st_pair_h_key, 0),
                    )
                    st_row: Dict[str, Any] = {
                        "staining_group": st,
                        "src_scanner": src,
                        "tgt_scanner": tgt,
                        "mean_pred_cos": st_pair_mean,
                        "mean_baseline_cos": st_pair_base,
                        "improvement": st_pair_mean - st_pair_base,
                        "n_samples": staining_pair_pred_count.get(st_pair_h_key, 0),
                    }
                    for idx, name in enumerate(sweep_aug_names):
                        st_row[name] = combo[idx]
                    staining_pair_rows.append(st_row)
                    st_pair_index.setdefault((st, src, tgt), []).append(st_row)

        for key, rows in sorted(st_pair_index.items()):
            rows_sorted = sorted(rows, key=lambda r: r["mean_pred_cos"], reverse=True)
            if rows_sorted:
                staining_pair_best_rows.append(rows_sorted[0])

        write_csv(
            output_dir / "staining_aug_sweep.csv",
            rows=staining_rows,
            fieldnames=["staining_group"]
            + list(sweep_aug_names)
            + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
        )
        write_csv(
            output_dir / "staining_best_aug.csv",
            rows=staining_best_rows,
            fieldnames=["staining_group"]
            + list(sweep_aug_names)
            + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
        )
        write_csv(
            output_dir / "staining_pair_aug_sweep.csv",
            rows=staining_pair_rows,
            fieldnames=["staining_group", "src_scanner", "tgt_scanner"]
            + list(sweep_aug_names)
            + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
        )
        write_csv(
            output_dir / "staining_pair_best_aug.csv",
            rows=staining_pair_best_rows,
            fieldnames=["staining_group", "src_scanner", "tgt_scanner"]
            + list(sweep_aug_names)
            + ["mean_pred_cos", "mean_baseline_cos", "improvement", "n_samples"],
        )

        if separate_by_staining and "hed" in sweep_aug_names:
            by_staining_heatmaps_dir = output_dir / "staining_heatmaps"
            by_staining_heatmaps_dir.mkdir(parents=True, exist_ok=True)
            for st in sorted(
                {row["staining_group"] for row in staining_pair_best_rows}
            ):
                st_best_cos: Dict[Tuple[str, str], float] = {}
                st_base: Dict[Tuple[str, str], float] = {}
                for row in staining_pair_best_rows:
                    if row["staining_group"] != st:
                        continue
                    key = (row["src_scanner"], row["tgt_scanner"])
                    st_best_cos[key] = float(row["mean_pred_cos"])
                    st_base[key] = float(row["mean_baseline_cos"])
                st_dir = by_staining_heatmaps_dir / st
                st_dir.mkdir(parents=True, exist_ok=True)
                save_heatmaps(
                    output_dir=st_dir,
                    scanner_names=scanner_names,
                    best_cos_by_pair=st_best_cos,
                    baseline_cos_by_pair=st_base,
                )

        if example_tile_enabled:
            save_staining_example_grids(
                output_dir=output_dir,
                scanner_names=scanner_names,
                staining_pair_best_rows=staining_pair_best_rows,
                example_tile_path=example_tile_path,
                grid_size=example_grid_size,
                sweep_aug_names=sweep_aug_names,
                num_example_tiles=example_num_tiles,
            )

    meta = {
        "config_path": str(args.config),
        "features_root": str(features_root),
        "model_id": model_id,
        "device": str(device),
        "embedding_dim": emb_dim,
        "num_slides": len(slides),
        "num_pairs": len(pairs),
        "batch_size": batch_size,
        "max_patches_per_pair": max_patches_per_pair,
        "max_pairs": args.max_pairs,
        "allow_cross_staining": bool(allow_cross_staining),
        "allow_same_scanner": bool(allow_same_scanner),
        "symmetric": bool(symmetric),
        "transform_names": transform_names,
        "sweep_aug_names": sweep_aug_names,
        "sweep_values": values_by_aug,
        "sweep_num_combinations": len(sweep_combos),
        "separate_by_staining": bool(separate_by_staining),
        "need_staining_breakdown": bool(need_staining_breakdown),
        "example_tile_path": str(example_tile_path) if example_tile_enabled else "",
        "example_grid_size": int(example_grid_size),
        "example_num_tiles": int(example_num_tiles),
        "heatmap_definition": {
            "best_mean_cosine": (
                "For each (source_scanner, target_scanner), select the sweep "
                "combination with the highest mean_pred_cos over sampled patches."
            ),
            "baseline_mean_cosine": (
                "Mean cosine similarity between source and target features without "
                "applying HistAug (identity/no augmentation)."
            ),
            "improvement": "best_mean_cosine - baseline_mean_cosine",
            "includes_hed_value_heatmap": False,
        },
        "global_baseline_cos": float(baseline_global),
        "best_global": best_global,
    }
    (output_dir / "run_config_and_best.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Sweep complete.")
    best_aug_str = ", ".join(
        f"{n}={float(best_global[n]):+.5f}" for n in sweep_aug_names
    )
    print(f"Best global params: {best_aug_str}")
    print(
        f"Best global mean cosine={best_global['mean_pred_cos']:.6f} "
        f"(baseline={best_global['mean_baseline_cos']:.6f}, "
        f"improvement={best_global['improvement']:+.6f})"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
