"""Prepare CycleGAN training metadata from the PLISM features root.

Applies the same organ-based 80/20 split and GVH holdout as the scanner-transfer
pipeline so train/val/test sets are directly comparable.

Output format (metadata, not feature tensors)
---------------------------------------------
Each split produces a small .pt file containing:
  scanner_vocab  — list of scanner names
  feature_dim    — int (e.g. 2560 for Virchow2)
  n_coord_cols   — leading coordinate columns in features.npy (always 3)
  chunks         — list of dicts, one per (src_scanner, tgt_scanner, staining):
                     path_a       str  — absolute path to src features.npy
                     path_b       str  — absolute path to tgt features.npy
                     src_id       int  — index into scanner_vocab
                     tgt_id       int  — index into scanner_vocab
                     staining     str
                     row_indices  LongTensor — which rows of the npy files to use
  (train only)
  norm_mean_a / norm_std_a  — per-dim stats over all training source features
  norm_mean_b / norm_std_b  — per-dim stats over all training target features

Feature files are never copied or aggregated.  The Dataset reads them at
training time via memory-mapping, so the OS page cache handles RAM usage.

Usage — all scanner pairs (recommended):
    python -m histaug.cyclegan_features.prepare_data \\
        --features_root /mnt/data/plismbench/features/virchow2 \\
        --all_pairs \\
        --output_dir /mnt/data/plismbench/features/CycleGAN/virchow2_meta

Usage — single pair:
    python -m histaug.cyclegan_features.prepare_data \\
        --features_root /mnt/data/plismbench/features/virchow2 \\
        --scanner_a AT2 --scanner_b GT450 \\
        --output_dir /mnt/data/plismbench/features/CycleGAN/virchow2_AT2_GT450_meta
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent.parent.parent))

from histaug.utils.organ_split import (
    build_organ_sets,
    build_valid_tile_keys,
    load_organ_map,
)

N_COORD_COLS = 3
HOLDOUT_STAININGS: FrozenSet[str] = frozenset(["GVH"])
TRAIN_FRACTION = 0.8
SPLIT_SEED = 2025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_features_root(features_root: Path) -> Dict[str, Dict[str, Path]]:
    """Return {scanner: {staining: dir_path}} for all entries in the features root."""
    index: Dict[str, Dict[str, Path]] = {}
    for d in features_root.iterdir():
        stem = d.name.replace(".tif", "")
        parts = stem.split("_to_")
        if len(parts) != 2:
            continue
        src_staining, src_scanner = parts[0].rsplit("_", 1)
        index.setdefault(src_scanner, {})[src_staining] = d
    return index


def _build_row_mask(features_root: Path, organ_loc_csv: Path, valid_organs) -> np.ndarray:
    organ_map = load_organ_map(organ_loc_csv)
    valid_tile_keys = build_valid_tile_keys(organ_map, frozenset(valid_organs))
    sample_dir = next(features_root.iterdir())
    coords = np.load(sample_dir / "features.npy", mmap_mode="r")[:, :N_COORD_COLS].astype(int)
    return np.array(
        [f"tile_{lv}_{l}_{t}" in valid_tile_keys for lv, l, t in coords], dtype=bool
    )


def _detect_feature_dim(index: Dict[str, Dict[str, Path]]) -> int:
    sample_path = next(iter(next(iter(index.values())).values())) / "features.npy"
    return int(np.load(sample_path, mmap_mode="r").shape[1]) - N_COORD_COLS


def _compute_norm_stats(
    chunks: list,
    n_coord_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-dim mean and std by reading each unique feature file once.

    Uses Chan's parallel-Welford algorithm so no data needs to be held in RAM
    beyond one file at a time.  Reading each unique path once (rather than once
    per pair) reduces I/O by ~6x compared to iterating over all chunks.
    """
    row_indices = np.asarray(chunks[0]["row_indices"])  # same mask for every chunk

    unique_src = sorted({c["path_a"] for c in chunks})
    unique_tgt = sorted({c["path_b"] for c in chunks})

    def _welford(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = 0
        mean = None
        M2 = None
        for path in paths:
            x = np.load(path, mmap_mode="r")[row_indices, n_coord_cols:]  # float32 view
            x = x.astype(np.float64, copy=True)                           # owned copy
            n_b = len(x)
            mean_b = x.mean(axis=0)
            M2_b = ((x - mean_b) ** 2).sum(axis=0)
            if mean is None:
                mean, M2, n = mean_b, M2_b, n_b
            else:
                n_new = n + n_b
                delta = mean_b - mean
                mean = (n * mean + n_b * mean_b) / n_new
                M2 += M2_b + delta ** 2 * n * n_b / n_new
                n = n_new
            del x
        std = np.sqrt(M2 / max(n - 1, 1)).clip(min=1e-6)
        return mean.astype(np.float32), std.astype(np.float32)

    print("  Computing norm stats (src)...")
    mean_a, std_a = _welford(unique_src)
    print("  Computing norm stats (tgt)...")
    mean_b, std_b = _welford(unique_tgt)
    return mean_a, std_a, mean_b, std_b


def _save_metadata(output_dir: Path, split: str, payload: dict) -> None:
    out = output_dir / f"{split}.pt"
    torch.save(payload, out)
    n_samples = sum(len(c["row_indices"]) for c in payload["chunks"])
    size_kb = out.stat().st_size / 1e3
    print(f"  {split}: {len(payload['chunks'])} chunks, {n_samples:,} samples "
          f"→ {out}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# All-pairs mode (multi-target, mmap metadata)
# ---------------------------------------------------------------------------

def prepare_all_pairs(
    features_root: Path,
    organ_loc_csv: Path,
    output_dir: Path,
    scanners: Optional[List[str]] = None,
) -> None:
    """Save per-split metadata pointing at the existing features.npy files.

    No feature data is copied.  norm_mean/std are computed by streaming through
    the training files once, without loading everything into RAM.
    """
    index = _index_features_root(features_root)
    scanner_vocab = sorted(index.keys()) if scanners is None else sorted(scanners)
    feature_dim = _detect_feature_dim(index)
    print(f"Scanner vocab ({len(scanner_vocab)}): {scanner_vocab}")
    print(f"Feature dim: {feature_dim}")

    organ_map = load_organ_map(organ_loc_csv)
    train_organs, val_organs = build_organ_sets(organ_map, TRAIN_FRACTION, SPLIT_SEED)
    train_mask = _build_row_mask(features_root, organ_loc_csv, train_organs)
    val_mask   = _build_row_mask(features_root, organ_loc_csv, val_organs)
    all_mask   = np.ones(train_mask.shape, dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_chunks = None
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", all_mask)]:
        is_test = split == "test"
        row_indices = torch.from_numpy(np.where(mask)[0])
        chunks = []

        for sc_a, sc_b in itertools.permutations(scanner_vocab, 2):
            src_idx = scanner_vocab.index(sc_a)
            tgt_idx = scanner_vocab.index(sc_b)
            dirs_a = index.get(sc_a, {})
            dirs_b = index.get(sc_b, {})
            common = sorted(set(dirs_a) & set(dirs_b))
            stainings = [s for s in common if (s in HOLDOUT_STAININGS) == is_test]
            for staining in stainings:
                chunks.append({
                    "path_a":      str(dirs_a[staining] / "features.npy"),
                    "path_b":      str(dirs_b[staining] / "features.npy"),
                    "src_id":      src_idx,
                    "tgt_id":      tgt_idx,
                    "staining":    staining,
                    "row_indices": row_indices,
                })

        payload = {
            "scanner_vocab": scanner_vocab,
            "feature_dim":   feature_dim,
            "n_coord_cols":  N_COORD_COLS,
            "chunks":        chunks,
        }

        if split == "train":
            train_chunks = chunks
            mean_a, std_a, mean_b, std_b = _compute_norm_stats(chunks, N_COORD_COLS)
            payload.update({
                "norm_mean_a": torch.from_numpy(mean_a),
                "norm_std_a":  torch.from_numpy(std_a),
                "norm_mean_b": torch.from_numpy(mean_b),
                "norm_std_b":  torch.from_numpy(std_b),
            })

        _save_metadata(output_dir, split, payload)


# ---------------------------------------------------------------------------
# Single-pair mode (mmap metadata)
# ---------------------------------------------------------------------------

def prepare_single_pair(
    features_root: Path,
    scanner_a: str,
    scanner_b: str,
    organ_loc_csv: Path,
    output_dir: Path,
) -> None:
    """Save per-split metadata for a single A→B scanner pair."""
    index = _index_features_root(features_root)
    dirs_a = index.get(scanner_a, {})
    dirs_b = index.get(scanner_b, {})
    common = sorted(set(dirs_a) & set(dirs_b))
    if not common:
        raise ValueError(f"No stainings found for both {scanner_a} and {scanner_b}")

    feature_dim = _detect_feature_dim(index)
    scanner_vocab = sorted([scanner_a, scanner_b])
    src_idx = scanner_vocab.index(scanner_a)
    tgt_idx = scanner_vocab.index(scanner_b)

    train_st = [s for s in common if s not in HOLDOUT_STAININGS]
    test_st  = [s for s in common if s in HOLDOUT_STAININGS]
    print(f"Scanners {scanner_a} → {scanner_b}  |  "
          f"train stainings: {len(train_st)}, holdout: {test_st}")

    organ_map = load_organ_map(organ_loc_csv)
    train_organs, val_organs = build_organ_sets(organ_map, TRAIN_FRACTION, SPLIT_SEED)
    train_mask = _build_row_mask(features_root, organ_loc_csv, train_organs)
    val_mask   = _build_row_mask(features_root, organ_loc_csv, val_organs)
    all_mask   = np.ones(train_mask.shape, dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split, stainings, mask in [
        ("train", train_st, train_mask),
        ("val",   train_st, val_mask),
        ("test",  test_st,  all_mask),
    ]:
        if not stainings:
            continue
        row_indices = torch.from_numpy(np.where(mask)[0])
        chunks = [
            {
                "path_a":      str(dirs_a[s] / "features.npy"),
                "path_b":      str(dirs_b[s] / "features.npy"),
                "src_id":      src_idx,
                "tgt_id":      tgt_idx,
                "staining":    s,
                "row_indices": row_indices,
            }
            for s in stainings
        ]
        payload = {
            "scanner_vocab": scanner_vocab,
            "feature_dim":   feature_dim,
            "n_coord_cols":  N_COORD_COLS,
            "chunks":        chunks,
        }
        if split == "train":
            mean_a, std_a, mean_b, std_b = _compute_norm_stats(chunks, N_COORD_COLS)
            payload.update({
                "norm_mean_a": torch.from_numpy(mean_a),
                "norm_std_a":  torch.from_numpy(std_a),
                "norm_mean_b": torch.from_numpy(mean_b),
                "norm_std_b":  torch.from_numpy(std_b),
            })
        _save_metadata(output_dir, split, payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare CycleGAN metadata from PLISM features (mmap format)"
    )
    p.add_argument("--features_root", required=True)
    p.add_argument("--scanner_a", default=None, help="Source scanner (single-pair mode)")
    p.add_argument("--scanner_b", default=None, help="Target scanner (single-pair mode)")
    p.add_argument("--all_pairs", action="store_true",
                   help="Collect all symmetric scanner pairs (multi-target mode)")
    p.add_argument("--scanners", nargs="*", default=None,
                   help="Subset of scanners for --all_pairs (default: all found)")
    p.add_argument("--organ_loc_csv", default="plism_organ_loc.csv")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    features_root = Path(args.features_root)
    organ_loc_csv = Path(args.organ_loc_csv)
    output_dir    = Path(args.output_dir)

    if args.all_pairs:
        prepare_all_pairs(
            features_root=features_root,
            organ_loc_csv=organ_loc_csv,
            output_dir=output_dir,
            scanners=args.scanners,
        )
    elif args.scanner_a and args.scanner_b:
        prepare_single_pair(
            features_root=features_root,
            scanner_a=args.scanner_a,
            scanner_b=args.scanner_b,
            organ_loc_csv=organ_loc_csv,
            output_dir=output_dir,
        )
    else:
        p.error("Provide either --all_pairs or both --scanner_a and --scanner_b")


if __name__ == "__main__":
    main()
