from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


N_COORD_COLS = 3  # (level, loc, tile) columns prepended to every features.npy row


# ---------------------------------------------------------------------------
# Legacy tensor-based loading (used when .pt files contain Z_A / Z_B)
# ---------------------------------------------------------------------------

def load_paired_tensors(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Z_A, Z_B from a single-pair .pt or .npz file."""
    p = Path(path)
    if p.suffix == ".pt":
        data = torch.load(p, weights_only=True)
        return data["Z_A"].float(), data["Z_B"].float()
    elif p.suffix == ".npz":
        data = np.load(p)
        return torch.from_numpy(data["Z_A"]).float(), torch.from_numpy(data["Z_B"]).float()
    raise ValueError(f"Unsupported format: {p.suffix}. Use .pt or .npz")


def load_multitarget_tensors(
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Load Z_A, Z_B, src_ids, tgt_ids, scanner_vocab from a multi-target .pt file.

    Returns:
        z_src:         [N, D] source embeddings
        z_tgt:         [N, D] target embeddings
        src_ids:       [N] long tensor — source scanner index in scanner_vocab
        tgt_ids:       [N] long tensor — target scanner index in scanner_vocab
        scanner_vocab: list of all scanner names (source and target)
    """
    data = torch.load(Path(path), weights_only=True)
    return (
        data["Z_A"].float(),
        data["Z_B"].float(),
        data["src_ids"].long(),
        data["tgt_ids"].long(),
        data["scanner_vocab"],
    )


class PairedFeatureDataset(Dataset):
    """Row-aligned paired embeddings (z_a[i] corresponds to z_b[i])."""

    def __init__(self, z_a: torch.Tensor, z_b: torch.Tensor):
        assert z_a.shape == z_b.shape, f"Shape mismatch: {z_a.shape} vs {z_b.shape}"
        self.z_a = z_a
        self.z_b = z_b

    def __len__(self) -> int:
        return len(self.z_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.z_a[idx], self.z_b[idx]


class MultiTargetPairedDataset(Dataset):
    """Paired embeddings across all scanner combinations.

    Returns (z_src, z_tgt, src_id, tgt_id).  src_id is used only for cycle
    consistency during training; at inference only tgt_id is needed.
    """

    def __init__(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ):
        assert len(z_src) == len(z_tgt) == len(src_ids) == len(tgt_ids)
        self.z_src = z_src
        self.z_tgt = z_tgt
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.z_src)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z_src[idx], self.z_tgt[idx], self.src_ids[idx], self.tgt_ids[idx]


class UnpairedFeatureDataset(Dataset):
    """Independent domain A and domain B embeddings with no row correspondence."""

    def __init__(self, z_a: torch.Tensor, z_b: torch.Tensor):
        self.z_a = z_a
        self.z_b = z_b

    def __len__(self) -> int:
        return max(len(self.z_a), len(self.z_b))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.z_a[idx % len(self.z_a)], self.z_b[idx % len(self.z_b)]


# ---------------------------------------------------------------------------
# Memory-mapped dataset — reads directly from features.npy, no duplication
# ---------------------------------------------------------------------------

class MmapMultiTargetDataset(Dataset):
    """Multi-target paired dataset backed by memory-mapped features.npy files.

    Reads directly from the per-staining feature files produced by the
    foundation-model extraction pipeline.  No feature data is aggregated or
    duplicated on disk; the OS page cache decides what actually lives in RAM.

    Each "chunk" describes one (src_scanner, tgt_scanner, staining) triple and
    carries the row indices that belong to this split (train / val / test).

    Args:
        chunks:        list of dicts with keys:
                         path_a, path_b  (str) — paths to features.npy
                         src_id, tgt_id  (int) — scanner indices
                         row_indices     (array-like of int) — which rows to use
        feature_dim:   number of feature dimensions (excluding coord columns)
        n_coord_cols:  leading coordinate columns to skip (default 3)
        norm_stats:    optional NormStats for in-place z-score normalisation
        data_fraction: randomly subsample this fraction of rows per chunk (0 < x ≤ 1)
        seed:          RNG seed for data_fraction subsampling
    """

    def __init__(
        self,
        chunks: list,
        feature_dim: int,
        n_coord_cols: int = N_COORD_COLS,
        norm_stats=None,
        data_fraction: float = 1.0,
        seed: int = 0,
    ):
        self.feature_dim = feature_dim
        self.n_coord_cols = n_coord_cols
        self.norm_stats = norm_stats

        rng = np.random.default_rng(seed)
        processed = []
        for c in chunks:
            idx = np.asarray(c["row_indices"], dtype=np.int64)
            if 0.0 < data_fraction < 1.0:
                n_keep = max(1, int(len(idx) * data_fraction))
                idx = np.sort(rng.choice(idx, n_keep, replace=False))
            processed.append({
                "path_a": c["path_a"],
                "path_b": c["path_b"],
                "src_id": int(c["src_id"]),
                "tgt_id": int(c["tgt_id"]),
                "row_indices": idx,
            })
        self.chunks = processed

        lengths = [len(c["row_indices"]) for c in self.chunks]
        self._cumlen = np.cumsum([0] + lengths)

        # Populated lazily per DataLoader worker — never in __init__.
        self._mmaps: dict = {}

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def _get_mmap(self, path: str) -> np.ndarray:
        if path not in self._mmaps:
            self._mmaps[path] = np.load(path, mmap_mode="r")
        return self._mmaps[path]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_idx = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
        local_idx = idx - int(self._cumlen[chunk_idx])
        chunk = self.chunks[chunk_idx]

        row = int(chunk["row_indices"][local_idx])
        z_src = torch.from_numpy(
            self._get_mmap(chunk["path_a"])[row, self.n_coord_cols:].copy()
        )
        z_tgt = torch.from_numpy(
            self._get_mmap(chunk["path_b"])[row, self.n_coord_cols:].copy()
        )

        if self.norm_stats is not None:
            z_src = self.norm_stats.normalize_a(z_src)
            z_tgt = self.norm_stats.normalize_b(z_tgt)

        return (
            z_src,
            z_tgt,
            torch.tensor(chunk["src_id"], dtype=torch.long),
            torch.tensor(chunk["tgt_id"], dtype=torch.long),
        )
