import random
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data

from utils.organ_split import (
    build_organ_sets,
    build_valid_tile_keys,
    load_organ_map,
    tile_key_to_coords,
)


class PlismPairPrefeaturesDataset(data.Dataset):
    """
    Paired scanner-transfer dataset backed by pre-extracted feature arrays.

    Loads .npy feature files from a features root directory. Each directory
    corresponds to one (staining, scanner) combination and contains a single
    `features.npy` of shape (N_patches, 3 + feature_dim): the first three
    columns are spatial coordinates shared across all slides, and the remaining
    columns are the embedding vectors.

    Because all slides share the same patch layout and row order, patches are
    paired by row index — no tile-key matching required.

    Expected layout:
        features_root/
            <staining>_<scanner>_to_<ref>.tif/
                features.npy
            ...

    Returned 6-tuple (same API as PlismPairJpegDataset):
        (feat_a, feat_b, src_scanner_id, tgt_scanner_id,
         src_staining_id, tgt_staining_id)
    """

    N_COORD_COLS = 3

    def __init__(
        self,
        dataset_cfg,
        state: str = "train",
        transforms=None,  # ignored — no image augmentations on pre-features
        foundation_model=None,  # ignored — features already extracted
        general=None,
    ):
        if dataset_cfg is None:
            raise ValueError("`dataset_cfg` must be provided")

        self.state = state

        features_root = self._cfg_get(dataset_cfg, "features_root", None)
        if not features_root:
            raise ValueError(
                "`Data.features_root` must point to the pre-extracted feature directory "
                "(e.g. /mnt/data/plismbench/features/h0_mini)"
            )
        self.features_root = Path(features_root)
        if not self.features_root.exists():
            raise FileNotFoundError(
                f"Pre-extracted features root not found: {self.features_root}"
            )

        self.split_seed = int(
            self._cfg_get(
                dataset_cfg, "split_seed", self._cfg_get(general, "seed", 2025)
            )
        )
        self.train_fraction = float(self._cfg_get(dataset_cfg, "train_split", 0.8))
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("`Data.train_split` must be in the open interval (0, 1)")

        pairing_cfg = self._cfg_get(dataset_cfg, "pairing", {}) or {}
        self.allow_cross_staining = bool(
            self._cfg_get(pairing_cfg, "allow_cross_staining", False)
        )
        self.allow_same_scanner = bool(
            self._cfg_get(pairing_cfg, "allow_same_scanner", False)
        )
        self.symmetric = bool(self._cfg_get(pairing_cfg, "symmetric", True))
        # Raw value — may be a fraction (0, 1] or an absolute integer count (> 1).
        # Resolved to an integer after valid_row_indices is known (see end of __init__).
        self._tiles_raw = float(
            self._cfg_get(pairing_cfg, "tiles_per_pair_per_epoch", 0.5)
        )
        if self._tiles_raw <= 0:
            raise ValueError("`pairing.tiles_per_pair_per_epoch` must be > 0")

        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        selected_stainings = self._as_str_set(
            self._cfg_get(dataset_cfg, "stainings", [])
        )
        self.holdout_stainings: FrozenSet[str] = frozenset(
            self._as_str_set(self._cfg_get(dataset_cfg, "holdout_stainings", []))
        )

        if state not in ("train", "val", "test", "test_holdout_staining"):
            raise ValueError(f"Unknown dataset state: {state!r}")
        if state == "test_holdout_staining" and not self.holdout_stainings:
            raise ValueError(
                "state='test_holdout_staining' requires `Data.holdout_stainings` to be "
                "set (e.g. ['GVH'])."
            )

        slides_all = self._discover_slides(self.features_root)
        slides_all = self._filter_slides(
            slides_all, selected_scanners, selected_stainings
        )
        if not slides_all:
            raise ValueError(
                "No pre-extracted feature slides matched the provided filters "
                "in `Data.scanners` / `Data.stainings`."
            )

        # Vocabularies built from the full filtered set so IDs stay stable across splits,
        # even when a staining is held out of training.
        scanners_all = sorted({s["scanner"] for s in slides_all})
        stainings_all = sorted({s["staining"] for s in slides_all})
        self.scanner_to_id: Dict[str, int] = {n: i for i, n in enumerate(scanners_all)}
        self.staining_to_id: Dict[str, int] = {
            n: i for i, n in enumerate(stainings_all)
        }

        # Holdout-staining partition. `regular_slides` feed train/val; `holdout_slides`
        # feed the dedicated held-out-staining test phase (state='test_holdout_staining').
        regular_slides = [
            s for s in slides_all if s["staining"] not in self.holdout_stainings
        ]
        holdout_slides = [
            s for s in slides_all if s["staining"] in self.holdout_stainings
        ]
        if self.holdout_stainings and not holdout_slides:
            raise ValueError(
                f"`Data.holdout_stainings`={sorted(self.holdout_stainings)} matched no "
                f"slides under {self.features_root}. Check staining spelling."
            )

        if state == "test_holdout_staining":
            pair_slides = holdout_slides
        else:
            pair_slides = regular_slides
        if not pair_slides:
            raise ValueError(
                f"No slides available for state='{state}' after holdout filtering. "
                "Check `Data.holdout_stainings`, `Data.scanners`, `Data.stainings`."
            )

        self.slide_pairs = self._build_slide_pair_index(pair_slides)
        if not self.slide_pairs:
            raise ValueError(
                "Zero slide-pairs formed. Loosen `Data.pairing` filters or "
                "broaden scanner/staining selection."
            )

        # Determine patch count from a sample file and resolve feature width.
        sample_path = slides_all[0]["features_path"]
        sample_arr = np.load(sample_path, mmap_mode="r")
        self.n_patches: int = sample_arr.shape[0]
        if sample_arr.shape[1] <= self.N_COORD_COLS:
            raise ValueError(
                f"{sample_path} has {sample_arr.shape[1]} columns, expected > "
                f"{self.N_COORD_COLS} (coord columns only is invalid)."
            )

        configured_feature_dim = self._cfg_get(dataset_cfg, "feature_dim", None)
        if configured_feature_dim is None:
            self.feature_dim = int(sample_arr.shape[1] - self.N_COORD_COLS)
        else:
            self.feature_dim = int(configured_feature_dim)
            if self.feature_dim <= 0:
                raise ValueError("`Data.feature_dim` must be a positive integer.")

        expected_cols = self.N_COORD_COLS + self.feature_dim
        if sample_arr.shape[1] < expected_cols:
            raise ValueError(
                f"{sample_path} has {sample_arr.shape[1]} columns, "
                f"expected >= {expected_cols} (3 coord + {self.feature_dim} features). "
                "Check `Data.feature_dim` for this config."
            )

        # Per-worker cache of memory-mapped arrays (populated lazily in __getitem__).
        self._feature_cache: Dict[str, np.ndarray] = {}

        # Organ-based patch split:
        #   - train: patches from train organs
        #   - val / test: patches from held-out organs
        #   - test_holdout_staining: ALL patches (staining is the only holdout axis here;
        #     train never saw these slides, so no organ leakage to worry about)
        self.valid_organs: Optional[List[str]] = None
        organ_loc_csv = self._cfg_get(dataset_cfg, "organ_loc_csv", None)
        self._organ_loc_csv_path: Optional[str] = (
            str(organ_loc_csv) if organ_loc_csv else None
        )
        if state == "test_holdout_staining":
            self.valid_row_indices = list(range(self.n_patches))
        elif organ_loc_csv:
            organ_map = load_organ_map(self._resolve_path(str(organ_loc_csv)))
            train_organs, test_organs = build_organ_sets(
                organ_map, self.train_fraction, self.split_seed
            )
            valid_organs = train_organs if state == "train" else test_organs
            self.valid_organs = sorted(valid_organs)
            valid_keys = build_valid_tile_keys(organ_map, valid_organs)
            self.valid_row_indices = self._build_valid_row_indices(
                slides_all[0]["features_path"], valid_keys
            )
        else:
            import warnings

            if state in ("train", "val"):
                warnings.warn(
                    "`organ_loc_csv` not set: train and val will sample from the same "
                    "patches (no patch-level split). Set `Data.organ_loc_csv` to avoid "
                    "data leakage between splits.",
                    UserWarning,
                    stacklevel=2,
                )
            self.valid_row_indices = list(range(self.n_patches))

        # Resolve tiles_per_pair_per_epoch: fraction of train patch pool → int count.
        n_train_patches = len(self.valid_row_indices)
        if 0 < self._tiles_raw <= 1.0:
            self.tiles_per_pair_per_epoch = max(
                1, round(self._tiles_raw * n_train_patches)
            )
        else:
            self.tiles_per_pair_per_epoch = int(self._tiles_raw)

        # Stash slide membership for downstream manifest dumping.
        self._slides_all = slides_all
        self._regular_slides = regular_slides
        self._holdout_slides = holdout_slides

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def use_all_test_samples(self) -> bool:
        return True

    @property
    def scanner_vocab_size(self) -> int:
        return len(self.scanner_to_id)

    @property
    def staining_vocab_size(self) -> int:
        return len(self.staining_to_id)

    @property
    def has_holdout_staining(self) -> bool:
        return bool(self.holdout_stainings)

    def current_split_summary(self) -> Dict:
        """Summary of the dataset instance's current state (slide pool + patch pool)."""
        slides_in_pool = (
            self._holdout_slides
            if self.state == "test_holdout_staining"
            else self._regular_slides
        )
        return {
            "state": self.state,
            "n_slides": len(slides_in_pool),
            "n_pairs": len(self.slide_pairs),
            "n_valid_patches": len(self.valid_row_indices),
            "n_items": len(self),
            "slides": [
                {
                    "slide_id": s["slide_id"],
                    "staining": s["staining"],
                    "scanner": s["scanner"],
                }
                for s in sorted(
                    slides_in_pool, key=lambda s: (s["staining"], s["scanner"])
                )
            ],
            "valid_organs": (
                list(self.valid_organs) if self.valid_organs is not None else None
            ),
        }

    def describe_splits(self, organ_loc_csv: Optional[str] = None) -> Dict:
        """Deterministic description of all three splits (train / val / test_holdout_staining).

        Slide membership is fully determined by ``holdout_stainings``; organ membership
        by (split_seed, train_fraction, organ_loc_csv). This method can be called on any
        instance and returns the same dict regardless of the instance's own state.
        """
        csv_path = organ_loc_csv or self._cfg_get_default_organ_csv()

        train_organs: List[str] = []
        val_organs: List[str] = []
        if csv_path:
            try:
                organ_map = load_organ_map(self._resolve_path(str(csv_path)))
                t_organs, v_organs = build_organ_sets(
                    organ_map, self.train_fraction, self.split_seed
                )
                train_organs = sorted(t_organs)
                val_organs = sorted(v_organs)
            except FileNotFoundError:
                pass

        def _slide_entry(slide: Dict) -> Dict:
            return {
                "slide_id": slide["slide_id"],
                "staining": slide["staining"],
                "scanner": slide["scanner"],
            }

        regular = sorted(
            (_slide_entry(s) for s in self._regular_slides),
            key=lambda s: (s["staining"], s["scanner"]),
        )
        holdout = sorted(
            (_slide_entry(s) for s in self._holdout_slides),
            key=lambda s: (s["staining"], s["scanner"]),
        )

        return {
            "features_root": str(self.features_root),
            "feature_dim": self.feature_dim,
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "holdout_stainings": sorted(self.holdout_stainings),
            "pairing": {
                "allow_cross_staining": self.allow_cross_staining,
                "allow_same_scanner": self.allow_same_scanner,
                "symmetric": self.symmetric,
                "tiles_per_pair_per_epoch": self.tiles_per_pair_per_epoch,
            },
            "scanner_vocab": dict(self.scanner_to_id),
            "staining_vocab": dict(self.staining_to_id),
            "organs": {"train": train_organs, "val": val_organs},
            "slides": {
                "train": regular,  # same slide pool as val
                "val": regular,
                "test_holdout_staining": holdout,
            },
        }

    def _cfg_get_default_organ_csv(self) -> Optional[str]:
        """Recover the organ CSV path from the stash set during __init__ (if any)."""
        return getattr(self, "_organ_loc_csv_path", None)

    def __len__(self) -> int:
        if self.state == "train":
            # Training: stochastic — tiles_per_pair_per_epoch random patches per pair.
            return len(self.slide_pairs) * self.tiles_per_pair_per_epoch
        else:
            # Val / test: deterministic — every valid patch once per pair.
            return len(self.slide_pairs) * len(self.valid_row_indices)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )

        n_per_pair = (
            self.tiles_per_pair_per_epoch
            if self.state == "train"
            else len(self.valid_row_indices)
        )
        pair = self.slide_pairs[index // n_per_pair]
        if self.state == "train":
            row_idx = random.choice(self.valid_row_indices)
        else:
            row_idx = self.valid_row_indices[index % n_per_pair]

        feat_a = self._load_feature(pair["features_path_a"], row_idx)
        feat_b = self._load_feature(pair["features_path_b"], row_idx)

        return (
            feat_a,
            feat_b,
            self.scanner_to_id[pair["scanner_a"]],
            self.scanner_to_id[pair["scanner_b"]],
            self.staining_to_id[pair["staining_a"]],
            self.staining_to_id[pair["staining_b"]],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_feature(self, features_path: Path, row_idx: int) -> torch.Tensor:
        key = str(features_path)
        if key not in self._feature_cache:
            self._feature_cache[key] = np.load(features_path, mmap_mode="r")
        arr = self._feature_cache[key]
        feat = arr[
            row_idx, self.N_COORD_COLS : self.N_COORD_COLS + self.feature_dim
        ].copy()
        return torch.from_numpy(feat)

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.exists():
            return p
        repo_root = Path(__file__).resolve().parents[3]
        alt = repo_root / path_str
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Could not resolve path '{path_str}'. Tried '{p}' and '{alt}'."
        )

    def _build_valid_row_indices(
        self, sample_features_path: Path, valid_keys: Set[str]
    ) -> List[int]:
        """Map valid tile_key strings to row indices using the coord columns in features.npy."""
        arr = np.load(sample_features_path, mmap_mode="r")
        indices = []
        for i in range(arr.shape[0]):
            level, left, top = int(arr[i, 0]), int(arr[i, 1]), int(arr[i, 2])
            if f"tile_{level}_{left}_{top}" in valid_keys:
                indices.append(i)
        if not indices:
            raise ValueError(
                f"Organ split for state='{self.state}' produced zero valid patches. "
                "Check train_fraction and organ_loc_csv."
            )
        return indices

    def _discover_slides(self, root: Path) -> List[Dict]:
        slides = []
        for slide_dir in sorted(root.iterdir()):
            if not slide_dir.is_dir():
                continue
            features_file = slide_dir / "features.npy"
            if not features_file.exists():
                continue
            dir_name = slide_dir.name
            # Strip registration suffix like "_to_GMH_S60.tif"
            slide_id = dir_name.split("_to_")[0] if "_to_" in dir_name else dir_name
            staining, scanner = self._parse_staining_scanner(slide_id)
            slides.append(
                {
                    "slide_id": slide_id,
                    "staining": staining,
                    "scanner": scanner,
                    "features_path": features_file,
                }
            )
        if not slides:
            raise ValueError(
                f"No subdirectories with features.npy found under '{root}'"
            )
        return slides

    def _filter_slides(
        self,
        slides: Sequence[Dict],
        selected_scanners: set,
        selected_stainings: set,
    ) -> List[Dict]:
        output = []
        for slide in slides:
            scanner_ok = (not selected_scanners) or (
                slide["scanner"] in selected_scanners
            )
            staining_ok = (not selected_stainings) or (
                slide["staining"] in selected_stainings
            )
            if scanner_ok and staining_ok:
                output.append(slide)
        return output

    def _split_slides_stratified(
        self, slides: Sequence[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        rng = random.Random(self.split_seed)
        groups: Dict[str, List[Dict]] = {}
        for slide in slides:
            groups.setdefault(slide["staining"], []).append(slide)

        train_slides: List[Dict] = []
        test_slides: List[Dict] = []
        for group_slides in groups.values():
            shuffled = list(group_slides)
            rng.shuffle(shuffled)
            n_group = len(shuffled)
            if n_group == 1:
                n_test = 0
            else:
                n_test = max(1, int(round(n_group * (1.0 - self.train_fraction))))
                if n_group >= 4 and not self.allow_cross_staining:
                    n_test = max(2, n_test)
                    n_test = min(n_test, n_group - 2)
                else:
                    n_test = min(n_test, n_group - 1)
            test_slides.extend(shuffled[:n_test])
            train_slides.extend(shuffled[n_test:])

        if not train_slides:
            raise ValueError(
                "Train split ended up empty; not enough slides after filtering"
            )
        if not test_slides:
            moved_idx = rng.randrange(len(train_slides))
            test_slides.append(train_slides.pop(moved_idx))
        return train_slides, test_slides

    def _build_slide_pair_index(self, slides: Sequence[Dict]) -> List[Dict]:
        pairs: List[Dict] = []
        n = len(slides)
        for i in range(n):
            a = slides[i]
            j_range = range(n) if self.symmetric else range(i + 1, n)
            for j in j_range:
                if i == j:
                    continue
                b = slides[j]
                same_staining = a["staining"] == b["staining"]
                same_scanner = a["scanner"] == b["scanner"]
                if not same_staining and not self.allow_cross_staining:
                    continue
                if same_scanner and not self.allow_same_scanner:
                    continue
                pairs.append(
                    {
                        "features_path_a": a["features_path"],
                        "features_path_b": b["features_path"],
                        "scanner_a": a["scanner"],
                        "scanner_b": b["scanner"],
                        "staining_a": a["staining"],
                        "staining_b": b["staining"],
                    }
                )
        return pairs

    def _parse_staining_scanner(self, slide_id: str) -> Tuple[str, str]:
        tokens = slide_id.split("_")
        if len(tokens) < 2:
            raise ValueError(
                f"Invalid PLISM slide id '{slide_id}'. "
                "Expected pattern '<staining>_<scanner>[_...]'."
            )
        return tokens[0], tokens[1]

    def _cfg_get(self, cfg_obj, key: str, default):
        if cfg_obj is None:
            return default
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    def _as_str_set(self, values) -> set:
        if values is None:
            return set()
        return {str(v) for v in values if str(v).strip()}
