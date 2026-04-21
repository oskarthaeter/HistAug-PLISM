from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from models.foundation_models import get_fm_transform
from PIL import Image


class ScorpionDataset(data.Dataset):
    """
    Evaluation dataset for the SCORPION benchmark.

    Layout::

        data_path/
            slide_<N>/
                sample_<M>/
                    <SCANNER>.jpg    # one image per scanner, same tissue ROI

    Each item is one ordered (src_scanner, tgt_scanner) pair drawn from the same
    tissue sample, returning a 6-tuple that is directly compatible with
    ``ModelInterface._shared_step_pair``::

        (tensor_src, tensor_tgt, src_scanner_id, tgt_scanner_id,
         src_staining_id, tgt_staining_id)

    SCORPION uses a single staining protocol, so ``staining_id`` is always 0.

    :param dataset_cfg: Config object (or dict) with at least a ``data_path`` key.
        Optional keys:
        - ``scanners``: list of scanner names to include (default: all discovered).
        - ``scanner_to_id``: dict mapping scanner name → integer id (default: build
          from discovered scanners sorted alphabetically).
        - ``symmetric``: if ``True`` (default), both (A→B) and (B→A) are included.
    :param foundation_model: Foundation model name string, forwarded to
        ``get_fm_transform`` to obtain the preprocessing pipeline.
    :param state: Accepted for interface compatibility; ignored (all data is used).
    :param general: General config, unused.
    """

    def __init__(
        self,
        dataset_cfg,
        state: str = "test",
        transforms=None,
        foundation_model: Optional[str] = None,
        general=None,
    ):
        if dataset_cfg is None:
            raise ValueError("`dataset_cfg` must be provided")
        if not foundation_model:
            raise ValueError("`foundation_model` must be provided")

        self.foundation_model = foundation_model
        self.dataset_root = self._resolve_root(str(self._cfg_get(dataset_cfg, "data_path", None)))
        self.image_preprocessing_pipeline, _ = get_fm_transform(self.foundation_model)

        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        external_vocab: Optional[Dict[str, int]] = self._cfg_get(dataset_cfg, "scanner_to_id", None)
        symmetric: bool = bool(self._cfg_get(dataset_cfg, "symmetric", True))

        samples = self._discover_samples(self.dataset_root, selected_scanners)
        if not samples:
            raise ValueError(
                f"No SCORPION samples found under '{self.dataset_root}'. "
                "Check `data_path` and `scanners` config."
            )

        # Build scanner vocabulary from all scanner names that appear in the data.
        all_scanners = sorted({sc for _, _, sc in samples})
        if external_vocab is not None:
            self.scanner_to_id: Dict[str, int] = external_vocab
        else:
            self.scanner_to_id = {name: i for i, name in enumerate(all_scanners)}

        # Single staining class.
        self.staining_to_id: Dict[str, int] = {"SCORPION": 0}

        self._items: List[Tuple[str, str, int, int]] = self._build_pairs(
            samples, symmetric=symmetric
        )
        if not self._items:
            raise ValueError(
                "No valid scanner pairs found. Ensure at least two distinct scanner "
                "names are present and, if `scanner_to_id` is provided externally, "
                "that at least two SCORPION scanners map into that vocabulary."
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def use_all_test_samples(self) -> bool:
        return True

    @property
    def scanner_vocab_size(self) -> int:
        return len(self.scanner_to_id)

    @property
    def staining_vocab_size(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")

        path_src, path_tgt, src_id, tgt_id = self._items[index]
        img_src = self._load_jpeg(path_src)
        img_tgt = self._load_jpeg(path_tgt)

        tensor_src = self.image_preprocessing_pipeline(img_src)
        tensor_tgt = self.image_preprocessing_pipeline(img_tgt)

        return tensor_src, tensor_tgt, src_id, tgt_id, 0, 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(
        self, root: Path, selected_scanners: set
    ) -> List[Tuple[str, str, str]]:
        """
        Walk the SCORPION directory tree and collect all (sample_dir, jpeg_path, scanner_name)
        triples, filtered to ``selected_scanners`` when non-empty.

        Returns a list of (sample_key, jpeg_path_str, scanner_name).
        """
        entries: List[Tuple[str, str, str]] = []
        for slide_dir in sorted(root.iterdir()):
            if not slide_dir.is_dir():
                continue
            for sample_dir in sorted(slide_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sample_key = f"{slide_dir.name}/{sample_dir.name}"
                for jpg in sorted(sample_dir.glob("*.jpg")):
                    scanner_name = jpg.stem
                    if selected_scanners and scanner_name not in selected_scanners:
                        continue
                    entries.append((sample_key, str(jpg), scanner_name))
        return entries

    def _build_pairs(
        self,
        samples: List[Tuple[str, str, str]],
        symmetric: bool,
    ) -> List[Tuple[str, str, int, int]]:
        """
        Group entries by sample_key, then enumerate ordered scanner pairs.
        Pairs where either scanner name is absent from ``scanner_to_id`` are skipped.
        """
        # Group by sample_key → {scanner_name: jpeg_path}
        by_sample: Dict[str, Dict[str, str]] = {}
        for sample_key, jpeg_path, scanner_name in samples:
            by_sample.setdefault(sample_key, {})[scanner_name] = jpeg_path

        pairs: List[Tuple[str, str, int, int]] = []
        for scanner_map in by_sample.values():
            scanner_names = sorted(scanner_map)
            n = len(scanner_names)
            for i in range(n):
                j_range = range(n) if symmetric else range(i + 1, n)
                for j in j_range:
                    if i == j:
                        continue
                    src_name = scanner_names[i]
                    tgt_name = scanner_names[j]
                    if src_name not in self.scanner_to_id or tgt_name not in self.scanner_to_id:
                        continue
                    pairs.append((
                        scanner_map[src_name],
                        scanner_map[tgt_name],
                        self.scanner_to_id[src_name],
                        self.scanner_to_id[tgt_name],
                    ))
        return pairs

    def _load_jpeg(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _resolve_root(self, data_path: str) -> Path:
        if data_path is None:
            raise ValueError("`dataset_cfg.data_path` must be set for ScorpionDataset")
        root = Path(data_path)
        if root.exists():
            return root
        repo_root = Path(__file__).resolve().parents[3]
        alt = repo_root / data_path
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Could not resolve SCORPION data path '{data_path}'. "
            f"Tried '{root}' and '{alt}'."
        )

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


class ScorpionPrefeaturesDataset(data.Dataset):
    """
    SCORPION evaluation dataset backed by pre-extracted features.

    Layout::

        features_root/
            AT2.npy
            DP200.npy
            GT450.npy
            P1000.npy
            Philips.npy

    Each ``.npy`` has shape ``(N, 2 + D)`` where column 0 is the slide number,
    column 1 is the sample number, and the remaining columns are the feature
    vector.  Rows are sorted by (slide, sample) and therefore aligned across
    scanner files — patches are paired by row index.

    Returns the same 6-tuple as :class:`ScorpionDataset`::

        (feat_src, feat_tgt, src_scanner_id, tgt_scanner_id,
         src_staining_id, tgt_staining_id)

    ``staining_id`` is always 0 (single staining protocol).

    :param dataset_cfg: Config dict with at least a ``features_root`` key.
        Optional keys:
        - ``scanners``: list of scanner names to include (default: all found).
        - ``scanner_to_id``: external vocab mapping scanner name → integer id.
        - ``symmetric``: if ``True`` (default), both (A→B) and (B→A) are included.
    """

    N_COORD_COLS = 2

    def __init__(
        self,
        dataset_cfg,
        state: str = "test",
        transforms=None,
        foundation_model=None,
        general=None,
    ):
        if dataset_cfg is None:
            raise ValueError("`dataset_cfg` must be provided")

        features_root_str = self._cfg_get(dataset_cfg, "features_root", None)
        if not features_root_str:
            raise ValueError("`dataset_cfg.features_root` must be set for ScorpionPrefeaturesDataset")
        features_root = Path(features_root_str)
        if not features_root.exists():
            raise FileNotFoundError(f"SCORPION features root not found: {features_root}")

        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        external_vocab: Optional[Dict[str, int]] = self._cfg_get(dataset_cfg, "scanner_to_id", None)
        symmetric: bool = bool(self._cfg_get(dataset_cfg, "symmetric", True))

        npy_files = {p.stem: p for p in sorted(features_root.glob("*.npy"))}
        if selected_scanners:
            npy_files = {k: v for k, v in npy_files.items() if k in selected_scanners}
        if not npy_files:
            raise ValueError(f"No .npy feature files found under '{features_root}'")

        self._arrays: Dict[str, np.ndarray] = {
            name: np.load(path, mmap_mode="r") for name, path in npy_files.items()
        }

        ref_arr = next(iter(self._arrays.values()))
        self.n_samples: int = ref_arr.shape[0]
        self.feature_dim: int = ref_arr.shape[1] - self.N_COORD_COLS
        for name, arr in self._arrays.items():
            if arr.shape[0] != self.n_samples:
                raise ValueError(
                    f"{name}.npy has {arr.shape[0]} rows; expected {self.n_samples} "
                    "(all SCORPION scanner files must have the same number of samples)"
                )

        all_scanners = sorted(self._arrays)
        if external_vocab is not None:
            self.scanner_to_id: Dict[str, int] = external_vocab
        else:
            self.scanner_to_id = {name: i for i, name in enumerate(all_scanners)}
        self.staining_to_id: Dict[str, int] = {"SCORPION": 0}

        self._pairs: List[Tuple[str, str, int, int]] = self._build_pairs(all_scanners, symmetric)
        if not self._pairs:
            raise ValueError(
                "No valid scanner pairs found. Ensure at least two scanner names "
                "map into the provided scanner_to_id vocabulary."
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def use_all_test_samples(self) -> bool:
        return True

    @property
    def scanner_vocab_size(self) -> int:
        return len(self.scanner_to_id)

    @property
    def staining_vocab_size(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self._pairs) * self.n_samples

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")

        pair_idx = index // self.n_samples
        row_idx = index % self.n_samples
        src_name, tgt_name, src_id, tgt_id = self._pairs[pair_idx]

        feat_src = torch.from_numpy(
            self._arrays[src_name][row_idx, self.N_COORD_COLS:].copy()
        )
        feat_tgt = torch.from_numpy(
            self._arrays[tgt_name][row_idx, self.N_COORD_COLS:].copy()
        )
        return feat_src, feat_tgt, src_id, tgt_id, 0, 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pairs(
        self, scanner_names: List[str], symmetric: bool
    ) -> List[Tuple[str, str, int, int]]:
        pairs: List[Tuple[str, str, int, int]] = []
        n = len(scanner_names)
        for i in range(n):
            j_range = range(n) if symmetric else range(i + 1, n)
            for j in j_range:
                if i == j:
                    continue
                src_name = scanner_names[i]
                tgt_name = scanner_names[j]
                if src_name not in self.scanner_to_id or tgt_name not in self.scanner_to_id:
                    continue
                pairs.append((
                    src_name,
                    tgt_name,
                    self.scanner_to_id[src_name],
                    self.scanner_to_id[tgt_name],
                ))
        return pairs

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
