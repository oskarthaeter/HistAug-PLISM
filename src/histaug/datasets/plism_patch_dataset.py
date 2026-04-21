import bisect
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch.utils.data as data
from models.foundation_models import get_fm_transform
from PIL import Image
from utils.transform_factory import create_transform


class PlismPatchDataset(data.Dataset):
    """
    Dataset for PLISM tissue patches stored directly in `.tif.h5` files.

    Each `.tif.h5` file is treated as one slide and each key in that H5 file is one
    patch tile. By default, all tiles from selected slides are used.
    """

    def __init__(
        self,
        dataset_cfg,
        state="train",
        transforms=None,
        foundation_model=None,
        general=None,
    ):
        if dataset_cfg is None:
            raise ValueError("`dataset_cfg` must be provided")
        if not foundation_model:
            raise ValueError("Foundation model must be provided")

        self.state = state
        self.dataset_root = self._resolve_data_root(str(dataset_cfg.data_path))
        self.transforms = create_transform(transforms)
        self.foundation_model = foundation_model
        self.image_preprocessing_pipeline, _ = get_fm_transform(self.foundation_model)

        self.split_seed = int(
            self._cfg_get(
                dataset_cfg, "split_seed", self._cfg_get(general, "seed", 2025)
            )
        )
        self.train_fraction = float(self._cfg_get(dataset_cfg, "train_split", 0.8))
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("`Data.train_split` must be in the open interval (0, 1)")

        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        selected_stainings = self._as_str_set(
            self._cfg_get(dataset_cfg, "stainings", [])
        )

        all_slides = self._discover_slides(self.dataset_root)
        filtered_slides = self._filter_slides(
            all_slides,
            selected_scanners=selected_scanners,
            selected_stainings=selected_stainings,
        )
        if not filtered_slides:
            raise ValueError(
                "No PLISM slides matched the provided filters in `Data.scanners` / `Data.stainings`."
            )

        train_slides, test_slides = self._split_slides_stratified(filtered_slides)

        train_ids = {slide["slide_id"] for slide in train_slides}
        test_ids = {slide["slide_id"] for slide in test_slides}
        if train_ids.intersection(test_ids):
            raise RuntimeError(
                "Train/test slide overlap detected, which should never happen"
            )

        if state == "train":
            self.slides = train_slides
        elif state in ("val", "test"):
            self.slides = test_slides
        else:
            raise ValueError(f"Unknown dataset state: {state}")

        if not self.slides:
            raise ValueError(
                f"Split '{state}' is empty after filtering/splitting. "
                "Try broadening `Data.scanners`/`Data.stainings` filters."
            )

        self.slide_offsets = self._build_offsets(self.slides)
        self.tile_key_cache: Dict[str, List[str]] = {}

    @property
    def use_all_test_samples(self) -> bool:
        return True

    def __len__(self) -> int:
        return self.slide_offsets[-1] if self.slide_offsets else 0

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )

        slide_idx = bisect.bisect_right(self.slide_offsets, index) - 1
        local_index = index - self.slide_offsets[slide_idx]
        slide = self.slides[slide_idx]
        tile_keys = self._get_tile_keys(slide)
        tile_id = tile_keys[local_index]

        tile_array = self._read_tile(slide["h5_path"], tile_id)
        pil_image = self._array_to_pil(tile_array)

        augmented_image, augmentation_params = self.transforms(pil_image)
        original_tensor = self.image_preprocessing_pipeline(pil_image)
        augmented_tensor = self.image_preprocessing_pipeline(augmented_image)

        return original_tensor, augmented_tensor, augmentation_params

    def _resolve_data_root(self, data_path: str) -> Path:
        root = Path(data_path)
        if root.exists():
            return root

        repo_root = Path(__file__).resolve().parents[3]
        alt_root = repo_root / data_path
        if alt_root.exists():
            return alt_root

        raise FileNotFoundError(
            f"Could not resolve PLISM data path '{data_path}'. Tried '{root}' and '{alt_root}'."
        )

    def _discover_slides(self, h5_dir: Path) -> List[Dict]:
        slides = []
        for h5_path in sorted(h5_dir.glob("*.tif.h5")):
            slide_id = h5_path.stem
            staining, scanner = self._parse_staining_scanner(slide_id)
            n_tiles = self._count_tiles(h5_path)
            if n_tiles == 0:
                continue

            slides.append(
                {
                    "slide_id": slide_id,
                    "h5_path": str(h5_path),
                    "staining": staining,
                    "scanner": scanner,
                    "n_tiles": n_tiles,
                }
            )

        if not slides:
            raise ValueError(f"No '*.tif.h5' files found in '{h5_dir}'")
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
        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for slide in slides:
            key = (slide["staining"], slide["scanner"])
            groups.setdefault(key, []).append(slide)

        train_slides = []
        test_slides = []
        for group_slides in groups.values():
            shuffled = list(group_slides)
            rng.shuffle(shuffled)

            n_group = len(shuffled)
            if n_group == 1:
                n_test = 0
            else:
                n_test = max(1, int(round(n_group * (1.0 - self.train_fraction))))
                n_test = min(n_test, n_group - 1)

            test_slides.extend(shuffled[:n_test])
            train_slides.extend(shuffled[n_test:])

        if not train_slides:
            raise ValueError(
                "Train split ended up empty; not enough slides after filtering"
            )
        if not test_slides:
            # Fall back to moving one random slide from train to test.
            moved_idx = rng.randrange(len(train_slides))
            test_slides.append(train_slides.pop(moved_idx))

        return train_slides, test_slides

    def _build_offsets(self, slides: Sequence[Dict]) -> List[int]:
        offsets = [0]
        running = 0
        for slide in slides:
            running += int(slide["n_tiles"])
            offsets.append(running)
        return offsets

    def _get_tile_keys(self, slide: Dict) -> List[str]:
        cache_key = slide["slide_id"]
        if cache_key not in self.tile_key_cache:
            with h5py.File(slide["h5_path"], "r", libver="latest", swmr=True) as f:
                self.tile_key_cache[cache_key] = sorted(f.keys())

        tile_keys = self.tile_key_cache[cache_key]
        expected = int(slide["n_tiles"])
        if len(tile_keys) != expected:
            raise RuntimeError(
                f"Tile count mismatch for {slide['slide_id']}: expected {expected}, got {len(tile_keys)}"
            )
        return tile_keys

    def _read_tile(self, h5_path: str, tile_id: str) -> np.ndarray:
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
            return f[tile_id][:]

    def _count_tiles(self, h5_path: Path) -> int:
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
            return len(f.keys())

    def _parse_staining_scanner(self, slide_id: str) -> Tuple[str, str]:
        tokens = slide_id.split("_")
        if len(tokens) < 2:
            raise ValueError(
                f"Invalid PLISM slide id '{slide_id}'. Expected pattern '<staining>_<scanner>_...'."
            )
        return tokens[0], tokens[1]

    def _array_to_pil(self, image_array: np.ndarray) -> Image.Image:
        arr = np.asarray(image_array)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return Image.fromarray(arr[..., 0], mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return Image.fromarray(arr[..., :3], mode="RGB")

        raise ValueError(f"Unexpected PLISM tile shape: {arr.shape}")

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
