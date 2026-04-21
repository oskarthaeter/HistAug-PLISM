import bisect
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch.utils.data as data
from models.foundation_models import get_fm_transform
from PIL import Image
from utils.organ_split import (
    build_organ_sets,
    build_valid_tile_keys,
    load_organ_map,
    tile_key_from_path,
)
from utils.transform_factory import create_transform


class PlismJpegDataset(data.Dataset):
    """
    Dataset for PLISM tissue patches exported as individual JPEG files.

    Expected layout:
        data_path/
            <slide_id>/
                *.jpg
            ...

    Each slide directory is treated as one WSI, and each JPEG in that directory is one tile.
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

        self.manifest_path = self.dataset_root / "manifest.csv"
        self.manifest_tile_paths = self._load_manifest_index(self.manifest_path)

        # Organ-based tile filter: only tiles from train/test organs are valid.
        organ_loc_csv = self._cfg_get(dataset_cfg, "organ_loc_csv", None)
        if organ_loc_csv:
            organ_map = load_organ_map(self._resolve_path(str(organ_loc_csv)))
            train_organs, test_organs = build_organ_sets(
                organ_map, self.train_fraction, self.split_seed
            )
            valid_organs = train_organs if state == "train" else test_organs
            self._valid_tile_keys: Optional[Set[str]] = build_valid_tile_keys(
                organ_map, valid_organs
            )
            if self.manifest_tile_paths:
                self._apply_organ_filter_to_manifest()
        else:
            self._valid_tile_keys = None

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
                "No PLISM JPEG slides matched the provided filters in `Data.scanners` / `Data.stainings`."
            )

        if self._valid_tile_keys is not None:
            # Organ split already provides train/test separation at tile level.
            # All slides are valid for any state; tile filtering handles the rest.
            self.slides = filtered_slides
        else:
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
        self.tile_path_cache: Dict[str, List[Path]] = {}

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
        tile_paths = self._get_tile_paths(slide)
        tile_path = tile_paths[local_index]

        with Image.open(tile_path) as img:
            pil_image = img.convert("RGB")

        augmented_image, augmentation_params = self.transforms(pil_image)
        original_tensor = self.image_preprocessing_pipeline(pil_image)
        augmented_tensor = self.image_preprocessing_pipeline(augmented_image)

        return original_tensor, augmented_tensor, augmentation_params

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.exists():
            return p
        alt = Path(__file__).resolve().parents[3] / path_str
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Could not resolve path '{path_str}'. Tried '{p}' and '{alt}'."
        )

    def _apply_organ_filter_to_manifest(self) -> None:
        """Filter self.manifest_tile_paths in-place to only valid tile_keys."""
        filtered: Dict[str, List[Path]] = {}
        for slide_id, paths in self.manifest_tile_paths.items():
            valid = [p for p in paths if tile_key_from_path(p) in self._valid_tile_keys]
            if valid:
                filtered[slide_id] = valid
        self.manifest_tile_paths = filtered

    def _resolve_data_root(self, data_path: str) -> Path:
        root = Path(data_path)
        if root.exists():
            return root

        repo_root = Path(__file__).resolve().parents[3]
        alt_root = repo_root / data_path
        if alt_root.exists():
            return alt_root

        raise FileNotFoundError(
            f"Could not resolve PLISM JPEG data path '{data_path}'. Tried '{root}' and '{alt_root}'."
        )

    def _discover_slides(self, root_dir: Path) -> List[Dict]:
        if self.manifest_tile_paths:
            slides = []
            for slide_id in sorted(self.manifest_tile_paths.keys()):
                staining, scanner = self._parse_staining_scanner(slide_id)
                n_tiles = len(self.manifest_tile_paths[slide_id])
                if n_tiles == 0:
                    continue

                slides.append(
                    {
                        "slide_id": slide_id,
                        "slide_dir": root_dir / slide_id,
                        "staining": staining,
                        "scanner": scanner,
                        "n_tiles": n_tiles,
                    }
                )

            if not slides:
                raise ValueError(
                    f"Manifest '{self.manifest_path}' exists but has no usable rows"
                )
            return slides

        slides = []
        for slide_dir in sorted(root_dir.iterdir()):
            if not slide_dir.is_dir():
                continue

            slide_id = slide_dir.name
            if slide_id.startswith("."):
                continue

            staining, scanner = self._parse_staining_scanner(slide_id)
            n_tiles = self._count_tiles(slide_dir)
            if n_tiles == 0:
                continue

            slides.append(
                {
                    "slide_id": slide_id,
                    "slide_dir": slide_dir,
                    "staining": staining,
                    "scanner": scanner,
                    "n_tiles": n_tiles,
                }
            )

        if not slides:
            raise ValueError(f"No slide folders with JPEG tiles found in '{root_dir}'")
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

    def _get_tile_paths(self, slide: Dict) -> List[Path]:
        cache_key = slide["slide_id"]
        if cache_key not in self.tile_path_cache:
            if cache_key in self.manifest_tile_paths:
                paths = self.manifest_tile_paths[cache_key]
            else:
                paths = sorted(slide["slide_dir"].glob("*.jpg"))
                if self._valid_tile_keys is not None:
                    paths = [p for p in paths if tile_key_from_path(p) in self._valid_tile_keys]
            self.tile_path_cache[cache_key] = paths

        tile_paths = self.tile_path_cache[cache_key]
        expected = int(slide["n_tiles"])
        if len(tile_paths) != expected:
            raise RuntimeError(
                f"Tile count mismatch for {slide['slide_id']}: expected {expected}, got {len(tile_paths)}"
            )
        return tile_paths

    def _count_tiles(self, slide_dir: Path) -> int:
        if self._valid_tile_keys is None:
            return sum(1 for _ in slide_dir.glob("*.jpg"))
        return sum(
            1 for p in slide_dir.glob("*.jpg")
            if tile_key_from_path(p) in self._valid_tile_keys
        )

    def _load_manifest_index(self, manifest_path: Path) -> Dict[str, List[Path]]:
        """
        Load tile paths from manifest.csv generated by export_plism_h5_to_jpeg.py.

        The manifest is treated as the source of truth when present.
        """
        if not manifest_path.exists():
            return {}

        required_cols = {"slide_id", "jpeg_path", "tile_index"}
        per_slide = {}

        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not required_cols.issubset(
                set(reader.fieldnames)
            ):
                raise ValueError(
                    f"Manifest '{manifest_path}' is missing required columns: {sorted(required_cols)}"
                )

            for row in reader:
                slide_id = row["slide_id"]
                rel_path = row["jpeg_path"]
                tile_index = int(row["tile_index"])
                abs_path = self.dataset_root / rel_path
                per_slide.setdefault(slide_id, []).append((tile_index, abs_path))

        indexed_paths = {}
        for slide_id, entries in per_slide.items():
            entries.sort(key=lambda x: x[0])
            indexed_paths[slide_id] = [p for _, p in entries]

        return indexed_paths

    def _parse_staining_scanner(self, slide_id: str) -> Tuple[str, str]:
        tokens = slide_id.split("_")
        if len(tokens) < 2:
            raise ValueError(
                f"Invalid PLISM slide id '{slide_id}'. Expected pattern '<staining>_<scanner>_...'."
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
