import csv
import random
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import torch.utils.data as data
from models.foundation_models import get_fm_transform
from PIL import Image
from utils.organ_split import build_organ_sets, build_valid_tile_keys, load_organ_map
from utils.transform_factory import create_transform


class PlismPairJpegDataset(data.Dataset):
    """
    Paired dataset for scanner-transfer training on PLISM JPEG tiles.

    Each item is a matched pair (image_a, image_b) that share the same tile_key
    across two slides from different scanners (optionally different stainings).
    The same augmentation (for diversity only) is applied to both images.

    Expected layout:
        data_path/
            manifest.csv             # with columns: slide_id, tile_key, jpeg_path, tile_index, ...
            <slide_id>/*.jpg

    Returned tuple:
        (img_a, img_b, src_scanner_id, tgt_scanner_id, src_staining_id, tgt_staining_id)
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

        pairing_cfg = self._cfg_get(dataset_cfg, "pairing", {}) or {}
        self.allow_cross_staining = bool(
            self._cfg_get(pairing_cfg, "allow_cross_staining", False)
        )
        self.allow_same_scanner = bool(
            self._cfg_get(pairing_cfg, "allow_same_scanner", False)
        )
        # Symmetric pairing yields both (A->B) and (B->A) entries.
        self.symmetric = bool(self._cfg_get(pairing_cfg, "symmetric", True))
        # Virtual epoch size: number of (tile) samples drawn per slide-pair per epoch.
        # A tile_key is drawn uniformly at random from each slide-pair's common set
        # at __getitem__ time, so the materialised index only stores slide-pairs.
        self.tiles_per_pair_per_epoch = int(
            self._cfg_get(pairing_cfg, "tiles_per_pair_per_epoch", 256)
        )
        if self.tiles_per_pair_per_epoch <= 0:
            raise ValueError("`pairing.tiles_per_pair_per_epoch` must be > 0")

        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        selected_stainings = self._as_str_set(
            self._cfg_get(dataset_cfg, "stainings", [])
        )

        self.manifest_path = self.dataset_root / "manifest.csv"
        slides = self._load_manifest(self.manifest_path)

        slides = self._filter_slides(
            slides,
            selected_scanners=selected_scanners,
            selected_stainings=selected_stainings,
        )
        if not slides:
            raise ValueError(
                "No PLISM JPEG slides matched the provided filters in `Data.scanners` / `Data.stainings`."
            )

        train_slides, test_slides = self._split_slides_stratified(slides)

        if state == "train":
            split_slides = train_slides
        elif state in ("val", "test"):
            split_slides = test_slides
        else:
            raise ValueError(f"Unknown dataset state: {state}")

        # Organ-based tile filter: only tile_keys from train/test organs are valid.
        organ_loc_csv = self._cfg_get(dataset_cfg, "organ_loc_csv", None)
        if organ_loc_csv:
            organ_map = load_organ_map(self._resolve_data_path(str(organ_loc_csv)))
            train_organs, test_organs = build_organ_sets(
                organ_map, self.train_fraction, self.split_seed
            )
            valid_organs = train_organs if state == "train" else test_organs
            self._valid_tile_keys: Optional[Set[str]] = build_valid_tile_keys(
                organ_map, valid_organs
            )
        else:
            self._valid_tile_keys = None

        # Build slide-pair index within this split only (train/test pair disjointness
        # follows from slide-level disjointness above). Each entry stores the two
        # slides and their common tile_keys; a tile is sampled per __getitem__.
        self.slide_pairs = self._build_slide_pair_index(split_slides)
        if not self.slide_pairs:
            raise ValueError(
                f"Split '{state}' produced zero slide-pairs. Loosen `Data.pairing` "
                "filters or broaden scanner/staining selection."
            )

        # Vocabularies are built from the filtered (pre-split) slide set so train/test
        # share the same integer ids.
        scanners_all = sorted({s["scanner"] for s in slides})
        stainings_all = sorted({s["staining"] for s in slides})
        self.scanner_to_id = {name: i for i, name in enumerate(scanners_all)}
        self.staining_to_id = {name: i for i, name in enumerate(stainings_all)}

    @property
    def use_all_test_samples(self) -> bool:
        return True

    @property
    def scanner_vocab_size(self) -> int:
        return len(self.scanner_to_id)

    @property
    def staining_vocab_size(self) -> int:
        return len(self.staining_to_id)

    def __len__(self) -> int:
        return len(self.slide_pairs) * self.tiles_per_pair_per_epoch

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )

        pair = self.slide_pairs[index // self.tiles_per_pair_per_epoch]
        tile_key = random.choice(pair["common_tile_keys"])
        img_a = self._load_jpeg(pair["tile_map_a"][tile_key])
        img_b = self._load_jpeg(pair["tile_map_b"][tile_key])

        # Sample one set of augmentation parameters and apply to BOTH images.
        aug_params = self.transforms.sample_aug_params()
        img_a = self.transforms.apply_transform(img_a, aug_params)
        img_b = self.transforms.apply_transform(img_b, aug_params)

        tensor_a = self.image_preprocessing_pipeline(img_a)
        tensor_b = self.image_preprocessing_pipeline(img_b)

        return (
            tensor_a,
            tensor_b,
            self.scanner_to_id[pair["scanner_a"]],
            self.scanner_to_id[pair["scanner_b"]],
            self.staining_to_id[pair["staining_a"]],
            self.staining_to_id[pair["staining_b"]],
        )

    def _load_jpeg(self, rel_path: str) -> Image.Image:
        abs_path = self.dataset_root / rel_path
        with Image.open(abs_path) as img:
            return img.convert("RGB")

    def _resolve_data_path(self, path_str: str) -> Path:
        """Resolve a config path relative to repo root when not absolute."""
        p = Path(path_str)
        if p.exists():
            return p
        alt = Path(__file__).resolve().parents[3] / path_str
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Could not resolve path '{path_str}'. Tried '{p}' and '{alt}'."
        )

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

    def _load_manifest(self, manifest_path: Path) -> List[Dict]:
        """Return a list of slide dicts with a tile_key -> jpeg_path mapping."""
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest '{manifest_path}' not found. PlismPairJpegDataset requires the "
                "manifest.csv produced by export_plism_h5_to_jpeg.py (it provides `tile_key` "
                "which is the cross-slide match key)."
            )

        required_cols = {"slide_id", "tile_key", "jpeg_path"}
        per_slide: Dict[str, Dict[str, str]] = {}

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
                tile_key = row["tile_key"]
                jpeg_path = row["jpeg_path"]
                per_slide.setdefault(slide_id, {})[tile_key] = jpeg_path

        slides = []
        for slide_id, tile_map in per_slide.items():
            if not tile_map:
                continue
            staining, scanner = self._parse_staining_scanner(slide_id)
            slides.append(
                {
                    "slide_id": slide_id,
                    "staining": staining,
                    "scanner": scanner,
                    "tile_map": tile_map,
                }
            )
        if not slides:
            raise ValueError(f"Manifest '{manifest_path}' has no usable rows")
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

        train_slides, test_slides = [], []
        for group_slides in groups.values():
            shuffled = list(group_slides)
            rng.shuffle(shuffled)
            n_group = len(shuffled)
            if n_group == 1:
                n_test = 0
            else:
                n_test = max(1, int(round(n_group * (1.0 - self.train_fraction))))
                # Need ≥2 slides per group in each split to form intra-group pairs
                # when cross-group pairing is disallowed.
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
        """
        Enumerate ordered slide-pairs (a, b) subject to pairing filters. Each entry
        caches the common tile_keys and the two per-slide tile_key->jpeg_path maps.
        Individual tiles are sampled lazily in __getitem__.
        """
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

                common = sorted(set(a["tile_map"].keys()) & set(b["tile_map"].keys()))
                if self._valid_tile_keys is not None:
                    common = [k for k in common if k in self._valid_tile_keys]
                if not common:
                    continue

                pairs.append(
                    {
                        "tile_map_a": a["tile_map"],
                        "tile_map_b": b["tile_map"],
                        "common_tile_keys": common,
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
