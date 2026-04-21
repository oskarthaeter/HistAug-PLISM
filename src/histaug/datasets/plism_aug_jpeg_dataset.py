import bisect
from typing import Dict, Tuple

import torch
from PIL import Image

from datasets.plism_jpeg_dataset import PlismJpegDataset


class PlismAugJpegDataset(PlismJpegDataset):
    """
    Extends PlismJpegDataset to also return the scanner and staining identity of each patch.

    Returns a 5-tuple:
        (original_tensor, augmented_tensor, aug_params, scanner_id, staining_id)

    scanner_to_id and staining_to_id vocabularies are built from the full filtered slide set
    (before train/test split) so IDs are stable across splits.

    Exposes scanner_vocab_size and staining_vocab_size so train.py can inject them into
    cfg.Model before the model is built.
    """

    def __init__(self, dataset_cfg, state="train", transforms=None, foundation_model=None, general=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            state=state,
            transforms=transforms,
            foundation_model=foundation_model,
            general=general,
        )
        # Build vocabularies from all slides seen by this split (after filtering, before splitting).
        # We re-discover and filter here to get the full set independent of state.
        all_slides = self._discover_slides(self.dataset_root)
        selected_scanners = self._as_str_set(self._cfg_get(dataset_cfg, "scanners", []))
        selected_stainings = self._as_str_set(self._cfg_get(dataset_cfg, "stainings", []))
        filtered = self._filter_slides(all_slides, selected_scanners, selected_stainings)

        scanners_all = sorted({s["scanner"] for s in filtered})
        stainings_all = sorted({s["staining"] for s in filtered})
        self.scanner_to_id: Dict[str, int] = {n: i for i, n in enumerate(scanners_all)}
        self.staining_to_id: Dict[str, int] = {n: i for i, n in enumerate(stainings_all)}

    @property
    def scanner_vocab_size(self) -> int:
        return len(self.scanner_to_id)

    @property
    def staining_vocab_size(self) -> int:
        return len(self.staining_to_id)

    def __getitem__(self, index: int):
        slide_idx, local_index = self._resolve_index(index)
        slide = self.slides[slide_idx]
        tile_paths = self._get_tile_paths(slide)
        tile_path = tile_paths[local_index]

        with Image.open(tile_path) as img:
            pil_image = img.convert("RGB")

        augmented_image, augmentation_params = self.transforms(pil_image)
        original_tensor = self.image_preprocessing_pipeline(pil_image)
        augmented_tensor = self.image_preprocessing_pipeline(augmented_image)

        scanner_id = self.scanner_to_id[slide["scanner"]]
        staining_id = self.staining_to_id[slide["staining"]]

        return original_tensor, augmented_tensor, augmentation_params, scanner_id, staining_id

    def _resolve_index(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        slide_idx = bisect.bisect_right(self.slide_offsets, index) - 1
        local_index = index - self.slide_offsets[slide_idx]
        return slide_idx, local_index
