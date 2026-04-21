"""
Shared utilities for organ-based train/test splitting.

All PLISM datasets share the same 16,278 patches identified by
(level, left, top) coordinates. The organ CSV maps each coordinate
to an organ type. This module provides:

  - load_organ_map:     {(level, left, top): organ}
  - build_organ_sets:   deterministic 80/20 split of organ labels
  - tile_key_to_coords: parse 'tile_{level}_{left}_{top}' → (level, left, top)
  - tile_key_from_path: extract tile_key from a JPEG filename stem
"""

import csv
import random
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple


Coords = Tuple[int, int, int]


def load_organ_map(csv_path: Path) -> Dict[Coords, str]:
    """Return {(level, left, top): organ} from plism_organ_loc.csv."""
    organ_map: Dict[Coords, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["level"]), int(row["left"]), int(row["top"]))
            organ_map[key] = row["organ"]
    if not organ_map:
        raise ValueError(f"organ_loc_csv '{csv_path}' produced an empty mapping")
    return organ_map


def build_organ_sets(
    organ_map: Dict[Coords, str],
    train_fraction: float,
    split_seed: int,
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """
    Deterministically split organ labels into train and test sets.

    Returns (train_organs, test_organs) as frozensets of organ name strings.
    The split is at the organ level so no organ appears in both sets.
    """
    all_organs = sorted(set(organ_map.values()))
    rng = random.Random(split_seed)
    shuffled = list(all_organs)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * (1.0 - train_fraction)))
    test_organs: FrozenSet[str] = frozenset(shuffled[:n_test])
    train_organs: FrozenSet[str] = frozenset(shuffled[n_test:])
    return train_organs, test_organs


def tile_key_to_coords(tile_key: str) -> Coords:
    """Parse 'tile_{level}_{left}_{top}' → (level, left, top)."""
    parts = tile_key.split("_")
    # Expected: ['tile', level, left, top]
    if len(parts) != 4 or parts[0] != "tile":
        raise ValueError(
            f"Unexpected tile_key format {tile_key!r}. Expected 'tile_<level>_<left>_<top>'."
        )
    return int(parts[1]), int(parts[2]), int(parts[3])


def tile_key_from_path(jpeg_path: Path) -> str:
    """
    Extract tile_key from a JPEG path whose stem is '{idx}__{hash}__{tile_key}'.

    Works with both absolute paths and bare filenames.
    """
    stem = jpeg_path.stem  # drop '.jpg'
    parts = stem.split("__")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse tile_key from JPEG path {jpeg_path!r}. "
            "Expected stem format '{idx}__{hash}__{tile_key}'."
        )
    return parts[-1]


def build_valid_tile_keys(
    organ_map: Dict[Coords, str],
    valid_organs: FrozenSet[str],
) -> Set[str]:
    """
    Return the set of tile_key strings (e.g. 'tile_16_8_20') whose organ is in valid_organs.
    """
    result: Set[str] = set()
    for (level, left, top), organ in organ_map.items():
        if organ in valid_organs:
            result.add(f"tile_{level}_{left}_{top}")
    return result
