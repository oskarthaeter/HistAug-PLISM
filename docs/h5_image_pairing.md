# Working with PLISM `.h5` tiles and matched pairs

Path: `plism/` (symlink to dataset directory)

This note explains how to:

1. read tile images from PLISM `.h5` files,
2. identify slide metadata (staining and scanner),
3. match the same tile location across slides,
4. build pairs for training on matched images in another repository.

The conventions below are based on the extraction/evaluation pipeline in this repository.

## 1) What is inside one `.h5` file

Each `.h5` file corresponds to one slide variant (one staining + one scanner) and contains tile datasets.

- File pattern used in this repo: `*.tif.h5`
- Number of tiles per slide: `16278`
- Tile dataset key: a `tile_id` string
- Tile dataset value: image array (typically `H x W x C`, uint8)

In `plismbench`, each batch keeps `(tile_id, tile_array)`, and tile coordinates are parsed from `tile_id` with:

```python
tile_coords = tile_id.split("_")[1:]  # 3 integer coordinates
```

So the tile key is the source of spatial correspondence.

## 2) Read images from an `.h5` file

```python
from pathlib import Path
import h5py
import numpy as np


def load_h5_tiles(h5_path: Path):
    """Return a dict: tile_id -> image array."""
    out = {}
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        for tile_id in f.keys():
            out[tile_id] = f[tile_id][:]  # numpy array
    return out


def load_one_tile(h5_path: Path, tile_id: str) -> np.ndarray:
    """Read a single tile by key."""
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        return f[tile_id][:]
```

For large-scale training, avoid loading all tiles into RAM. Stream by key/batch instead.

## 3) Parse slide-level metadata (staining/scanner)

In the benchmark code, slide metadata is extracted from the slide directory name using:

```python
staining, scanner = slide_name.split("_")[:2]
```

So if your slide id is `GMH_S60_to_GMH_S60.tif`, then:

- staining = `GMH`
- scanner = `S60`

Use the same convention in your training repo to keep compatibility.

## 4) Match images between scanners and stainings

### Core rule

Two images are matched if they have the same `tile_id` in two different slide files.

This is exactly the assumption used by the benchmark metrics:

- tile coordinates are extracted from tile ids,
- slide features are sorted by coordinates,
- and coordinates must match across slide pairs.

### Pairing strategy

1. Build the set of tile keys for each slide file.
2. For each slide pair `(A, B)`, compute `common_keys = keys(A) ∩ keys(B)`.
3. Use each common key to create one training pair:
   - source image: `A[key]`
   - target image: `B[key]`
4. Optionally filter slide pairs by metadata:
   - cross-scanner: same staining, different scanner
   - cross-staining: same scanner, different staining
   - cross-scanner + cross-staining: both different

## 5) End-to-end pair index builder

This example writes a parquet index that another repo can load to sample matched pairs lazily.

```python
from pathlib import Path
import h5py
import pandas as pd


def slide_id_from_path(h5_path: Path) -> str:
    # Example: "GMH_S60_to_GMH_S60.tif.h5" -> "GMH_S60_to_GMH_S60.tif"
    return h5_path.stem


def parse_staining_scanner(slide_id: str) -> tuple[str, str]:
    staining, scanner = slide_id.split("_")[:2]
    return staining, scanner


def keys_of_h5(h5_path: Path) -> set[str]:
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        return set(f.keys())


def pair_type(meta_a: tuple[str, str], meta_b: tuple[str, str]) -> str:
    staining_a, scanner_a = meta_a
    staining_b, scanner_b = meta_b
    same_staining = staining_a == staining_b
    same_scanner = scanner_a == scanner_b

    if same_staining and not same_scanner:
        return "cross_scanner"
    if not same_staining and same_scanner:
        return "cross_staining"
    if not same_staining and not same_scanner:
        return "cross_scanner_cross_staining"
    return "same_scanner_same_staining"


def build_pair_index(h5_dir: Path, output_path: Path) -> pd.DataFrame:
    h5_paths = sorted(h5_dir.glob("*.tif.h5"))
    slide_info = []

    for p in h5_paths:
        sid = slide_id_from_path(p)
        slide_info.append(
            {
                "slide_id": sid,
                "h5_path": str(p),
                "staining": parse_staining_scanner(sid)[0],
                "scanner": parse_staining_scanner(sid)[1],
                "keys": keys_of_h5(p),
            }
        )

    rows = []
    for i in range(len(slide_info)):
        for j in range(i + 1, len(slide_info)):
            a = slide_info[i]
            b = slide_info[j]
            common = sorted(a["keys"].intersection(b["keys"]))
            ptype = pair_type((a["staining"], a["scanner"]), (b["staining"], b["scanner"]))

            for tile_id in common:
                rows.append(
                    {
                        "slide_a": a["slide_id"],
                        "slide_b": b["slide_id"],
                        "h5_path_a": a["h5_path"],
                        "h5_path_b": b["h5_path"],
                        "tile_id": tile_id,
                        "pair_type": ptype,
                    }
                )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df
```

## 6) Lazy loading in a training dataset

In your training repo, store only paths and tile ids in the index. During `__getitem__`, open both files and read one matched tile:

```python
import h5py


def read_matched_pair(h5_path_a: str, h5_path_b: str, tile_id: str):
    with h5py.File(h5_path_a, "r", libver="latest", swmr=True) as fa, h5py.File(h5_path_b, "r", libver="latest", swmr=True) as fb:
        img_a = fa[tile_id][:]
        img_b = fb[tile_id][:]
    return img_a, img_b
```

This pattern scales much better than materializing all images first.

## 7) Practical checks before training

Run these checks once when generating the pair index:

- Verify expected slide count and tile count per slide.
- Check each candidate pair has enough `common_keys`.
- Assert image shape consistency for sampled matched keys.
- Keep a fixed random seed when subsampling pairs.

Minimal consistency check:

```python
assert len(common_keys) > 0
assert img_a.shape == img_b.shape
```

## 8) Recommended export format for the other repo

Export a table with at least:

- `h5_path_a`
- `h5_path_b`
- `tile_id`
- `pair_type`
- optional: `slide_a`, `slide_b`, `scanner_a`, `scanner_b`, `staining_a`, `staining_b`

This keeps your training code simple and makes stratified sampling by pair type easy.
