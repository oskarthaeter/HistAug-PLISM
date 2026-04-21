#!/usr/bin/env python3
"""Export PLISM .tif.h5 tile datasets to individual JPEG files.

Each tile is saved as one JPEG, and a manifest CSV is written to keep a stable
mapping between exported image files and their source (slide file + tile key).

Example:
    cd src/histaug
    python scripts/export_plism_h5_to_jpeg.py \
        --input-dir ../../plism \
        --output-dir ../../plism_jpeg \
        --quality 95
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PLISM .tif.h5 tiles to JPEG files with a manifest CSV"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PLISM .tif.h5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JPEG files and manifest CSV will be written",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100), default: 95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPEG files if present",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.csv",
        help="Output manifest CSV filename (default: manifest.csv)",
    )
    return parser.parse_args()


def discover_h5_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.tif.h5"))


def array_to_pil(image_array: np.ndarray) -> Image.Image:
    arr = np.asarray(image_array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return Image.fromarray(arr[..., :3], mode="RGB")

    raise ValueError(f"Unexpected tile shape: {arr.shape}")


def sanitize_for_filename(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned[:max_len] if cleaned else "tile"


def tile_filename(tile_index: int, tile_key: str) -> str:
    tile_hash = hashlib.sha1(tile_key.encode("utf-8")).hexdigest()[:10]
    safe_key = sanitize_for_filename(tile_key)
    return f"{tile_index:08d}__{tile_hash}__{safe_key}.jpg"


def iter_tile_keys(h5_file: h5py.File) -> Iterable[str]:
    return sorted(h5_file.keys())


def export_slide(
    h5_path: Path,
    output_dir: Path,
    csv_writer: csv.DictWriter,
    quality: int,
    overwrite: bool,
) -> int:
    slide_id = h5_path.stem
    slide_out_dir = output_dir / slide_id
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as h5f:
        keys = list(iter_tile_keys(h5f))
        for tile_index, tile_key in enumerate(keys):
            tile_array = h5f[tile_key][:]
            pil_img = array_to_pil(tile_array)

            jpg_name = tile_filename(tile_index, tile_key)
            jpg_path = slide_out_dir / jpg_name
            rel_jpg_path = jpg_path.relative_to(output_dir)

            if overwrite or not jpg_path.exists():
                pil_img.save(
                    jpg_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    subsampling=0,
                )

            csv_writer.writerow(
                {
                    "image_id": f"{slide_id}::{tile_key}",
                    "slide_id": slide_id,
                    "tile_key": tile_key,
                    "tile_index": tile_index,
                    "source_h5": str(h5_path.resolve()),
                    "jpeg_path": str(rel_jpg_path),
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "channels": 3,
                }
            )
            exported_count += 1

    return exported_count


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    if not (1 <= args.quality <= 100):
        raise ValueError("--quality must be between 1 and 100")

    h5_files = discover_h5_files(args.input_dir)
    if not h5_files:
        raise FileNotFoundError(
            f"No '.tif.h5' files found in input directory: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_name

    fieldnames = [
        "image_id",
        "slide_id",
        "tile_key",
        "tile_index",
        "source_h5",
        "jpeg_path",
        "width",
        "height",
        "channels",
    ]

    total_tiles = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, h5_path in enumerate(h5_files, start=1):
            print(f"[{i}/{len(h5_files)}] Exporting {h5_path.name}...")
            n_tiles = export_slide(
                h5_path=h5_path,
                output_dir=args.output_dir,
                csv_writer=writer,
                quality=args.quality,
                overwrite=args.overwrite,
            )
            total_tiles += n_tiles
            print(f"    -> {n_tiles} tiles")

    print("Export complete")
    print(f"Slides processed: {len(h5_files)}")
    print(f"Tiles exported: {total_tiles}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
