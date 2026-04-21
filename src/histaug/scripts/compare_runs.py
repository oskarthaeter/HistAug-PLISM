"""
compare_runs.py — side-by-side comparison of two HistAug run directories.

Usage:
    python compare_runs.py <run_dir_a> <run_dir_b> \\
        [--label-a NAME] [--label-b NAME] \\
        [--output comparison.pdf]

Produces a PDF (or PNG when the path ends in .png) containing:
  1. A scalar-metrics table (key test / holdout / scorpion metrics, Δ coloured).
  2. For each predictions_*.h5 file present in both run directories: scanner
     heatmaps for run A and run B (pred cosine), plus an exact numerical B-A
     difference heatmap.

Each H5 file is fully self-contained: it stores cosine_similarity alongside
src/tgt scanner and staining IDs, and the vocabulary as JSON attributes.
No splits.json or any other sidecar file is required.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Scalar metrics shown in the comparison table
# ---------------------------------------------------------------------------

KEY_METRICS = [
    "test/mean_origtrans_cos",
    "test/mean_imgaug_cos",
    "test/mean_imgaug_cos_ci_low",
    "test/mean_imgaug_cos_ci_high",
    "test/relative_improvement",
    "test/cos_same_vendor",
    "test/cos_diff_vendor",
    "test_holdout_staining/mean_origtrans_cos",
    "test_holdout_staining/mean_imgaug_cos",
    "test_holdout_staining/relative_improvement",
    "test_holdout_staining/cos_same_vendor",
    "test_holdout_staining/cos_diff_vendor",
    "scorpion/mean_origtrans_cos",
    "scorpion/mean_imgaug_cos",
    "scorpion/relative_improvement",
    "scorpion/cos_same_vendor",
    "scorpion/cos_diff_vendor",
]

# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

CMAP_PRED = "RdYlGn"
CMAP_DIFF = "RdBu_r"


def find_summary(run_dir: Path) -> dict:
    candidates = list(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
    return json.loads(candidates[-1].read_text()) if candidates else {}


def load_h5_matrix(h5_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """
    Build an NxN mean-cosine matrix from a self-contained predictions H5 file.

    Returns (matrix, scanner_names) or None if the file lacks metadata.
    Samples with src_scanner_id == -1 or tgt_scanner_id == -1 are skipped
    (these come from basic / conditioned HistAug steps with no pair context).
    """
    with h5py.File(h5_path, "r") as f:
        if "src_scanner_id" not in f or "scanner_names" not in f.attrs:
            return None
        cos = f["cosine_similarity"][:]
        src_ids = f["src_scanner_id"][:].astype(np.int16)
        tgt_ids = f["tgt_scanner_id"][:].astype(np.int16)
        scanner_names: list[str] = json.loads(f.attrs["scanner_names"])

    n = len(scanner_names)
    sums = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    valid = (src_ids >= 0) & (tgt_ids >= 0)
    valid_idx = np.where(valid)[0]
    np.add.at(sums, (src_ids[valid_idx], tgt_ids[valid_idx]), cos[valid_idx])
    np.add.at(counts, (src_ids[valid_idx], tgt_ids[valid_idx]), 1)

    mat = np.full((n, n), np.nan)
    mask = counts > 0
    mat[mask] = sums[mask] / counts[mask]
    return mat, scanner_names


def discover_phases(run_a: Path, run_b: Path) -> list[str]:
    """Return phase names (h5 stem without 'predictions_') present in both runs."""
    stems_a = {p.stem for p in run_a.glob("predictions_*.h5")}
    stems_b = {p.stem for p in run_b.glob("predictions_*.h5")}
    shared = stems_a & stems_b
    # Deterministic ordering: test first, then holdout, then scorpion, then rest
    priority = [
        "predictions_test",
        "predictions_test_holdout_staining",
        "predictions_scorpion",
    ]
    ordered = [s for s in priority if s in shared]
    ordered += sorted(shared - set(priority))
    return [s.removeprefix("predictions_") for s in ordered]


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    names: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    vcenter: float | None = None,
) -> None:
    n = len(names)
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        if vcenter is not None
        else None
    )
    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        vmin=(None if norm else vmin),
        vmax=(None if norm else vmax),
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Target scanner", fontsize=8)
    ax.set_ylabel("Source scanner", fontsize=8)
    ax.set_title(title, fontsize=9)
    span = max(vmax - vmin, 1e-6)
    for si in range(n):
        for ti in range(n):
            v = mat[si, ti]
            if not np.isnan(v):
                brightness = (v - vmin) / span
                txt_color = "black" if 0.3 < brightness < 0.75 else "white"
                ax.text(
                    ti,
                    si,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=txt_color,
                )


def build_heatmap_page(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    names: list[str],
    phase: str,
    label_a: str,
    label_b: str,
) -> plt.Figure:
    diff = mat_b - mat_a

    finite_ab = np.concatenate([mat_a[~np.isnan(mat_a)], mat_b[~np.isnan(mat_b)]])
    vmin_pred = max(float(finite_ab.min()) - 0.02, 0.0) if finite_ab.size else 0.8

    finite_diff = diff[~np.isnan(diff)]
    abs_max = float(np.abs(finite_diff).max()) if finite_diff.size else 0.01
    abs_max = max(abs_max, 1e-4)

    n = len(names)
    # cell_size inches per cell; extra room for tick labels, title, colorbar
    cell_size = 0.675
    panel = n * cell_size + 2.5
    fig, axes = plt.subplots(1, 3, figsize=(panel * 3 + 1, panel))

    _draw_heatmap(
        axes[0], mat_a, names, f"{label_a}\npred cosine", vmin_pred, 1.0, CMAP_PRED
    )
    _draw_heatmap(
        axes[1], mat_b, names, f"{label_b}\npred cosine", vmin_pred, 1.0, CMAP_PRED
    )
    _draw_heatmap(
        axes[2],
        diff,
        names,
        f"B − A  (exact)\n{label_b} minus {label_a}",
        -abs_max,
        abs_max,
        CMAP_DIFF,
        vcenter=0.0,
    )

    fig.suptitle(f"Scanner heatmaps — phase: {phase}", fontsize=12)
    fig.tight_layout()
    return fig


def build_metrics_table(
    summary_a: dict,
    summary_b: dict,
    label_a: str,
    label_b: str,
    fig_width: float = 14.0,
) -> plt.Figure:
    rows = []
    for key in KEY_METRICS:
        va, vb = summary_a.get(key), summary_b.get(key)
        if va is None and vb is None:
            continue
        fmt = lambda v: (
            f"{v:.6f}" if isinstance(v, float) else (str(v) if v is not None else "—")
        )
        rows.append((key, fmt(va), fmt(vb)))

    if not rows:
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.text(
            0.5,
            0.5,
            "No scalar metrics found in wandb-summary.json.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(fig_width, max(2.0, len(rows) * 0.30 + 1.4)))
    ax.axis("off")

    table_data = []
    for key, va_s, vb_s in rows:
        try:
            delta = f"{float(vb_s) - float(va_s):+.6f}"
        except (ValueError, TypeError):
            delta = "—"
        table_data.append([key, va_s, vb_s, delta])

    tbl = ax.table(
        cellText=table_data,
        colLabels=["Metric", label_a, label_b, "Δ (B − A)"],
        loc="center",
        cellLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1, 2, 3])

    for row_idx, row in enumerate(table_data, start=1):
        cell = tbl[row_idx, 3]
        try:
            val = float(row[3])
            cell.set_facecolor(
                "#d4edda" if val > 1e-7 else "#f8d7da" if val < -1e-7 else "white"
            )
        except ValueError:
            pass

    fig.suptitle("Scalar metrics comparison", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--output", type=Path, default=Path("comparison.pdf"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_a = args.run_a.expanduser().resolve()
    run_b = args.run_b.expanduser().resolve()

    for d in (run_a, run_b):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    label_a = args.label_a or run_a.name
    label_b = args.label_b or run_b.name

    figures: list[plt.Figure] = []

    print("Building metrics table...")
    figures.append(
        build_metrics_table(
            find_summary(run_a),
            find_summary(run_b),
            label_a,
            label_b,
        )
    )

    for phase in discover_phases(run_a, run_b):
        print(f"Building heatmap page for '{phase}'...")
        h5_a = run_a / f"predictions_{phase}.h5"
        h5_b = run_b / f"predictions_{phase}.h5"

        result_a = load_h5_matrix(h5_a)
        result_b = load_h5_matrix(h5_b)

        if result_a is None or result_b is None:
            print(
                f"  Skipping '{phase}': H5 lacks scanner metadata (re-run evaluation to regenerate)."
            )
            continue

        mat_a, names_a = result_a
        mat_b, names_b = result_b

        if names_a != names_b:
            print(f"  Skipping '{phase}': scanner vocabularies differ between runs.")
            continue

        figures.append(
            build_heatmap_page(mat_a, mat_b, names_a, phase, label_a, label_b)
        )

    output = args.output.expanduser().resolve()
    use_pdf = output.suffix.lower() == ".pdf"

    if use_pdf:
        print(f"Saving to {output}...")
        with PdfPages(output) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches="tight", dpi=200)
                plt.close(fig)
    else:
        stem, suffix, parent = output.stem, output.suffix, output.parent
        for i, fig in enumerate(figures):
            path = parent / f"{stem}_{i:02d}{suffix}"
            print(f"Saving to {path}...")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
