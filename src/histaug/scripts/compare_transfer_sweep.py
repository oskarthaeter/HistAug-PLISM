"""
compare_transfer_sweep.py — compare a HistAug transfer run with an aug sweep.

Usage:
    python compare_transfer_sweep.py <transfer_run_dir> <sweep_dir> \\
        [--label-transfer NAME] [--label-sweep NAME] \\
        [--phase test] \\
        [--output comparison.pdf]

Produces a PDF (or PNG when the path ends in .png) containing:
  1. A scalar-metrics table: transfer run wandb-summary metrics alongside
     sweep global-best stats from run_config_and_best.json.
  2. Absolute-cosine heatmaps: HistAug model prediction vs sweep best-aug
     cosine per scanner pair, with a B-A difference panel.
  3. Improvement heatmaps: per-pair improvement over each method's own
     baseline (HistAug model - origtrans, sweep best - sweep baseline),
     with a difference panel showing which method improves more.

Transfer run directory must contain predictions_<phase>.h5 (self-contained,
with scanner_names attribute and cosine_similarity + origtrans_cosine_similarity
datasets).

Sweep directory must contain pair_best_aug.csv (columns: src_scanner,
tgt_scanner, mean_pred_cos, mean_baseline_cos, improvement) and optionally
run_config_and_best.json for global summary metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Scalar metrics shown in the comparison table
# ---------------------------------------------------------------------------

TRANSFER_KEY_METRICS = [
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
    "scorpion/mean_origtrans_cos",
    "scorpion/mean_imgaug_cos",
    "scorpion/relative_improvement",
]

CMAP_PRED = "RdYlGn"
CMAP_DIFF = "RdBu_r"
CMAP_IMPR = "RdYlGn"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_wandb_summary(run_dir: Path) -> dict:
    candidates = list(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
    return json.loads(candidates[-1].read_text()) if candidates else {}


def load_transfer_matrices(
    h5_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load HistAug model cosine and origtrans cosine matrices from H5.

    Returns (model_mat, baseline_mat, scanner_names) or None if metadata absent.
    """
    with h5py.File(h5_path, "r") as f:
        if "src_scanner_id" not in f or "scanner_names" not in f.attrs:
            return None
        cos_model = f["cosine_similarity"][:]
        cos_baseline = (
            f["origtrans_cosine_similarity"][:]
            if "origtrans_cosine_similarity" in f
            else None
        )
        src_ids = f["src_scanner_id"][:].astype(np.int16)
        tgt_ids = f["tgt_scanner_id"][:].astype(np.int16)
        scanner_names: list[str] = json.loads(f.attrs["scanner_names"])

    n = len(scanner_names)
    valid = (src_ids >= 0) & (tgt_ids >= 0)
    idx = np.where(valid)[0]

    def _mean_mat(values: np.ndarray) -> np.ndarray:
        sums = np.zeros((n, n), dtype=np.float64)
        counts = np.zeros((n, n), dtype=np.int64)
        np.add.at(sums, (src_ids[idx], tgt_ids[idx]), values[idx])
        np.add.at(counts, (src_ids[idx], tgt_ids[idx]), 1)
        mat = np.full((n, n), np.nan)
        mat[counts > 0] = sums[counts > 0] / counts[counts > 0]
        return mat

    model_mat = _mean_mat(cos_model)
    baseline_mat = _mean_mat(cos_baseline) if cos_baseline is not None else None
    return model_mat, baseline_mat, scanner_names


def active_row_col_indices(
    mat: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Return (row_indices, col_indices) that have at least one non-NaN entry."""
    rows = [i for i in range(mat.shape[0]) if not np.all(np.isnan(mat[i, :]))]
    cols = [j for j in range(mat.shape[1]) if not np.all(np.isnan(mat[:, j]))]
    return rows, cols


def subselect(
    mat: np.ndarray, row_idx: list[int], col_idx: list[int]
) -> np.ndarray:
    return mat[np.ix_(row_idx, col_idx)]


def load_sweep_matrices(
    sweep_dir: Path, scanner_names: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build sweep best-aug and baseline matrices aligned to scanner_names.

    Returns (sweep_best_mat, sweep_baseline_mat) or None if CSV absent.
    """
    csv_path = sweep_dir / "pair_best_aug.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"src_scanner", "tgt_scanner", "mean_pred_cos", "mean_baseline_cos"}
    if not required.issubset(df.columns):
        return None

    name_to_idx = {name: i for i, name in enumerate(scanner_names)}
    n = len(scanner_names)
    sweep_best = np.full((n, n), np.nan)
    sweep_baseline = np.full((n, n), np.nan)

    for _, row in df.iterrows():
        si = name_to_idx.get(row["src_scanner"])
        ti = name_to_idx.get(row["tgt_scanner"])
        if si is None or ti is None:
            continue
        sweep_best[si, ti] = row["mean_pred_cos"]
        sweep_baseline[si, ti] = row["mean_baseline_cos"]

    return sweep_best, sweep_baseline


def load_sweep_summary(sweep_dir: Path) -> dict:
    cfg_path = sweep_dir / "run_config_and_best.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    vcenter: float | None = None,
    fontsize_cell: int = 6,
) -> None:
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
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(row_names, fontsize=7)
    ax.set_xlabel("Target scanner", fontsize=8)
    ax.set_ylabel("Source scanner", fontsize=8)
    ax.set_title(title, fontsize=9)
    span = max(vmax - vmin, 1e-6)
    for si in range(len(row_names)):
        for ti in range(len(col_names)):
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
                    fontsize=fontsize_cell,
                    color=txt_color,
                )


def build_absolute_heatmap_page(
    model_mat: np.ndarray,
    sweep_best_mat: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    label_transfer: str,
    label_sweep: str,
    phase: str,
) -> plt.Figure:
    """Three-panel page: HistAug model | sweep best aug | difference."""
    diff = model_mat - sweep_best_mat

    all_vals = np.concatenate(
        [
            model_mat[~np.isnan(model_mat)],
            sweep_best_mat[~np.isnan(sweep_best_mat)],
        ]
    )
    vmin_pred = max(float(all_vals.min()) - 0.02, 0.0) if all_vals.size else 0.8

    finite_diff = diff[~np.isnan(diff)]
    abs_max = float(np.abs(finite_diff).max()) if finite_diff.size else 0.01
    abs_max = max(abs_max, 1e-4)

    cell_size = 0.675
    panel_h = len(row_names) * cell_size + 2.5
    panel_w = len(col_names) * cell_size + 2.5
    fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3 + 1, panel_h))

    _draw_heatmap(
        axes[0],
        model_mat,
        row_names,
        col_names,
        f"{label_transfer}\nHistAug pred cosine",
        vmin_pred,
        1.0,
        CMAP_PRED,
    )
    _draw_heatmap(
        axes[1],
        sweep_best_mat,
        row_names,
        col_names,
        f"{label_sweep}\nSweep best-aug cosine",
        vmin_pred,
        1.0,
        CMAP_PRED,
    )
    _draw_heatmap(
        axes[2],
        diff,
        row_names,
        col_names,
        f"Transfer - Sweep (exact)\n{label_transfer} minus {label_sweep}",
        -abs_max,
        abs_max,
        CMAP_DIFF,
        vcenter=0.0,
    )

    fig.suptitle(
        f"Absolute cosine comparison — phase: {phase}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def build_improvement_heatmap_page(
    model_mat: np.ndarray,
    transfer_baseline_mat: np.ndarray,
    sweep_best_mat: np.ndarray,
    sweep_baseline_mat: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    label_transfer: str,
    label_sweep: str,
    phase: str,
) -> plt.Figure:
    """Three-panel page: HistAug improvement | sweep improvement | difference."""
    transfer_impr = model_mat - transfer_baseline_mat
    sweep_impr = sweep_best_mat - sweep_baseline_mat
    impr_diff = transfer_impr - sweep_impr

    all_impr = np.concatenate(
        [
            transfer_impr[~np.isnan(transfer_impr)],
            sweep_impr[~np.isnan(sweep_impr)],
        ]
    )
    if all_impr.size:
        abs_max_impr = max(float(np.abs(all_impr).max()), 1e-4)
    else:
        abs_max_impr = 0.05

    finite_diff = impr_diff[~np.isnan(impr_diff)]
    abs_max_diff = float(np.abs(finite_diff).max()) if finite_diff.size else 0.01
    abs_max_diff = max(abs_max_diff, 1e-4)

    cell_size = 0.675
    panel_h = len(row_names) * cell_size + 2.5
    panel_w = len(col_names) * cell_size + 2.5
    fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3 + 1, panel_h))

    _draw_heatmap(
        axes[0],
        transfer_impr,
        row_names,
        col_names,
        f"{label_transfer}\nHistAug improvement\n(model - origtrans baseline)",
        -abs_max_impr,
        abs_max_impr,
        CMAP_DIFF,
        vcenter=0.0,
    )
    _draw_heatmap(
        axes[1],
        sweep_impr,
        row_names,
        col_names,
        f"{label_sweep}\nSweep improvement\n(best-aug - sweep baseline)",
        -abs_max_impr,
        abs_max_impr,
        CMAP_DIFF,
        vcenter=0.0,
    )
    _draw_heatmap(
        axes[2],
        impr_diff,
        row_names,
        col_names,
        "Transfer impr. - Sweep impr.\n(positive = HistAug improves more)",
        -abs_max_diff,
        abs_max_diff,
        CMAP_DIFF,
        vcenter=0.0,
    )

    fig.suptitle(
        f"Improvement over baseline — phase: {phase}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v) if v is not None else "—"


def build_metrics_table(
    transfer_summary: dict,
    sweep_cfg: dict,
    label_transfer: str,
    label_sweep: str,
    fig_width: float = 14.0,
) -> plt.Figure:
    rows: list[tuple[str, str, str]] = []

    # Transfer-run wandb metrics
    for key in TRANSFER_KEY_METRICS:
        v = transfer_summary.get(key)
        if v is not None:
            rows.append((key, _fmt(v), "—"))

    # Sweep global best metrics
    best = sweep_cfg.get("best_global", {})
    sweep_metric_map = {
        "sweep/global_baseline_cos": sweep_cfg.get("global_baseline_cos"),
        "sweep/best_mean_pred_cos": best.get("mean_pred_cos"),
        "sweep/best_improvement": best.get("improvement"),
        "sweep/best_hed": best.get("hed"),
        "sweep/best_saturation": best.get("saturation"),
        "sweep/best_brightness": best.get("brightness"),
        "sweep/best_hue": best.get("hue"),
        "sweep/best_contrast": best.get("contrast"),
        "sweep/n_combinations": sweep_cfg.get("sweep_num_combinations"),
    }
    for key, v in sweep_metric_map.items():
        if v is not None:
            rows.append((key, "—", _fmt(v)))

    if not rows:
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.text(0.5, 0.5, "No metrics found.", ha="center", va="center")
        ax.axis("off")
        return fig

    # Colour the delta column green/red where both values are numeric
    table_data = []
    for metric, va_s, vb_s in rows:
        try:
            delta = f"{float(vb_s) - float(va_s):+.6f}"
        except (ValueError, TypeError):
            delta = "—"
        table_data.append([metric, va_s, vb_s, delta])

    fig, ax = plt.subplots(
        figsize=(fig_width, max(2.0, len(rows) * 0.30 + 1.4))
    )
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=["Metric", label_transfer, label_sweep, "Δ (sweep - transfer)"],
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

    fig.suptitle("Scalar metrics: transfer run vs aug sweep", fontsize=11, y=0.98)
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
    p.add_argument("transfer_run", type=Path, help="HistAug transfer run directory")
    p.add_argument("sweep_dir", type=Path, help="Aug sweep output directory")
    p.add_argument("--label-transfer", default=None)
    p.add_argument("--label-sweep", default=None)
    p.add_argument(
        "--phase",
        default="test",
        help="H5 phase to compare (default: test)",
    )
    p.add_argument(
        "--scanners",
        nargs="+",
        default=None,
        metavar="SCANNER",
        help=(
            "Restrict heatmaps to these scanners (both axes). "
            "Overrides the automatic active-pair detection. "
            "Example: --scanners AT2 GT450 P"
        ),
    )
    p.add_argument("--output", type=Path, default=Path("comparison.pdf"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    transfer_dir = args.transfer_run.expanduser().resolve()
    sweep_dir = args.sweep_dir.expanduser().resolve()

    for d in (transfer_dir, sweep_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    label_transfer = args.label_transfer or transfer_dir.name
    label_sweep = args.label_sweep or sweep_dir.name
    phase = args.phase

    figures: list[plt.Figure] = []

    # --- Metrics table --------------------------------------------------
    print("Loading metrics...")
    transfer_summary = find_wandb_summary(transfer_dir)
    sweep_cfg = load_sweep_summary(sweep_dir)
    figures.append(
        build_metrics_table(transfer_summary, sweep_cfg, label_transfer, label_sweep)
    )

    # --- Heatmaps -------------------------------------------------------
    h5_path = transfer_dir / f"predictions_{phase}.h5"
    if not h5_path.exists():
        print(
            f"WARNING: {h5_path} not found — skipping heatmap pages.",
            file=sys.stderr,
        )
    else:
        print(f"Loading transfer H5 for phase '{phase}'...")
        result = load_transfer_matrices(h5_path)
        if result is None:
            print(
                f"  Skipping: H5 lacks scanner metadata. Re-run evaluation to regenerate.",
                file=sys.stderr,
            )
        else:
            model_mat, baseline_mat, scanner_names = result

            print("Loading sweep pair_best_aug.csv...")
            sweep_result = load_sweep_matrices(sweep_dir, scanner_names)
            if sweep_result is None:
                print(
                    "  Skipping heatmaps: pair_best_aug.csv not found or missing columns.",
                    file=sys.stderr,
                )
            else:
                sweep_best_mat, sweep_baseline_mat = sweep_result

                # Determine which scanner rows/cols to show.
                if args.scanners is not None:
                    keep = set(args.scanners)
                    unknown = keep - set(scanner_names)
                    if unknown:
                        print(
                            f"  WARNING: unknown scanners ignored: {sorted(unknown)}",
                            file=sys.stderr,
                        )
                    row_idx = [i for i, n in enumerate(scanner_names) if n in keep]
                    col_idx = row_idx  # same set on both axes
                    print(f"  Scanner filter: {[scanner_names[i] for i in row_idx]}")
                else:
                    # Fall back to auto-detecting active pairs from the transfer H5.
                    row_idx, col_idx = active_row_col_indices(model_mat)

                row_names = [scanner_names[i] for i in row_idx]
                col_names = [scanner_names[j] for j in col_idx]
                if row_names != col_names or len(row_names) < len(scanner_names):
                    print(
                        f"  Layout: {len(row_names)} sources x {len(col_names)} targets"
                    )

                m_model = subselect(model_mat, row_idx, col_idx)
                m_sweep = subselect(sweep_best_mat, row_idx, col_idx)
                m_sweep_base = subselect(sweep_baseline_mat, row_idx, col_idx)

                print("Building absolute cosine heatmap page...")
                figures.append(
                    build_absolute_heatmap_page(
                        m_model,
                        m_sweep,
                        row_names,
                        col_names,
                        label_transfer,
                        label_sweep,
                        phase,
                    )
                )

                if baseline_mat is not None:
                    m_baseline = subselect(baseline_mat, row_idx, col_idx)
                    print("Building improvement heatmap page...")
                    figures.append(
                        build_improvement_heatmap_page(
                            m_model,
                            m_baseline,
                            m_sweep,
                            m_sweep_base,
                            row_names,
                            col_names,
                            label_transfer,
                            label_sweep,
                            phase,
                        )
                    )
                else:
                    print(
                        "  Skipping improvement page: origtrans_cosine_similarity absent in H5."
                    )

    # --- Save -----------------------------------------------------------
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
