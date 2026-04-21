"""
run_report.py — single-run overview report.

Usage:
    python run_report.py <run_dir> \\
        [--label NAME] \\
        [--output report.pdf] \\
        [--explain-metrics]

Produces a PDF (or PNG when the path ends in .png) containing:
  1. Run info page: conditioning encodings, model architecture, training
     hyperparameters, and data configuration read from hparams.yaml.
  2. A scalar-metrics table (key test / holdout / scorpion metrics).
  3. All scanner heatmaps (test, test_holdout_staining, scorpion) on one page,
     each cell square.
  4. Scanner improvement heatmaps (imgaug - origtrans), one per phase,
      when origtrans cosine is present in predictions_*.h5.
  5. A single staining bar chart: per-staining mean pred cosine from the
     "test" phase (blue) and "test_holdout_staining" phase (grey).
     Scorpion is excluded from staining plots.
  6. Optionally (--explain-metrics): an appended page with rendered metrics
     documentation from docs/logged_metrics.md.

Each H5 file is fully self-contained: it stores cosine_similarity alongside
src/tgt scanner and staining IDs, and the vocabularies as JSON attributes.
No splits.json or any other sidecar file is required.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Scalar metrics shown in the summary table
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

CMAP_PRED = "RdYlGn"
CMAP_DIFF = "RdYlGn"

_DOCS_PATH = Path(__file__).parents[3] / "docs" / "logged_metrics.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_summary(run_dir: Path) -> dict:
    candidates = list(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
    return json.loads(candidates[-1].read_text()) if candidates else {}


def find_hparams(run_dir: Path) -> dict:
    """Return parsed hparams.yaml from the run dir, or {} if not found."""
    path = run_dir / "hparams.yaml"
    if not path.exists():
        candidates = sorted(run_dir.glob("**/hparams.yaml"))
        if not candidates:
            return {}
        path = candidates[0]
    try:
        import yaml  # pyyaml is available via lightning's deps

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _build_mean_matrix(
    cos: np.ndarray,
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    n: int,
) -> np.ndarray:
    sums = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    valid = (src_ids >= 0) & (tgt_ids >= 0)
    idx = np.where(valid)[0]
    np.add.at(sums, (src_ids[idx], tgt_ids[idx]), cos[idx])
    np.add.at(counts, (src_ids[idx], tgt_ids[idx]), 1)
    mat = np.full((n, n), np.nan)
    mask = counts > 0
    mat[mask] = sums[mask] / counts[mask]
    return mat


def _build_improvement_matrix(
    imgaug_mat: np.ndarray,
    origtrans_mat: np.ndarray,
) -> np.ndarray:
    """Return imgaug-origtrans per scanner pair, matching W&B off-diagonal logic."""
    if imgaug_mat.shape != origtrans_mat.shape:
        raise ValueError("imgaug and origtrans matrices must have identical shapes")

    n = imgaug_mat.shape[0]
    diff = np.full((n, n), np.nan, dtype=np.float64)
    for si in range(n):
        for ti in range(n):
            if si == ti:
                continue
            a = imgaug_mat[si, ti]
            b = origtrans_mat[si, ti]
            if not (np.isnan(a) or np.isnan(b)):
                diff[si, ti] = a - b
    return diff


def load_h5_data(h5_path: Path) -> dict | None:
    """
    Load cosine similarity data from a self-contained predictions H5 file.

    Returns a dict with keys:
      scanner_mat       NxN ndarray or None
      scanner_diff_mat  NxN ndarray or None (imgaug - origtrans)
      scanner_names     list[str] or None
      staining_means    dict[str, float] - per-staining mean pred cosine (row-mean)
    or None if the file is unreadable.
    """
    result: dict = {
        "scanner_mat": None,
        "scanner_diff_mat": None,
        "scanner_names": None,
        "staining_means": {},
    }
    with h5py.File(h5_path, "r") as f:
        if "cosine_similarity" not in f:
            return None
        cos_imgaug = f["cosine_similarity"][:].astype(np.float32)

        if "src_scanner_id" in f and "scanner_names" in f.attrs:
            src_ids = f["src_scanner_id"][:].astype(np.int16)
            tgt_ids = f["tgt_scanner_id"][:].astype(np.int16)
            names: list[str] = json.loads(f.attrs["scanner_names"])
            pred_mat = _build_mean_matrix(cos_imgaug, src_ids, tgt_ids, len(names))
            result["scanner_mat"] = pred_mat
            result["scanner_names"] = names

            # Optional baseline cosine (orig->trans) for improvement heatmaps.
            cos_orig = None
            for key in (
                "origtrans_cosine_similarity",
                "cosine_similarity_origtrans",
                "origtrans_cosine",
            ):
                if key in f:
                    cos_orig = f[key][:].astype(np.float32)
                    break

            if cos_orig is not None:
                orig_mat = _build_mean_matrix(cos_orig, src_ids, tgt_ids, len(names))
                result["scanner_diff_mat"] = _build_improvement_matrix(
                    pred_mat, orig_mat
                )

        if "src_staining_id" in f and "staining_names" in f.attrs:
            src_ids = f["src_staining_id"][:].astype(np.int16)
            tgt_ids = f["tgt_staining_id"][:].astype(np.int16)
            staining_names: list[str] = json.loads(f.attrs["staining_names"])
            mat = _build_mean_matrix(cos_imgaug, src_ids, tgt_ids, len(staining_names))
            result["staining_means"] = {
                name: float(np.nanmean(mat[i, :]))
                for i, name in enumerate(staining_names)
                if not np.all(np.isnan(mat[i, :]))
            }

    return result


def discover_phases(run_dir: Path) -> list[str]:
    stems = {p.stem for p in run_dir.glob("predictions_*.h5")}
    priority = [
        "predictions_test",
        "predictions_test_holdout_staining",
        "predictions_scorpion",
    ]
    ordered = [s for s in priority if s in stems]
    ordered += sorted(stems - set(priority))
    return [s.removeprefix("predictions_") for s in ordered]


def find_splits(run_dir: Path) -> dict:
    path = run_dir / "splits.json"
    if not path.exists():
        candidates = sorted(run_dir.glob("**/splits.json"))
        if not candidates:
            return {}
        path = candidates[0]
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Run info page
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "none"
    return str(v)


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def build_run_info_page(
    run_dir: Path, label: str, hparams: dict, splits: dict
) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Title
    fig.text(
        0.5,
        0.96,
        label,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#2c3e50",
    )
    fig.text(
        0.5,
        0.925,
        str(run_dir),
        ha="center",
        va="top",
        fontsize=7,
        color="#666",
        family="monospace",
    )

    if not hparams:
        fig.text(
            0.5,
            0.85,
            "hparams.yaml not found.",
            ha="center",
            va="top",
            fontsize=10,
            color="#c0392b",
        )
        return fig

    # ---- Conditioning highlight box ----------------------------------------
    conditioning = _deep_get(hparams, "model", "conditioning") or []
    cond_str = _fmt(conditioning) if conditioning else "none"
    cond_color = "#2ecc71" if conditioning else "#e74c3c"

    box_ax = fig.add_axes([0.05, 0.82, 0.9, 0.085])
    box_ax.axis("off")
    box_ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=cond_color,
            facecolor=cond_color + "22",
            transform=box_ax.transAxes,
            clip_on=False,
        )
    )
    box_ax.text(
        0.5,
        0.72,
        "Conditioning",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#555",
        transform=box_ax.transAxes,
    )
    box_ax.text(
        0.5,
        0.28,
        cond_str,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=cond_color,
        transform=box_ax.transAxes,
    )

    # ---- Three-column info table -------------------------------------------
    col_defs: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Model",
            [
                ("architecture", _fmt(_deep_get(hparams, "model", "name"))),
                ("input_dim", _fmt(_deep_get(hparams, "model", "input_dim"))),
                ("hidden_dim", _fmt(_deep_get(hparams, "model", "hidden_dim"))),
                (
                    "hidden_layers",
                    _fmt(_deep_get(hparams, "model", "num_hidden_layers")),
                ),
                (
                    "scanner_vocab",
                    _fmt(_deep_get(hparams, "model", "scanner_vocab_size")),
                ),
                (
                    "staining_vocab",
                    _fmt(_deep_get(hparams, "model", "staining_vocab_size")),
                ),
                ("foundation", _fmt(_deep_get(hparams, "foundation_model", "name"))),
                ("histaug_ckpt", _fmt(_deep_get(hparams, "histaug", "ckpt_path"))),
            ],
        ),
        (
            "Training",
            [
                ("epochs", _fmt(_deep_get(hparams, "general", "epochs"))),
                ("optimizer", _fmt(_deep_get(hparams, "optimizer", "name"))),
                ("lr", _fmt(_deep_get(hparams, "optimizer", "parameters", "lr"))),
                (
                    "weight_decay",
                    _fmt(_deep_get(hparams, "optimizer", "parameters", "weight_decay")),
                ),
                ("scheduler", _fmt(_deep_get(hparams, "scheduler", "name"))),
                ("precision", _fmt(_deep_get(hparams, "general", "precision"))),
                ("grad_acc", _fmt(_deep_get(hparams, "general", "grad_acc"))),
                ("loss", _fmt(_deep_get(hparams, "loss", "base_loss"))),
            ],
        ),
        (
            "Data",
            [
                ("dataset", _fmt(_deep_get(hparams, "data", "dataset_name"))),
                ("features_root", _fmt(_deep_get(hparams, "data", "features_root"))),
                (
                    "holdout_stain",
                    _fmt(_deep_get(hparams, "data", "holdout_stainings")),
                ),
                ("scanners", _fmt(_deep_get(hparams, "data", "scanners") or "all")),
                ("stainings", _fmt(_deep_get(hparams, "data", "stainings") or "all")),
                (
                    "cross_staining",
                    _fmt(_deep_get(hparams, "data", "pairing", "allow_cross_staining")),
                ),
                ("symmetric", _fmt(_deep_get(hparams, "data", "pairing", "symmetric"))),
                ("seed", _fmt(_deep_get(hparams, "general", "seed"))),
            ],
        ),
    ]

    col_xs = [0.03, 0.36, 0.69]
    top_y = 0.78
    row_h = 0.033
    header_h = 0.04

    for col_i, (header, rows) in enumerate(col_defs):
        x = col_xs[col_i]
        y = top_y
        fig.text(
            x,
            y,
            header,
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
            va="top",
            transform=fig.transFigure,
        )
        y -= header_h
        fig.add_artist(
            plt.Line2D(
                [x, x + 0.30],
                [y + 0.005, y + 0.005],
                transform=fig.transFigure,
                color="#aaa",
                linewidth=0.8,
            )
        )
        for key, val in rows:
            if len(val) > 38:
                val = val[:35] + "..."
            fig.text(
                x, y, key, fontsize=7, color="#555", va="top", transform=fig.transFigure
            )
            fig.text(
                x + 0.14,
                y,
                val,
                fontsize=7,
                color="#222",
                va="top",
                transform=fig.transFigure,
                family="monospace",
            )
            y -= row_h

    # ---- Holdout organs (val split) ----------------------------------------
    val_organs: list[str] = (splits.get("organs") or {}).get("val") or []
    if val_organs:
        # strip leading "NN_" number prefix for display
        def _organ_label(s: str) -> str:
            parts = s.split("_", 1)
            return (
                parts[1].replace("_", " ")
                if len(parts) == 2 and parts[0].isdigit()
                else s
            )

        organs_y = top_y - header_h - len(col_defs[0][1]) * row_h - 0.045
        fig.text(
            0.03,
            organs_y,
            "Holdout organs (test / val split)",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
            va="top",
            transform=fig.transFigure,
        )
        organs_y -= header_h
        fig.add_artist(
            plt.Line2D(
                [0.03, 0.97],
                [organs_y + 0.005, organs_y + 0.005],
                transform=fig.transFigure,
                color="#aaa",
                linewidth=0.8,
            )
        )

        # lay out as a grid: 4 columns across the full width
        n_cols = 4
        col_w = 0.235
        x0 = 0.03
        for i, organ in enumerate(val_organs):
            ox = x0 + (i % n_cols) * col_w
            oy = organs_y - (i // n_cols) * row_h
            fig.text(
                ox,
                oy,
                f"\u2022  {_organ_label(organ)}",
                fontsize=7,
                color="#333",
                va="top",
                transform=fig.transFigure,
            )

    # ---- Comment -----------------------------------------------------------
    comment = _deep_get(hparams, "general", "comment")
    if comment:
        fig.text(
            0.5,
            0.035,
            f'"{comment}"',
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555",
            style="italic",
            transform=fig.transFigure,
        )

    return fig


# ---------------------------------------------------------------------------
# Scanner heatmaps (all phases on one page)
# ---------------------------------------------------------------------------

_CELL = 0.54  # inches per scanner cell (about 25% smaller)


def _draw_square_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    names: list[str],
    title: str,
    vmin: float,
) -> None:
    n = len(names)
    im = ax.imshow(
        mat,
        cmap=CMAP_PRED,
        vmin=vmin,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Target scanner", fontsize=8)
    ax.set_ylabel("Source scanner", fontsize=8)
    ax.set_title(title, fontsize=9)
    span = max(1.0 - vmin, 1e-6)
    # Keep matrix labels readable as scanner vocab grows.
    fs = max(12, 8 - n // 6)
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
                    fontsize=fs,
                    color=txt_color,
                )


def build_scanner_heatmaps_page(
    scanner_data: dict[str, tuple[np.ndarray, list[str]]],
    label: str,
) -> plt.Figure | None:
    """One figure with all available scanner heatmaps in a single row."""
    if not scanner_data:
        return None

    phases = list(scanner_data.keys())
    k = len(phases)

    # Use the largest N to size the figure; all phases should share the same
    # scanner vocabulary, but we handle differences gracefully.
    max_n = max(len(names) for _, names in scanner_data.values())
    cell_total = max_n * _CELL
    ax_w = cell_total + 2.5  # colorbar + y-labels
    ax_h = cell_total + 2.2  # title + x-labels
    fig_w = k * ax_w + 0.4
    fig_h = ax_h + 0.6  # suptitle margin

    fig, axes = plt.subplots(1, k, figsize=(fig_w, fig_h), constrained_layout=True)
    if k == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        mat, names = scanner_data[phase]
        finite = mat[~np.isnan(mat)]
        vmin = max(float(finite.min()) - 0.02, 0.0) if finite.size else 0.8
        _draw_square_heatmap(ax, mat, names, phase.replace("_", " "), vmin)

    fig.suptitle(f"Scanner heatmaps — {label}", fontsize=11)
    return fig


def _draw_square_diff_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    names: list[str],
    title: str,
) -> None:
    n = len(names)
    finite = mat[~np.isnan(mat)]
    abs_max = max(float(np.abs(finite).max()), 1e-6) if finite.size else 0.1
    vmin, vmax = -abs_max, abs_max
    im = ax.imshow(
        mat,
        cmap=CMAP_DIFF,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
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
    fs = max(12, 8 - n // 6)
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
                    fontsize=fs,
                    color=txt_color,
                )


def build_scanner_diff_heatmaps_page(
    diff_data: dict[str, tuple[np.ndarray, list[str]]],
    label: str,
) -> plt.Figure | None:
    """One figure with all available scanner improvement heatmaps in a single row."""
    if not diff_data:
        return None

    phases = list(diff_data.keys())
    k = len(phases)

    max_n = max(len(names) for _, names in diff_data.values())
    cell_total = max_n * _CELL
    ax_w = cell_total + 2.5
    ax_h = cell_total + 2.2
    fig_w = k * ax_w + 0.4
    fig_h = ax_h + 0.6

    fig, axes = plt.subplots(1, k, figsize=(fig_w, fig_h), constrained_layout=True)
    if k == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        mat, names = diff_data[phase]
        _draw_square_diff_heatmap(
            ax,
            mat,
            names,
            f"{phase.replace('_', ' ')} (imgaug - origtrans)",
        )

    fig.suptitle(f"Scanner improvement heatmaps — {label}", fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Staining bar chart
# ---------------------------------------------------------------------------


def build_staining_barchart(
    test_means: dict[str, float],
    holdout_means: dict[str, float],
) -> plt.Figure | None:
    all_names = sorted(set(test_means) | set(holdout_means))
    if not all_names:
        return None

    x = np.arange(len(all_names))
    has_test = bool(test_means)
    has_holdout = bool(holdout_means)
    n_series = has_test + has_holdout
    width = 0.35 if n_series == 2 else 0.6

    offsets = [-width / 2, width / 2] if (has_test and has_holdout) else [0.0]

    series = []
    if has_test:
        series.append(("test", test_means, "#4C72B0", offsets[0]))
    if has_holdout:
        series.append(("test_holdout_staining", holdout_means, "#888888", offsets[-1]))

    all_vals = [v for m in (test_means, holdout_means) for v in m.values()]
    vmin = max(0.0, min(all_vals) - 0.02) if all_vals else 0.8

    fig, ax = plt.subplots(figsize=(max(6.0, len(all_names) * 0.55 + 2.0), 4.5))
    for phase_label, means, color, offset in series:
        vals = [means.get(name, float("nan")) for name in all_names]
        ax.bar(x + offset, vals, width, label=phase_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean pred cosine", fontsize=9)
    ax.set_ylim(bottom=vmin)
    ax.legend(fontsize=8)
    ax.set_title("Per-staining mean predicted cosine similarity", fontsize=10)
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scalar metrics table
# ---------------------------------------------------------------------------


def build_metrics_table(
    summary: dict,
    label: str,
    fig_width: float = 10.0,
) -> plt.Figure:
    rows = []
    for key in KEY_METRICS:
        v = summary.get(key)
        if v is None:
            continue
        fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
        rows.append((key, fmt))

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
    tbl = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", label],
        loc="center",
        cellLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1])
    fig.suptitle("Scalar metrics", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics explanation page (rendered markdown)
# ---------------------------------------------------------------------------

# All heights are in inches.  line_h = size * 1.35 / 72.
_STYLE: dict[str, dict] = {
    "h1": {"size": 13, "weight": "bold", "color": "#1a252f", "skip_after_in": 0.10},
    "h2": {"size": 10, "weight": "bold", "color": "#2c3e50", "skip_after_in": 0.07},
    "h3": {"size": 8, "weight": "bold", "color": "#555", "skip_after_in": 0.05},
    "rule": {"size": 0, "weight": "normal", "color": "#ccc", "skip_after_in": 0.07},
    "bullet": {"size": 7, "weight": "normal", "color": "#222", "skip_after_in": 0.00},
    "code": {"size": 6, "weight": "normal", "color": "#333", "skip_after_in": 0.00},
    "text": {"size": 7, "weight": "normal", "color": "#333", "skip_after_in": 0.00},
    "blank": {"size": 4, "weight": "normal", "color": "#fff", "skip_after_in": 0.00},
}


def _line_h_in(kind: str) -> float:
    return max(_STYLE[kind]["size"], 1) * 1.35 / 72


def _parse_md(text: str) -> list[tuple[str, str, int]]:
    """Return list of (kind, content, indent) from markdown text."""
    elements: list[tuple[str, str, int]] = []
    in_code = False
    code_lines: list[str] = []
    table_buf: list[list[str]] = []  # buffered table rows awaiting alignment

    def _flush_table() -> None:
        if not table_buf:
            return
        n_cols = max(len(r) for r in table_buf)
        widths = [
            max(len(r[i]) if i < len(r) else 0 for r in table_buf)
            for i in range(n_cols)
        ]
        for row in table_buf:
            cells = [
                f"{(row[i] if i < len(row) else ''):<{widths[i]}}"
                for i in range(n_cols)
            ]
            elements.append(("code", "   ".join(cells).rstrip(), 0))
        table_buf.clear()

    for line in text.splitlines():
        stripped = line.rstrip()

        if stripped.startswith("```"):
            _flush_table()
            if in_code:
                for cl in code_lines:
                    elements.append(("code", cl, 1))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(stripped)
            continue

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # drop separator rows (only dashes/colons)
            if all(
                set(c.replace("-", "").replace(":", "").replace(" ", "")) == set()
                for c in cells
            ):
                continue
            # strip inline markdown from cells
            cells = [c.replace("**", "").replace("`", "") for c in cells]
            table_buf.append(cells)
            continue

        # Non-table line: flush any buffered table first
        _flush_table()

        if not stripped:
            elements.append(("blank", "", 0))
        elif stripped.startswith("### "):
            elements.append(("h3", stripped[4:], 0))
        elif stripped.startswith("## "):
            elements.append(("h2", stripped[3:], 0))
        elif stripped.startswith("# "):
            elements.append(("h1", stripped[2:], 0))
        elif stripped.startswith(("---", "===")) and len(set(stripped)) <= 2:
            elements.append(("rule", "", 0))
        elif stripped.startswith(("- ", "* ")):
            content = stripped[2:].replace("**", "").replace("`", "")
            elements.append(("bullet", content, 0))
        else:
            content = stripped.replace("**", "").replace("`", "")
            elements.append(("text", content, 0))

    _flush_table()
    return elements


def build_explanation_page() -> plt.Figure:
    if _DOCS_PATH.exists():
        raw = _DOCS_PATH.read_text()
    else:
        raw = "# Metrics Reference\n\n(logged_metrics.md not found)"

    elements = _parse_md(raw)

    FIG_W = 11.0
    X_MARGIN = 0.04
    X_INDENT = 0.06
    WRAP_TEXT = 130
    WRAP_CODE = 110
    TOP_MARGIN_IN = 0.35
    BOT_MARGIN_IN = 0.25

    # First pass: expand wrapping into (kind, display_text, height_in) triples.
    # height_in includes line height + skip_after only on the last wrapped line.
    display: list[tuple[str, str, float]] = []
    for kind, content, _indent in elements:
        lh = _line_h_in(kind)
        skip = _STYLE[kind]["skip_after_in"]

        if kind == "blank":
            display.append((kind, "", lh + skip))
            continue
        if kind == "rule":
            display.append((kind, "", 0.04 + skip))
            continue

        prefix = "\u2022  " if kind == "bullet" else ""
        ww = WRAP_CODE if kind == "code" else WRAP_TEXT
        wrapped = textwrap.wrap(prefix + content, width=ww) or [""]
        for j, wline in enumerate(wrapped):
            # skip_after only after the last wrapped line of an element
            extra = skip if j == len(wrapped) - 1 else 0.0
            display.append((kind, wline, lh + extra))

    # Second pass: compute exact figure height from display line heights.
    total_content_in = sum(h for _, _, h in display)
    fig_h = total_content_in + TOP_MARGIN_IN + BOT_MARGIN_IN

    fig = plt.figure(figsize=(FIG_W, fig_h))

    # y_in: distance from bottom of figure in inches
    y_in = fig_h - TOP_MARGIN_IN

    for kind, text, h_in in display:
        y_frac = y_in / fig_h
        style = _STYLE[kind]

        if kind == "blank":
            y_in -= h_in
            continue

        if kind == "rule":
            fig.add_artist(
                plt.Line2D(
                    [X_MARGIN, 1 - X_MARGIN],
                    [y_frac, y_frac],
                    transform=fig.transFigure,
                    color="#ccc",
                    linewidth=0.7,
                )
            )
            y_in -= h_in
            continue

        x_frac = X_MARGIN + (X_INDENT if kind in ("bullet", "code") else 0)
        fig.text(
            x_frac,
            y_frac,
            text,
            va="top",
            ha="left",
            fontsize=style["size"],
            fontweight=style["weight"],
            color=style["color"],
            family="monospace" if kind == "code" else "sans-serif",
            transform=fig.transFigure,
        )
        y_in -= h_in

    fig.patch.set_facecolor("white")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument(
        "--label", default=None, help="Human-readable run label (defaults to dir name)"
    )
    p.add_argument("--output", type=Path, default=Path("report.pdf"))
    p.add_argument(
        "--explain-metrics",
        action="store_true",
        help="Append a page explaining all logged metrics (from docs/logged_metrics.md)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    label = args.label or run_dir.name
    figures: list[plt.Figure] = []

    print("Building run info page...")
    hparams = find_hparams(run_dir)
    splits = find_splits(run_dir)
    figures.append(build_run_info_page(run_dir, label, hparams, splits))

    print("Building metrics table...")
    figures.append(build_metrics_table(find_summary(run_dir), label))

    scanner_data: dict[str, tuple[np.ndarray, list[str]]] = {}
    scanner_diff_data: dict[str, tuple[np.ndarray, list[str]]] = {}
    staining_means: dict[str, dict[str, float]] = {}

    for phase in discover_phases(run_dir):
        h5_path = run_dir / f"predictions_{phase}.h5"
        print(f"Loading '{phase}'...")
        data = load_h5_data(h5_path)

        if data is None:
            print(f"  Skipping '{phase}': unreadable H5 file.")
            continue

        if data["scanner_mat"] is not None:
            scanner_data[phase] = (data["scanner_mat"], data["scanner_names"])
            if data["scanner_diff_mat"] is not None:
                scanner_diff_data[phase] = (
                    data["scanner_diff_mat"],
                    data["scanner_names"],
                )
        else:
            print(f"  No scanner metadata in '{phase}'.")

        if phase in ("test", "test_holdout_staining") and data["staining_means"]:
            staining_means[phase] = data["staining_means"]

    if scanner_data:
        print("Building scanner heatmaps page...")
        fig = build_scanner_heatmaps_page(scanner_data, label)
        if fig is not None:
            figures.append(fig)

    if scanner_diff_data:
        print("Building scanner improvement heatmaps page...")
        fig = build_scanner_diff_heatmaps_page(scanner_diff_data, label)
        if fig is not None:
            figures.append(fig)
    elif scanner_data:
        print(
            "Skipping scanner improvement heatmaps: origtrans cosine not found in H5."
        )

    staining_fig = build_staining_barchart(
        staining_means.get("test", {}),
        staining_means.get("test_holdout_staining", {}),
    )
    if staining_fig is not None:
        print("Building staining bar chart...")
        figures.append(staining_fig)

    if args.explain_metrics:
        print("Appending metrics explanation page...")
        figures.append(build_explanation_page())

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
