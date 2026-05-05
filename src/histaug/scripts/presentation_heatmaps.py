"""
presentation_heatmaps.py — Generate clean, presentation-ready scanner heatmaps.

Outputs PNGs to presentation_figures/ (relative to the repo root, or --outdir).

Usage:
    python presentation_heatmaps.py [--outdir path/to/figures]

Figures produced
----------------
  scanner_cosine_virchow2.png     — 1×3 cosine heatmaps (test / holdout / SCORPION)
  scanner_cosine_h0mini.png       — same for H-optimus-1 mini
  scanner_improvement_virchow2.png — 1×3 improvement (pred − baseline) heatmaps
  scanner_improvement_h0mini.png   — same for H-optimus-1 mini
  comparison_cosine.png            — 2×3 overview: both models × all phases
  comparison_improvement.png       — 2×3 overview: both models × all phases (delta)
  staining_virchow2.png            — per-staining cosine bar chart (test + holdout)
  staining_h0mini.png              — same for H-optimus-1 mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_LOGS = Path(__file__).parents[3] / "src" / "histaug" / "logs"
_REPO_ROOT = Path(__file__).parents[3]

MODELS = {
    "Virchow2": {
        "run": _LOGS / "virchow2-transfer-bottleneck-tgt_spectral_norm_all_no_HR_capped_alpha",
        "color": "#2563EB",
        "slug": "virchow2",
    },
    "H0-mini": {
        "run": _LOGS / "h0mini-transfer-bottleneck-tgt_spectral_norm_all_no_HR_capped_alpha",
        "color": "#059669",
        "slug": "h0mini",
    },
}

PHASES = ["test", "test_holdout_staining", "scorpion"]
PHASE_LABELS = {
    "test": "PLISM test\n(unseen organs)",
    "test_holdout_staining": "PLISM holdout staining\n(unseen GVH)",
    "scorpion": "SCORPION\n(external dataset)",
}

# SCORPION: source rows and target columns to show
_SCORPION_SRC_ORDER = ["AT2", "GT450", "P", "S210", "S360"]
_SCORPION_TGT_ORDER = ["AT2", "GT450", "P"]
# Display labels for SCORPION source scanners (surrogate mapping)
_SCORPION_SRC_DISPLAY = {
    "AT2": "AT2",
    "GT450": "GT450",
    "P": "Philips",
    "S210": "DP200",
    "S360": "P1000",
}

# Matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

CMAP_COS = "YlGnBu"
CMAP_DIFF = "RdYlGn"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_mat(
    values: np.ndarray,
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    n: int,
) -> np.ndarray:
    sums = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    valid = (src_ids >= 0) & (tgt_ids >= 0)
    idx = np.where(valid)[0]
    np.add.at(sums, (src_ids[idx], tgt_ids[idx]), values[idx])
    np.add.at(counts, (src_ids[idx], tgt_ids[idx]), 1)
    mat = np.full((n, n), np.nan)
    mask = counts > 0
    mat[mask] = sums[mask] / counts[mask]
    return mat


def load_phase(run_dir: Path, phase: str) -> dict | None:
    path = run_dir / f"predictions_{phase}.h5"
    if not path.exists():
        return None

    with h5py.File(path, "r") as f:
        scanner_names: list[str] = json.loads(f.attrs["scanner_names"])
        staining_names: list[str] = json.loads(f.attrs["staining_names"])
        n_sc = len(scanner_names)
        n_st = len(staining_names)

        cos = f["cosine_similarity"][:].astype(np.float32)
        src_sc = f["src_scanner_id"][:].astype(np.int16)
        tgt_sc = f["tgt_scanner_id"][:].astype(np.int16)
        src_st = f["src_staining_id"][:].astype(np.int16)
        tgt_st = f["tgt_staining_id"][:].astype(np.int16)

        pred_mat = _build_mat(cos, src_sc, tgt_sc, n_sc)

        # Fill diagonal from identity cosine
        if "identity_cosine_similarity" in f:
            id_cos = f["identity_cosine_similarity"][:].astype(np.float32)
            for i, mean_val in enumerate(
                np.array(
                    [
                        np.mean(id_cos[src_sc == i]) if np.any(src_sc == i) else np.nan
                        for i in range(n_sc)
                    ]
                )
            ):
                if not np.isnan(mean_val):
                    pred_mat[i, i] = mean_val

        orig_mat = None
        if "origtrans_cosine_similarity" in f:
            orig_cos = f["origtrans_cosine_similarity"][:].astype(np.float32)
            orig_mat = _build_mat(orig_cos, src_sc, tgt_sc, n_sc)
            if "orig_identity_cosine_similarity" in f:
                orig_id = f["orig_identity_cosine_similarity"][:].astype(np.float32)
                for i, mean_val in enumerate(
                    np.array(
                        [
                            np.mean(orig_id[src_sc == i])
                            if np.any(src_sc == i)
                            else np.nan
                            for i in range(n_sc)
                        ]
                    )
                ):
                    if not np.isnan(mean_val):
                        orig_mat[i, i] = mean_val

        # Per-staining mean predicted cosine (row-wise over target scanners)
        staining_cos = {}
        if n_st > 0:
            st_mat = _build_mat(cos, src_st, tgt_st, n_st)
            for i, name in enumerate(staining_names):
                row_vals = st_mat[i, :]
                if not np.all(np.isnan(row_vals)):
                    staining_cos[name] = float(np.nanmean(row_vals))

    return {
        "pred_mat": pred_mat,
        "orig_mat": orig_mat,
        "scanner_names": scanner_names,
        "staining_cos": staining_cos,
    }


def scorpion_slice(
    mat: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Extract the rectangular SCORPION-focused view plus a same-scanner diagonal mask."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    src_names = [s for s in _SCORPION_SRC_ORDER if s in name_to_idx]
    tgt_names = [t for t in _SCORPION_TGT_ORDER if t in name_to_idx]
    src_idx = [name_to_idx[s] for s in src_names]
    tgt_idx = [name_to_idx[t] for t in tgt_names]
    sliced = mat[np.ix_(src_idx, tgt_idx)]
    row_labels = [_SCORPION_SRC_DISPLAY.get(s, s) for s in src_names]
    col_labels = [_SCORPION_SRC_DISPLAY.get(t, t) for t in tgt_names]
    # Diagonal mask: True where src and tgt are the same PLISM scanner
    dmask = np.zeros((len(src_names), len(tgt_names)), dtype=bool)
    for ri, sn in enumerate(src_names):
        for ci, tn in enumerate(tgt_names):
            if sn == tn:
                dmask[ri, ci] = True
    return sliced, row_labels, col_labels, dmask


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

# Vendor label colors — used to tint tick labels
_VENDOR_COLORS = {
    "AT2": "#B45309",    # Leica — amber
    "GT450": "#B45309",
    "P": "#6D28D9",      # Philips — violet
    "Philips": "#6D28D9",
    "S210": "#0369A1",   # Hamamatsu — steel blue
    "S360": "#0369A1",
    "S60": "#0369A1",
    "SQ": "#0369A1",
    "DP200": "#0F766E",  # Roche — teal
    "P1000": "#BE185D",  # 3DHistech — magenta
}


def _color_tick_labels(ax: plt.Axes, col_names: list[str], row_names: list[str]) -> None:
    """Color x and y tick labels by vendor and draw a thin vendor-colored underline."""
    # X tick labels
    for tick, name in zip(ax.get_xticklabels(), col_names):
        base = name.split("\n")[0]
        color = _VENDOR_COLORS.get(base, "#222")
        tick.set_color(color)
        tick.set_fontweight("bold")

    # Y tick labels
    for tick, name in zip(ax.get_yticklabels(), row_names):
        base = name.split("\n")[0]
        color = _VENDOR_COLORS.get(base, "#222")
        tick.set_color(color)
        tick.set_fontweight("bold")

    # Thin vendor-colored strip just outside the plot border (not overlapping labels)
    n_cols = len(col_names)
    n_rows = len(row_names)
    ax_bbox = ax.get_position()  # normalized figure coords — use data coords instead

    for i, name in enumerate(col_names):
        base = name.split("\n")[0]
        color = _VENDOR_COLORS.get(base, "#888")
        ax.axvspan(i - 0.5, i + 0.5,
                   ymin=1.0, ymax=1.025,
                   color=color, clip_on=False, lw=0, transform=ax.transData,
                   zorder=0)

    for i, name in enumerate(row_names):
        base = name.split("\n")[0]
        color = _VENDOR_COLORS.get(base, "#888")
        ax.axhspan(i - 0.5, i + 0.5,
                   xmin=1.0, xmax=1.025,
                   color=color, clip_on=False, lw=0, transform=ax.transData,
                   zorder=0)


def draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    show_values: bool = True,
    fmt: str = ".3f",
    diag_mask: np.ndarray | None = None,
) -> mpl.cm.ScalarMappable:
    n_rows, n_cols = mat.shape

    # Plot the matrix; shade diagonal (identity) cells distinctly
    plot_mat = mat.copy()
    if diag_mask is not None:
        # Replace diagonal with NaN for separate rendering
        plot_mat[diag_mask] = np.nan

    im = ax.imshow(plot_mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    # Draw identity diagonal cells with a neutral gray overlay
    if diag_mask is not None:
        diag_display = np.full_like(mat, np.nan)
        diag_display[diag_mask] = mat[diag_mask]
        ax.imshow(diag_display, cmap="Greys_r", vmin=0.97, vmax=1.01,
                  aspect="equal", interpolation="nearest", alpha=0.55)
        # Thin diagonal border
        rows_d, cols_d = np.where(diag_mask)
        for r, c in zip(rows_d, cols_d):
            ax.add_patch(
                plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                               fill=False, edgecolor="#555", linewidth=1.2, zorder=3)
            )

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Target scanner", fontsize=11, labelpad=6)
    ax.set_ylabel("Source scanner", fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=14, pad=10)

    # Color tick labels by vendor and add thin strips outside the plot border
    _color_tick_labels(ax, col_labels, row_labels)

    if show_values:
        cmap_obj = plt.get_cmap(cmap)
        span = max(vmax - vmin, 1e-6)
        for ri in range(n_rows):
            for ci in range(n_cols):
                v = mat[ri, ci]
                if not np.isnan(v):
                    is_diag = diag_mask is not None and diag_mask[ri, ci]
                    if is_diag:
                        # Grey overlay makes these cells medium-light grey — use dark text
                        ax.text(
                            ci, ri, format(v, fmt),
                            ha="center", va="center",
                            fontsize=9, color="#333", fontweight="normal",
                            zorder=5,
                        )
                        continue
                    norm_v = (v - vmin) / span
                    # Determine text color from actual colormap luminance
                    r, g, b, _ = cmap_obj(norm_v)
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    txt_color = "white" if luminance < 0.50 else "#111"
                    ax.text(
                        ci, ri, format(v, fmt),
                        ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="normal",
                        zorder=4,
                    )

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


# ---------------------------------------------------------------------------
# Individual per-model scanner cosine / improvement figures
# ---------------------------------------------------------------------------

def _diag_mask(n_rows: int, n_cols: int) -> np.ndarray:
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(min(n_rows, n_cols)):
        mask[i, i] = True
    return mask


def fig_scanner_cosine(
    model_name: str,
    run_dir: Path,
    vmin: float | None = None,
    vmax: float = 1.0,
) -> plt.Figure:
    data = {p: load_phase(run_dir, p) for p in PHASES}
    data = {p: d for p, d in data.items() if d is not None}

    # Use only off-diagonal values for vmin so diagonal ~0.997 doesn't anchor the scale
    all_off_diag = []
    for p, d in data.items():
        m = d["pred_mat"].copy()
        n = m.shape[0]
        for i in range(n):
            m[i, i] = np.nan
        all_off_diag.extend(m[~np.isnan(m)].tolist())
    _vmin = max(float(np.percentile(all_off_diag, 1)) - 0.01, 0.7) if vmin is None else vmin

    n_phases = len(data)
    fig, axes = plt.subplots(1, n_phases, figsize=(5.2 * n_phases, 5.8),
                             constrained_layout=True)
    if n_phases == 1:
        axes = [axes]

    for ax, phase in zip(axes, data):
        d = data[phase]
        mat = d["pred_mat"]
        names = d["scanner_names"]

        if phase == "scorpion":
            mat, row_labels, col_labels, dmask = scorpion_slice(mat, names)
        else:
            row_labels = col_labels = names
            dmask = _diag_mask(mat.shape[0], mat.shape[1])

        im = draw_heatmap(
            ax, mat, row_labels, col_labels,
            title=PHASE_LABELS[phase],
            vmin=_vmin, vmax=vmax, cmap=CMAP_COS,
            diag_mask=dmask,
        )

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=20)
    cbar.set_label("Mean cosine similarity", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Scanner transfer — {model_name}", fontsize=16, fontweight="bold", y=1.01)
    return fig


def fig_scanner_improvement(
    model_name: str,
    run_dir: Path,
    abs_max: float | None = None,
) -> plt.Figure:
    data = {p: load_phase(run_dir, p) for p in PHASES}
    data = {p: d for p, d in data.items() if d is not None and d["orig_mat"] is not None}

    if not data:
        return None

    diffs = {}
    for phase, d in data.items():
        diff = d["pred_mat"] - d["orig_mat"]
        names = d["scanner_names"]
        if phase == "scorpion":
            diff, row_labels, col_labels, dmask = scorpion_slice(diff, names)
        else:
            row_labels = col_labels = names
            dmask = _diag_mask(diff.shape[0], diff.shape[1])
        diffs[phase] = (diff, row_labels, col_labels, dmask)

    # Scale from off-diagonal values only (diagonal improvement is unrelated)
    all_off_diag = []
    for diff, _, _, dmask in diffs.values():
        tmp = diff.copy()
        if dmask is not None:
            tmp[dmask] = np.nan
        all_off_diag.extend(tmp[~np.isnan(tmp)].tolist())
    _abs_max = float(np.abs(all_off_diag).max()) * 1.05 if abs_max is None else abs_max

    n_phases = len(diffs)
    fig, axes = plt.subplots(1, n_phases, figsize=(5.2 * n_phases, 5.8),
                             constrained_layout=True)
    if n_phases == 1:
        axes = [axes]

    for ax, phase in zip(axes, diffs):
        diff, row_labels, col_labels, dmask = diffs[phase]
        im = draw_heatmap(
            ax, diff, row_labels, col_labels,
            title=PHASE_LABELS[phase],
            vmin=-_abs_max, vmax=_abs_max, cmap=CMAP_DIFF,
            fmt="+.3f",
            diag_mask=dmask,
        )

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=20)
    cbar.set_label("Cosine improvement (pred − baseline)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Scanner transfer improvement — {model_name}", fontsize=16, fontweight="bold", y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Combined comparison figures (both models side-by-side)
# ---------------------------------------------------------------------------

def fig_comparison_cosine(
    models: dict[str, dict],
) -> plt.Figure:
    all_data = {}
    for mname, mcfg in models.items():
        for phase in PHASES:
            d = load_phase(mcfg["run"], phase)
            if d is not None:
                all_data[(mname, phase)] = d

    model_names = list(models.keys())
    phases_present = [p for p in PHASES if any((m, p) in all_data for m in model_names)]

    all_vals = [v for d in all_data.values() for v in d["pred_mat"][~np.isnan(d["pred_mat"])]]
    vmin = max(float(np.percentile(all_vals, 1)) - 0.01, 0.7)
    vmax = 1.0

    n_rows = len(model_names)
    n_cols = len(phases_present)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.5 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    last_im = None
    for ri, mname in enumerate(model_names):
        for ci, phase in enumerate(phases_present):
            ax = axes[ri, ci]
            key = (mname, phase)
            if key not in all_data:
                ax.axis("off")
                continue
            d = all_data[key]
            mat, names = d["pred_mat"], d["scanner_names"]
            if phase == "scorpion":
                mat, row_labels, col_labels, dmask = scorpion_slice(mat, names)
            else:
                row_labels = col_labels = names
                dmask = _diag_mask(mat.shape[0], mat.shape[1])

            title = PHASE_LABELS[phase] if ri == 0 else ""
            last_im = draw_heatmap(ax, mat, row_labels, col_labels,
                                   title=title, vmin=vmin, vmax=vmax, cmap=CMAP_COS,
                                   diag_mask=dmask)

        # Row label (model name)
        axes[ri, 0].set_ylabel(
            f"{mname}\n\nSource scanner", fontsize=12, labelpad=8
        )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.02, aspect=30)
        cbar.set_label("Mean cosine similarity", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Scanner transfer — cosine similarity to target-scanner embeddings",
                 fontsize=15, fontweight="bold")
    return fig


def fig_comparison_improvement(
    models: dict[str, dict],
) -> plt.Figure:
    all_data = {}
    for mname, mcfg in models.items():
        for phase in PHASES:
            d = load_phase(mcfg["run"], phase)
            if d is not None and d["orig_mat"] is not None:
                all_data[(mname, phase)] = d

    model_names = list(models.keys())
    phases_present = [p for p in PHASES if any((m, p) in all_data for m in model_names)]

    all_diffs = []
    for (mname, phase), d in all_data.items():
        diff = d["pred_mat"] - d["orig_mat"]
        if phase != "scorpion":
            n = diff.shape[0]
            for i in range(n):
                diff[i, i] = np.nan
        all_diffs.extend(diff[~np.isnan(diff)].tolist())
    abs_max = float(np.abs(all_diffs).max()) * 1.05

    n_rows = len(model_names)
    n_cols = len(phases_present)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.5 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    last_im = None
    for ri, mname in enumerate(model_names):
        for ci, phase in enumerate(phases_present):
            ax = axes[ri, ci]
            key = (mname, phase)
            if key not in all_data:
                ax.axis("off")
                continue
            d = all_data[key]
            diff = d["pred_mat"] - d["orig_mat"]
            names = d["scanner_names"]
            if phase == "scorpion":
                diff, row_labels, col_labels, dmask = scorpion_slice(diff, names)
            else:
                row_labels = col_labels = names
                dmask = _diag_mask(diff.shape[0], diff.shape[1])

            title = PHASE_LABELS[phase] if ri == 0 else ""
            last_im = draw_heatmap(ax, diff, row_labels, col_labels,
                                   title=title, vmin=-abs_max, vmax=abs_max,
                                   cmap=CMAP_DIFF, fmt="+.3f", diag_mask=dmask)

        axes[ri, 0].set_ylabel(f"{mname}\n\nSource scanner", fontsize=12, labelpad=8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.02, aspect=30)
        cbar.set_label("Cosine improvement (pred − baseline)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Scanner transfer — improvement over baseline (pred − without transfer)",
                 fontsize=15, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Staining bar chart
# ---------------------------------------------------------------------------

# Map staining codes to Solution Category (strip trailing H, except MY)
def _stain_category(name: str) -> str:
    if name.endswith("H") and name != "MY":
        return name[:-1]
    return name


_CATEGORY_PALETTE = {
    "GIV": "#4C72B0",
    "GM": "#55A868",
    "GV": "#C44E52",
    "MY": "#8172B2",
    "HR": "#CCB974",
    "KR": "#64B5CD",
    "LM": "#DD8452",
    "LMH": "#DD8452",
}


_HOLDOUT_STAININGS = {"GVH"}


def fig_staining(
    model_name: str,
    run_dir: Path,
) -> plt.Figure:
    # test has all trained stainings; test_holdout_staining has holdout (GVH) pairs
    test_data = load_phase(run_dir, "test")
    holdout_data = load_phase(run_dir, "test_holdout_staining")

    test_means = test_data["staining_cos"] if test_data else {}
    holdout_means = holdout_data["staining_cos"] if holdout_data else {}

    if not test_means and not holdout_means:
        return None

    # Order: all stainings alphabetically, trained first then holdout
    trained_stains = sorted(s for s in test_means if s not in _HOLDOUT_STAININGS)
    holdout_stains = sorted(_HOLDOUT_STAININGS & set(holdout_means))
    all_stains = trained_stains + holdout_stains

    categories = [_stain_category(s) for s in all_stains]
    bar_colors = [_CATEGORY_PALETTE.get(c, "#aaa") for c in categories]

    x = np.arange(len(all_stains))
    width = 0.62
    all_vals = list(test_means.values()) + list(holdout_means.values())
    vmin = max(0.0, float(np.min(all_vals)) - 0.03)

    fig, ax = plt.subplots(figsize=(max(8.5, len(all_stains) * 0.75 + 2.5), 5.2))

    # Draw bars — each staining appears once; holdout stainings get hatching
    bars_list = []
    for i, stain in enumerate(all_stains):
        is_holdout = stain in _HOLDOUT_STAININGS
        val = holdout_means.get(stain, float("nan")) if is_holdout else test_means.get(stain, float("nan"))
        color = bar_colors[i]
        b = ax.bar(
            x[i], val, width,
            color=color,
            alpha=1.0,
            hatch="////" if is_holdout else "",
            edgecolor="#fff" if not is_holdout else "#333",
            linewidth=0.5 if not is_holdout else 1.2,
            zorder=2,
        )
        bars_list.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels(all_stains, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Mean cosine similarity", fontsize=12)
    ax.set_ylim(bottom=vmin)
    ax.set_title(f"Per-staining cosine similarity — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, color="#ccc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)

    # Vertical separator between trained and holdout stainings
    if trained_stains and holdout_stains:
        sep_x = len(trained_stains) - 0.5
        ax.axvline(sep_x, color="#aaa", linewidth=1.2, linestyle="--", zorder=1)

    # Series legend (top right)
    legend_handles = [
        mpl.patches.Patch(facecolor="#aaa", label="Trained staining (test, unseen organs)"),
        mpl.patches.Patch(facecolor="#aaa", hatch="////", edgecolor="#333",
                          label="Holdout staining (GVH — unseen during training)"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="upper right",
              framealpha=0.9, edgecolor="#ccc")

    # Category color legend — top left inset
    seen_cats: dict[str, str] = {}
    for cat, col in zip(categories, bar_colors):
        if cat not in seen_cats:
            seen_cats[cat] = col
    cat_handles = [
        mpl.patches.Patch(facecolor=c, label=cat)
        for cat, c in seen_cats.items()
    ]
    ax2 = ax.inset_axes([0.01, 0.62, 0.20, 0.36])
    ax2.legend(
        handles=cat_handles, title="Hematoxylin\ncategory",
        fontsize=11, title_fontsize=11, loc="center",
        framealpha=0.9,
    )
    ax2.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=_REPO_ROOT / "presentation_figures")
    return p.parse_args()


def fig_single_cosine(
    model_name: str,
    run_dir: Path,
    phase: str,
    vmin: float | None = None,
    vmax: float = 1.0,
) -> plt.Figure | None:
    """Single-panel cosine heatmap for one phase, no figure title."""
    d = load_phase(run_dir, phase)
    if d is None:
        return None

    mat = d["pred_mat"]
    names = d["scanner_names"]

    if phase == "scorpion":
        mat, row_labels, col_labels, dmask = scorpion_slice(mat, names)
    else:
        row_labels = col_labels = names
        dmask = _diag_mask(mat.shape[0], mat.shape[1])

    # vmin from off-diagonal values
    if vmin is None:
        off = mat.copy()
        if dmask is not None:
            off[dmask] = np.nan
        valid = off[~np.isnan(off)]
        _vmin = max(float(np.percentile(valid, 1)) - 0.01, 0.7) if valid.size else 0.85
    else:
        _vmin = vmin

    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(1, 1,
                           figsize=(max(n_cols * 0.70 + 1.8, 3.5),
                                    max(n_rows * 0.70 + 1.8, 3.5)),
                           constrained_layout=True)
    im = draw_heatmap(ax, mat, row_labels, col_labels,
                      title=PHASE_LABELS[phase],
                      vmin=_vmin, vmax=vmax, cmap=CMAP_COS, diag_mask=dmask)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean cosine similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    return fig


def fig_single_improvement(
    model_name: str,
    run_dir: Path,
    phase: str,
    abs_max: float | None = None,
) -> plt.Figure | None:
    """Single-panel improvement heatmap for one phase, no figure title."""
    d = load_phase(run_dir, phase)
    if d is None or d["orig_mat"] is None:
        return None

    diff = d["pred_mat"] - d["orig_mat"]
    names = d["scanner_names"]

    if phase == "scorpion":
        diff, row_labels, col_labels, dmask = scorpion_slice(diff, names)
    else:
        row_labels = col_labels = names
        dmask = _diag_mask(diff.shape[0], diff.shape[1])

    if abs_max is None:
        off = diff.copy()
        if dmask is not None:
            off[dmask] = np.nan
        valid = off[~np.isnan(off)]
        _abs_max = float(np.abs(valid).max()) * 1.05 if valid.size else 0.1
    else:
        _abs_max = abs_max

    n_rows, n_cols = diff.shape
    fig, ax = plt.subplots(1, 1,
                           figsize=(max(n_cols * 0.70 + 1.8, 3.5),
                                    max(n_rows * 0.70 + 1.8, 3.5)),
                           constrained_layout=True)
    im = draw_heatmap(ax, diff, row_labels, col_labels,
                      title=PHASE_LABELS[phase],
                      vmin=-_abs_max, vmax=_abs_max, cmap=CMAP_DIFF,
                      fmt="+.3f", diag_mask=dmask)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine improvement (pred − baseline)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    return fig


def save(fig: plt.Figure | None, path: Path) -> None:
    if fig is None:
        print(f"  Skipping {path.name} (no data)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


_PHASE_SLUGS = {
    "test": "plism_test",
    "test_holdout_staining": "plism_holdout_gvh",
    "scorpion": "scorpion",
}


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for model_name, mcfg in MODELS.items():
        slug = mcfg["slug"]
        run_dir = mcfg["run"]
        print(f"\n{model_name}")

        # Multi-phase combined figures (with suptitle)
        save(fig_scanner_cosine(model_name, run_dir),
             outdir / f"scanner_cosine_{slug}.png")
        save(fig_scanner_improvement(model_name, run_dir),
             outdir / f"scanner_improvement_{slug}.png")
        save(fig_staining(model_name, run_dir),
             outdir / f"staining_{slug}.png")

        # Individual per-phase figures in model subfolder (no suptitle)
        sub = outdir / slug
        for phase in PHASES:
            phase_slug = _PHASE_SLUGS[phase]
            save(fig_single_cosine(model_name, run_dir, phase),
                 sub / f"cosine_{phase_slug}.png")
            save(fig_single_improvement(model_name, run_dir, phase),
                 sub / f"improvement_{phase_slug}.png")

    print("\nCombined figures")
    save(fig_comparison_cosine(MODELS), outdir / "comparison_cosine.png")
    save(fig_comparison_improvement(MODELS), outdir / "comparison_improvement.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
