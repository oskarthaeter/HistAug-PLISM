"""
Visualize h0-mini patch embeddings across scanners using PCA and UMAP.

Each file in the features directory encodes one slide as:
  {tissue}_{scanner}_to_GMH_S60.tif/features.npy
  shape: (N_patches, 771)  — first 3 cols are spatial coords, rest are 768-dim embeddings.

All slides are H&E stained.  The tissue abbreviation encodes the hematoxylin product and
conditions (see CLAUDE.md).  Stripping a trailing "H" gives the Solution Category, which
groups slides by hematoxylin product and is used for colour-coding in plots.

Usage:
    python visualize_scanner_embeddings.py \
        --features_dir /mnt/data/plismbench/features/h0_mini \
        --out_dir ./scanner_viz \
        [--patches_per_slide 500] \
        [--pca_components 50] \
        [--seed 42]
"""

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


# ── helpers ──────────────────────────────────────────────────────────────────

SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]

# Colorblind-friendly palette (7 scanners)
SCANNER_COLORS = {
    "AT2":   "#E69F00",
    "GT450": "#56B4E9",
    "P":     "#009E73",
    "S210":  "#F0E442",
    "S360":  "#0072B2",
    "S60":   "#D55E00",
    "SQ":    "#CC79A7",
}

# Distinct marker per scanner
SCANNER_MARKERS = {
    "AT2":   "o",
    "GT450": "s",
    "P":     "^",
    "S210":  "D",
    "S360":  "v",
    "S60":   "P",
    "SQ":    "*",
}

# Solution Category = hematoxylin product group (strip trailing H from abbreviation).
# 7 categories; colorblind-friendly palette.
STAINING_ORDER = ["GIV", "GM", "GV", "MY", "HR", "KR", "LM"]
STAINING_COLORS = {
    "GIV": "#E69F00",
    "GM":  "#56B4E9",
    "GV":  "#009E73",
    "MY":  "#F0E442",
    "HR":  "#0072B2",
    "KR":  "#D55E00",
    "LM":  "#CC79A7",
}


def parse_filename(name: str):
    """Return (tissue, scanner) from '{tissue}_{scanner}_to_GMH_S60.tif'."""
    stem = re.sub(r"_to_GMH_S60\.tif$", "", name)
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse filename: {name!r}")
    return parts[0], parts[1]


def staining_from_tissue(tissue: str) -> str:
    """Return Solution Category (hematoxylin product group) for a tissue abbreviation.

    Strip a trailing 'H' to get the category (e.g. GIVH → GIV, MY → MY).
    """
    return tissue[:-1] if tissue.endswith("H") else tissue


def load_features(features_dir: Path, patches_per_slide: int, rng: np.random.Generator):
    """Load subsampled embeddings from all slides.

    Returns:
        embeddings : (N, 768) float32
        scanners   : (N,) str
        stainings  : (N,) str   — Solution Category (hematoxylin product group)
        tissues    : (N,) str   — raw tissue code
        slide_ids  : (N,) str
    """
    embeddings, scanners, stainings, tissues, slide_ids = [], [], [], [], []

    slide_dirs = sorted(p for p in features_dir.iterdir() if p.is_dir())
    print(f"Found {len(slide_dirs)} slides.")

    for slide_dir in slide_dirs:
        npy_path = slide_dir / "features.npy"
        if not npy_path.exists():
            print(f"  [warn] no features.npy in {slide_dir.name}, skipping.")
            continue

        try:
            tissue, scanner = parse_filename(slide_dir.name)
        except ValueError as e:
            print(f"  [warn] {e}, skipping.")
            continue

        feat = np.load(npy_path)  # (N, 771)
        emb = feat[:, 3:].astype(np.float32)  # drop coord cols → (N, 768)

        n = len(emb)
        idx = rng.choice(n, size=min(patches_per_slide, n), replace=False)
        emb = emb[idx]

        stain = staining_from_tissue(tissue)
        embeddings.append(emb)
        scanners.extend([scanner] * len(emb))
        stainings.extend([stain] * len(emb))
        tissues.extend([tissue] * len(emb))
        slide_ids.extend([slide_dir.name] * len(emb))

    return (
        np.concatenate(embeddings, axis=0),
        np.array(scanners),
        np.array(stainings),
        np.array(tissues),
        np.array(slide_ids),
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def scatter_staining_marker(ax, xy, scanners, stainings, scanner_subset, title,
                             alpha=0.4, s=12):
    """Scatter colored by staining, shaped by scanner."""
    for scanner in scanner_subset:
        for stain in STAINING_ORDER:
            mask = (scanners == scanner) & (stainings == stain)
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                c=STAINING_COLORS[stain],
                marker=SCANNER_MARKERS.get(scanner, "o"),
                label=f"{scanner} / {stain}",
                alpha=alpha,
                s=s,
                linewidths=0,
                rasterized=True,
            )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def make_combined_legend(ax, scanner_subset):
    """Legend with staining patches + scanner marker lines."""
    stain_handles = [
        mpatches.Patch(color=STAINING_COLORS[s], label=s)
        for s in STAINING_ORDER
        if s in STAINING_COLORS
    ]
    marker_handles = [
        mlines.Line2D([], [], color="grey",
                      marker=SCANNER_MARKERS.get(sc, "o"),
                      linestyle="None", markersize=6, label=sc)
        for sc in scanner_subset
        if sc in SCANNER_MARKERS
    ]
    ax.legend(
        handles=stain_handles + marker_handles,
        loc="best",
        fontsize=7,
        framealpha=0.7,
        ncol=2,
    )


def scatter(ax, xy, labels, color_map, order, title, alpha=0.35, s=6):
    """Generic scatter plot colored by discrete label."""
    for label in order:
        mask = labels == label
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=color_map[label],
            label=label,
            alpha=alpha,
            s=s,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def make_legend(ax, color_map, order):
    handles = [
        mpatches.Patch(color=color_map[k], label=k)
        for k in order
        if k in color_map
    ]
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=8,
        markerscale=1.5,
        framealpha=0.7,
        ncol=1,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize h0-mini scanner embeddings.")
    parser.add_argument(
        "--features_dir",
        default="/mnt/data/plismbench/features/h0_mini",
        help="Root directory containing one sub-folder per slide.",
    )
    parser.add_argument(
        "--out_dir", default="scanner_viz",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--patches_per_slide", type=int, default=500,
        help="Max patches to sample per slide.",
    )
    parser.add_argument(
        "--pca_components", type=int, default=50,
        help="Number of PCA components fed into UMAP.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading features …")
    emb, scanners, stainings, tissues, slide_ids = load_features(
        Path(args.features_dir), args.patches_per_slide, rng
    )
    print(f"Total patches loaded: {emb.shape[0]:,}  |  embedding dim: {emb.shape[1]}")

    # ── 2. PCA ────────────────────────────────────────────────────────────────
    print(f"Fitting PCA ({args.pca_components} components) …")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)

    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    emb_pca = pca.fit_transform(emb_scaled)

    explained = pca.explained_variance_ratio_
    pc12 = emb_pca[:, :2]  # first two PCs for direct scatter

    # ── 3. UMAP ───────────────────────────────────────────────────────────────
    print("Fitting UMAP …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=args.seed,
        verbose=True,
    )
    emb_umap = reducer.fit_transform(emb_pca)

    present_scanners = [s for s in SCANNER_ORDER if s in set(scanners)]

    # ── 4. Explained variance bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    top_k = min(20, args.pca_components)
    ax.bar(range(1, top_k + 1), explained[:top_k] * 100, color="#0072B2", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Explained Variance — h0-mini embeddings")
    ax.set_xticks(range(1, top_k + 1))
    fig.tight_layout()
    path = out_dir / "pca_explained_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── 5. PCA all scanners — staining colour, scanner marker ────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_staining_marker(
        ax, pc12, scanners, stainings, present_scanners,
        f"PCA (PC1 {explained[0]*100:.1f}% / PC2 {explained[1]*100:.1f}%)\n"
        "Colour = H&E protocol  |  Marker = scanner",
    )
    make_combined_legend(ax, present_scanners)
    fig.tight_layout()
    path = out_dir / "pca_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── 6. Pairwise PCA plots (one per scanner pair) ──────────────────────────
    pair_dir = out_dir / "pca_pairwise"
    pair_dir.mkdir(exist_ok=True)

    pairs = list(itertools.combinations(present_scanners, 2))
    print(f"Generating {len(pairs)} pairwise PCA plots …")
    for sc_a, sc_b in pairs:
        mask = np.isin(scanners, [sc_a, sc_b])
        xy_pair = pc12[mask]
        sc_pair = scanners[mask]
        st_pair = stainings[mask]

        fig, ax = plt.subplots(figsize=(5, 5))
        scatter_staining_marker(
            ax, xy_pair, sc_pair, st_pair, [sc_a, sc_b],
            f"PCA  {sc_a} vs {sc_b}\n"
            f"(PC1 {explained[0]*100:.1f}% / PC2 {explained[1]*100:.1f}%)",
            alpha=0.5,
            s=18,
        )
        make_combined_legend(ax, [sc_a, sc_b])
        fig.tight_layout()
        path = pair_dir / f"pca_{sc_a}_vs_{sc_b}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(pairs)} pairwise plots to {pair_dir}/")

    # ── 7. UMAP — staining colour, scanner marker ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_staining_marker(
        ax, emb_umap, scanners, stainings, present_scanners,
        "UMAP — Colour = H&E protocol  |  Marker = scanner",
    )
    make_combined_legend(ax, present_scanners)
    fig.tight_layout()
    path = out_dir / "umap_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── 8. Per-scanner UMAP facet ─────────────────────────────────────────────
    ncols = 4
    nrows = (len(present_scanners) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    bg_color = "#cccccc"
    for ax, scanner in zip(axes, present_scanners):
        ax.scatter(
            emb_umap[:, 0], emb_umap[:, 1],
            color=bg_color, alpha=0.15, s=3, linewidths=0, rasterized=True,
        )
        mask = scanners == scanner
        for stain in STAINING_ORDER:
            smask = mask & (stainings == stain)
            if not smask.any():
                continue
            ax.scatter(
                emb_umap[smask, 0], emb_umap[smask, 1],
                c=STAINING_COLORS[stain],
                marker=SCANNER_MARKERS.get(scanner, "o"),
                alpha=0.6, s=10, linewidths=0, rasterized=True,
                label=stain,
            )
        ax.set_title(scanner, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.6)

    for ax in axes[len(present_scanners):]:
        ax.set_visible(False)

    fig.suptitle("UMAP facets — colour = H&E protocol, marker = scanner", fontsize=13)
    fig.tight_layout()
    path = out_dir / "umap_per_scanner.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── 9. PC1 distribution per scanner (violin) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    data_by_scanner = [emb_pca[scanners == s, 0] for s in present_scanners]
    parts = ax.violinplot(
        data_by_scanner,
        positions=range(len(present_scanners)),
        showmedians=True,
        showextrema=False,
    )
    for body, scanner in zip(parts["bodies"], present_scanners):
        body.set_facecolor(SCANNER_COLORS.get(scanner, "grey"))
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(present_scanners)))
    ax.set_xticklabels(present_scanners)
    ax.set_ylabel("PC1 projection")
    ax.set_title(f"PC1 distribution per scanner  (explains {explained[0]*100:.1f}% variance)")
    fig.tight_layout()
    path = out_dir / "pc1_violin_by_scanner.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\nAll figures written to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
