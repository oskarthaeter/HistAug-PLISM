"""
Tests for PlismPairPrefeaturesDataset.

Requires real data at /mnt/data/plismbench/features/phikon and the
plism_organ_loc.csv file at the repo root.  Run from the repo root:

    pytest tests/test_plism_pair_prefeatures_dataset.py -v
"""

import sys
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "histaug"
sys.path.insert(0, str(SRC))

from datasets.plism_pair_prefeatures_dataset import PlismPairPrefeaturesDataset

FEATURES_ROOT = Path("/mnt/data/plismbench/features/phikon")
ORGAN_CSV = REPO_ROOT / "plism_organ_loc.csv"

pytestmark = pytest.mark.skipif(
    not FEATURES_ROOT.exists(),
    reason="PLISM feature data not available",
)

DATASET_CFG = {
    "features_root": str(FEATURES_ROOT),
    "organ_loc_csv": str(ORGAN_CSV),
    "train_split": 0.8,
    "split_seed": 2025,
    "scanners": [],
    "stainings": [],
    "pairing": {
        "allow_cross_staining": False,
        "allow_same_scanner": True,
        "symmetric": True,
        "tiles_per_pair_per_epoch": 50,  # small for fast tests
    },
}

ALL_SCANNERS = {"AT2", "GT450", "P", "S210", "S360", "S60", "SQ"}
ALL_STAININGS = {"GIV", "GIVH", "GM", "GMH", "GV", "GVH", "HR", "HRH",
                 "KR", "KRH", "LM", "LMH", "MY"}


@pytest.fixture(scope="module")
def train_ds():
    return PlismPairPrefeaturesDataset(DATASET_CFG, state="train")


@pytest.fixture(scope="module")
def val_ds():
    return PlismPairPrefeaturesDataset(DATASET_CFG, state="val")


# ---------------------------------------------------------------------------
# Coverage: all scanners and stainings in both splits
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_train_has_all_scanners(self, train_ds):
        # slide_pairs dicts store scanner/staining names directly as strings
        scanners_in_pairs = {p["scanner_a"] for p in train_ds.slide_pairs}
        scanners_in_pairs |= {p["scanner_b"] for p in train_ds.slide_pairs}
        assert scanners_in_pairs == ALL_SCANNERS, (
            f"Missing from train pairs: {ALL_SCANNERS - scanners_in_pairs}"
        )

    def test_val_has_all_scanners(self, val_ds):
        scanners_in_pairs = {p["scanner_a"] for p in val_ds.slide_pairs}
        scanners_in_pairs |= {p["scanner_b"] for p in val_ds.slide_pairs}
        assert scanners_in_pairs == ALL_SCANNERS, (
            f"Missing from val pairs: {ALL_SCANNERS - scanners_in_pairs}"
        )

    def test_train_has_all_stainings(self, train_ds):
        stainings_in_pairs = {p["staining_a"] for p in train_ds.slide_pairs}
        assert stainings_in_pairs == ALL_STAININGS, (
            f"Missing from train pairs: {ALL_STAININGS - stainings_in_pairs}"
        )

    def test_val_has_all_stainings(self, val_ds):
        stainings_in_pairs = {p["staining_a"] for p in val_ds.slide_pairs}
        assert stainings_in_pairs == ALL_STAININGS, (
            f"Missing from val pairs: {ALL_STAININGS - stainings_in_pairs}"
        )

    def test_same_pairs_in_train_and_val(self, train_ds, val_ds):
        """Both splits must expose the same set of slide pairs."""
        def pair_set(ds):
            return {
                (p["scanner_a"], p["scanner_b"], p["staining_a"])
                for p in ds.slide_pairs
            }
        assert pair_set(train_ds) == pair_set(val_ds)

    def test_pair_count(self, train_ds):
        """13 stainings × 7×6 ordered scanner pairs = 546 pairs."""
        expected = len(ALL_STAININGS) * len(ALL_SCANNERS) * (len(ALL_SCANNERS) - 1)
        assert len(train_ds.slide_pairs) == expected, (
            f"Expected {expected} pairs, got {len(train_ds.slide_pairs)}"
        )


# ---------------------------------------------------------------------------
# Patch-level split: train and val row indices must be disjoint
# ---------------------------------------------------------------------------

class TestPatchSplit:
    def test_row_indices_disjoint(self, train_ds, val_ds):
        train_rows = set(train_ds.valid_row_indices)
        val_rows = set(val_ds.valid_row_indices)
        overlap = train_rows & val_rows
        assert not overlap, (
            f"{len(overlap)} patch row(s) appear in both train and val valid_row_indices"
        )

    def test_row_indices_cover_all_patches(self, train_ds, val_ds):
        """Union of train + val rows should equal the full patch count."""
        train_rows = set(train_ds.valid_row_indices)
        val_rows = set(val_ds.valid_row_indices)
        all_rows = train_rows | val_rows
        assert len(all_rows) == train_ds.n_patches, (
            f"Union has {len(all_rows)} rows, expected {train_ds.n_patches}"
        )

    def test_split_ratio(self, train_ds, val_ds):
        n_train = len(train_ds.valid_row_indices)
        n_val = len(val_ds.valid_row_indices)
        total = n_train + n_val
        ratio = n_train / total
        assert 0.70 <= ratio <= 0.90, (
            f"Train patch fraction {ratio:.2f} is outside expected 70–90% range"
        )


# ---------------------------------------------------------------------------
# Dataset length
# ---------------------------------------------------------------------------

class TestLength:
    def test_train_length(self, train_ds):
        expected = len(train_ds.slide_pairs) * train_ds.tiles_per_pair_per_epoch
        assert len(train_ds) == expected

    def test_val_length(self, val_ds):
        expected = len(val_ds.slide_pairs) * len(val_ds.valid_row_indices)
        assert len(val_ds) == expected

    def test_val_longer_than_train_tiles_would_give(self, train_ds, val_ds):
        """Val enumerates all valid patches; train subsamples — lengths differ."""
        val_per_pair = len(val_ds.valid_row_indices)
        train_per_pair = train_ds.tiles_per_pair_per_epoch
        # val patches ≈ 20% of 16k ≈ 3200; train at 0.5 fraction ≈ 6500
        assert val_per_pair != train_per_pair

    def test_fraction_resolves_correctly(self):
        """tiles_per_pair_per_epoch=0.5 should resolve to ~50% of train patch count."""
        cfg = {**DATASET_CFG, "pairing": {**DATASET_CFG["pairing"], "tiles_per_pair_per_epoch": 0.5}}
        ds = PlismPairPrefeaturesDataset(cfg, state="train")
        n_train = len(ds.valid_row_indices)
        expected = max(1, round(0.5 * n_train))
        assert ds.tiles_per_pair_per_epoch == expected

    def test_integer_tiles_unchanged(self):
        """An integer value > 1 should pass through unchanged."""
        cfg = {**DATASET_CFG, "pairing": {**DATASET_CFG["pairing"], "tiles_per_pair_per_epoch": 200}}
        ds = PlismPairPrefeaturesDataset(cfg, state="train")
        assert ds.tiles_per_pair_per_epoch == 200


# ---------------------------------------------------------------------------
# __getitem__ correctness
# ---------------------------------------------------------------------------

class TestGetItem:
    def test_train_item_shape(self, train_ds):
        feat_a, feat_b, src_sc, tgt_sc, src_st, tgt_st = train_ds[0]
        assert feat_a.shape == (PlismPairPrefeaturesDataset.FEATURE_DIM,)
        assert feat_b.shape == (PlismPairPrefeaturesDataset.FEATURE_DIM,)
        assert isinstance(src_sc, int)

    def test_val_item_shape(self, val_ds):
        feat_a, feat_b, src_sc, tgt_sc, src_st, tgt_st = val_ds[0]
        assert feat_a.shape == (PlismPairPrefeaturesDataset.FEATURE_DIM,)
        assert feat_b.shape == (PlismPairPrefeaturesDataset.FEATURE_DIM,)

    def test_val_is_deterministic(self, val_ds):
        """Same index must return the same features on repeated calls."""
        item1 = val_ds[42]
        item2 = val_ds[42]
        assert torch.equal(item1[0], item2[0])
        assert torch.equal(item1[1], item2[1])

    def test_train_stochastic(self, train_ds):
        """Different indices within the same pair should sometimes differ (stochastic rows)."""
        n_per_pair = train_ds.tiles_per_pair_per_epoch
        # Collect 20 row samples from the first pair
        feats = [train_ds[i][0] for i in range(min(20, n_per_pair))]
        unique = {tuple(f.tolist()) for f in feats}
        # With 3000+ valid train rows and 20 draws, we almost certainly get >1 unique row
        assert len(unique) > 1, "Train sampling appears non-stochastic (all rows identical)"

    def test_index_out_of_range(self, train_ds):
        with pytest.raises(IndexError):
            train_ds[len(train_ds)]

    def test_scanner_ids_in_vocab(self, train_ds):
        feat_a, feat_b, src_sc, tgt_sc, src_st, tgt_st = train_ds[0]
        assert 0 <= src_sc < train_ds.scanner_vocab_size
        assert 0 <= tgt_sc < train_ds.scanner_vocab_size

    def test_staining_ids_in_vocab(self, train_ds):
        feat_a, feat_b, src_sc, tgt_sc, src_st, tgt_st = train_ds[0]
        assert 0 <= src_st < train_ds.staining_vocab_size
        assert 0 <= tgt_st < train_ds.staining_vocab_size

    def test_val_all_pairs_reachable(self, val_ds):
        """Every slide pair must be reachable from some index in val."""
        n_pairs = len(val_ds.slide_pairs)
        n_per_pair = len(val_ds.valid_row_indices)
        pair_indices_hit = set()
        for i in range(0, len(val_ds), n_per_pair):
            pair_indices_hit.add(i // n_per_pair)
        assert pair_indices_hit == set(range(n_pairs))


# ---------------------------------------------------------------------------
# Vocab stability across splits
# ---------------------------------------------------------------------------

class TestVocab:
    def test_same_scanner_vocab(self, train_ds, val_ds):
        assert train_ds.scanner_to_id == val_ds.scanner_to_id

    def test_same_staining_vocab(self, train_ds, val_ds):
        assert train_ds.staining_to_id == val_ds.staining_to_id

    def test_scanner_vocab_size(self, train_ds):
        assert train_ds.scanner_vocab_size == len(ALL_SCANNERS)

    def test_staining_vocab_size(self, train_ds):
        assert train_ds.staining_vocab_size == len(ALL_STAININGS)


# ---------------------------------------------------------------------------
# Warning when organ_loc_csv absent
# ---------------------------------------------------------------------------

class TestMissingOrganCsv:
    def test_warns_without_organ_csv(self):
        cfg = {**DATASET_CFG, "organ_loc_csv": None}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PlismPairPrefeaturesDataset(cfg, state="train")
        assert any(issubclass(x.category, UserWarning) for x in w), (
            "Expected a UserWarning when organ_loc_csv is absent"
        )

    def test_no_warn_for_test_without_organ_csv(self):
        cfg = {**DATASET_CFG, "organ_loc_csv": None}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PlismPairPrefeaturesDataset(cfg, state="test")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings, "Unexpected UserWarning for test state without organ_loc_csv"


# ---------------------------------------------------------------------------
# Held-out staining (GVH) test split
# ---------------------------------------------------------------------------

HOLDOUT_CFG = {**DATASET_CFG, "holdout_stainings": ["GVH"]}


@pytest.fixture(scope="module")
def holdout_train_ds():
    return PlismPairPrefeaturesDataset(HOLDOUT_CFG, state="train")


@pytest.fixture(scope="module")
def holdout_val_ds():
    return PlismPairPrefeaturesDataset(HOLDOUT_CFG, state="val")


@pytest.fixture(scope="module")
def holdout_test_ds():
    return PlismPairPrefeaturesDataset(HOLDOUT_CFG, state="test_holdout_staining")


class TestHoldoutStaining:
    def test_train_excludes_holdout(self, holdout_train_ds):
        stainings = {p["staining_a"] for p in holdout_train_ds.slide_pairs}
        stainings |= {p["staining_b"] for p in holdout_train_ds.slide_pairs}
        assert "GVH" not in stainings
        assert stainings == ALL_STAININGS - {"GVH"}

    def test_val_excludes_holdout(self, holdout_val_ds):
        stainings = {p["staining_a"] for p in holdout_val_ds.slide_pairs}
        stainings |= {p["staining_b"] for p in holdout_val_ds.slide_pairs}
        assert "GVH" not in stainings

    def test_holdout_test_only_holdout(self, holdout_test_ds):
        stainings = {p["staining_a"] for p in holdout_test_ds.slide_pairs}
        stainings |= {p["staining_b"] for p in holdout_test_ds.slide_pairs}
        assert stainings == {"GVH"}

    def test_holdout_test_all_scanners(self, holdout_test_ds):
        """Option A pairing: GVH ↔ GVH across all scanners."""
        scanners = {p["scanner_a"] for p in holdout_test_ds.slide_pairs}
        scanners |= {p["scanner_b"] for p in holdout_test_ds.slide_pairs}
        assert scanners == ALL_SCANNERS

    def test_holdout_test_pair_count(self, holdout_test_ds):
        """1 staining × 7×6 ordered scanner pairs = 42 pairs."""
        expected = len(ALL_SCANNERS) * (len(ALL_SCANNERS) - 1)
        assert len(holdout_test_ds.slide_pairs) == expected

    def test_holdout_test_uses_all_patches(self, holdout_test_ds):
        """Staining is the only holdout axis; all 16,278 patches are valid."""
        assert len(holdout_test_ds.valid_row_indices) == holdout_test_ds.n_patches

    def test_vocab_stable_across_holdout_states(
        self, holdout_train_ds, holdout_val_ds, holdout_test_ds
    ):
        """Vocab built from all slides — GVH staining ID must still exist in train."""
        assert holdout_train_ds.scanner_to_id == holdout_val_ds.scanner_to_id
        assert holdout_train_ds.scanner_to_id == holdout_test_ds.scanner_to_id
        assert holdout_train_ds.staining_to_id == holdout_test_ds.staining_to_id
        assert "GVH" in holdout_train_ds.staining_to_id

    def test_describe_splits_partitions_slides(self, holdout_train_ds):
        splits = holdout_train_ds.describe_splits()
        train_slides = {s["slide_id"] for s in splits["slides"]["train"]}
        holdout_slides = {s["slide_id"] for s in splits["slides"]["test_holdout_staining"]}
        assert train_slides.isdisjoint(holdout_slides)
        assert all(s["staining"] == "GVH" for s in splits["slides"]["test_holdout_staining"])
        assert all(s["staining"] != "GVH" for s in splits["slides"]["train"])
        assert splits["holdout_stainings"] == ["GVH"]

    def test_holdout_state_rejected_without_config(self):
        """state='test_holdout_staining' must error if holdout_stainings is empty."""
        with pytest.raises(ValueError, match="holdout_stainings"):
            PlismPairPrefeaturesDataset(DATASET_CFG, state="test_holdout_staining")

    def test_unknown_holdout_staining_rejected(self):
        cfg = {**DATASET_CFG, "holdout_stainings": ["NOPE"]}
        with pytest.raises(ValueError, match="matched no slides"):
            PlismPairPrefeaturesDataset(cfg, state="train")
