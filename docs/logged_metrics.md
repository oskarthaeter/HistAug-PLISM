# Logged Metrics and Scanner Heatmaps

All logging is centralized in `src/histaug/models/model_interface.py`. Scanner heatmaps (described in the final section) visualize pairwise cosine similarities across all scanner combinations.

---

## Step-Level Metrics (Training Only)

| Key           | Phase | Frequency |
|---------------|-------|-----------|
| `train/loss`  | train | per step  |

**`train/loss`** — The loss for the current batch, displayed in the progress bar. For standard and conditioned batches this is `loss_imgaug + loss_id`; for pair batches a residual penalty may be added.

---

## Epoch-Level Scalar Metrics

All scalar metrics are logged at the end of every epoch and synced across distributed processes.

### Loss

| Key                            | Phases                               |
|--------------------------------|--------------------------------------|
| `{phase}/epoch_loss`           | train, val, test                     |
| `{phase}/loss_{component}`     | train, val, test (CombinedLoss only) |

**`{phase}/epoch_loss`** — Mean batch loss over the epoch, computed as `mean(batch_losses)`.

**`{phase}/loss_{component}`** — When the loss is a `CombinedLoss`, each weighted sub-term (e.g. `loss_SimCLR`, `loss_ResidualPenalty`) is logged separately as the mean of its per-batch value over the epoch. Component names come from the loss config.

---

### Core Cosine Similarity Metrics

These are the primary measures of model quality. All three compare embeddings using `F.cosine_similarity` and are accumulated element-wise across the epoch before averaging.

| Key                           | Phases            | Numerator (per sample)          |
|-------------------------------|-------------------|---------------------------------|
| `{phase}/mean_imgaug_cos`     | train, val, test  | `cos(pred_imgaug, feats_trans)` |
| `{phase}/mean_origtrans_cos`  | train, val, test  | `cos(feats_orig, feats_trans)`  |
| `{phase}/mean_id_cos`         | val, test         | `cos(pred_id, feats_orig)`      |

**`mean_imgaug_cos`** — The core quality metric. Measures how well the model's predicted augmented embedding matches the ground-truth augmented embedding (produced by applying image-space augmentation then re-encoding through the frozen foundation model). Higher is better; 1.0 is perfect prediction.

**`mean_origtrans_cos`** — Baseline cosine similarity between the original embedding and the transformed embedding, i.e. how similar two differently-prepared scans of the same tissue already are _without_ any correction. This is the starting point the model tries to improve upon. In the scanner-transfer setting it represents raw scanner similarity.

**`mean_id_cos`** — Cosine similarity of the identity pass: how close the predicted embedding under identity augmentation parameters is to the original. A well-trained model should score near 1.0 here.

---

### Relative Improvement

| Key                            | Phases    | Condition                       |
|--------------------------------|-----------|---------------------------------|
| `{phase}/relative_improvement` | val, test | `mean_origtrans_cos < 1 − 1e-6` |

**`relative_improvement`** — Normalizes the cosine gain against the maximum possible improvement:

```
(mean_imgaug_cos − mean_origtrans_cos) / (1.0 − mean_origtrans_cos)
```

A value of 0 means the model adds nothing; 1.0 means the model perfectly closes the gap between baseline and ideal. Values can be negative if the model makes alignment worse.

---

### Per-Group Cosine Breakdowns (val and test)

These metrics appear only when pair-mode data is active and scanner metadata is available (`self.id_to_scanner` is populated). Each sample is bucketed by the relationship between its source scanner and target scanner.

| Key pattern                   | Bucket description                            |
|-------------------------------|-----------------------------------------------|
| `{phase}/cos_same_scanner`    | Source and target scanner are identical       |
| `{phase}/cos_same_vendor`     | Different scanner models from the same vendor |
| `{phase}/cos_diff_vendor`     | Scanners from different vendors               |
| `{phase}/cos_{scanner_name}`  | Per-scanner mean (conditioned models only)    |

The mean cosine similarity is computed independently for each bucket. These metrics reveal whether the model handles within-vendor differences differently from cross-vendor ones.

---

### Bootstrap Confidence Intervals (test only)

| Key                            | Phase |
|--------------------------------|-------|
| `test/mean_imgaug_cos_ci_low`  | test  |
| `test/mean_imgaug_cos_ci_high` | test  |

**`mean_imgaug_cos_ci_low` / `ci_high`** — 95% percentile bootstrap confidence interval around `mean_imgaug_cos`, computed on the full set of per-sample cosine values collected during the test epoch. Parameters: 3,000 resamples, percentile method (avoids the memory overhead of BCa jackknife). These are logged as scalars alongside the point estimate.

---

## Source and Target Scanner

In the PLISM dataset every tissue sample is physically scanned on multiple scanners, producing one feature file per `(staining, scanner)` combination. All files share the same patch layout and row order, so patches are paired by row index with no tile-key matching required.

**Source scanner** is the scanner that produced the input embedding (`feat_a`) given to the model. **Target scanner** is the scanner whose embedding (`feat_b`) the model must predict. The model receives `feat_a` together with the two scanner identity tokens `(src_scanner_id, tgt_scanner_id)` and outputs a predicted `feat_b`.

The task is therefore: given a patch embedding extracted under scanner A's imaging conditions, predict what that same patch's embedding would look like under scanner B's imaging conditions. With `symmetric=True` in the pairing config both `(A→B)` and `(B→A)` are included as separate training examples.

The identity regularization pass uses `tgt_scanner = src_scanner`, enforcing that the model is a no-op when asked to translate a scanner to itself. This is what fills the diagonal of the scanner heatmaps.

---

## Scanner Heatmaps (Weights & Biases Images)

Logged to W&B as `wandb.Image` objects at the end of val and test epochs when pair data with scanner metadata is available. They are not scalar metrics but visual matrices analogous to confusion matrices: rows are source scanners, columns are target scanners, and cell values are mean cosine similarities.

### `{phase}/scanner_pair_heatmap`

**All phases (val + test).** Each off-diagonal cell `(i, j)` shows the mean predicted cosine similarity when translating from scanner `i` to scanner `j`. Diagonal cells show the identity-pass cosine similarity (same scanner, regularization objective). Colormap is one-sided (vmin = data min − 0.02, vmax = 1.0).

### `{phase}/scanner_origtrans_heatmap`

**Test phase only.** Each off-diagonal cell shows the raw baseline cosine similarity (`mean_origtrans_cos`) for the pair `(i, j)`, i.e. how similar the embeddings already are before the model acts. Diagonal is always 1.0. Colormap is one-sided (0 to 1).

### `{phase}/scanner_diff_heatmap`

**Test phase only.** Difference matrix: `pred_mat[i,j] − orig_mat[i,j]` for off-diagonal entries. Positive values (warm colors) mean the model improves scanner alignment; negative values (cool colors) mean it degrades it. Diagonal is masked. Colormap is diverging, symmetric around 0.

### `{phase}/scanner_rel_heatmap`

**Test phase only.** Relative improvement per scanner pair, same normalization as the scalar `relative_improvement` metric:

```
(pred_mat[i,j] − orig_mat[i,j]) / (1.0 − orig_mat[i,j])
```

This controls for pairs that are already very similar (high baseline), making the values comparable across easy and hard scanner combinations. Colormap is diverging, symmetric around 0.
