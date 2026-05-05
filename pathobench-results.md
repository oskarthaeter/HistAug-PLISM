# Patho-bench results

For Virchow2 on 11 CPTAC mutation tasks. Metric: Macro OvR AUC


## Tasks

11 binary mutation-status prediction tasks (mutated vs. wild-type) from the Clinical Proteomic Tumor Analysis Consortium (CPTAC), predicting somatic mutations directly from H&E whole-slide images. Tasks span 6 cancer cohorts:

| Cohort      | Cancer type                               | Mutations         |
|-------------|-------------------------------------------|-------------------|
| CPTAC CCRCC | Clear-cell renal cell carcinoma           | BAP1, PBRM1       |
| CPTAC LUAD  | Lung adenocarcinoma                       | EGFR, STK11, TP53 |
| CPTAC LSCC  | Lung squamous cell carcinoma              | KEAP1, ARID1A     |
| CPTAC HNSC  | Head and neck squamous cell carcinoma     | CASP8             |
| CPTAC GBM   | Glioblastoma                              | EGFR, TP53        |
| CPTAC PDA   | Pancreatic ductal adenocarcinoma          | SMAD4             |

Canonical train/test splits are loaded from the HuggingFace Patho-Bench dataset via `SplitFactory.from_hf`. Results are reported as mean ± SE across cross-validation folds.


## Scanner-transfer model checkpoint

Run: `src/histaug/logs/virchow2-transfer-bottleneck-tgt_spectral_norm_all`

| Parameter | Value |
|---|---|
| Architecture | `scanner_transfer_layer_model` (FiLM-conditioned residual MLP) |
| Conditioning | `tgt_scanner` only |
| Hidden dims | [128, 64] |
| Spectral norm | All layers (`spectral_norm_all=True`) |
| Residual scale alpha | 0.1 (fixed) |
| Epochs | 10 (cosine schedule, 2-epoch warmup) |
| Loss | 0.7 × CosineSimilarity + 0.3 × SmoothL1 + 0.01 × residual penalty |
| Feature dim | 2560 (Virchow2) |
| Scanner vocab | AT2, GT450, P, S210, S360, S60, SQ |
| Staining vocab | 13 PLISM staining protocols (same-staining pairs only) |
| Holdout staining | GVH (excluded from training) |


## PLISM intrinsic evaluation (cosine similarity)

Metrics: `cos(pred, tgt)` is the cosine similarity between the predicted target-scanner embedding and the ground-truth target-scanner embedding. `cos(src, tgt)` is the oracle baseline (raw cross-scanner similarity without any model). `cos(pred, src)` measures how much the prediction drifts from the source embedding.

| Split | N pairs | cos(pred, tgt) | cos(src, tgt) oracle | cos(pred, src) | Relative improvement |
|---|---:|---:|---:|---:|---|
| Test (held-out organs, seen stainings) | 1,638,504 | **0.9123** | 0.8756 | 0.9965 | +29.6% of gap closed |
| Test-holdout staining (GVH, unseen) | 683,676 | **0.9347** | 0.9125 | 0.9961 | +25.3% of gap closed |
| SCORPION (external, OOD scanners) | 9,600 | 0.9487 | **0.9575** | 0.9868 | below oracle |

Relative improvement = (cos(pred,tgt) - cos(src,tgt)) / (1 - cos(src,tgt)); measures how much of the remaining gap to 1.0 the model closes beyond the oracle baseline.

Per target-scanner breakdown (test split):

| Target scanner | cos(pred, tgt) | cos(src, tgt) oracle |
|---|---:|---:|
| AT2   | 0.9197 | 0.8820 |
| GT450 | 0.9111 | 0.8684 |
| P     | 0.8959 | 0.8449 |
| S210  | 0.9272 | 0.9000 |
| S360  | 0.9262 | 0.8915 |
| S60   | 0.8990 | 0.8606 |
| SQ    | 0.9074 | 0.8816 |

Notes:
- On PLISM test and holdout-staining splits, the model consistently outperforms the oracle (raw cross-scanner similarity), confirming it learns a meaningful alignment.
- On SCORPION, the model falls slightly below the oracle. SCORPION scanners (DP200, P1000) are mapped to PLISM surrogates (S210, S360) and were never seen during training; the model applies a small residual correction that slightly overshoots, suggesting the surrogate mapping is imperfect.
- `cos(pred, src)` is very close to 1.0 in all splits, confirming the residual correction is small and the model does not distort the input.


## Training and evaluation procedure (Patho-bench)

All variants finetune an ABMIL slide encoder on top of frozen Virchow2 patch embeddings (dim 2560). Shared hyperparameters: AdamW, LR 3 × 10⁻⁴, weight decay 10⁻⁵, cosine schedule, 20 epochs, balanced cross-entropy loss, bag size 2048, `combine_slides_per_patient=True`. Feature-space augmentations are applied with probability 0.5 per slide **at both train and test time**.

- **Reproduced (no aug.)** — baseline finetuning with no augmentation.
- **Scanner-transfer aug.** — patch features are stochastically mapped to one of seven target scanner domains (AT2, GT450, P, S210, S360, S60, SQ) using the checkpoint above.
- **HistAug** — patch features are perturbed in feature space by a pretrained HistAug-Virchow2 model (`sofieneb/histaug-virchow2` on HuggingFace), trained by the original authors. Perturbations include brightness/contrast/saturation/hue jitter, HED stain variation, power-law gamma, Gaussian blur, and spatial transforms (flips, rotation, crop), all applied WSI-wise.
- **Scanner-transfer + HistAug / HistAug + Scanner-transfer** — both augmentors chained in the stated order.


## Results

| Dataset     | Organ     | Mutation | Atlas2 paper | Reproduced (no aug.) | Scanner-transfer aug. | HistAug     | Scanner-transfer + HistAug | HistAug + Scanner-transfer |
|-------------|-----------|----------|--------------|----------------------|-----------------------|-------------|----------------------------|----------------------------|
| CPTAC CCRCC | Kidney    | BAP1     | 67.4         | 62.9 ± 01.8          | 66.5 ± 02.1           | 67.3 ± 02.2 | 65.1 ± 02.6                | 62.0 ± 02.7                |
| CPTAC CCRCC | Kidney    | PBRM1    | 49.4         | 50.5 ± 01.8          | 52.2 ± 01.7           | 51.0 ± 01.6 | 52.7 ± 01.8                | 51.6 ± 01.6                |
| CPTAC LUAD  | Lung      | EGFR     | 81.7         | 81.0 ± 01.4          | 80.3 ± 01.5           | 77.8 ± 01.7 | 66.5 ± 01.8                | 67.2 ± 02.0                |
| CPTAC LUAD  | Lung      | STK11    | 82.4         | 82.3 ± 01.6          | 83.6 ± 01.6           | 79.5 ± 01.7 | 70.6 ± 02.1                | 71.3 ± 02.1                |
| CPTAC LUAD  | Lung      | TP53     | 77.3         | 77.2 ± 01.5          | 77.5 ± 01.5           | 75.9 ± 01.8 | 66.1 ± 01.8                | 67.3 ± 01.7                |
| CPTAC LSCC  | Lung      | KEAP1    | 63.1         | 64.5 ± 02.1          | 65.2 ± 02.2           | 59.6 ± 02.0 | 54.3 ± 02.2                | 51.7 ± 02.4                |
| CPTAC LSCC  | Lung      | ARID1A   | 41.9         | 38.5 ± 02.4          | 43.3 ± 02.2           | 46.7 ± 02.0 | 44.3 ± 02.1                | 45.6 ± 02.1                |
| CPTAC HNSC  | Head&Neck | CASP8    | 56.6         | 58.4 ± 02.6          | 58.4 ± 02.8           | 57.2 ± 03.0 | 61.2 ± 03.1                | 61.1 ± 02.9                |
| CPTAC GBM   | Brain     | EGFR     | 62.1         | 64.0 ± 01.7          | 63.8 ± 01.8           | 58.6 ± 01.9 | 56.8 ± 01.9                | 56.4 ± 02.1                |
| CPTAC GBM   | Brain     | TP53     | 74.8         | 70.2 ± 02.2          | 73.8 ± 01.9           | 65.1 ± 02.0 | 57.5 ± 02.3                | 57.7 ± 02.1                |
| CPTAC PDA   | Pancreas  | SMAD4    | 44.6         | 50.3 ± 02.0          | 51.4 ± 02.3           | 47.3 ± 02.1 | 46.9 ± 02.0                | 46.4 ± 02.0                |
