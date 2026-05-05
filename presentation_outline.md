# HistAug-PLISM: Scanner Transfer for Digital Pathology
### Internal Work Meeting

---

## 1. Motivation

**Problem:** Clinical AI models trained on slides from one scanner fail to generalise to others. Scanner and staining variability introduces domain shift in the feature space even when the underlying tissue content is identical.

**Why feature-space, not image-space?**
- WSIs contain hundreds of thousands of patches; reprocessing them at image level is prohibitively expensive
- Foundation models already compress patches into rich, semantically stable embeddings
- A lightweight model operating on embeddings can transfer scanner style at a fraction of the cost

**Goal:** Learn a mapping that, given patch embeddings from scanner A, predicts what those embeddings would look like from scanner B, enabling scanner-agnostic downstream classifiers.

---

## 2. Background: HistAug (ICCV 2025)

- Lightweight **cross-attention transformer** conditioned on explicit augmentation parameters (hue, erosion, HED transform, ...)
- Operates directly on **patch embeddings** from frozen foundation models (CONCH, UNI, Virchow2, H-optimus-1)
- Trained to predict ground-truth augmented embeddings; at inference generates WSI-consistent augmented embeddings for MIL training
- Key properties: fast, low-memory (200k patches in one forward pass on a V100), controllable, WSI-consistent (bag-wise or patch-wise conditioning)

---

## 3. Adaptation to Scanner Transfer

**Core idea:** replace augmentation-parameter conditioning with scanner / staining identity conditioning.

### 3.1 Architecture

The **ScannerTransferModel** mirrors the HistAug architecture:
- Input embedding is chunked into tokens and processed through cross-attention blocks
- Conditioning tokens encode `(src_scanner, tgt_scanner, src_staining, tgt_staining)` as learned embeddings
- Residual design: `output = input + α · Δ(input, condition)` — the model learns only the correction, not the full target
- Three architecture variants explored: linear/MLP, FiLM-conditioned MLP, low-rank bottleneck

### 3.2 Training Setup

| Aspect | Detail |
|---|---|
| Features | Pre-extracted Virchow2 (2560-d), H-optimus-1 mini, Phikon |
| Conditioning | Target-scanner only (`tgt_scanner`) in primary config; full `(src, tgt)` pair in ablations |
| Training objective | Cosine similarity + Smooth L1 + MSE + residual magnitude penalty |
| Data pairing | Same-staining cross-scanner pairs; symmetric; 0.5× tiles per pair per epoch |

---

## 4. Evaluation Data and Procedure

### 4.1 PLISM Dataset

- Multi-scanner, multi-staining H&E dataset: **7 scanners** × **13 staining protocols** (see abbreviation table)
- **Organ-based train/test split:** train on 80% of organs, test on held-out organs (organ types never seen during training)
- **Held-out staining:** `GVH` removed from train/val entirely; used only at test time to assess generalisation to unseen staining conditions

### 4.2 SCORPION Dataset

- External independent dataset with scanner and staining variability
- Scanner name mapping to PLISM vocabulary (e.g. DP200 → S210, P1000 → S360)
- Used to assess out-of-distribution generalisation without any fine-tuning

### 4.3 Downstream Evaluation: Patho-bench

- **11 CPTAC mutation prediction tasks** across 5 organs (kidney, lung, head & neck, brain, pancreas)
- Foundation model: Virchow2
- Metric: Macro One-vs-Rest AUC (± std over 5-fold CV)
- Comparison conditions:
  - Atlas2 paper baseline
  - Reproduced (no augmentation)
  - Scanner-transfer augmentation (this work)
  - HistAug augmentation only
  - Combined (scanner-transfer + HistAug, both orders)

### 4.4 Heatmap Visualisation

- Attention heatmaps from H-optimus-1 mini and Virchow2 before and after scanner transfer
- Phikon heatmaps as backup / additional evidence
- Objective: confirm that pathologically relevant tissue regions remain salient after transfer

---

## 5. Results

### 5.1 PLISM (Left-out Organs and Held-out Staining)

- Cosine similarity between transferred and ground-truth target-scanner embeddings
- Breakdown by: seen vs. unseen organs, seen vs. held-out staining (GVH), scanner pair

### 5.2 SCORPION

- Cosine similarity under scanner name surrogate mapping
- Qualitative embedding space visualisation (UMAP / t-SNE)

### 5.3 Patho-bench (Virchow2, CPTAC Mutations)

| Dataset | Organ | Mutation | No aug. | Scanner-transfer | HistAug | ST + HistAug |
|---|---|---|---|---|---|---|
| CPTAC CCRCC | Kidney | BAP1 | 62.9 | **66.5** | 67.3 | 65.1 |
| CPTAC LUAD | Lung | STK11 | 82.3 | **83.6** | 79.5 | 70.6 |
| CPTAC LSCC | Lung | ARID1A | 38.5 | 43.3 | **46.7** | 44.3 |
| CPTAC HNSC | H&N | CASP8 | 58.4 | 58.4 | 57.2 | **61.2** |
| ... | | | | | | |

Key observations:
- Scanner-transfer augmentation is consistently competitive with or better than no augmentation
- Combined ST + HistAug does not always stack additively; task-dependent interaction
- Scanner-transfer alone gives the most reliable gains across tasks

---

## 6. Next Steps

### 6.1 Staining Transfer

- Same architectural approach, but the conditioning now targets **staining protocol** rather than scanner
- Training data: PLISM pairs with cross-staining capability (`allow_cross_staining: true`)
- Open question: should staining transfer be trained jointly with scanner transfer, or as a separate model?
- Potential use case: normalise slides to a canonical staining before downstream classification

### 6.2 Broader Internal Evaluation

**Downstream clinical tasks:**
- **RlapsRisk** (relapse risk prediction): apply scanner-transfer augmentation at training and/or test time; measure AUC stability across acquisition sites
- **BRCAura**: similar protocol; assess whether scanner transfer closes the cross-site performance gap

**Test-time augmentation (TTA) in feature space:**
- At inference, augment each slide's embeddings to N target-scanner domains, average predictions
- Potential robustness gain without retraining

**Training augmentation strategy:**
- Ablate: augment only at training, only at test, or both
- Augment before or after MIL aggregation?

**Variability / stability analysis:**
- Given the same tissue block scanned on all PLISM scanners, measure prediction variance across scanners and staining conditions before and after transfer
- Quantify how much of the scanner-induced variability the model removes

**RNOH Dataset:**
- Overview: cohort composition, scanner and staining distribution, available labels
- Discuss suitability as an evaluation target and data access constraints

### 6.3 Public Benchmarking and Publication

**Additional benchmarks to run:**
- Patho-bench with features re-extracted using the **tilingtool pipeline** (fairer comparison to published baselines)
- Eva benchmark (lower priority)

**Open questions for the group:**
- On which datasets should results be published? (PLISM is public; SCORPION, RNOH, RlapsRisk, BRCAura are internal or restricted)
- Target venue: journal extension of ICCV work, or independent clinical AI paper?
- Authorship and collaboration structure given multi-site data requirements

---

## Appendix

### PLISM Staining Abbreviations

| Abbrev. | Product | H time | Dehydrations |
|---|---|---|---|
| GIVH / GIV | Gill IV | 0.5 min / overnight | 1 / 4 |
| GMH / GM | GM / New Type G | 2 min / 5 min | 1 / 4 |
| GVH / GV | Gill V | 5 min / 60 min | 1 / 4 |
| MY | Mayer | 3 min | 1 |
| HRH / HR | Harris | 2 min / overnight | 1 / 1 |
| KRH / KR | Carrazi | 5 min / 60 min | 1 / 4 |
| LMH / LM | Lillie-Mayer | 2 min / 2 min | 1 / 5 |

### PLISM Scanners

| Vendor | Model | Abbreviation |
|---|---|---|
| Hamamatsu | NanoZoomer-S360 | S360 |
| Hamamatsu | NanoZoomer-S210 | S210 |
| Hamamatsu | NanoZoomer-SQ | SQ |
| Hamamatsu | NanoZoomer-S60 | S60 |
| Leica | Aperio AT2 | AT2 |
| Leica | Aperio GT450 | GT450 |
| Philips | Ultrafast Scanner | P |
