# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HistAug** is a lightweight transformer-based generator for controllable feature-space augmentations in digital pathology (ICCV 2025). It augments patch embeddings from foundation models (CONCH, UNI, Virchow2, H-optimus-1) directly in latent space, avoiding expensive image-space transformations on whole-slide image (WSI) patches.

## Environment Setup

The project runs inside a Docker container (see `Dockerfile`). PyTorch, CUDA, and most system dependencies are pre-installed in the image. To set up the project packages:

```bash
# Inside the container — install the project in editable mode
pip install -e .

# With dev extras (pytest, ruff, black, isort, ipdb)
pip install -e ".[dev]"
```

The Dockerfile base image (`nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04`) pre-installs:
- `torch==2.10.0+cu129`, `torchvision==0.25.0+cu129`
- `lightning`, `timm`, `huggingface_hub`, `xformers`, `openslide-python`, `h5py`, and other common ML libraries

`torch` and `torchvision` are intentionally absent from `pyproject.toml` to avoid overwriting the CUDA-enabled versions from the image.

## Key Commands

```bash
# Training
cd src/histaug
python train.py --stage=train --config config/Histaug_conch.yaml

# Testing
python train.py --stage=test --config config/Histaug_conch.yaml

# Linting/formatting
black src/
ruff check src/
isort src/
```

## Architecture

The system is a **PyTorch Lightning** pipeline with a configuration-driven design using YAML + OmegaConf.

### Core Components

**`src/histaug/train.py`** — Main entry point. Orchestrates `build_datamodule()`, `build_model()`, and `build_trainer()`, then calls `trainer.fit()` or `trainer.test()`.

**`models/histaug_model.py`** — The core transformer model (`HistaugModel`). Takes a patch embedding + augmentation parameters and outputs a transformed embedding. Uses standard attention blocks with Q/K normalization. `sample_aug_params()` generates transformation parameters either patch-wise or bag-wise (WSI-consistent).

**`models/model_interface.py`** — PyTorch Lightning `LightningModule` wrapper (`ModelInterface`). Holds the frozen foundation model (feature extractor) and HistaugModel. Training step: extracts embeddings for original and image-augmented patches, runs HistaugModel, computes loss against ground-truth augmented embeddings.

**`models/foundation_models.py`** — Abstract `FoundationBackend` + concrete implementations for CONCH, UNI, Virchow2, H-optimus-1. Use `get_foundation_model(name, ckpt_path)` factory.

**`datasets/patch_dataset.py`** — `PatchDataset` loads WSI patches via H5 coordinate files (CLAM format). Returns `(original_patch, augmented_patch, aug_params)` tuples.

**`datasets/data_interface.py`** — `DataInterface` (`LightningDataModule`). Reads CSV split files from `dataset_csv/` and dynamically instantiates the dataset.

**`utils/transform_factory.py`** — `PatchAugmentation` applies image-space augmentations and records parameter values + shuffle order for conditioning the model. `TRANSFORM_REGISTRY` maps names to transform classes.

**`utils/loss_factory.py`** — `create_loss()` instantiates `CosineSimilarityLoss`, MSE, or `CombinedLoss` from config.

**`utils/optim_factory.py`** — `create_optimizer()` and `create_scheduler()` instantiate from config names.

### Data Flow

```
WSI H5 coordinates
  → PatchDataset: load patches, apply image augmentations, record aug params
  → Foundation model (frozen): extract embeddings for original + augmented patches
  → HistaugModel: predict augmented embedding from (original embedding + aug params)
  → Loss: cosine similarity between prediction and ground-truth augmented embedding
```

### Configuration

YAML configs in `src/histaug/config/` (one per foundation model). Top-level keys:
- `General` — seed, epochs, precision, devices
- `Data` — dataset paths, patch dirs, transforms, dataloader settings
- `Foundation_model` — model name and checkpoint path
- `Model` — HistaugModel architecture (depth, num_heads, chunk_size, etc.)
- `Optimizer` / `Scheduler` / `Loss` — dynamically instantiated from names

### Data Splits

Train/val/test slide IDs stored in `src/histaug/dataset_csv/` as CSV files. `check_data_leak.py` validates no overlap between splits.

## PLISM Staining Protocol Abbreviations

All PLISM slides are H&E stained. The slide-name abbreviations encode the hematoxylin
product and staining conditions.  The trailing **H** in an abbreviation (e.g. `GIVH`)
marks the *short-exposure / low-dehydration* variant of the same product; the non-H
sibling (e.g. `GIV`) uses a longer / overnight exposure with more dehydration steps.
`MY` (Mayer) has only one variant.

| Abbrev. | Solution Category | Hematoxylin product   | Hematoxylin time | Eosin time | Dehydrations |
|---------|-------------------|-----------------------|------------------|------------|--------------|
| GIVH    | GIV               | Gill IV (8647)        | 0.5 min          | 15 min     | 1            |
| GIV     | GIV               | Gill IV (8647)        | Overnight        | 15 min     | 4            |
| GMH     | GM                | GM (30081)            | 2 min            | 15 min     | 1            |
| GM      | GM                | New Type G (30161)    | 5 min            | 15 min     | 4            |
| GVH     | GV                | Gill V (20032)        | 5 min            | 15 min     | 1            |
| GV      | GV                | Gill V (20032)        | 60 min           | 15 min     | 4            |
| MY      | MY                | Mayer (30002)         | 3 min            | 3 min      | 1            |
| HRH     | HR                | Harris (20022)        | 2 min            | 15 min     | 1            |
| HR      | HR                | Harris (20022)        | Overnight        | 15 min     | 1            |
| KRH     | KR                | Carrazi (30131)       | 5 min            | 15 min     | 1            |
| KR      | KR                | Carrazi (30131)       | 60 min           | 15 min     | 4            |
| LMH     | LMH               | Lillie-Mayer (30072)  | 2 min            | 15 min     | 1            |
| LM      | LM                | Lillie-Mayer (30072)  | 2 min            | 15 min     | 5            |

**In code:** strip a trailing `H` from the tissue abbreviation to get the Solution
Category (e.g. `GIVH` → `GIV`, `MY` stays `MY`).  Use Solution Category for coloring
in visualizations — it groups slides by hematoxylin product, the dominant staining
variable.

## PLISM Scanners
| Vendor | Model | Abbreviation |
|--------|-------|--------------|
| Hamamatsu | NanoZoomer-S360 |  C13220-01 scanner S360 |
| Hamamatsu | NanoZoomer-S210 |  C13239-01 scanner S210 |
| Hamamatsu | NanoZoomer-SQ | C13140-D03 SQ |
| Hamamatsu | NanoZoomer-S60 | C13210-01 S60 |
| Leica | Aperio AT2 | AT2 |
| Leica | Aperio GT450 | GT450 |
| Phillips | Ultrafast Scanner | P |

## Notes

- Foundation models are loaded with `trust_remote_code=True` from HuggingFace Hub
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for HPC clusters without internet
- Code style: Black formatting
- Pretrained HistAug checkpoints are available on HuggingFace Hub
