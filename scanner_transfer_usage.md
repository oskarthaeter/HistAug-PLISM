# Using Trained Scanner-Transfer Models for Patch Feature Augmentation

This document explains how to load a trained PLISM scanner-transfer model from a checkpoint and apply it to patch embeddings in your own codebase.

## What these models do

The scanner-transfer models map patch embeddings from one scanner/staining domain to another directly in feature space. Given pre-extracted embeddings from scanner A with staining X, the model predicts what those embeddings would look like if they had been scanned by scanner B with staining Y.

All three architectures share the same interface:

```
output = model(x, src_scanner, tgt_scanner, src_staining, tgt_staining)
```

where all conditioning arguments are integer tensors (vocabulary indices) and the output has the same shape as `x`. The transformation is residual: `output = x + alpha * delta(x, condition)`.

### Architecture variants

| Class | Description |
|---|---|
| `ScannerTransferLinearModel` | Concatenates one-hot condition to features, runs linear/MLP projection with residual connection |
| `ScannerTransferLayerModel` | Residual MLP with FiLM conditioning at every hidden layer; learnable residual scale `alpha` |
| `ScannerTransferBottleneckModel` | Low-rank bottleneck encoder, single FiLM layer, zero-init decoder; learnable `alpha` |

The architecture used for a given checkpoint is stored inside the checkpoint itself.

## Checkpoint format

Checkpoints are saved by PyTorch Lightning. The relevant keys are:

```
ckpt["hyper_parameters"]["model"]
    .name              # e.g. "scanner_transfer_layer_model"
    .input_dim         # feature dimensionality (e.g. 2560 for Virchow2)
    .scanner_vocab_size
    .staining_vocab_size
    .scanner_names     # list[str], index i -> scanner name
    .staining_names    # list[str], index i -> staining name
    ... (architecture hyperparameters)

ckpt["state_dict"]
    # keys are prefixed with "model.", e.g. "model.layers.0.weight"
```

## Minimal usage

Copy [scanner_transfer_inference.py](scanner_transfer_inference.py) into your project (it requires only `torch`, no Lightning or PLISM dependencies).

### Loading

```python
from scanner_transfer_inference import load_scanner_transfer_model

augmentor = load_scanner_transfer_model(
    ckpt_path="path/to/checkpoint.ckpt",
    device="cuda",   # or "cpu"
)
```

### Augmenting features

```python
import torch

# features: (N, D) tensor of patch embeddings from the source scanner/staining
features = torch.load("slide_features.pt")  # shape (N, D), e.g. (4096, 2560)

augmented = augmentor.augment(
    features,
    src_scanner="S360",
    tgt_scanner="GT450",
    src_staining="GIV",
    tgt_staining="GIV",
)
# augmented: same shape as features, in the target domain
```

Scanner and staining names must match the vocabulary stored in the checkpoint. The augmentor raises a `KeyError` with the available names if a lookup fails.

### Checking the vocabulary

```python
print(augmentor.scanner_names)   # e.g. ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
print(augmentor.staining_names)  # e.g. ["GIV", "GIVH", "GM", "GMH", ...]
```

### Identity pass (sanity check)

Passing the same scanner and staining for source and target should return features close to the input (all models are identity-initialized):

```python
identity = augmentor.augment(features, "S360", "S360", "GIV", "GIV")
cos = torch.nn.functional.cosine_similarity(features, identity).mean()
print(f"Identity cosine similarity: {cos:.4f}")  # should be close to 1.0
```

## Using in a MIL pipeline

The typical usage is to augment patch embeddings before aggregation, effectively simulating a different scanner/staining during training or test-time augmentation:

```python
from scanner_transfer_inference import load_scanner_transfer_model
import torch

augmentor = load_scanner_transfer_model("virchow2_transfer.ckpt", device="cuda")

def augment_slide(patch_features, src_scanner, tgt_scanner, src_staining, tgt_staining):
    """Drop-in replacement for patch features in a MIL forward pass."""
    with torch.inference_mode():
        return augmentor.augment(
            patch_features,
            src_scanner=src_scanner,
            tgt_scanner=tgt_scanner,
            src_staining=src_staining,
            tgt_staining=tgt_staining,
        )

# Example: normalise all slides to GT450 / GIV during test
slide_feats = torch.load("slide_AT2_GIV.pt").cuda()
normalised  = augment_slide(slide_feats, "AT2", "GT450", "GIV", "GIV")
```

## Conditioning variants

Some checkpoints are trained with a subset of the four conditioning signals. The `conditioning` field in `model_cfg` lists which signals are active (e.g. `[tgt_scanner]` for target-only models). The `augment()` method handles this automatically:

- `src_staining` and `tgt_staining` are **optional** (default `None`)
- If the model does not condition on staining, omit them and they are silently ignored
- If the model does condition on staining and you omit them, a `ValueError` is raised

All four arguments (`src_scanner`, `tgt_scanner`, `src_staining`, `tgt_staining`) are optional. Each is required only if the model conditions on that signal; otherwise it is silently ignored and a zero index is passed to satisfy the forward signature.

```python
# tgt-scanner-only checkpoint
augmented = augmentor.augment(features, tgt_scanner="GT450")

# src+tgt scanner, no staining
augmented = augmentor.augment(features, src_scanner="S360", tgt_scanner="GT450")

# full conditioning
augmented = augmentor.augment(
    features,
    src_scanner="S360", tgt_scanner="GT450",
    src_staining="GIV", tgt_staining="GIV",
)
```

A `ValueError` is raised if a required argument (one the model actually conditions on) is omitted.

Check which signals the loaded model uses:

```python
print(augmentor.model.conditioning)        # e.g. ["tgt_scanner"]
print(augmentor.conditions_on_src_scanner) # True / False
print(augmentor.conditions_on_tgt_scanner) # True / False
print(augmentor.conditions_on_staining)    # True / False
```

## PLISM scanner and staining names

Scanner abbreviations used in PLISM checkpoints:

| Abbreviation | Scanner |
|---|---|
| `AT2` | Leica Aperio AT2 |
| `GT450` | Leica Aperio GT450 |
| `P` | Philips Ultrafast Scanner |
| `S210` | Hamamatsu NanoZoomer-S210 |
| `S360` | Hamamatsu NanoZoomer-S360 |
| `S60` | Hamamatsu NanoZoomer-S60 |
| `SQ` | Hamamatsu NanoZoomer-SQ |

Staining abbreviations (all H&E, encoding hematoxylin product and exposure):

`GIVH`, `GIV`, `GMH`, `GM`, `GVH`, `GV`, `MY`, `HRH`, `HR`, `KRH`, `KR`, `LMH`, `LM`

See `CLAUDE.md` for the full protocol table.
