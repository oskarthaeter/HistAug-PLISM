# Using the Pretrained HistAug Virchow2 Model

HistAug augments patch embeddings directly in feature space. Given pre-extracted Virchow2 embeddings, it synthesises what those embeddings would look like under a random stain/imaging perturbation, without loading any images.

The pretrained model lives at [`sofieneb/histaug-virchow2`](https://huggingface.co/sofieneb/histaug-virchow2) on HuggingFace.

## Installation

```bash
pip install transformers timm
# torch must already be installed
```

## How it works

```
patch_embeddings (N, 2560)
        │
        ▼
sample_aug_params()   ← random augmentation parameters (brightness, HED, flip, …)
        │
        ▼
HistaugModel.forward(patch_embeddings, aug_params)
        │
        ▼
augmented_embeddings (N, 2560)   ← same shape, perturbed in latent space
```

The model is a cross-attention transformer. The patch embedding sequence is the query; the augmentation parameter tokens are the keys/values. The output is a new embedding of the same dimensionality.

## Quickstart

```python
import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once; cached in ~/.cache/huggingface after the first download
model = AutoModel.from_pretrained(
    "sofieneb/histaug-virchow2", trust_remote_code=True
).to(device).eval()

# Your pre-extracted Virchow2 features for one slide
patch_embeddings = torch.load("slide_features.pt").to(device)  # (N, 2560)

# Sample a random augmentation — same transform applied to all patches (MIL-safe)
aug_params = model.sample_aug_params(
    batch_size=len(patch_embeddings),
    device=device,
    mode="wsi_wise",
)

with torch.inference_mode():
    augmented = model(patch_embeddings, aug_params)  # (N, 2560)
```

## Augmentation modes

| Mode | Behaviour | Use case |
|---|---|---|
| `"wsi_wise"` | One shared transformation sampled for the whole slide. All patches shift together. | MIL training: preserves slide-level distribution, consistent with how image augmentation works on tiles. |
| `"instance_wise"` | Independent random transformation per patch. | Patch-level contrastive pre-training; when slide-level coherence is not required. |

For MIL bag classification, prefer `"wsi_wise"`.

## Wrapper for large slides

[histaug_virchow2_augmentor.py](histaug_virchow2_augmentor.py) provides a thin wrapper that handles chunked processing so GPU memory is bounded for very large slides:

```python
from histaug_virchow2_augmentor import load_histaug_virchow2, augment_slide_features

model = load_histaug_virchow2(device="cuda")

features = torch.load("slide_features.pt")        # (N, 2560), on CPU or GPU
augmented = augment_slide_features(model, features, mode="wsi_wise")
```

`augment_slide_features` samples aug_params for the full slide once, then processes in chunks of `batch_size` (default 4096) to cap peak memory.

## MIL training integration

The typical pattern is to randomly apply HistAug augmentation at the bag level during training:

```python
from histaug_virchow2_augmentor import load_histaug_virchow2, augment_slide_features

histaug = load_histaug_virchow2(device="cuda")
for p in histaug.parameters():
    p.requires_grad_(False)   # frozen throughout MIL training

# Inside your training loop:
for bag_features, label in dataloader:        # bag_features: (N, 2560)
    if torch.rand(1).item() < 0.5:            # augment with 50% probability
        bag_features = augment_slide_features(histaug, bag_features, mode="wsi_wise")

    logits = mil_model(bag_features)
    loss = criterion(logits, label)
    ...
```

## What the augmentation parameters cover

`sample_aug_params` samples from all transforms the model was trained on:

| Transform | Type | Config |
|---|---|---|
| `rotation` | discrete (0/90/180/270°) | applied with probability 0.75 |
| `h_flip`, `v_flip` | discrete | applied with probability 0.75 |
| `crop` | discrete (5 crop positions) | applied with probability 0.75 |
| `gaussian_blur`, `erosion`, `dilation` | discrete | applied with probability 0.75 |
| `brightness`, `contrast`, `saturation`, `hue`, `gamma` | continuous | uniform in [−0.5, 0.5] |
| `hed` | continuous (stain vector shift) | uniform in [−0.5, 0.5] |

The return value of `sample_aug_params` is an `OrderedDict` mapping each transform name to a `(value_tensor, position_tensor)` tuple. You do not need to interpret this directly — just pass it to `model()`.

## Using a local checkpoint instead of HuggingFace

If you are on an offline cluster, download the model first:

```bash
# On a machine with internet access
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sofieneb/histaug-virchow2', trust_remote_code=True)"
# Then copy ~/.cache/huggingface to the cluster, or:
HF_HUB_OFFLINE=1 python your_script.py
```

Or point directly to a local directory:

```python
model = AutoModel.from_pretrained(
    "/path/to/local/histaug-virchow2", trust_remote_code=True
).to(device).eval()
```

## Expected output quality

Mean cosine similarity between HistAug-generated Virchow2 embeddings and the corresponding image-augmented embeddings (evaluated on BLCA, BRCA, LUSC across 10×/20×/40× magnifications):

| Metric | Value |
|---|---|
| Mean cosine similarity | 90.5% |
| 95% bootstrap CI | [90.3%, 90.6%] |

This measures how well the latent augmentation approximates the actual image-augmented embedding. Higher is better; perfect would be 100%.
