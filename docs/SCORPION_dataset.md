# SCORPION Features Structure

## Dataset layout

The raw SCORPION dataset lives at `/mnt/data/plismbench/SCORPION/SCORPION_dataset/` and is organised as:

```
SCORPION_dataset/
└── slide_{i}/
    └── sample_{j}/
        ├── AT2.jpg
        ├── DP200.jpg
        ├── GT450.jpg
        ├── P1000.jpg
        └── Philips.jpg
```

Each leaf image is the same tissue region captured by a different scanner.

## Extracted features

One `.npy` file per scanner under `/mnt/data/plismbench/features/scorpion/{extractor}/`:

```
/mnt/data/plismbench/features/scorpion/{extractor}/
├── AT2.npy
├── DP200.npy
├── GT450.npy
├── P1000.npy
└── Philips.npy
```

### Array format

Each file has shape `(N, 2 + D)` where:

| Columns | Content |
|---------|---------|
| `0` | slide number (int, from `slide_{i}`) |
| `1` | sample number (int, from `sample_{j}`) |
| `2 : 2+D` | feature vector (float32, dimension D depends on the model) |

Rows are sorted by slide number then sample number. This mirrors the PLISM feature convention where leading integer columns encode spatial or structural coordinates.

