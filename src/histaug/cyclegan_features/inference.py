"""Translate a tensor of embeddings using a trained generator.

For multi-target models, supply --tgt_scanner.  No source-scanner identity needed.

Usage:
    # Single-pair model (unconditioned):
    python -m histaug.cyclegan_features.inference \\
        --checkpoint runs/cyclegan/ckpt_best.pt \\
        --norm_stats  runs/cyclegan/norm_stats.pt \\
        --input embeddings.pt --direction fwd --output translated.pt

    # Multi-target model (conditioned on target scanner):
    python -m histaug.cyclegan_features.inference \\
        --checkpoint runs/cyclegan/ckpt_best.pt \\
        --norm_stats  runs/cyclegan/norm_stats.pt \\
        --scanner_vocab runs/cyclegan/scanner_vocab.json \\
        --tgt_scanner GT450 \\
        --input embeddings_any_scanner.pt \\
        --output translated_to_GT450.pt
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from .models import FeatureGenerator
from .utils import NormStats


def load_generator(
    checkpoint_path: Path,
    direction: str,
    dim: int,
    cond_dim: int,
    device: torch.device,
) -> FeatureGenerator:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # In multi-target mode g_ab == g_ba (same state_dict); either key works.
    key = "g_ab" if direction == "fwd" else "g_ba"
    g = FeatureGenerator(dim, dim, cond_dim).to(device)
    g.load_state_dict(ckpt[key])
    g.eval()
    return g


@torch.no_grad()
def translate(
    embeddings: torch.Tensor,
    checkpoint_path: str,
    norm_stats_path: str,
    direction: str = "fwd",
    scanner_vocab: Optional[List[str]] = None,
    tgt_scanner: Optional[str] = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Translate embeddings to a target scanner domain.

    Args:
        embeddings:      Float tensor [N, D].
        checkpoint_path: Path to .pt checkpoint saved by train.py.
        norm_stats_path: Path to norm_stats.pt.
        direction:       "fwd" or "bwd" (only relevant for single-pair models).
        scanner_vocab:   List of scanner names (multi-target models).
        tgt_scanner:     Target scanner name for conditioning (multi-target models).
        batch_size:      Processing batch size.

    Returns:
        Translated embeddings [N, D] in the original (unnormalized) scale.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = embeddings.shape[1]
    cond_dim = len(scanner_vocab) if scanner_vocab else 0

    norm_stats = NormStats.load(Path(norm_stats_path))
    g = load_generator(Path(checkpoint_path), direction, dim, cond_dim, device)

    normalize   = norm_stats.normalize_a   if direction == "fwd" else norm_stats.normalize_b
    denormalize = norm_stats.denormalize_b if direction == "fwd" else norm_stats.denormalize_a

    x = normalize(embeddings.float())

    one_hot: Optional[torch.Tensor] = None
    if cond_dim > 0:
        if tgt_scanner is None:
            raise ValueError("tgt_scanner is required for conditioned (multi-target) models")
        if tgt_scanner not in scanner_vocab:
            raise ValueError(f"{tgt_scanner!r} not in scanner_vocab {scanner_vocab}")
        idx = scanner_vocab.index(tgt_scanner)
        one_hot = F.one_hot(torch.tensor(idx), num_classes=cond_dim).float().to(device)

    results = []
    for chunk in x.split(batch_size):
        chunk = chunk.to(device)
        cond = one_hot.unsqueeze(0).expand(len(chunk), -1) if one_hot is not None else None
        results.append(denormalize(g(chunk, cond).cpu()))

    return torch.cat(results, dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Translate embeddings with a trained CycleGAN generator")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--norm_stats",    required=True)
    p.add_argument("--input",         required=True, help="Input .pt tensor [N, D]")
    p.add_argument("--output",        required=True)
    p.add_argument("--direction",     choices=["fwd", "bwd"], default="fwd",
                   help="Only relevant for single-pair models")
    p.add_argument("--scanner_vocab", default=None,
                   help="Path to scanner_vocab.json (multi-target models)")
    p.add_argument("--tgt_scanner",   default=None,
                   help="Target scanner name for conditioning (multi-target models)")
    p.add_argument("--batch_size",    type=int, default=1024)
    args = p.parse_args()

    embeddings = torch.load(args.input, weights_only=True)
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("--input must point to a plain torch.Tensor")

    vocab = None
    if args.scanner_vocab:
        with open(args.scanner_vocab) as f:
            vocab = json.load(f)

    out = translate(
        embeddings,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        direction=args.direction,
        scanner_vocab=vocab,
        tgt_scanner=args.tgt_scanner,
        batch_size=args.batch_size,
    )
    torch.save(out, args.output)
    print(f"Saved {out.shape} translated embeddings to {args.output}")


if __name__ == "__main__":
    main()
