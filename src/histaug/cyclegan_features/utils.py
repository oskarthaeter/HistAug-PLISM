import json
import random
from pathlib import Path
from typing import Dict, Tuple

import torch


class ReplayBuffer:
    """Stores a history of generated samples and returns a random mix of history
    and current batch.  Keeps discriminator training more stable by breaking
    temporal correlation between generator updates.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        result = []
        for element in batch:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element.clone())
                result.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element.clone()
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


class NormStats:
    """Per-domain mean/std for z-score normalization."""

    def __init__(self, mean_a: torch.Tensor, std_a: torch.Tensor,
                 mean_b: torch.Tensor, std_b: torch.Tensor):
        self.mean_a = mean_a
        self.std_a = std_a
        self.mean_b = mean_b
        self.std_b = std_b

    @classmethod
    def from_data(cls, z_a: torch.Tensor, z_b: torch.Tensor) -> "NormStats":
        return cls(
            mean_a=z_a.mean(0),
            std_a=z_a.std(0).clamp(min=1e-6),
            mean_b=z_b.mean(0),
            std_b=z_b.std(0).clamp(min=1e-6),
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "NormStats":
        """Load from a metadata .pt dict produced by prepare_data in mmap mode."""
        return cls(
            mean_a=meta["norm_mean_a"],
            std_a=meta["norm_std_a"],
            mean_b=meta["norm_mean_b"],
            std_b=meta["norm_std_b"],
        )

    def normalize_a(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_a.to(x)) / self.std_a.to(x)

    def normalize_b(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_b.to(x)) / self.std_b.to(x)

    def denormalize_a(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_a.to(x) + self.mean_a.to(x)

    def denormalize_b(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_b.to(x) + self.mean_b.to(x)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "mean_a": self.mean_a,
                "std_a": self.std_a,
                "mean_b": self.mean_b,
                "std_b": self.std_b,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = torch.load(path, weights_only=True)
        return cls(d["mean_a"], d["std_a"], d["mean_b"], d["std_b"])


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    g_ab: torch.nn.Module,
    g_ba: torch.nn.Module,
    d_a: torch.nn.Module,
    d_b: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    tag: str = "",
) -> None:
    name = f"ckpt_epoch{epoch:04d}{('_' + tag) if tag else ''}.pt"
    torch.save(
        {
            "epoch": epoch,
            "g_ab": g_ab.state_dict(),
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "d_b": d_b.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        output_dir / name,
    )
