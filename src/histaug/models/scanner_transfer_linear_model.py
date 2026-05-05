from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

VALID_CONDITIONING = {"src_scanner", "tgt_scanner", "src_staining", "tgt_staining"}
ALL_CONDITIONING = ["src_scanner", "tgt_scanner", "src_staining", "tgt_staining"]


class ScannerTransferLinearModel(nn.Module):
    """
    Minimal scanner-transfer baseline: a linear projection (or MLP) from
    feats_A to feats_B, conditioned on a configurable subset of
    (src_scanner, tgt_scanner, src_staining, tgt_staining) by concatenating
    one-hot encodings to the input features.

    :param input_dim: Foundation-model embedding dimensionality.
    :param scanner_vocab_size: Number of distinct scanners.
    :param staining_vocab_size: Number of distinct stainings.
    :param conditioning: List of conditioning signals to use. Allowed values:
        src_scanner, tgt_scanner, src_staining, tgt_staining. Defaults to all four.
    :param hidden_dim: If None, a pure linear projection is used. Otherwise an MLP
                      with this hidden width and GELU activations is used.
    :param num_hidden_layers: Number of hidden layers when hidden_dim is set (default 1).
    """

    def __init__(
        self,
        input_dim: int,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[list[str]] = None,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.scanner_vocab_size = scanner_vocab_size
        self.staining_vocab_size = staining_vocab_size

        self.conditioning = (
            conditioning if conditioning is not None else ALL_CONDITIONING
        )
        unknown = set(self.conditioning) - VALID_CONDITIONING
        if unknown:
            raise ValueError(
                f"Unknown conditioning signals: {unknown}. Valid: {VALID_CONDITIONING}"
            )

        cond_dim = sum(
            (
                scanner_vocab_size
                if s in ("src_scanner", "tgt_scanner")
                else staining_vocab_size
            )
            for s in self.conditioning
        )
        joined_dim = input_dim + cond_dim

        if hidden_dim is None:
            self.proj = nn.Linear(joined_dim, input_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(joined_dim, hidden_dim), nn.GELU()]
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, input_dim))
            self.proj = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        cond = torch.cat([inputs[s] for s in self.conditioning], dim=-1)
        return self.proj(torch.cat([x, cond], dim=-1)) + x
