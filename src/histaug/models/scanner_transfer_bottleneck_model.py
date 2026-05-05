from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

VALID_CONDITIONING = {"src_scanner", "tgt_scanner", "src_staining", "tgt_staining"}
ALL_CONDITIONING = ["src_scanner", "tgt_scanner", "src_staining", "tgt_staining"]


class ScannerTransferBottleneckModel(nn.Module):
    """
    Residual feature-space scanner transfer model with:
    - bottleneck MLP (low-rank constraint)
    - FiLM conditioning (clean separation of content vs condition)
    - zero-init last layer (starts as identity)
    - residual scaling parameter alpha
    - optional spectral normalization (first layer)

    x -> x + alpha * f_theta(x | condition)
    """

    def __init__(
        self,
        input_dim: int,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[List[str]] = None,
        hidden_dim: int = 64,
        use_spectral_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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

        # ----- conditioning dimension -----
        cond_dim = sum(
            (
                scanner_vocab_size
                if s in ("src_scanner", "tgt_scanner")
                else staining_vocab_size
            )
            for s in self.conditioning
        )

        # ----- encoder (low-rank bottleneck) -----
        fc1 = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            fc1 = spectral_norm(fc1)

        self.encoder = nn.Sequential(
            fc1,
            nn.GELU(),
        )

        # ----- FiLM generator -----
        self.film = nn.Linear(cond_dim, 2 * hidden_dim)

        # ----- decoder -----
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # zero-init decoder → start as identity
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        # ----- residual scaling -----
        # Store raw (unconstrained); alpha property applies sigmoid → (0, 1)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.197))  # sigmoid(-2.197) ≈ 0.1

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def forward(
        self,
        x: torch.Tensor,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
    ) -> torch.Tensor:

        # ----- build conditioning vector -----
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        cond = torch.cat([inputs[s] for s in self.conditioning], dim=-1)

        # ----- encode -----
        h = self.encoder(x)  # (B, hidden_dim)

        # ----- FiLM modulation -----
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = 1.0 + gamma  # ensures start at 1
        h = gamma * h + beta  # beta starts at 0

        # ----- decode -----
        delta = self.decoder(h)

        return x + self.alpha * delta
