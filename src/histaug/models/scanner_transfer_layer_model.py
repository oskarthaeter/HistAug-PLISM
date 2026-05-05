from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


VALID_CONDITIONING = {"src_scanner", "tgt_scanner", "src_staining", "tgt_staining"}
ALL_CONDITIONING = ["src_scanner", "tgt_scanner", "src_staining", "tgt_staining"]


class FiLMLayer(nn.Module):
    """Applies FiLM modulation: gamma * x + beta"""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)

        # Identity initialization: gamma=1, beta=0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        gamma = 1.0 + gamma
        return gamma * x + beta


class ScannerTransferLayerModel(nn.Module):
    """
    Flexible residual MLP with FiLM conditioning.

    x -> x + alpha * f_theta(x | condition)

    Features:
    - configurable hidden_dims
    - flexible conditioning (src/tgt scanner + staining)
    - FiLM at each hidden layer
    - zero-init decoder (identity start)
    - residual scaling
    - optional spectral normalization
    """

    def __init__(
        self,
        input_dim: int,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[List[str]] = None,
        hidden_dims: Optional[List[int]] = None,
        use_spectral_norm: bool = False,
        spectral_norm_all: bool = False,
        alpha: float = 0.1,
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

        # ----- conditioning dimension -----
        cond_dim = sum(
            (
                scanner_vocab_size
                if s in ("src_scanner", "tgt_scanner")
                else staining_vocab_size
            )
            for s in self.conditioning
        )

        hidden_dims = hidden_dims or []
        dims = [input_dim] + hidden_dims

        # ----- backbone -----
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i in range(len(hidden_dims)):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            linear = nn.Linear(in_dim, out_dim)

            if use_spectral_norm and (spectral_norm_all or i == 0):
                linear = spectral_norm(linear)

            self.layers.append(linear)
            self.film_layers.append(FiLMLayer(cond_dim, out_dim))

        # ----- decoder -----
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.decoder = nn.Linear(last_dim, input_dim)

        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        # ----- residual scaling -----
        # Store raw (unconstrained); alpha property applies sigmoid → (0, 1)
        init_raw = torch.logit(torch.tensor(alpha).clamp(1e-6, 1 - 1e-6))
        self.alpha_raw = nn.Parameter(init_raw)

        self.activation = nn.GELU()

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def _build_condition(
        self,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        return torch.cat([inputs[s] for s in self.conditioning], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
    ) -> torch.Tensor:

        cond = self._build_condition(
            src_scanner, tgt_scanner, src_staining, tgt_staining
        )

        h = x

        # ----- MLP with FiLM -----
        for linear, film in zip(self.layers, self.film_layers):
            h = linear(h)
            h = film(h, cond)
            h = self.activation(h)

        delta = self.decoder(h)

        return x + self.alpha * delta
