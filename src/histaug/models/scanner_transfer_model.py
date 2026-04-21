from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.histaug_model import Block
from models.scanner_transfer_linear_model import ALL_CONDITIONING, VALID_CONDITIONING


class ScannerTransferModel(nn.Module):
    """
    Scanner-transfer model. Given a foundation-model embedding from scanner A, predicts
    the embedding the same tissue content would have under scanner B. Conditioning is
    provided as learnable tokens for a configurable subset of
    (src_scanner, tgt_scanner, src_staining, tgt_staining).

    Architecture mirrors HistaugModel: the input embedding is split into `chunk_size`
    tokens of dimension `embed_dim = input_dim // chunk_size` with sinusoidal chunk
    positional encodings, then passed through cross-attention blocks where the
    conditioning tokens form the `z` sequence. No image augmentation conditioning is
    used (augmentations are applied identically to A and B as data diversity only).
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[list[str]] = None,
        chunk_size: int = 16,
        final_activation: str = "Identity",
        **kwargs,
    ):
        super().__init__()
        assert input_dim % chunk_size == 0, "input_dim must be divisible by chunk_size"

        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.embed_dim = input_dim // chunk_size

        self.conditioning = conditioning if conditioning is not None else ALL_CONDITIONING
        unknown = set(self.conditioning) - VALID_CONDITIONING
        if unknown:
            raise ValueError(f"Unknown conditioning signals: {unknown}. Valid: {VALID_CONDITIONING}")

        self.features_embed = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        self.chunk_pos_embeddings = self._get_sinusoidal_embeddings(
            chunk_size, self.embed_dim
        )
        self.register_buffer("chunk_pos_embeddings_buffer", self.chunk_pos_embeddings)

        self.scanner_vocab_size = scanner_vocab_size
        self.staining_vocab_size = staining_vocab_size
        # One-hot projectors — only instantiated for selected conditioning signals.
        self._cond_projs = nn.ModuleDict()
        for signal in self.conditioning:
            vocab_size = scanner_vocab_size if signal in ("src_scanner", "tgt_scanner") else staining_vocab_size
            self._cond_projs[signal] = nn.Linear(vocab_size, self.embed_dim, bias=False)
        # Role embeddings distinguish the conditioning token slots.
        self.role_embed = nn.Embedding(len(self.conditioning), self.embed_dim)

        self.blocks = nn.ModuleList(
            [
                Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        if hasattr(nn, final_activation):
            self.final_activation = getattr(nn, final_activation)()
        else:
            raise ValueError(f"Activation {final_activation} is not found in torch.nn")

        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim), self.final_activation
        )

    @staticmethod
    def _get_sinusoidal_embeddings(num_positions: int, embed_dim: int) -> torch.Tensor:
        import math

        assert embed_dim % 2 == 0, "embed_dim must be even"
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(num_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _build_conditioning(
        self,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
    ) -> torch.Tensor:
        """Return conditioning tokens of shape (B, len(conditioning), embed_dim)."""
        raw = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        role_ids = torch.arange(len(self.conditioning), device=src_scanner.device)
        role = self.role_embed(role_ids)
        tokens = torch.stack(
            [self._cond_projs[s](raw[s]) + role[i] for i, s in enumerate(self.conditioning)],
            dim=1,
        )
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        src_scanner: torch.Tensor,
        tgt_scanner: torch.Tensor,
        src_staining: torch.Tensor,
        tgt_staining: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        :param x: (B, input_dim) source-scanner embedding.
        :param src_scanner, tgt_scanner, src_staining, tgt_staining: (B,) long tensors.
        :return: (B, input_dim) predicted target-scanner embedding.
        """
        B = x.shape[0]
        x = x.view(B, self.chunk_size, self.embed_dim)
        x = x + self.chunk_pos_embeddings_buffer.unsqueeze(0)

        z = self._build_conditioning(src_scanner, tgt_scanner, src_staining, tgt_staining)

        for block in self.blocks:
            x = block(x, z)
        x = self.norm(x)

        x = x.view(B, 1, -1)
        x = self.head(x)
        return x[:, 0, :]
