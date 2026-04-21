import torch
import torch.nn.functional as F
from torch import nn

from models.histaug_model import HistaugModel


class HistaugConditionedModel(HistaugModel):
    """
    HistAug model additionally conditioned on the source scanner and staining.

    The scanner and staining identities are projected to embed_dim tokens and
    appended to the augmentation token sequence z before each cross-attention
    block. This lets the model adapt its augmentation behaviour to the imaging
    conditions of the input patch.

    Can be used for inference-time feature augmentation: given a stored feature
    and known scanner/staining metadata, sample random aug_params and call
    forward() to synthesise an augmented feature without loading images.

    :param scanner_vocab_size: Number of distinct scanner identities.
    :param staining_vocab_size: Number of distinct staining identities.
    All other parameters are identical to HistaugModel.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        chunk_size: int = 16,
        use_transform_pos_embeddings: bool = True,
        positional_encoding_type: str = "learnable",
        final_activation: str = "Identity",
        transforms=None,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            chunk_size=chunk_size,
            use_transform_pos_embeddings=use_transform_pos_embeddings,
            positional_encoding_type=positional_encoding_type,
            final_activation=final_activation,
            transforms=transforms,
            **kwargs,
        )
        self.scanner_vocab_size = scanner_vocab_size
        self.staining_vocab_size = staining_vocab_size

        # One-hot → embed_dim projectors (no bias keeps them symmetric w.r.t. classes).
        self.scanner_proj = nn.Linear(scanner_vocab_size, self.embed_dim, bias=False)
        self.staining_proj = nn.Linear(staining_vocab_size, self.embed_dim, bias=False)
        # Role embeddings distinguish scanner token from staining token.
        self.context_role_embed = nn.Embedding(2, self.embed_dim)

    def _build_context_tokens(
        self,
        scanner_id: torch.Tensor,
        staining_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, 2, embed_dim) conditioning tokens for scanner and staining."""
        roles = self.context_role_embed(
            torch.arange(2, device=scanner_id.device)
        )  # (2, E)
        scanner_tok = (
            self.scanner_proj(F.one_hot(scanner_id, self.scanner_vocab_size).float())
            + roles[0]
        )
        staining_tok = (
            self.staining_proj(F.one_hot(staining_id, self.staining_vocab_size).float())
            + roles[1]
        )
        return torch.stack([scanner_tok, staining_tok], dim=1)  # (B, 2, E)

    def forward(
        self,
        x: torch.Tensor,
        aug_params,
        scanner_id: torch.Tensor = None,
        staining_id: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        :param x: (B, input_dim) input embedding.
        :param aug_params: Augmentation parameter dict from sample_aug_params().
        :param scanner_id: (B,) long tensor of scanner indices (optional at inference).
        :param staining_id: (B,) long tensor of staining indices (optional at inference).
        :return: (B, input_dim) predicted augmented embedding.
        """
        B = x.shape[0]
        x = x.view(B, self.chunk_size, self.embed_dim)
        x = x + self.chunk_pos_embeddings_buffer.unsqueeze(0)

        z_aug = self.forward_aug_params_embed(aug_params)  # (B, K, E)

        if scanner_id is not None and staining_id is not None:
            z_ctx = self._build_context_tokens(scanner_id, staining_id)  # (B, 2, E)
            z = torch.cat([z_aug, z_ctx], dim=1)  # (B, K+2, E)
        else:
            z = z_aug

        for block in self.blocks:
            x = block(x, z)
        x = self.norm(x)

        x = x.view(B, 1, -1)
        x = self.head(x)
        return x[:, 0, :]
