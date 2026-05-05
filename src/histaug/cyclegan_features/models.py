from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: out = (1 + gamma) * x + beta.

    Gamma and beta are predicted from the conditioning vector.
    Zero-initialised so the layer starts as identity.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class FeatureGenerator(nn.Module):
    """Residual MLP generator: out = x + MLP(x, cond).

    Two conditioning modes (selected by use_film):

    concat (default):
        cond is concatenated to x before the first layer:
            x_in = cat(x, cond)  →  hidden_dims  →  dim
        LayerNorm + GELU between hidden layers.

    film:
        cond modulates each hidden layer via FiLM (gamma/beta).
        The feature path does NOT receive the raw cond vector, so the
        conditioning has no effect at depth=0 initialisation (identity start).
        Zero-initialised decoder ensures the residual starts at zero.

    In both modes the generator is conditioned on the TARGET scanner one-hot.
    The residual skip always adds the original x, so the output dimension
    equals dim regardless of hidden_dims.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Optional[List[int]] = None,
        cond_dim: int = 0,
        use_film: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.use_film = use_film
        hidden_dims = hidden_dims or [dim, dim]

        if use_film:
            dims = [dim] + hidden_dims
            self.layers = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
            )
            self.film_layers = (
                nn.ModuleList([FiLMLayer(cond_dim, h) for h in hidden_dims])
                if cond_dim > 0
                else None
            )
            self.decoder = nn.Linear(hidden_dims[-1], dim)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
            self.activation = nn.GELU()
        else:
            in_dim = dim + cond_dim
            layer_list: list[nn.Module] = []
            prev = in_dim
            for h in hidden_dims:
                layer_list += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
                prev = h
            layer_list.append(nn.Linear(prev, dim))
            self.mlp = nn.Sequential(*layer_list)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_film:
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if self.film_layers is not None and cond is not None:
                    h = self.film_layers[i](h, cond)
                h = self.activation(h)
            return x + self.decoder(h)
        else:
            x_in = torch.cat([x, cond], dim=-1) if self.cond_dim > 0 else x
            return x + self.mlp(x_in)


class FeatureDiscriminator(nn.Module):
    """2-layer MLP discriminator, LeakyReLU(0.2), scalar LSGAN output.

    When cond_dim > 0 the target-scanner one-hot is appended to the input.
    No normalisation — empirically more stable for feature-space GANs.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        cond_dim: int = 0,
        dropout: float = 0.3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        linear = (
            (lambda module: spectral_norm(module))
            if use_spectral_norm
            else (lambda module: module)
        )
        self.net = nn.Sequential(
            linear(nn.Linear(dim + cond_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            linear(nn.Linear(hidden_dim, 1)),
        )

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_in = torch.cat([x, cond], dim=-1) if self.cond_dim > 0 else x
        return self.net(x_in)
