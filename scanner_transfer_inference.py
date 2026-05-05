"""
Minimal standalone loader for trained PLISM scanner-transfer models.

Supports all three architectures:
  - ScannerTransferLinearModel  (scanner_transfer_linear_model)
  - ScannerTransferLayerModel   (scanner_transfer_layer_model)
  - ScannerTransferBottleneckModel (scanner_transfer_bottleneck_model)

No dependency on PyTorch Lightning or the training codebase — only torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Inline model definitions (copy from training code so this file is self-contained)
# ---------------------------------------------------------------------------

VALID_CONDITIONING = {"src_scanner", "tgt_scanner", "src_staining", "tgt_staining"}
ALL_CONDITIONING = ["src_scanner", "tgt_scanner", "src_staining", "tgt_staining"]


class ScannerTransferLinearModel(nn.Module):
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
        self.conditioning = conditioning if conditioning is not None else ALL_CONDITIONING

        cond_dim = sum(
            scanner_vocab_size if s in ("src_scanner", "tgt_scanner") else staining_vocab_size
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

    def forward(self, x, src_scanner, tgt_scanner, src_staining, tgt_staining, **kwargs):
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        cond = torch.cat([inputs[s] for s in self.conditioning], dim=-1)
        return self.proj(torch.cat([x, cond], dim=-1)) + x


class _FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class ScannerTransferLayerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[list[str]] = None,
        hidden_dims: Optional[list[int]] = None,
        use_spectral_norm: bool = False,
        spectral_norm_all: bool = False,
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        from torch.nn.utils import spectral_norm

        self.input_dim = input_dim
        self.scanner_vocab_size = scanner_vocab_size
        self.staining_vocab_size = staining_vocab_size
        self.conditioning = conditioning if conditioning is not None else ALL_CONDITIONING

        cond_dim = sum(
            scanner_vocab_size if s in ("src_scanner", "tgt_scanner") else staining_vocab_size
            for s in self.conditioning
        )
        hidden_dims = hidden_dims or []
        dims = [input_dim] + hidden_dims

        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm and (spectral_norm_all or i == 0):
                linear = spectral_norm(linear)
            self.layers.append(linear)
            self.film_layers.append(_FiLMLayer(cond_dim, dims[i + 1]))

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.decoder = nn.Linear(last_dim, input_dim)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        init_raw = torch.logit(torch.tensor(alpha).clamp(1e-6, 1 - 1e-6))
        self.alpha_raw = nn.Parameter(init_raw)
        self.activation = nn.GELU()

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def _build_condition(self, src_scanner, tgt_scanner, src_staining, tgt_staining):
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        return torch.cat([inputs[s] for s in self.conditioning], dim=-1)

    def forward(self, x, src_scanner, tgt_scanner, src_staining, tgt_staining, **kwargs):
        cond = self._build_condition(src_scanner, tgt_scanner, src_staining, tgt_staining)
        h = x
        for linear, film in zip(self.layers, self.film_layers):
            h = linear(h)
            h = film(h, cond)
            h = self.activation(h)
        return x + self.alpha * self.decoder(h)


class ScannerTransferBottleneckModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scanner_vocab_size: int,
        staining_vocab_size: int,
        conditioning: Optional[list[str]] = None,
        hidden_dim: int = 64,
        use_spectral_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        from torch.nn.utils import spectral_norm

        self.input_dim = input_dim
        self.scanner_vocab_size = scanner_vocab_size
        self.staining_vocab_size = staining_vocab_size
        self.conditioning = conditioning if conditioning is not None else ALL_CONDITIONING

        cond_dim = sum(
            scanner_vocab_size if s in ("src_scanner", "tgt_scanner") else staining_vocab_size
            for s in self.conditioning
        )
        fc1 = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            fc1 = spectral_norm(fc1)
        self.encoder = nn.Sequential(fc1, nn.GELU())
        self.film = nn.Linear(cond_dim, 2 * hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.197))  # sigmoid(-2.197) ≈ 0.1

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x, src_scanner, tgt_scanner, src_staining, tgt_staining, **kwargs):
        inputs = {
            "src_scanner": F.one_hot(src_scanner, self.scanner_vocab_size).float(),
            "tgt_scanner": F.one_hot(tgt_scanner, self.scanner_vocab_size).float(),
            "src_staining": F.one_hot(src_staining, self.staining_vocab_size).float(),
            "tgt_staining": F.one_hot(tgt_staining, self.staining_vocab_size).float(),
        }
        cond = torch.cat([inputs[s] for s in self.conditioning], dim=-1)
        h = self.encoder(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        return x + self.alpha * self.decoder(h)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "scanner_transfer_linear_model": ScannerTransferLinearModel,
    "scanner_transfer_layer_model": ScannerTransferLayerModel,
    "scanner_transfer_bottleneck_model": ScannerTransferBottleneckModel,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ScannerTransferAugmentor:
    """
    Wraps a loaded scanner-transfer model with name-to-ID lookup.

    Parameters
    ----------
    model : nn.Module
        The loaded, eval-mode transfer model.
    scanner_names : list[str]
        Ordered scanner vocabulary from the checkpoint (index = ID).
    staining_names : list[str]
        Ordered staining vocabulary from the checkpoint (index = ID).
    device : str or torch.device
        Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        scanner_names: list[str],
        staining_names: list[str],
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.scanner_to_id = {name: i for i, name in enumerate(scanner_names)}
        self.staining_to_id = {name: i for i, name in enumerate(staining_names)}
        self.scanner_names = scanner_names
        self.staining_names = staining_names

    @property
    def conditions_on_src_scanner(self) -> bool:
        return "src_scanner" in getattr(self.model, "conditioning", [])

    @property
    def conditions_on_tgt_scanner(self) -> bool:
        return "tgt_scanner" in getattr(self.model, "conditioning", [])

    @property
    def conditions_on_staining(self) -> bool:
        conditioning = getattr(self.model, "conditioning", [])
        return any(s in conditioning for s in ("src_staining", "tgt_staining"))

    @torch.inference_mode()
    def augment(
        self,
        features: torch.Tensor,
        src_scanner: Optional[str] = None,
        tgt_scanner: Optional[str] = None,
        src_staining: Optional[str] = None,
        tgt_staining: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Transform a batch of patch features from source to target domain.

        Parameters
        ----------
        features : torch.Tensor, shape (N, D)
            Pre-extracted patch embeddings from the source scanner/staining.
        src_scanner : str, optional
            Name of the source scanner. Required only when the model conditions
            on src_scanner; silently ignored otherwise.
        tgt_scanner : str, optional
            Name of the target scanner. Required only when the model conditions
            on tgt_scanner; silently ignored otherwise.
        src_staining : str, optional
            Name of the source staining protocol (e.g. "GIV", "GMH").
            Required only when the model conditions on src_staining.
        tgt_staining : str, optional
            Name of the target staining protocol. Same rules as src_staining.

        Returns
        -------
        torch.Tensor, shape (N, D)
            Transformed patch embeddings in the target domain.
        """
        def _resolve_scanner(name, signal):
            if signal in getattr(self.model, "conditioning", []):
                if name is None:
                    raise ValueError(
                        f"This checkpoint conditions on {signal}. Provide {signal}."
                    )
                return self._lookup_scanner(name)
            return 0

        def _resolve_staining(name, signal):
            if signal in getattr(self.model, "conditioning", []):
                if name is None:
                    raise ValueError(
                        f"This checkpoint conditions on {signal}. Provide {signal}."
                    )
                return self._lookup_staining(name)
            return 0

        src_sc_id = _resolve_scanner(src_scanner, "src_scanner")
        tgt_sc_id = _resolve_scanner(tgt_scanner, "tgt_scanner")
        src_st_id = _resolve_staining(src_staining, "src_staining")
        tgt_st_id = _resolve_staining(tgt_staining, "tgt_staining")

        n = features.shape[0]
        x = features.to(self.device)
        src_sc = torch.full((n,), src_sc_id, dtype=torch.long, device=self.device)
        tgt_sc = torch.full((n,), tgt_sc_id, dtype=torch.long, device=self.device)
        src_st = torch.full((n,), src_st_id, dtype=torch.long, device=self.device)
        tgt_st = torch.full((n,), tgt_st_id, dtype=torch.long, device=self.device)

        return self.model(x, src_sc, tgt_sc, src_st, tgt_st)

    def _lookup_scanner(self, name: str) -> int:
        if name not in self.scanner_to_id:
            raise KeyError(
                f"Scanner '{name}' not in checkpoint vocabulary. "
                f"Available: {self.scanner_names}"
            )
        return self.scanner_to_id[name]

    def _lookup_staining(self, name: str) -> int:
        if name not in self.staining_to_id:
            raise KeyError(
                f"Staining '{name}' not in checkpoint vocabulary. "
                f"Available: {self.staining_names}"
            )
        return self.staining_to_id[name]


def load_scanner_transfer_model(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> ScannerTransferAugmentor:
    """
    Load a trained scanner-transfer model from a Lightning checkpoint.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the `.ckpt` file produced by train.py.
    device : str or torch.device
        Device for inference.

    Returns
    -------
    ScannerTransferAugmentor
        Ready-to-use augmentor with scanner/staining name lookup.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hp = ckpt["hyper_parameters"]
    model_cfg = hp["model"]

    def _get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    model_name = _get(model_cfg, "name")
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported: {list(_MODEL_REGISTRY)}"
        )
    ModelClass = _MODEL_REGISTRY[model_name]

    # Collect all constructor kwargs from the checkpoint
    import inspect
    sig = inspect.signature(ModelClass.__init__)
    init_keys = {
        k for k, p in sig.parameters.items()
        if k != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    kwargs = {k: _get(model_cfg, k) for k in init_keys if _get(model_cfg, k) is not None}

    model = ModelClass(**kwargs)

    # Strip "model." (Lightning) and optional "_orig_mod." (torch.compile) prefixes
    state = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("model."):
            continue
        k = k[len("model."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        state[k] = v
    model.load_state_dict(state, strict=True)

    scanner_names = list(_get(model_cfg, "scanner_names") or [])
    staining_names = list(_get(model_cfg, "staining_names") or [])

    return ScannerTransferAugmentor(model, scanner_names, staining_names, device=device)
