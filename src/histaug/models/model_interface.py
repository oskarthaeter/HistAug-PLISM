import importlib
import inspect
import logging
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import bootstrap

from models.foundation_models import get_foundation_model
from utils.loss_factory import CombinedLoss, create_loss
from utils.optim_factory import create_optimizer, create_scheduler

# PLISM scanner → vendor mapping (Hamamatsu NanoZoomer family vs Leica Aperio)
_SCANNER_VENDOR: dict[str, str] = {
    "AT2": "Leica",
    "P": "Philips",
    "S210": "Hamamatsu",
    "S360": "Hamamatsu",
    "S60": "Hamamatsu",
    "SQ": "Hamamatsu",
    "GT450": "Leica",
}

_SCORPION_FOCUS_SOURCE_SCANNERS = {"DP200", "P1000"}
_SCORPION_FOCUS_TARGET_SCANNERS = {"AT2", "GT450", "P"}
_SCORPION_SOURCE_ORDER = ["AT2", "GT450", "P", "DP200", "P1000"]
_SCORPION_TARGET_ORDER = ["AT2", "GT450", "P"]


def _canonical_scanner_name(scanner: str) -> str:
    """Normalize scanner aliases so SCORPION-focused filtering is robust."""
    s = "".join(ch for ch in str(scanner).upper() if ch.isalnum())
    alias_to_canonical = {
        "AT2": "AT2",
        "APERIOAT2": "AT2",
        "GT450": "GT450",
        "APERIOGT450": "GT450",
        "P": "P",
        "PHILIPS": "P",
        "ULTRAFASTSCANNER": "P",
        "DP200": "DP200",
        "VENTANADP200": "DP200",
        "S210": "DP200",
        "P1000": "P1000",
        "3DHISTECHP1000": "P1000",
        "HISTECHP1000": "P1000",
        "S360": "P1000",
    }
    return alias_to_canonical.get(s, s)


def _get_scorpion_focus_axes(names: list[str]) -> tuple[list[int], list[int]]:
    """Return (source_indices, target_indices) for the requested SCORPION subset."""
    src_idx = [
        i
        for i, n in enumerate(names)
        if _canonical_scanner_name(n) in _SCORPION_FOCUS_SOURCE_SCANNERS
    ]
    tgt_idx = [
        i
        for i, n in enumerate(names)
        if _canonical_scanner_name(n) in _SCORPION_FOCUS_TARGET_SCANNERS
    ]
    return src_idx, tgt_idx


def _get_scorpion_rect_axes(
    names: list[str],
) -> tuple[list[int], list[str], list[int], list[str]]:
    """Return ordered row/col indices+labels for rectangular SCORPION heatmaps."""
    canonical_to_idx: dict[str, int] = {}
    for i, name in enumerate(names):
        canon = _canonical_scanner_name(name)
        if canon not in canonical_to_idx:
            canonical_to_idx[canon] = i

    src_labels = [s for s in _SCORPION_SOURCE_ORDER if s in canonical_to_idx]
    tgt_labels = [t for t in _SCORPION_TARGET_ORDER if t in canonical_to_idx]
    src_idx = [canonical_to_idx[s] for s in src_labels]
    tgt_idx = [canonical_to_idx[t] for t in tgt_labels]
    return src_idx, src_labels, tgt_idx, tgt_labels


def _apply_scorpion_focus_mask(mat: np.ndarray, names: list[str]) -> np.ndarray:
    """Keep only requested SCORPION target columns; preserve all source rows."""
    _, tgt_idx = _get_scorpion_focus_axes(names)
    if not tgt_idx:
        return mat
    out = np.full_like(mat, np.nan, dtype=np.float64)
    out[:, tgt_idx] = mat[:, tgt_idx]
    return out


def _staining_to_category(staining: str) -> str:
    """Strip trailing H to get the solution category (e.g. GIVH → GIV, MY stays MY)."""
    if staining != "MY" and staining.endswith("H"):
        return staining[:-1]
    return staining


class ModelInterface(pl.LightningModule):
    """
    PyTorch Lightning module wrapping a foundation-based adaptation model.

    :param model: Configuration dict for the adaptation model architecture.
    :param loss: Loss configuration for training objective.
    :param optimizer: Optimizer configuration dict.
    :param scheduler: Learning rate scheduler configuration dict.
    :param transforms: Data augmentation and preprocessing parameters.
    """

    def __init__(
        self,
        model,
        loss,
        optimizer,
        scheduler,
        transforms,
        histaug=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparameter containers
        self.optimizer_params = optimizer
        self.scheduler_params = scheduler
        self.transforms = transforms

        # Instantiate components
        self.load_model()
        self.loss_fn = create_loss(loss)

        # Feature extractor — skipped when using pre-extracted features (name is null/empty)
        fm_cfg = self.hparams.foundation_model or {}
        fm_name = (
            fm_cfg.get("name")
            if isinstance(fm_cfg, dict)
            else getattr(fm_cfg, "name", None)
        )
        if fm_name:
            self.feature_extractor = get_foundation_model(
                self.hparams.foundation_model,
                device=self.device,
            )
        else:
            self.feature_extractor = None

        # Frozen HistAug augmentor for feature-space augmentation (prefeatures training)
        self.histaug_augmentor = None
        self.histaug_aug_prob = 0.5
        self.histaug_scanner_to_id: dict[str, int] = {}
        self.histaug_staining_to_id: dict[str, int] = {}
        if histaug is not None:
            self._load_histaug_augmentor(histaug)

        # Vocabularies for per-group breakdown (populated from cfg.Model.*_names)
        model_cfg = self.hparams.model
        self.id_to_scanner: list[str] = list(
            getattr(model_cfg, "scanner_names", None) or []
        )
        self.id_to_staining: list[str] = list(
            getattr(model_cfg, "staining_names", None) or []
        )

        # Metric namespace for the current test run; set to "scorpion" externally
        # before calling trainer.test() on the SCORPION dataloader.
        self._test_log_prefix: str = "test"

        # Structured containers for metrics
        self.cosine_stats = {
            phase: {"imgaug_sum": 0.0, "id_sum": 0.0, "orig_trans_sum": 0.0, "count": 0}
            for phase in ("train", "val", "test")
        }
        self.losses = {phase: [] for phase in ("train", "val", "test")}
        self._test_imgaug_cos_values = []
        # Per-sample cosine similarity buffer for HDF5 export (test / test_holdout_staining / scorpion)
        self._cos_buffer: list[torch.Tensor] = []
        self._origtrans_cos_buffer: list[torch.Tensor] = []
        self._identity_cos_buffer: list[torch.Tensor] = []
        self._orig_identity_cos_buffer: list[torch.Tensor] = []
        # Parallel scanner/staining ID buffers — int16, -1 when not applicable
        self._src_scanner_buffer: list[torch.Tensor] = []
        self._tgt_scanner_buffer: list[torch.Tensor] = []
        self._src_staining_buffer: list[torch.Tensor] = []
        self._tgt_staining_buffer: list[torch.Tensor] = []
        # Per (src_staining→tgt_staining) and (src_scanner→tgt_scanner) cosine tracking
        self.pair_group_cos: dict[str, dict[str, list[float]]] = {
            phase: {} for phase in ("train", "val", "test")
        }
        # Individual weighted loss component accumulators (for CombinedLoss)
        self.component_loss_sums: dict[str, dict[str, list[torch.Tensor]]] = {
            phase: {} for phase in ("train", "val", "test")
        }

        # Residual magnitude penalty weight (penalises ||pred_b - feats_a||²)
        loss_cfg = loss
        self.residual_penalty_weight = float(
            loss_cfg.get("residual_penalty_weight", 0.0)
            if hasattr(loss_cfg, "get")
            else getattr(loss_cfg, "residual_penalty_weight", 0.0)
        )

        # Per (src_scanner_idx, tgt_scanner_idx) cosine tracking for heatmaps
        self.pair_scanner_cos: dict[str, dict[tuple[int, int], list[float]]] = {
            phase: {} for phase in ("val", "test")
        }
        # Per src_scanner_idx identity cosine (fills heatmap diagonal)
        self.pair_scanner_id_cos: dict[str, dict[int, list[float]]] = {
            phase: {} for phase in ("val", "test")
        }
        # Per (src, tgt) origtrans cosine (raw baseline, second heatmap)
        self.pair_scanner_orig_cos: dict[str, dict[tuple[int, int], list[float]]] = {
            phase: {} for phase in ("val", "test")
        }

        # torch.compile state (fit-stage only)
        self._models_compiled: bool = False

    def setup(self, stage: str) -> None:
        """Compile model components for training when requested via config."""
        if stage != "fit" or self._models_compiled:
            return

        general_cfg = self.hparams.get("general", {})
        compile_enabled = bool(
            general_cfg.get("compile", True)
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile", True)
        )
        if not compile_enabled:
            logging.info("torch.compile disabled by config (General.compile=false).")
            return

        if not hasattr(torch, "compile"):
            logging.warning(
                "torch.compile is unavailable in this PyTorch build; skipping compilation."
            )
            return

        backend = (
            general_cfg.get("compile_backend", "inductor")
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile_backend", "inductor")
        )
        mode = (
            general_cfg.get("compile_mode", "default")
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile_mode", "default")
        )
        fullgraph = bool(
            general_cfg.get("compile_fullgraph", False)
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile_fullgraph", False)
        )
        dynamic = bool(
            general_cfg.get("compile_dynamic", False)
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile_dynamic", False)
        )

        compile_frozen = bool(
            general_cfg.get("compile_frozen", True)
            if hasattr(general_cfg, "get")
            else getattr(general_cfg, "compile_frozen", True)
        )

        self.model = self._compile_module(
            module=self.model,
            module_name="model",
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

        if compile_frozen and self.feature_extractor is not None:
            self.feature_extractor = self._compile_module(
                module=self.feature_extractor,
                module_name="feature_extractor",
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        if compile_frozen and self.histaug_augmentor is not None:
            self.histaug_augmentor = self._compile_module(
                module=self.histaug_augmentor,
                module_name="histaug_augmentor",
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )

        self._models_compiled = True

    def _compile_module(
        self,
        module: torch.nn.Module,
        module_name: str,
        backend: str,
        mode: str,
        fullgraph: bool,
        dynamic: bool,
    ) -> torch.nn.Module:
        """Best-effort torch.compile wrapper with logging and fallback."""
        if module is None:
            return module
        if module.__class__.__name__ == "OptimizedModule":
            return module

        try:
            compiled = torch.compile(
                module,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            logging.info(
                f"Compiled {module_name} with torch.compile "
                f"(backend={backend}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic})."
            )
            return compiled
        except Exception as e:
            logging.warning(
                f"torch.compile failed for {module_name}; using eager module instead. "
                f"Reason: {e}"
            )
            return module

    def get_progress_bar_dict(self) -> dict:
        """
        Customize the progress bar to hide version number information.

        :return: Dictionary of progress bar metrics.
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning training step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for backpropagation.
        """
        return self._shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning validation step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for validation.
        """
        return self._shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning test step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for testing.
        """
        return self._shared_step(batch, batch_idx, phase="test")

    def _shared_step(self, batch: tuple, batch_idx: int, phase: str) -> torch.Tensor:
        """
        Shared logic for train/val/test steps: feature extraction, forward pass, loss and metric computation.

        Dispatches on batch shape:
        - 3-tuple (patch_orig, patch_trans, aug_params) -> original HistAug training.
        - 5-tuple (patch_orig, patch_trans, aug_params, scanner_id, staining_id)
          -> conditioned HistAug training (scanner/staining context).
        - 6-tuple (img_a, img_b, src_scanner, tgt_scanner, src_staining, tgt_staining)
          -> paired scanner-transfer training.
        """
        if len(batch) == 6:
            return self._shared_step_pair(batch, phase)
        if len(batch) == 5:
            return self._shared_step_conditioned(batch, phase)

        patch_orig, patch_trans, aug_params = batch
        patch_orig = patch_orig.to(self.device, non_blocking=True)
        patch_trans = patch_trans.to(self.device, non_blocking=True)
        bsz = patch_orig.shape[0]

        with torch.amp.autocast("cuda"):
            if self.feature_extractor is not None:
                with torch.no_grad():
                    feats = self.feature_extractor(
                        torch.cat([patch_orig, patch_trans], dim=0)
                    )
                feats_orig, feats_trans = feats[:bsz], feats[bsz:]
            else:
                feats_orig, feats_trans = patch_orig, patch_trans

            # Predict transformed features
            pred_imgaug = self.model(feats_orig, aug_params["img_aug"])
            loss_imgaug = self.loss_fn(pred_imgaug, feats_trans)

            # Predict identity features
            pred_id = self.model(feats_orig, aug_params["id"])
            loss_id = self.loss_fn(pred_id, feats_orig)

            batch_loss = loss_imgaug + loss_id

            # Cosine similarities per sample
            cos_imgaug = F.cosine_similarity(pred_imgaug, feats_trans)
            cos_id = F.cosine_similarity(pred_id, feats_orig)
            cos_orig_trans = F.cosine_similarity(feats_orig, feats_trans)

        if phase == "test":
            cos_cpu = cos_imgaug.detach().float().cpu()
            cos_orig_cpu = cos_orig_trans.detach().float().cpu()
            cos_id_cpu = cos_id.detach().float().cpu()
            cos_orig_id_cpu = (
                F.cosine_similarity(feats_orig, feats_orig).detach().float().cpu()
            )
            self._test_imgaug_cos_values.append(cos_cpu)
            self._cos_buffer.append(cos_cpu)
            self._origtrans_cos_buffer.append(cos_orig_cpu)
            self._identity_cos_buffer.append(cos_id_cpu)
            self._orig_identity_cos_buffer.append(cos_orig_id_cpu)
            _fill = torch.full((cos_cpu.shape[0],), -1, dtype=torch.int16)
            self._src_scanner_buffer.append(_fill)
            self._tgt_scanner_buffer.append(_fill)
            self._src_staining_buffer.append(_fill)
            self._tgt_staining_buffer.append(_fill)

        # Accumulate sums and counts in dict
        stats = self.cosine_stats[phase]
        stats["imgaug_sum"] += cos_imgaug.sum().item()
        stats["id_sum"] += cos_id.sum().item()
        stats["orig_trans_sum"] += cos_orig_trans.sum().item()
        stats["count"] += cos_imgaug.numel()
        # Store for epoch aggregations
        self.losses[phase].append(batch_loss.detach())
        if phase == "train":
            self.log(
                "train/loss",
                batch_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return batch_loss

    def _shared_step_conditioned(self, batch: tuple, phase: str) -> torch.Tensor:
        """
        Conditioned HistAug step. Batch from PlismAugJpegDataset:
        (patch_orig, patch_trans, aug_params, scanner_id, staining_id).

        Forwards scanner/staining IDs to the model alongside aug_params.
        """
        patch_orig, patch_trans, aug_params, scanner_id, staining_id = batch
        patch_orig = patch_orig.to(self.device, non_blocking=True)
        patch_trans = patch_trans.to(self.device, non_blocking=True)
        scanner_id = scanner_id.to(self.device, non_blocking=True).long()
        staining_id = staining_id.to(self.device, non_blocking=True).long()
        bsz = patch_orig.shape[0]

        with torch.amp.autocast("cuda"):
            if self.feature_extractor is not None:
                with torch.no_grad():
                    feats = self.feature_extractor(
                        torch.cat([patch_orig, patch_trans], dim=0)
                    )
                feats_orig, feats_trans = feats[:bsz], feats[bsz:]
            else:
                feats_orig, feats_trans = patch_orig, patch_trans

            pred_imgaug = self.model(
                feats_orig,
                aug_params["img_aug"],
                scanner_id=scanner_id,
                staining_id=staining_id,
            )
            loss_imgaug = self.loss_fn(pred_imgaug, feats_trans)

            pred_id = self.model(
                feats_orig,
                aug_params["id"],
                scanner_id=scanner_id,
                staining_id=staining_id,
            )
            loss_id = self.loss_fn(pred_id, feats_orig)

            batch_loss = loss_imgaug + loss_id

            cos_imgaug = F.cosine_similarity(pred_imgaug, feats_trans)
            cos_id = F.cosine_similarity(pred_id, feats_orig)
            cos_orig_trans = F.cosine_similarity(feats_orig, feats_trans)

        if phase == "test":
            cos_cpu = cos_imgaug.detach().float().cpu()
            cos_orig_cpu = cos_orig_trans.detach().float().cpu()
            cos_id_cpu = cos_id.detach().float().cpu()
            cos_orig_id_cpu = (
                F.cosine_similarity(feats_orig, feats_orig).detach().float().cpu()
            )
            self._test_imgaug_cos_values.append(cos_cpu)
            self._cos_buffer.append(cos_cpu)
            self._origtrans_cos_buffer.append(cos_orig_cpu)
            self._identity_cos_buffer.append(cos_id_cpu)
            self._orig_identity_cos_buffer.append(cos_orig_id_cpu)
            # conditioned mode: augments within the same scanner — no separate target
            _sc = scanner_id.detach().cpu().short()
            _st = staining_id.detach().cpu().short()
            _fill = torch.full((bsz,), -1, dtype=torch.int16)
            self._src_scanner_buffer.append(_sc)
            self._tgt_scanner_buffer.append(_fill)
            self._src_staining_buffer.append(_st)
            self._tgt_staining_buffer.append(_fill)

        stats = self.cosine_stats[phase]
        stats["imgaug_sum"] += cos_imgaug.sum().item()
        stats["id_sum"] += cos_id.sum().item()
        stats["orig_trans_sum"] += cos_orig_trans.sum().item()
        stats["count"] += cos_imgaug.numel()

        # Per-scanner and per-staining-category cosine breakdown for val/test
        if phase in ("val", "test"):
            cos_cpu = cos_imgaug.detach().float().cpu()
            group_dict = self.pair_group_cos[phase]
            if self.id_to_scanner:
                for i, sc_id in enumerate(scanner_id.cpu().tolist()):
                    sc = self.id_to_scanner[sc_id]
                    group_dict.setdefault(f"scanner_{sc}", []).append(cos_cpu[i].item())
            # if self.id_to_staining:
            #     for i, st_id in enumerate(staining_id.cpu().tolist()):
            #         cat = _staining_to_category(self.id_to_staining[st_id])
            #         group_dict.setdefault(f"stain_{cat}", []).append(cos_cpu[i].item())

        self.losses[phase].append(batch_loss.detach())
        if phase == "train":
            self.log(
                "train/loss",
                batch_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return batch_loss

    def _shared_step_pair(self, batch: tuple, phase: str) -> torch.Tensor:
        """
        Scanner-transfer step. Batch is produced by PlismPairJpegDataset:
        (img_a, img_b, src_scanner, tgt_scanner, src_staining, tgt_staining).

        Predicts feats_B from feats_A conditioned on scanner/staining tokens.
        Also runs an identity pass (src==tgt tokens) as regularisation.
        """
        img_a, img_b, src_scanner, tgt_scanner, src_staining, tgt_staining = batch
        img_a = img_a.to(self.device, non_blocking=True)
        img_b = img_b.to(self.device, non_blocking=True)
        src_scanner = src_scanner.to(self.device, non_blocking=True).long()
        tgt_scanner = tgt_scanner.to(self.device, non_blocking=True).long()
        src_staining = src_staining.to(self.device, non_blocking=True).long()
        tgt_staining = tgt_staining.to(self.device, non_blocking=True).long()
        bsz = img_a.shape[0]

        with torch.amp.autocast("cuda"):
            if self.feature_extractor is not None:
                with torch.no_grad():
                    feats = self.feature_extractor(torch.cat([img_a, img_b], dim=0))
                feats_a, feats_b = feats[:bsz], feats[bsz:]
            else:
                feats_a, feats_b = img_a, img_b

            # Feature-space augmentation via frozen HistAug (training only)
            if (
                self.histaug_augmentor is not None
                and self.training
                and self.id_to_scanner
                and self.id_to_staining
            ):
                ha_sc = torch.tensor(
                    [
                        self.histaug_scanner_to_id.get(self.id_to_scanner[i], -1)
                        for i in src_scanner.cpu().tolist()
                    ],
                    device=self.device,
                )
                ha_st = torch.tensor(
                    [
                        self.histaug_staining_to_id.get(self.id_to_staining[i], -1)
                        for i in src_staining.cpu().tolist()
                    ],
                    device=self.device,
                )
                valid = (ha_sc >= 0) & (ha_st >= 0)
                apply = valid & (
                    torch.rand(bsz, device=self.device) < self.histaug_aug_prob
                )
                if apply.any():
                    with torch.no_grad():
                        aug_params = self.histaug_augmentor.sample_aug_params(
                            bsz, device=self.device, mode="instance_wise"
                        )
                        # sample_aug_params includes zero-valued "missing" transforms that
                        # have no embedding in this model — strip them before forwarding.
                        aug_params = {
                            k: v
                            for k, v in aug_params.items()
                            if k in self.histaug_augmentor.transform_embeddings
                        }
                        feats_a_aug = self.histaug_augmentor(
                            feats_a,
                            aug_params,
                            scanner_id=ha_sc.clamp(min=0),
                            staining_id=ha_st.clamp(min=0),
                        )
                    feats_a = torch.where(apply.unsqueeze(-1), feats_a_aug, feats_a)

            # Scanner-transfer prediction
            pred_b = self.model(
                feats_a,
                src_scanner=src_scanner,
                tgt_scanner=tgt_scanner,
                src_staining=src_staining,
                tgt_staining=tgt_staining,
            )

            # Identity regularisation: same-scanner/staining tokens should be a no-op
            pred_id = self.model(
                feats_a,
                src_scanner=src_scanner,
                tgt_scanner=src_scanner,
                src_staining=src_staining,
                tgt_staining=src_staining,
            )

            # Losses
            if isinstance(self.loss_fn, CombinedLoss):
                loss_pair, components = self.loss_fn.forward_with_components(
                    pred_b, feats_b
                )
            else:
                loss_pair = self.loss_fn(pred_b, feats_b)
                components = {}
            loss_id = self.loss_fn(pred_id, feats_a)
            batch_loss = loss_pair + loss_id

            if self.residual_penalty_weight > 0.0:
                loss_residual = F.mse_loss(pred_b, feats_a.detach())
                batch_loss = batch_loss + self.residual_penalty_weight * loss_residual
                components["ResidualPenalty"] = (
                    self.residual_penalty_weight * loss_residual
                ).detach()

            cos_pred = F.cosine_similarity(pred_b, feats_b)
            cos_ab = F.cosine_similarity(feats_a, feats_b)
            cos_id = F.cosine_similarity(pred_id, feats_a)

        if phase == "test":
            cos_cpu = cos_pred.detach().float().cpu()
            cos_orig_cpu = cos_ab.detach().float().cpu()
            cos_id_cpu = cos_id.detach().float().cpu()
            cos_orig_id_cpu = (
                F.cosine_similarity(feats_a, feats_a).detach().float().cpu()
            )
            self._test_imgaug_cos_values.append(cos_cpu)
            self._cos_buffer.append(cos_cpu)
            self._origtrans_cos_buffer.append(cos_orig_cpu)
            self._identity_cos_buffer.append(cos_id_cpu)
            self._orig_identity_cos_buffer.append(cos_orig_id_cpu)
            self._src_scanner_buffer.append(src_scanner.detach().cpu().short())
            self._tgt_scanner_buffer.append(tgt_scanner.detach().cpu().short())
            self._src_staining_buffer.append(src_staining.detach().cpu().short())
            self._tgt_staining_buffer.append(tgt_staining.detach().cpu().short())

        stats = self.cosine_stats[phase]
        stats["imgaug_sum"] += cos_pred.sum().item()
        stats["id_sum"] += cos_id.sum().item()
        stats["orig_trans_sum"] += cos_ab.sum().item()
        stats["count"] += cos_pred.numel()

        # Accumulate weighted component losses for separate logging
        for name, val in components.items():
            self.component_loss_sums[phase].setdefault(name, []).append(val.detach())

        # Per-group cosine tracking for val/test
        if phase in ("val", "test"):
            cos_cpu = cos_pred.detach().float().cpu()
            cos_id_cpu = cos_id.detach().float().cpu()
            cos_ab_cpu = cos_ab.detach().float().cpu()
            group_dict = self.pair_group_cos[phase]

            # Scanner-distance buckets + per-(src, tgt) pair tracking for heatmaps
            if self.id_to_scanner:
                src_sc_list = src_scanner.cpu().tolist()
                tgt_sc_list = tgt_scanner.cpu().tolist()
                pair_dict = self.pair_scanner_cos[phase]
                id_dict = self.pair_scanner_id_cos[phase]
                orig_dict = self.pair_scanner_orig_cos[phase]
                for i in range(bsz):
                    si, ti = src_sc_list[i], tgt_sc_list[i]
                    if si == ti:
                        bucket = "same_scanner"
                    else:
                        sv = _SCANNER_VENDOR.get(self.id_to_scanner[si], "")
                        tv = _SCANNER_VENDOR.get(self.id_to_scanner[ti], "")
                        bucket = "same_vendor" if sv == tv else "diff_vendor"
                    group_dict.setdefault(bucket, []).append(cos_cpu[i].item())
                    pair_dict.setdefault((si, ti), []).append(cos_cpu[i].item())
                    id_dict.setdefault(si, []).append(cos_id_cpu[i].item())
                    orig_dict.setdefault((si, ti), []).append(cos_ab_cpu[i].item())

            # Per-source-staining-category breakdown
            # if self.id_to_staining:
            #     src_st_list = src_staining.cpu().tolist()
            #     for i in range(bsz):
            #         cat = _staining_to_category(self.id_to_staining[src_st_list[i]])
            #         group_dict.setdefault(f"stain_{cat}", []).append(cos_cpu[i].item())

        self.losses[phase].append(batch_loss.detach())
        if phase == "train":
            self.log(
                "train/loss",
                batch_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return batch_loss

    def _log_scanner_heatmap(self, phase: str, log_prefix: str) -> None:
        """Build scanner-pair cosine heatmaps (pred + origtrans) and log them to wandb.

        :param phase: Internal accumulator key ('val' or 'test').
        :param log_prefix: Metric namespace used for W&B keys (e.g. 'val', 'test', 'scorpion').
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = self.id_to_scanner
        n = len(names)

        # --- Matrix 1: pred cosine off-diagonal, identity cosine on diagonal ---
        pred_mat = np.full((n, n), np.nan)
        for (si, ti), vals in self.pair_scanner_cos[phase].items():
            if vals:
                pred_mat[si, ti] = sum(vals) / len(vals)
        for si, vals in self.pair_scanner_id_cos[phase].items():
            if vals:
                pred_mat[si, si] = sum(vals) / len(vals)

        # --- Matrix 2: origtrans cosine (raw baseline) ---
        # Diagonal is trivially 1.0 (same scanner → same features file → same tile).
        # No same-scanner pairs form in PLISM (one slide per staining-scanner combo),
        # so we fill the diagonal explicitly rather than relying on accumulated data.
        orig_mat = np.full((n, n), np.nan)
        for i in range(n):
            orig_mat[i, i] = 1.0
        for (si, ti), vals in self.pair_scanner_orig_cos[phase].items():
            if vals:
                orig_mat[si, ti] = sum(vals) / len(vals)

        # Each entry: (wandb_key, matrix, title, diverging)
        # diverging=True  → symmetric colormap centred at 0 (RdYlGn, vmin=-abs_max, vmax=abs_max)
        # diverging=False → one-sided colormap (vmin=data_min-0.02, vmax=1.0)
        heatmaps = [
            (
                "scanner_pair_heatmap",
                pred_mat,
                f"{log_prefix} pred cosine (diagonal = identity pass)",
                False,
            ),
        ]
        if phase != "val":
            # --- Matrix 3: absolute improvement (pred − origtrans, off-diagonal only) ---
            diff_mat = np.full((n, n), np.nan)
            for si in range(n):
                for ti in range(n):
                    if si != ti and not (
                        np.isnan(pred_mat[si, ti]) or np.isnan(orig_mat[si, ti])
                    ):
                        diff_mat[si, ti] = pred_mat[si, ti] - orig_mat[si, ti]

            # --- Matrix 4: relative improvement (pred − orig) / (1 − orig), off-diagonal ---
            rel_mat = np.full((n, n), np.nan)
            for si in range(n):
                for ti in range(n):
                    if si != ti and not (
                        np.isnan(pred_mat[si, ti]) or np.isnan(orig_mat[si, ti])
                    ):
                        gap = 1.0 - orig_mat[si, ti]
                        if gap > 1e-6:
                            rel_mat[si, ti] = (
                                pred_mat[si, ti] - orig_mat[si, ti]
                            ) / gap

            heatmaps += [
                (
                    "scanner_origtrans_heatmap",
                    orig_mat,
                    f"{log_prefix} origtrans cosine (raw baseline)",
                    False,
                ),
                (
                    "scanner_diff_heatmap",
                    diff_mat,
                    f"{log_prefix} improvement (pred − origtrans)",
                    True,
                ),
                (
                    "scanner_rel_heatmap",
                    rel_mat,
                    f"{log_prefix} relative improvement (pred − orig) / (1 − orig)",
                    True,
                ),
            ]

        figs = {}

        def _build_heatmap_figure(
            mat: np.ndarray,
            row_names: list[str],
            col_names: list[str],
            title: str,
            diverging: bool,
        ) -> plt.Figure:
            finite = mat[~np.isnan(mat)]
            if diverging:
                abs_max = (
                    max(float(np.abs(finite).max()), 1e-6) if finite.size > 0 else 0.1
                )
                vmin, vmax = -abs_max, abs_max
            else:
                vmin = max(float(finite.min()) - 0.02, 0.0) if finite.size > 0 else 0.7
                vmax = 1.0

            fig, ax = plt.subplots(
                figsize=(max(4, len(col_names) + 1), max(3, len(row_names)))
            )
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(col_names)))
            ax.set_yticks(range(len(row_names)))
            ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(row_names, fontsize=8)
            ax.set_xlabel("Target scanner")
            ax.set_ylabel("Source scanner")

            span = max(vmax - vmin, 1e-6)
            for si in range(len(row_names)):
                for ti in range(len(col_names)):
                    v = mat[si, ti]
                    if not np.isnan(v):
                        brightness = (v - vmin) / span
                        txt_color = "black" if 0.3 < brightness < 0.75 else "white"
                        ax.text(
                            ti,
                            si,
                            f"{v:.3f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=txt_color,
                        )
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig

        scorpion_src_idx: list[int] = []
        scorpion_src_labels: list[str] = []
        scorpion_tgt_idx: list[int] = []
        scorpion_tgt_labels: list[str] = []
        if log_prefix == "scorpion":
            (
                scorpion_src_idx,
                scorpion_src_labels,
                scorpion_tgt_idx,
                scorpion_tgt_labels,
            ) = _get_scorpion_rect_axes(names)
            if not scorpion_src_idx or not scorpion_tgt_idx:
                logging.warning(
                    "SCORPION rectangular view not applied: required source/target "
                    f"scanners not found in vocabulary {names}. "
                    "Need sources {AT2, GT450, P, DP200, P1000} and targets {AT2, GT450, P/Philips}."
                )

        for key, mat, title, diverging in heatmaps:
            draw_mat = mat
            row_names = names
            col_names = names
            if log_prefix == "scorpion" and scorpion_src_idx and scorpion_tgt_idx:
                draw_mat = mat[np.ix_(scorpion_src_idx, scorpion_tgt_idx)]
                row_names = scorpion_src_labels
                col_names = scorpion_tgt_labels
            figs[key] = _build_heatmap_figure(
                draw_mat,
                row_names,
                col_names,
                title,
                diverging,
            )

        logged = False
        for logger in self.trainer.loggers if self.trainer else []:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                try:
                    import wandb

                    logger.experiment.log(
                        {f"{log_prefix}/{k}": wandb.Image(f) for k, f in figs.items()},
                        commit=False,
                    )
                    logged = True
                except Exception as e:
                    logging.warning(f"Could not log scanner heatmaps to wandb: {e}")
        if not logged:
            logging.debug(
                f"Scanner heatmaps for {log_prefix} not logged (no compatible logger)."
            )
        for fig in figs.values():
            plt.close(fig)

    def _epoch_end(self, phase: str) -> None:
        """
        Aggregate losses and cosine similarities at epoch end and log metrics.

        :param phase: One of 'train', 'val', 'test'.
        :return: None.
        """
        # Use _test_log_prefix so SCORPION metrics land under "scorpion/" rather
        # than "test/" while still sharing the same internal accumulators.
        log_phase = self._test_log_prefix if phase == "test" else phase

        # Aggregate loss
        losses = self.losses[phase]
        mean_loss = torch.stack(losses).mean().item()

        # Compute mean similarities
        stats = self.cosine_stats[phase]
        count = stats["count"]
        mean_imgaug_cos = stats["imgaug_sum"] / count
        mean_orig_cos = stats["orig_trans_sum"] / count

        metrics = {
            f"{log_phase}/epoch_loss": mean_loss,
            f"{log_phase}/mean_imgaug_cos": mean_imgaug_cos,
            f"{log_phase}/mean_origtrans_cos": mean_orig_cos,
        }
        if stats["id_sum"] != 0.0:
            metrics[f"{log_phase}/mean_id_cos"] = stats["id_sum"] / count

        # Fraction of the achievable gap closed: (pred_cos - baseline) / (1 - baseline)
        if mean_orig_cos < 1.0 - 1e-6:
            metrics[f"{log_phase}/relative_improvement"] = (
                mean_imgaug_cos - mean_orig_cos
            ) / (1.0 - mean_orig_cos)

        # Per-component loss breakdown (CombinedLoss only)
        for name, vals in self.component_loss_sums[phase].items():
            metrics[f"{log_phase}/loss_{name}"] = torch.stack(vals).mean().item()
        self.component_loss_sums[phase].clear()

        # Per-group cosine breakdown (val/test only)
        for key, values in self.pair_group_cos[phase].items():
            metrics[f"{log_phase}/cos_{key}"] = sum(values) / len(values)
        self.pair_group_cos[phase].clear()

        # Scanner-pair heatmaps (val/test only, requires scanner vocab)
        if (
            phase in ("val", "test")
            and self.id_to_scanner
            and self.pair_scanner_cos.get(phase)
        ):
            self._log_scanner_heatmap(phase, log_phase)
            self.pair_scanner_cos[phase].clear()
            self.pair_scanner_id_cos[phase].clear()
            self.pair_scanner_orig_cos[phase].clear()

        if phase == "test":
            if len(self._test_imgaug_cos_values) > 0:
                vals = torch.cat(
                    self._test_imgaug_cos_values, dim=0
                ).numpy()  # 1D array
                self._test_imgaug_cos_values.clear()
                if vals.size > 0:
                    # method="percentile" avoids the BCa jackknife, whose default
                    # batch=None allocates an (n, n-1) matrix — ~700 GB for n≈3e5.
                    # batch caps the bootstrap resample matrix at batch*n*8 bytes.
                    res = bootstrap(
                        (vals,),
                        np.mean,
                        confidence_level=0.95,
                        n_resamples=3000,
                        vectorized=True,
                        batch=100,
                        method="percentile",
                        random_state=None,
                    )
                    metrics[f"{log_phase}/mean_imgaug_cos_ci_low"] = float(
                        res.confidence_interval.low
                    )
                    metrics[f"{log_phase}/mean_imgaug_cos_ci_high"] = float(
                        res.confidence_interval.high
                    )
                    del res
                del vals

        # Consolidated logging
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=(phase in ["train", "val"]),
            logger=True,
            sync_dist=True,
        )

        # Reset for next epoch
        self.losses[phase].clear()
        self.cosine_stats[phase] = {
            "imgaug_sum": 0.0,
            "id_sum": 0.0,
            "orig_trans_sum": 0.0,
            "count": 0,
        }

    def on_train_epoch_end(self) -> None:
        """
        Hook called at the end of training epoch to perform epoch-level logging.
        """
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        """
        Hook called at the end of validation epoch to perform epoch-level logging.
        """
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """
        Hook called at the end of testing epoch to perform epoch-level logging.
        """
        self._epoch_end("test")
        self._save_predictions_hdf5()

    def _save_predictions_hdf5(self) -> None:
        """Flush per-sample cosine similarities and pair metadata to an HDF5 file.

        Datasets written:
          cosine_similarity  float32 (N,)   — predicted cosine per sample
                    origtrans_cosine_similarity float32 (N,) — raw baseline cosine per sample
                    identity_cosine_similarity float32 (N,) — identity-pass cosine per sample
                    orig_identity_cosine_similarity float32 (N,) — baseline self-identity cosine per sample
          src_scanner_id     int16   (N,)   — source scanner vocab index (-1 if N/A)
          tgt_scanner_id     int16   (N,)   — target scanner vocab index (-1 if N/A)
          src_staining_id    int16   (N,)   — source staining vocab index (-1 if N/A)
          tgt_staining_id    int16   (N,)   — target staining vocab index (-1 if N/A)

        Attributes:
          scanner_names  JSON list mapping index → scanner name
          staining_names JSON list mapping index → staining name
        """
        if not self._cos_buffer:
            return
        ckpt_path = getattr(self, "current_model_path", None)
        if ckpt_path:
            log_dir = Path(ckpt_path).parent
        else:
            log_dir = Path(getattr(self.hparams, "log", None) or ".")
        path = log_dir / f"predictions_{self._test_log_prefix}.h5"
        cos = torch.cat(self._cos_buffer, dim=0).numpy()
        with h5py.File(path, "w") as f:
            f.create_dataset("cosine_similarity", data=cos, compression="gzip")
            if self._origtrans_cos_buffer:
                cos_orig = torch.cat(self._origtrans_cos_buffer, dim=0).numpy()
                f.create_dataset(
                    "origtrans_cosine_similarity", data=cos_orig, compression="gzip"
                )
            if self._identity_cos_buffer:
                cos_id = torch.cat(self._identity_cos_buffer, dim=0).numpy()
                f.create_dataset(
                    "identity_cosine_similarity", data=cos_id, compression="gzip"
                )
            if self._orig_identity_cos_buffer:
                cos_orig_id = torch.cat(self._orig_identity_cos_buffer, dim=0).numpy()
                f.create_dataset(
                    "orig_identity_cosine_similarity",
                    data=cos_orig_id,
                    compression="gzip",
                )
            for name, buf in (
                ("src_scanner_id", self._src_scanner_buffer),
                ("tgt_scanner_id", self._tgt_scanner_buffer),
                ("src_staining_id", self._src_staining_buffer),
                ("tgt_staining_id", self._tgt_staining_buffer),
            ):
                if buf:
                    f.create_dataset(
                        name, data=torch.cat(buf).numpy(), compression="gzip"
                    )
            if self.id_to_scanner:
                import json as _json

                f.attrs["scanner_names"] = _json.dumps(self.id_to_scanner)
            if self.id_to_staining:
                import json as _json

                f.attrs["staining_names"] = _json.dumps(self.id_to_staining)
        logging.info(f"Saved {len(cos)} cosine similarities to {path}")
        self._cos_buffer.clear()
        self._origtrans_cos_buffer.clear()
        self._identity_cos_buffer.clear()
        self._orig_identity_cos_buffer.clear()
        self._src_scanner_buffer.clear()
        self._tgt_scanner_buffer.clear()
        self._src_staining_buffer.clear()
        self._tgt_staining_buffer.clear()

    def configure_optimizers(self):
        """
        Set up optimizer and optional scheduler for training.

        :return: Single or tuple of lists: [optimizers], [schedulers] if scheduler exists.
        """
        optimizer = create_optimizer(self.optimizer_params, self.model)
        lr_scheduler = create_scheduler(self.scheduler_params, optimizer)

        if lr_scheduler is None:
            return [optimizer]
        return [optimizer], [lr_scheduler]

    def named_children(self):
        for name, module in super().named_children():
            if name == "loss_fn":
                continue
            yield name, module

    def _load_histaug_augmentor(self, histaug_cfg) -> None:
        """Load a frozen HistaugConditionedModel from checkpoint for feature-space augmentation."""

        def _get(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        ckpt_path = _get(histaug_cfg, "ckpt_path")
        if not ckpt_path:
            return

        self.histaug_aug_prob = float(_get(histaug_cfg, "aug_prob", 0.5))

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        model_cfg = hp["model"]
        transforms_cfg = hp.get("transforms")

        from models.histaug_conditioned_model import HistaugConditionedModel

        aug_model = HistaugConditionedModel(
            input_dim=_get(model_cfg, "input_dim"),
            depth=_get(model_cfg, "depth"),
            num_heads=_get(model_cfg, "num_heads"),
            mlp_ratio=_get(model_cfg, "mlp_ratio"),
            scanner_vocab_size=_get(model_cfg, "scanner_vocab_size"),
            staining_vocab_size=_get(model_cfg, "staining_vocab_size"),
            chunk_size=_get(model_cfg, "chunk_size", 16),
            use_transform_pos_embeddings=_get(
                model_cfg, "use_transform_pos_embeddings", True
            ),
            positional_encoding_type=_get(
                model_cfg, "positional_encoding_type", "learnable"
            ),
            final_activation=_get(model_cfg, "final_activation", "Identity"),
            transforms=transforms_cfg,
        )

        state = {
            k[len("model.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        aug_model.load_state_dict(state)
        aug_model.eval()
        for p in aug_model.parameters():
            p.requires_grad_(False)

        self.histaug_augmentor = aug_model

        # Build name→ID remapping from HistAug's training vocabulary
        histaug_scanner_names = list(_get(model_cfg, "scanner_names", []) or [])
        histaug_staining_names = list(_get(model_cfg, "staining_names", []) or [])
        self.histaug_scanner_to_id = {n: i for i, n in enumerate(histaug_scanner_names)}
        self.histaug_staining_to_id = {
            n: i for i, n in enumerate(histaug_staining_names)
        }

        logging.info(
            f"Loaded HistAug augmentor from {ckpt_path} "
            f"(aug_prob={self.histaug_aug_prob}, "
            f"scanners={histaug_scanner_names}, stainings={histaug_staining_names})"
        )

    def load_model(self) -> None:
        """
        Dynamically import and instantiate the model class defined in hyperparameters.

        :raises ValueError: If module or class name is invalid.
        """
        name = self.hparams.model.name
        model_class_name = "".join(part.capitalize() for part in name.split("_"))
        try:
            module = importlib.import_module(f"models.{name}")
            Model = getattr(module, model_class_name)
        except (ImportError, AttributeError):
            raise ValueError("Invalid model module or class name")

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """
        Instantiate a model using matching hyperparameters and additional arguments.

        :param Model: Class of the model to instantiate.
        :param other_args: Additional keyword arguments to override defaults.
        :return: An instance of the given Model.
        """
        signature = inspect.signature(Model.__init__)
        class_args = [
            name
            for name, param in signature.parameters.items()
            if name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        args1["transforms"] = self.transforms

        return Model(**args1)
