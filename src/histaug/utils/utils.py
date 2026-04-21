import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import List

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from utils.constants import TransformCategoryConstants

logger = logging.getLogger(__name__)


class NohupProgressBar(Callback):
    """Line-based progress logger for non-interactive runs (e.g. nohup)."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._epoch_start_time = None

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start_time = time.time()
        if trainer.is_global_zero:
            print(
                f"[progress] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started",
                flush=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero:
            return
        total = int(trainer.num_training_batches)
        step = batch_idx + 1
        if step % self.log_every_n_steps != 0 and step != total:
            return

        pct = (100.0 * step / total) if total > 0 else 0.0
        loss_value = self._to_float(outputs)
        loss_text = f" loss={loss_value:.5f}" if loss_value is not None else ""
        print(
            f"[progress] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"step {step}/{total} ({pct:.1f}%) global_step={trainer.global_step}{loss_text}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        elapsed = 0.0
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time
        val_loss = self._to_float(trainer.callback_metrics.get("val/loss"))
        val_text = f" val/loss={val_loss:.5f}" if val_loss is not None else ""
        print(
            f"[progress] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"finished in {elapsed:.1f}s{val_text}",
            flush=True,
        )


def ensure_dir(path: Path) -> None:
    """
    Create a directory and all necessary parent directories if they do not exist.

    :param path: Path object representing the directory to ensure.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(fpath=None) -> DictConfig:
    """
    Read a YAML configuration file into an OmegaConf DictConfig object.

    :param fpath: Path to the YAML file.
    :return: DictConfig containing the parsed YAML contents.
    """
    cfg = OmegaConf.load(str(fpath))
    return cfg


def load_loggers(cfg) -> List[WandbLogger]:
    """
    Set up the run directory and instantiate the W&B logger (train only).

    The log path is formed as: cfg.General.log_path / <Wandb.run_name>.
    Set Wandb.run_name in the config to a short descriptive name before each run.
    If unset, falls back to the config file stem.

    :param cfg: Configuration object with attributes:
                - General.log_path: Base directory for logs.
                - config: Path string to the config file.
                - Wandb.run_name: Short descriptive run name (set before each run).
                - Wandb.group: W&B group (e.g. "histaug" or "scanner-transfer").
                - Wandb.tags: List of W&B tags (e.g. ["h0-mini"]).
    :return: List of loggers.
    """
    base_log = Path(cfg.General.log_path)
    ensure_dir(base_log)

    config_path = Path(cfg.config)

    wandb_cfg = getattr(cfg, "Wandb", None)
    project = (
        getattr(wandb_cfg, "project", "HistAug-PLISM") if wandb_cfg else "HistAug-PLISM"
    )
    run_name = (
        getattr(wandb_cfg, "run_name", None) if wandb_cfg else None
    ) or config_path.stem
    group = getattr(wandb_cfg, "group", None) if wandb_cfg else None
    tags = list(getattr(wandb_cfg, "tags", None) or []) if wandb_cfg else []

    run_dir = base_log / run_name

    if cfg.General.server == "train" and run_dir.exists():
        logger.warning(
            f"Run directory already exists: {run_dir}. "
            "Set a unique Wandb.run_name to avoid mixing run artifacts."
        )

    ensure_dir(run_dir)
    cfg.log_path = str(run_dir)
    logger.info(f"Logs folder: {run_dir}")

    params_file_to_save = run_dir / "config_used.json"
    cfg_dict_to_save = OmegaConf.to_container(cfg, resolve=True)
    with open(params_file_to_save, "w") as f:
        json.dump(cfg_dict_to_save, f, indent=2)

    resume_run_id = getattr(cfg.General, "wandb_run_id", None)
    csv_logger = CSVLogger(save_dir=str(run_dir), name="", version="")

    if cfg.General.server != "train":
        if not resume_run_id:
            return [csv_logger]
        ckpt_override = getattr(cfg.General, "ckpt_path_override", None)
        save_dir = str(Path(ckpt_override).parent) if ckpt_override else str(run_dir)
        wb = WandbLogger(
            project=project,
            id=resume_run_id,
            resume="must",
            save_dir=save_dir,
        )
        return [wb, csv_logger]

    wb = WandbLogger(
        project=project,
        name=run_name,
        group=group,
        tags=tags or None,
        save_dir=str(run_dir),
        config=cfg_dict_to_save,
    )

    return [wb, csv_logger]


# ---->load Callback
def load_callbacks(cfg):
    """
    Create PyTorch Lightning callbacks for model checkpointing and learning rate monitoring.

    :param cfg: Configuration object with attributes:
                - log_path: Directory where checkpoints will be saved.
                - General.server: String indicating runtime mode ('train' for training).
                - Scheduler.name: Name of the LR scheduler (optional).
    :return: List of instantiated callback objects.
    """
    progress_every = int(getattr(cfg.General, "progress_log_every_n_steps", 50))
    is_interactive = sys.stdout.isatty() and sys.stderr.isatty()
    if is_interactive:
        Mycallbacks = [RichProgressBar()]
    else:
        Mycallbacks = [NohupProgressBar(log_every_n_steps=progress_every)]

    # Make output path
    output_path = cfg.log_path
    Path(output_path).mkdir(exist_ok=True, parents=True)

    if cfg.General.server == "train":
        Mycallbacks.append(
            ModelCheckpoint(
                monitor=None,
                dirpath=str(cfg.log_path),
                verbose=True,
                save_last=True,
                save_top_k=0,
            )
        )
        if cfg.Scheduler.name is not None:
            Mycallbacks.append(LearningRateMonitor())

    return Mycallbacks


def check_parameters_validity(transformation_params: dict | DictConfig) -> None:
    """
    Validate augmentation parameter ranges for discrete and continuous transforms.

    Discrete transforms must be float in [0,1] or int in {0,1}.
    Continuous transforms must be list/tuple of two numbers in [-0.5, 0.5] with min ≤ max.

    :param transformation_params: Dict mapping transform names to parameter values.
    :raises KeyError: If a transform name is unrecognized.
    :raises ValueError: If a parameter value is outside its valid range.
    """
    discrete_keys = TransformCategoryConstants.DISCRETE_TRANSFORMATIONS.value
    continuous_keys = TransformCategoryConstants.CONTINOUS_TRANSFORMATIONS.value

    for k, v in transformation_params.items():
        if k in discrete_keys:
            if not (
                (isinstance(v, float) and 0.0 <= v <= 1.0)
                or (isinstance(v, int) and v in (0, 1))
            ):
                raise ValueError(
                    f"Discrete transformation '{k}' must be between 0 and 1 (float or int 0/1), got {v!r}"
                )

        elif k in continuous_keys:
            # must be a list or tuple of two items
            if not isinstance(v, (list, tuple, ListConfig)) or len(v) != 2:
                raise ValueError(
                    f"Continuous transformation '{k}' must be a list or tuple of two numbers."
                )
            # each element must be float in [-0.5, 0.5]
            if not all(
                (
                    (isinstance(i, float) or (isinstance(i, int) and i == 0))
                    and -0.5 <= i <= 0.5
                )
                for i in v
            ):
                raise ValueError(
                    f"Continuous transformation '{k}' values must be floats between -0.5 and 0.5, got {v}"
                )
            # ensure min ≤ max
            if v[0] > v[1]:
                raise ValueError(
                    f"Continuous transformation '{k}' must have min <= max, got {v}"
                )

        else:
            raise KeyError(f"Transformation '{k}' is not recognized.")


def write_split_manifest(datamodule, log_dir, loggers=None) -> dict | None:
    """Dump a deterministic description of the train/val/test_holdout_staining splits.

    Writes ``<log_dir>/splits.json`` with slide membership, organ membership, per-split
    counts, and vocabularies. If a W&B logger is passed, also logs summary scalars and
    a per-slide assignment table so the run page shows the splits at a glance.

    Silently no-ops when the underlying dataset does not expose ``describe_splits`` —
    e.g. legacy image-based datasets — since the manifest is dataset-specific.

    Returns the manifest dict, or ``None`` if skipped.
    """
    import json as _json

    reference = (
        getattr(datamodule, "train_dataset", None)
        or getattr(datamodule, "test_dataset", None)
        or getattr(datamodule, "val_dataset", None)
    )
    if reference is None or not hasattr(reference, "describe_splits"):
        return None

    manifest = reference.describe_splits()

    # Enrich with per-split counts by asking each instantiated dataset for its summary.
    counts: dict = {}
    for attr, key in [
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "val"),  # "test" dataset is the organ-split val pool
        ("test_holdout_staining_dataset", "test_holdout_staining"),
    ]:
        ds = getattr(datamodule, attr, None)
        if ds is None or not hasattr(ds, "current_split_summary"):
            continue
        summary = ds.current_split_summary()
        counts.setdefault(key, {}).update(
            {
                "n_slides": summary["n_slides"],
                "n_pairs": summary["n_pairs"],
                "n_valid_patches": summary["n_valid_patches"],
                "n_items_per_epoch": summary["n_items"],
            }
        )
        if summary.get("valid_organs") is not None:
            manifest.setdefault("organs", {})[key] = summary["valid_organs"]
    manifest["counts"] = counts

    log_dir_path = Path(log_dir)
    ensure_dir(log_dir_path)
    manifest_path = log_dir_path / "splits.json"
    with manifest_path.open("w") as f:
        _json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Wrote split manifest to {manifest_path}")

    wb = None
    if loggers:
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                wb = lg.experiment
                break

    if wb is not None:
        try:
            import wandb

            scalars: dict = {}
            for split_name, c in counts.items():
                for metric, value in c.items():
                    scalars[f"split/{split_name}/{metric}"] = value
            scalars["split/holdout_stainings"] = (
                ", ".join(manifest["holdout_stainings"]) or "(none)"
            )
            wb.summary.update(scalars)

            rows: list[list] = []
            for split_name in ("train", "val", "test_holdout_staining"):
                for slide in manifest["slides"].get(split_name, []):
                    rows.append(
                        [
                            slide["slide_id"],
                            slide["staining"],
                            slide["scanner"],
                            split_name,
                        ]
                    )
            if rows:
                table = wandb.Table(
                    columns=["slide_id", "staining", "scanner", "split"],
                    data=rows,
                )
                wb.log({"splits/slides": table}, commit=False)
        except Exception as e:
            logger.warning(f"Could not log split manifest to W&B: {e}")

    return manifest


def print_run_summary(cfg, color_mode: str = "auto") -> None:
    """
    Colorized summary of cfg.
    """
    force = color_mode == "always"
    never = color_mode == "never"
    use_colors = (sys.stdout.isatty() or force) and not never

    RESET = "\033[0m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    WHITE = "\033[97m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    GREEN = "\033[92m" if use_colors else ""
    TOPLEVEL_COLOR = "\033[94m" if use_colors else ""

    SKIP_KEYS = {"config", "log_path", "load_loggers", "callbacks"}

    def section(title: str, indent: int = 0):
        pad = "  " * indent
        color = TOPLEVEL_COLOR if indent == 0 else CYAN
        print(f"{pad}{BOLD}{color}{title}:{RESET}")

    def kv(key: str, value, indent: int = 0, val_color: str = GREEN):
        pad = "  " * indent
        # Keep sequences as a single list representation; don't expand items.
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            rendered = pformat(list(value), width=120, compact=True)
        else:
            rendered = pformat(value, width=120, compact=True)
        print(f"{pad}{BOLD}{WHITE}{key}{RESET}: {val_color}{rendered}{RESET}")

    def as_mapping(obj):
        """Treat dicts or objects-with-__dict__ as mappings; preserve insertion order."""
        if isinstance(obj, Mapping):
            return True, obj
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            clean = {
                k: v for k, v in d.items() if not k.startswith("_") and not callable(v)
            }
            return True, clean
        return False, None

    def recurse(key, value, indent: int):
        if key in SKIP_KEYS:
            return
        is_map, mapobj = as_mapping(value)
        if is_map:
            section(key, indent)
            for k, v in mapobj.items():
                if k in SKIP_KEYS:
                    continue
                recurse(k, v, indent + 1)
        else:
            kv(key, value, indent)

    # Start at root
    is_root_map, root = as_mapping(cfg)
    if is_root_map:
        for top_key, top_val in root.items():
            if top_key in SKIP_KEYS:
                continue
            recurse(top_key, top_val, indent=0)
    else:
        kv("cfg", cfg, indent=0)
