import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import DataInterface
from datasets.scorpion_dataset import ScorpionDataset, ScorpionPrefeaturesDataset
from models import ModelInterface
from models.foundation_models import get_foundation_model
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from utils.utils import (
    load_callbacks,
    load_loggers,
    print_run_summary,
    read_yaml,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and Test Pipeline with PyTorch Lightning"
    )
    parser.add_argument(
        "--stage",
        choices=["train", "test"],
        default="train",
        help="Execution stage: train or test",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Resume logging to an existing W&B run (test stage only)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Exact checkpoint to evaluate (test stage only); skips the glob search",
    )
    return parser.parse_args()


def configure(cfg) -> None:
    """
    Set seeds, logging, callbacks, and normalize and general settings.
    """
    pl.seed_everything(cfg.General.seed)
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)

    # General defaults
    cfg.General.devices = cfg.General.devices or 1
    cfg.General.accelerator = getattr(cfg.General, "accelerator", "gpu")
    cfg.General.strategy = getattr(cfg.General, "strategy", "auto")
    matmul_precision = getattr(cfg.General, "matmul_precision", "medium")
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cudnn.benchmark = bool(getattr(cfg.General, "cudnn_benchmark", True))

    return loggers, callbacks


def build_datamodule(cfg) -> pl.LightningDataModule:
    """
    Instantiate the LightningDataModule based on configuration.

    :param cfg: Configuration object containing Data and Model parameters.
    :return: Configured DataInterface instance.
    """
    return DataInterface(
        train_batch_size=cfg.Data.train_dataloader.batch_size,
        train_num_workers=cfg.Data.train_dataloader.num_workers,
        test_batch_size=cfg.Data.test_dataloader.batch_size,
        test_num_workers=cfg.Data.test_dataloader.num_workers,
        train_dataloader_cfg=cfg.Data.train_dataloader,
        test_dataloader_cfg=cfg.Data.test_dataloader,
        test_max_samples=cfg.Data.get("test_max_samples", 10000),
        dataset_name=cfg.Data.dataset_name,
        shuffle_data=cfg.Data.shuffle_data,
        transforms=cfg.Data.Transforms,
        dataset_cfg=cfg.Data,
        general=cfg.General,
        model=cfg.Model,
        foundation_model=cfg.Foundation_model,
    )


def build_model(cfg) -> pl.LightningModule:
    """
    Instantiate the LightningModule for training or testing.

    :param cfg: Configuration object containing Model, Loss, Optimizer, Scheduler, and Data settings.
    :return: Configured ModelInterface instance.
    """
    return ModelInterface(
        general=cfg.General,
        model=cfg.Model,
        loss=cfg.Loss,
        optimizer=cfg.Optimizer,
        scheduler=cfg.Scheduler,
        transforms=cfg.Data.Transforms,
        data=cfg.Data,
        log=cfg.log_path,
        foundation_model=cfg.Foundation_model,
        histaug=cfg.get("HistAug", None),
    )


def build_trainer(cfg, loggers, callbacks) -> Trainer:
    """
    Create a PyTorch Lightning Trainer using the configuration.

    :param cfg: Configuration object with keys:
                - load_loggers: list of loggers
                - callbacks: list of callbacks
                - General.epochs, devices, accelerator, strategy, precision, grad_acc
    :return: Configured Trainer instance.
    """

    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.General.epochs,
        devices=cfg.General.devices,
        accelerator=cfg.General.accelerator,
        strategy=cfg.General.strategy,
        precision=cfg.General.precision,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        accumulate_grad_batches=cfg.General.get("grad_acc", 1),
        num_sanity_val_steps=0,
    )


def build_scorpion_eval_loader(cfg, reference_dataset):
    """
    Build a DataLoader for SCORPION evaluation using the training vocabulary.

    Maps SCORPION scanner names to the PLISM integer IDs from ``reference_dataset``
    via the optional ``cfg.Scorpion.scanner_name_map`` (e.g. ``Philips: P``).

    When ``cfg.Scorpion.features_root`` is set, uses pre-extracted feature ``.npy``
    files (no foundation model required).  Otherwise falls back to the image-based
    ``ScorpionDataset`` which loads JPEG patches and runs feature extraction on the fly.

    Returns ``None`` when the ``Scorpion`` section is absent or fewer than two
    SCORPION scanners can be mapped into the training vocabulary.
    """
    scorpion_cfg = cfg.get("Scorpion", None)
    if not scorpion_cfg:
        return None

    plism_scanner_to_id: dict = getattr(reference_dataset, "scanner_to_id", {})
    if not plism_scanner_to_id:
        logging.warning("SCORPION eval skipped: training dataset has no scanner_to_id.")
        return None

    raw_name_map = scorpion_cfg.get("scanner_name_map", None) or {}
    scanner_name_map = {str(k): str(v) for k, v in raw_name_map.items()}

    # Map each SCORPION scanner name to its PLISM integer ID, skipping unknowns.
    scorpion_scanner_to_id: dict = {}
    for sc_name in ["AT2", "DP200", "GT450", "P1000", "Philips"]:
        plism_name = scanner_name_map.get(sc_name, sc_name)
        if plism_name in plism_scanner_to_id:
            scorpion_scanner_to_id[sc_name] = plism_scanner_to_id[plism_name]

    if len(scorpion_scanner_to_id) < 2:
        logging.warning(
            "SCORPION eval skipped: fewer than 2 SCORPION scanner names map into "
            f"the PLISM vocabulary. Mapped: {scorpion_scanner_to_id}. "
            "Add entries to cfg.Scorpion.scanner_name_map if needed."
        )
        return None

    test_cfg = cfg.Data.test_dataloader

    features_root = scorpion_cfg.get("features_root", None)
    if features_root:
        # Pre-extracted features — no foundation model loading required.
        dataset = ScorpionPrefeaturesDataset(
            dataset_cfg={
                "features_root": str(features_root),
                "scanner_to_id": scorpion_scanner_to_id,
            }
        )
        logging.info(
            f"SCORPION prefeatures eval: {len(dataset)} items, "
            f"scanners={list(scorpion_scanner_to_id)}"
        )
    else:
        # Image-based dataset: extract features on the fly.
        fm_cfg = scorpion_cfg.get("foundation_model", None) or cfg.Foundation_model
        dataset = ScorpionDataset(
            dataset_cfg={
                "data_path": str(scorpion_cfg.data_path),
                "scanner_to_id": scorpion_scanner_to_id,
            },
            foundation_model=fm_cfg,
        )
        logging.info(
            f"SCORPION eval: {len(dataset)} pairs, scanners={list(scorpion_scanner_to_id)}"
        )

    return DataLoader(
        dataset,
        batch_size=int(test_cfg.batch_size),
        shuffle=False,
        num_workers=int(test_cfg.num_workers),
        pin_memory=bool(test_cfg.get("pin_memory", True)),
    )


def run_scorpion_eval(
    trainer: Trainer,
    model: pl.LightningModule,
    cfg,
    reference_dataset,
) -> None:
    """
    Evaluate the model on SCORPION if ``cfg.Scorpion`` is configured.

    For prefeatures models whose ``feature_extractor`` is ``None``, temporarily
    loads the foundation model specified in ``cfg.Scorpion.foundation_model``
    (or ``cfg.Foundation_model``) so that raw SCORPION JPEG patches can be
    embedded before passing through the scanner-transfer head.
    """
    scorpion_loader = build_scorpion_eval_loader(cfg, reference_dataset)
    if scorpion_loader is None:
        return
    logging.info("Preparings SCORPION evaluation ...")

    scorpion_cfg = cfg.get("Scorpion", None)

    # No foundation model needed when the loader uses pre-extracted features.
    uses_prefeatures = bool(scorpion_cfg and scorpion_cfg.get("features_root", None))
    if uses_prefeatures:
        fm_name = None
    else:
        fm_cfg = (
            scorpion_cfg.get("foundation_model", None) if scorpion_cfg else None
        ) or cfg.Foundation_model
        fm_name = (
            fm_cfg.get("name")
            if isinstance(fm_cfg, dict)
            else getattr(fm_cfg, "name", None)
        )

    needs_temp_fm = model.feature_extractor is None and bool(fm_name)
    if needs_temp_fm:
        logging.info(f"Loading {fm_name} temporarily for SCORPION evaluation …")
        model.feature_extractor = get_foundation_model(fm_cfg, device=model.device)

    model._test_log_prefix = "scorpion"
    try:
        logging.info("Starting SCORPION evaluation ...")
        trainer.test(model=model, dataloaders=scorpion_loader)
    finally:
        model._test_log_prefix = "test"
        if needs_temp_fm:
            model.feature_extractor = None


def run_holdout_staining_eval(
    trainer: Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> None:
    """
    Evaluate on the held-out-staining test set if the datamodule built one.

    Logs metrics under the ``test_holdout_staining/`` namespace (reuses the
    ``_test_log_prefix`` routing that SCORPION also uses).
    """
    loader = datamodule.test_holdout_staining_dataloader()
    if loader is None:
        return
    model._test_log_prefix = "test_holdout_staining"
    try:
        logging.info("Starting held-out-staining evaluation ...")
        trainer.test(model=model, dataloaders=loader)
    finally:
        model._test_log_prefix = "test"


def run_training(
    trainer: Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg,
    reference_dataset=None,
) -> None:
    """
    Run the training loop, then evaluate the best checkpoint on the validation set.

    After fit completes, runs trainer.test on the val dataloader so that detailed
    test-phase metrics (bootstrap CI, per-group cosine breakdowns, relative improvement)
    are logged under "test/" for the final model state.  If ``cfg.Scorpion`` is
    configured, also runs a SCORPION evaluation pass.

    :param trainer: Trainer instance.
    :param model: ModelInterface instance.
    :param datamodule: DataInterface instance (must have val_dataset from stage='fit').
    :param cfg: Configuration object possibly containing resume checkpoint path.
    :param reference_dataset: Training dataset, used to extract the scanner/staining vocab
        for SCORPION evaluation.
    """
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.General.get("ckpt_path_resume_training", None),
    )

    # Test 1: organ-split validation (same pool as val, but routed through test_step
    # for bootstrap CI, per-pair heatmaps, etc.).
    trainer.test(model=model, dataloaders=datamodule.val_dataloader())
    # Test 2: held-out-staining (e.g. GVH) — only runs if configured.
    run_holdout_staining_eval(trainer, model, datamodule)
    # Test 3: SCORPION — only runs if cfg.Scorpion is configured.
    run_scorpion_eval(trainer, model, cfg, reference_dataset)


def run_testing(
    trainer: Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg,
    reference_dataset=None,
) -> None:
    """
    Run the testing loop on all checkpoints in the log directory.

    For each .ckpt file (excluding 'temp'), load the model and call Trainer.test.
    If ``cfg.Scorpion`` is configured, also runs a SCORPION evaluation pass per
    checkpoint.

    :param trainer: Trainer instance.
    :param model: Base ModelInterface class (used to load from checkpoint).
    :param datamodule: DataInterface instance.
    :param cfg: Configuration object with attribute log_path.
    :param reference_dataset: Dataset used to extract the scanner/staining vocab
        for SCORPION evaluation.
    """
    override = getattr(cfg.General, "ckpt_path_override", None)
    if override:
        ckpt_paths = [Path(override)]
    else:
        ckpt_dir = Path(cfg.log_path)
        ckpt_paths = [p for p in ckpt_dir.glob("*/*.ckpt") if "temp" not in p.name]
    for ckpt in ckpt_paths:
        logging.info(f"Testing checkpoint: {ckpt}")
        test_model = model.__class__.load_from_checkpoint(
            checkpoint_path=ckpt, cfg=cfg, strict=False, weights_only=False
        )
        test_model.current_model_path = ckpt
        trainer.test(model=test_model, datamodule=datamodule)
        run_holdout_staining_eval(trainer, test_model, datamodule)
        run_scorpion_eval(trainer, test_model, cfg, reference_dataset)


def main() -> None:
    """
    Main entrypoint: parse arguments, load config, build components, and execute train or test.
    """
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.config = str(args.config)
    cfg.General.server = args.stage
    cfg.General.wandb_run_id = args.wandb_run_id
    cfg.General.ckpt_path_override = str(args.ckpt_path) if args.ckpt_path else None

    loggers, callbacks = configure(cfg)
    print_run_summary(cfg)
    datamodule = build_datamodule(cfg)

    # Eagerly set up the datamodule so the model can pick up dataset-derived values
    # (e.g. scanner/staining vocab sizes for scanner-transfer). Lightning would call
    # setup() later, but we need the values before build_model(). Subsequent calls
    # by Lightning are no-ops as long as setup is idempotent (it is: it just rebuilds
    # dataset objects from the same config).
    datamodule.setup("fit" if cfg.General.server == "train" else "test")
    reference_ds = getattr(datamodule, "train_dataset", None) or getattr(
        datamodule, "test_dataset", None
    )
    if hasattr(reference_ds, "scanner_vocab_size"):
        cfg.Model.scanner_vocab_size = reference_ds.scanner_vocab_size
    if hasattr(reference_ds, "staining_vocab_size"):
        cfg.Model.staining_vocab_size = reference_ds.staining_vocab_size
    if hasattr(reference_ds, "scanner_to_id"):
        cfg.Model.scanner_names = sorted(
            reference_ds.scanner_to_id, key=reference_ds.scanner_to_id.get
        )
    if hasattr(reference_ds, "staining_to_id"):
        cfg.Model.staining_names = sorted(
            reference_ds.staining_to_id, key=reference_ds.staining_to_id.get
        )

    model = build_model(cfg)
    trainer = build_trainer(cfg, loggers, callbacks)

    # Dump train / val / test_holdout_staining membership so the run is reproducible
    # and each split's contents are visible in the W&B run page.
    write_split_manifest(datamodule, cfg.log_path, loggers=loggers)

    if cfg.General.server == "train":
        run_training(trainer, model, datamodule, cfg, reference_dataset=reference_ds)
    else:
        run_testing(trainer, model, datamodule, cfg, reference_dataset=reference_ds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
