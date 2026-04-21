import importlib
import inspect
import math
import multiprocessing as mp

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Subset


class DataInterface(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for dynamic dataset instantiation and standardized dataloader creation.

    This module loads a dataset class by name, instantiates training, validation, and testing datasets,
    and provides DataLoaders with consistent behavior (shuffling, batching, and worker management).

    :param train_batch_size: Batch size for training and validation loaders.
    :param train_num_workers: Number of worker processes for training and validation.
    :param test_batch_size: Batch size for testing loader (default often 1 for patch-based evaluation).
    :param test_num_workers: Number of worker processes for testing loader.
    :param shuffle_data: Whether to shuffle training data each epoch.
    :param dataset_name: Name of the dataset module under 'datasets' (e.g., 'my_dataset').
    :param kwargs: Additional keyword arguments passed to the dataset constructor (e.g., data paths, transforms).
    """

    def __init__(
        self,
        train_batch_size=64,
        train_num_workers=8,
        test_batch_size=1,
        test_num_workers=1,
        test_max_samples=10000,
        shuffle_data=True,
        dataset_name=None,
        train_dataloader_cfg=None,
        test_dataloader_cfg=None,
        **kwargs,
    ):
        super().__init__()
        # DataLoader parameters
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.test_max_samples = test_max_samples
        self.dataset_name = dataset_name
        self.shuffle = shuffle_data
        self.train_loader_cfg = train_dataloader_cfg or {}
        self.test_loader_cfg = test_dataloader_cfg or {}
        self.kwargs = kwargs
        self.load_data_module()

    def _build_loader_kwargs(self, num_workers: int, loader_cfg: dict) -> dict:
        """
        Build DataLoader kwargs with safe defaults and worker-aware options.

        :param num_workers: Number of workers for this loader.
        :param loader_cfg: Optional loader configuration dictionary.
        :return: Keyword arguments for torch.utils.data.DataLoader.
        """
        pin_memory = bool(loader_cfg.get("pin_memory", True))
        persistent_workers = bool(loader_cfg.get("persistent_workers", True))
        drop_last = bool(loader_cfg.get("drop_last", False))
        prefetch_factor = loader_cfg.get("prefetch_factor", 4)

        kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)
            # Avoid fork-after-thread deadlocks: timm/HF/OpenMP loaded in the
            # parent leave locked mutexes that fork() would duplicate.
            kwargs["multiprocessing_context"] = mp.get_context("forkserver")

        return kwargs

    def setup(self, stage: str = None) -> None:
        """
        Instantiate datasets for the given stage.

        :param stage: One of 'fit' or 'test'.
                      'fit' loads train and validation splits;
                      'test' loads test split only.
        :raises ValueError: If stage is not 'fit' or 'test'.
        """
        if stage == "fit":
            self.train_dataset = self.instancialize(state="train")
            self.val_dataset = self.instancialize(state="val")
            self.test_holdout_staining_dataset = self._maybe_build_holdout_staining_dataset()
        elif stage == "test":
            # Organ-split "test" = same pool as val (regular slides, held-out organs).
            self.test_dataset = self.instancialize(state="val")
            self.test_holdout_staining_dataset = self._maybe_build_holdout_staining_dataset()
        else:
            raise ValueError(
                f"Invalid stage provided: {stage}. Must be either train' or 'test'."
            )

    def _maybe_build_holdout_staining_dataset(self):
        """Instantiate the held-out-staining test dataset if the dataset config requests it.

        Returns ``None`` when ``Data.holdout_stainings`` is empty or the dataset class
        does not support this state (e.g. legacy image-based datasets).
        """
        holdout_cfg = self.kwargs.get("dataset_cfg", None)
        stainings = None
        if holdout_cfg is not None:
            stainings = (
                holdout_cfg.get("holdout_stainings", None)
                if isinstance(holdout_cfg, dict)
                else getattr(holdout_cfg, "holdout_stainings", None)
            )
        if not stainings:
            return None
        try:
            return self.instancialize(state="test_holdout_staining")
        except ValueError as e:
            # Dataset may not support this state or the staining may not match.
            import logging as _logging
            _logging.warning(f"Skipping test_holdout_staining dataset: {e}")
            return None

    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training dataset.

        :return: DataLoader yielding batches of training data.
        """
        loader_kwargs = self._build_loader_kwargs(
            self.train_num_workers, self.train_loader_cfg
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            **loader_kwargs,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation dataset.

        :return: DataLoader yielding batches of validation data.
        """
        loader_kwargs = self._build_loader_kwargs(
            self.test_num_workers, self.test_loader_cfg
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    def test_holdout_staining_dataloader(self):
        """
        DataLoader for the held-out-staining test set (all patches of held-out stainings).

        Returns ``None`` when no holdout was configured or the dataset was not built.
        """
        dataset = getattr(self, "test_holdout_staining_dataset", None)
        if dataset is None:
            return None
        loader_kwargs = self._build_loader_kwargs(
            self.test_num_workers, self.test_loader_cfg
        )
        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    def test_dataloader(self):
        """
        Create DataLoader for testing dataset.

        If `self.test_max_samples` is None or the dataset explicitly requests using
        all test samples, the full test dataset is returned. Otherwise, this loader
        concatenates and truncates to a fixed number of samples.

        :return: DataLoader yielding test samples without shuffling.
        """
        if self.test_max_samples is None or getattr(
            self.test_dataset, "use_all_test_samples", False
        ):
            loader_kwargs = self._build_loader_kwargs(
                self.test_num_workers, self.test_loader_cfg
            )
            return DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                **loader_kwargs,
            )

        # Determine repetition factor to reach 10,000 samples
        L = len(self.test_dataset)
        repeats = math.ceil(self.test_max_samples / L)

        # Concatenate and truncate to the requested number of examples
        long_ds = ConcatDataset([self.test_dataset] * repeats)
        truncated = Subset(long_ds, list(range(self.test_max_samples)))

        # DataLoader configured to return exactly `test_max_samples` random patches.
        # Each call to __getitem__ draws a fresh random patch from each WSI.
        # By concatenating the dataset multiple times, we ensure we sample
        # enough unique patches with different transformation parameters,
        # then truncate to the first requested number of examples.

        loader_kwargs = self._build_loader_kwargs(
            self.test_num_workers, self.test_loader_cfg
        )
        return DataLoader(
            truncated,
            batch_size=self.test_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    def load_data_module(self) -> None:
        """
        Dynamically import the dataset class from 'datasets' package based on dataset_name.

        :raises ValueError: If the module or class cannot be found.
        """
        dataset_class_name = "".join(
            [i.capitalize() for i in (self.dataset_name).split("_")]
        )
        try:
            self.data_module = getattr(
                importlib.import_module(f"datasets.{self.dataset_name}"),
                dataset_class_name,
            )
        except:
            raise ValueError("Invalid Dataset File Name or Invalid Class Name!")

    def instancialize(self, **other_args):
        """
        Instantiate the dataset class with proper split state and parameters.

        :param state: Split identifier, typically 'train', 'val', or 'test'.
        :param override_kwargs: Additional arguments to override defaults.
        :return: Instantiated dataset object.
        """
        signature = inspect.signature(self.data_module.__init__)
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
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
