from abc import ABC, abstractmethod
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
import torch

# BUG FIX: num_workers > 0 on Windows requires the 'spawn' start method,
# which fails when running as a package (WinError 1455).
# Linux/macOS use 'fork' which supports > 0.
_TRAIN_NUM_WORKERS = 0 if sys.platform == 'win32' else 4
_USE_PIN_MEMORY = torch.cuda.is_available()


class MusicDataset(ABC):
    """
    Abstract Base Class for music datasets
    """

    def __init__(self, cache_dir):
        self._tensor_dataset = None
        self.cache_dir = cache_dir

    @abstractmethod
    def iterator_gen(self): pass

    @abstractmethod
    def make_tensor_dataset(self): pass

    @abstractmethod
    def get_score_tensor(self, score): pass

    @abstractmethod
    def get_metadata_tensor(self, score): pass

    @abstractmethod
    def transposed_score_and_metadata_tensors(self, score, semi_tone): pass

    @abstractmethod
    def extract_score_tensor_with_padding(self, tensor_score, start_tick, end_tick): pass

    @abstractmethod
    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick): pass

    @abstractmethod
    def empty_score_tensor(self, score_length): pass

    @abstractmethod
    def random_score_tensor(self, score_length): pass

    @abstractmethod
    def tensor_to_score(self, tensor_score): pass

    @property
    def tensor_dataset(self):
        """Loads or builds (and caches) the TensorDataset."""
        if self._tensor_dataset is None:
            if self.tensor_dataset_is_cached():
                print(f'Loading TensorDataset for {self.__repr__()}')
                # BUG FIX: weights_only=False required for TensorDataset objects
                # (which are not pure-tensor dicts).  Without this flag PyTorch
                # 2.x emits a FutureWarning; a later release will make it an error.
                self._tensor_dataset = torch.load(
                    self.tensor_dataset_filepath, weights_only=False)
            else:
                print(f'Creating {self.__repr__()} TensorDataset'
                      f' since it is not cached')
                self._tensor_dataset = self.make_tensor_dataset()
                torch.save(self._tensor_dataset, self.tensor_dataset_filepath)
                print(f'TensorDataset for {self.__repr__()} '
                      f'saved in {self.tensor_dataset_filepath}')
        return self._tensor_dataset

    @tensor_dataset.setter
    def tensor_dataset(self, value):
        self._tensor_dataset = value

    def tensor_dataset_is_cached(self):
        return os.path.exists(self.tensor_dataset_filepath)

    @property
    def tensor_dataset_filepath(self):
        tensor_datasets_cache_dir = os.path.join(self.cache_dir, 'tensor_datasets')
        # BUG FIX: makedirs instead of mkdir — creates full path if missing
        os.makedirs(tensor_datasets_cache_dir, exist_ok=True)
        return os.path.join(tensor_datasets_cache_dir, self.__repr__())

    @property
    def filepath(self):
        datasets_cache_dir = os.path.join(self.cache_dir, 'datasets')
        os.makedirs(datasets_cache_dir, exist_ok=True)
        return os.path.join(datasets_cache_dir, self.__repr__())

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """
        Returns (train, val, eval) DataLoaders.

        GPU SPEEDUP: pin_memory lets the DataLoader pre-page tensors into
        pinned (non-pageable) host memory so the GPU DMA engine can transfer
        them without CPU involvement, overlapping transfer with compute.
        """
        assert sum(split) < 1

        dataset = self.tensor_dataset
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset   = TensorDataset(*dataset[int(a * num_examples):
                                               int((a + b) * num_examples)])
        eval_dataset  = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_TRAIN_NUM_WORKERS,
            pin_memory=_USE_PIN_MEMORY,
            persistent_workers=(_TRAIN_NUM_WORKERS > 0),
            drop_last=True,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl
