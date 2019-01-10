from abc import ABC, abstractmethod
import os
from torch.utils.data import TensorDataset, DataLoader
import torch


class MusicDataset(ABC):
    """
    Abstract Base Class for music datasets
    """

    def __init__(self, cache_dir):
        self._tensor_dataset = None
        self.cache_dir = cache_dir

    @abstractmethod
    def iterator_gen(self):
        """

        return: Iterator over the dataset
        """
        pass

    @abstractmethod
    def make_tensor_dataset(self):
        """

        :return: TensorDataset
        """
        pass

    @abstractmethod
    def get_score_tensor(self, score):
        """

        :param score: music21 score object
        :return: torch tensor, with the score representation
                 as a tensor
        """
        pass
    
    @abstractmethod
    def get_metadata_tensor(self, score):
        """

        :param score: music21 score object
        :return: torch tensor, with the metadata representation
                 as a tensor
        """
        pass

    @abstractmethod
    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        """

        :param score: music21 score object
        :param semi-tone: int, +12 to -12, semitones to transpose 
        :return: Transposed score shifted by the semi-tone
        """
        pass

    @abstractmethod
    def extract_score_tensor_with_padding(self, 
                                          tensor_score, 
                                          start_tick, 
                                          end_tick):
        """

        :param tensor_score: torch tensor containing the score representation
        :param start_tick:
        :param end_tick:
        :return: tensor_score[:, start_tick: end_tick]
        with padding if necessary
        i.e. if start_tick < 0 or end_tick > tensor_score length
        """
        pass

    @abstractmethod
    def extract_metadata_with_padding(self, 
                                      tensor_metadata,
                                      start_tick, 
                                      end_tick):
        """

        :param tensor_metadata: torch tensor containing metadata
        :param start_tick:
        :param end_tick:
        :return:
        """
        pass

    @abstractmethod
    def empty_score_tensor(self, score_length):
        """
        
        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices 
        """
        pass 

    @abstractmethod
    def random_score_tensor(self, score_length):
        """

        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with random indices
        """
        pass

    @abstractmethod
    def tensor_to_score(self, tensor_score):
        """

        :param tensor_score: torch tensor, tensor representation
                             of the score
        :return: music21 score object
        """
        pass

    @property
    def tensor_dataset(self):
        """
        Loads or computes TensorDataset
        :return: TensorDataset
        """
        if self._tensor_dataset is None:
            if self.tensor_dataset_is_cached():
                print(f'Loading TensorDataset for {self.__repr__()}')
                self._tensor_dataset = torch.load(self.tensor_dataset_filepath)
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
        tensor_datasets_cache_dir = os.path.join(
            self.cache_dir,
            'tensor_datasets')
        if not os.path.exists(tensor_datasets_cache_dir):
            os.mkdir(tensor_datasets_cache_dir)
        fp = os.path.join(
            tensor_datasets_cache_dir,
            self.__repr__()
        )
        return fp

    @property
    def filepath(self):
        tensor_datasets_cache_dir = os.path.join(
            self.cache_dir,
            'datasets')
        if not os.path.exists(tensor_datasets_cache_dir):
            os.mkdir(tensor_datasets_cache_dir)
        return os.path.join(
            self.cache_dir,
            'datasets',
            self.__repr__()
        )

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1

        dataset = self.tensor_dataset
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                             int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
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
