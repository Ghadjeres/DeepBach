import os

import music21
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.helpers import ShortChoraleIteratorGen
from DatasetManager.metadata import TickMetadata, \
    FermataMetadata, \
    KeyMetadata
from DatasetManager.music_dataset import MusicDataset

# Basically, all you have to do to use an existing dataset is to
# add an entry in the all_datasets variable
# and specify its base class and which music21 objects it uses
# by giving an iterator over music21 scores

all_datasets = {
    'bach_chorales':
        {
            'dataset_class_name': ChoraleDataset,
            'corpus_it_gen':      music21.corpus.chorales.Iterator
        },
    'bach_chorales_test':
        {
            'dataset_class_name': ChoraleDataset,
            'corpus_it_gen':      ShortChoraleIteratorGen()
        },
}


class DatasetManager:
    def __init__(self):
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_dir = os.path.join(self.package_dir,
                                      'dataset_cache')
        # create cache dir if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def get_dataset(self, name: str, **dataset_kwargs) -> MusicDataset:
        if name in all_datasets:
            return self.load_if_exists_or_initialize_and_save(
                name=name,
                **all_datasets[name],
                **dataset_kwargs
            )
        else:
            print('Dataset with name {name} is not registered in all_datasets variable')
            raise ValueError

    def load_if_exists_or_initialize_and_save(self,
                                              dataset_class_name,
                                              corpus_it_gen,
                                              name,
                                              **kwargs):
        """

        :param dataset_class_name:
        :param corpus_it_gen:
        :param name:
        :param kwargs: parameters specific to an implementation
        of MusicDataset (ChoraleDataset for instance)
        :return:
        """
        kwargs.update(
            {'name':          name,
             'corpus_it_gen': corpus_it_gen,
             'cache_dir':     self.cache_dir
             })
        dataset = dataset_class_name(**kwargs)
        if os.path.exists(dataset.filepath):
            print(f'Loading {dataset.__repr__()} from {dataset.filepath}')
            dataset = torch.load(dataset.filepath)
            dataset.cache_dir = self.cache_dir
            print(f'(the corresponding TensorDataset is not loaded)')
        else:
            print(f'Creating {dataset.__repr__()}, '
                  f'both tensor dataset and parameters')
            # initialize and force the computation of the tensor_dataset
            # first remove the cached data if it exists
            if os.path.exists(dataset.tensor_dataset_filepath):
                os.remove(dataset.tensor_dataset_filepath)
            # recompute dataset parameters and tensor_dataset
            # this saves the tensor_dataset in dataset.tensor_dataset_filepath
            tensor_dataset = dataset.tensor_dataset
            # save all dataset parameters EXCEPT the tensor dataset
            # which is stored elsewhere
            dataset.tensor_dataset = None
            torch.save(dataset, dataset.filepath)
            print(f'{dataset.__repr__()} saved in {dataset.filepath}')
            dataset.tensor_dataset = tensor_dataset
        return dataset


if __name__ == '__main__':
    # Usage example

    dataset_manager = DatasetManager()
    subdivision = 4
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]

    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales_test',
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=subdivision
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = bach_chorales_dataset.data_loaders(
        batch_size=128,
        split=(0.85, 0.10)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))
