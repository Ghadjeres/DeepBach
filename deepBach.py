"""
@author: Gaetan Hadjeres

"""
 
import os
import click
 
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
 
from DeepBach.model_manager import DeepBach
 
 
@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=512,
              help='training batch size')
@click.option('--num_epochs', default=30,
              help='number of training epochs')
@click.option('--lr', default=1e-3,
              help='initial learning rate for Adam')
@click.option('--lr_patience', default=3,
              help='ReduceLROnPlateau patience (epochs without val improvement)')
@click.option('--lr_factor', default=0.5,
              help='ReduceLROnPlateau reduction factor')
@click.option('--train', 'do_train', is_flag=True,
              help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         lr,
         lr_patience,
         lr_factor,
         do_train,
         num_iterations,
         sequence_length_ticks,
         ):
    dataset_manager = DatasetManager()
 
    metadatas = [
        FermataMetadata(),
        TickMetadata(subdivision=4),
        KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
    )
    dataset = bach_chorales_dataset
 
    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )
 
    if do_train:
        print(f'Training {dataset.num_voices} voice models for {num_epochs} epochs '
              f'(batch_size={batch_size}, lr={lr}) ...')
        deepbach.cuda()
        deepbach.train(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )
        print('All voice models saved (best val-loss checkpoints).')
        missing = []
        for vm in deepbach.voice_models:
            load_path = os.path.join(vm.models_dir, vm.__repr__())
            if not os.path.exists(load_path):
                missing.append(load_path)
 
        if missing:
            print('\n[ERROR] Pretrained model weights not found:')
            for p in missing:
                print(f'  {p}')
            print('\nTrain from scratch first with:')
            print('  python deepBach.py --train --num_epochs 30 --batch_size 512\n')
            raise SystemExit(1)
 
        deepbach.load()
        deepbach.cuda()
 
        print('Generation')
        score, tensor_chorale, tensor_metadata = deepbach.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
        )
        score.show('txt')
        score.show()
 

DO_TRAIN = True
 
if __name__ == '__main__':
    args = [
        '--num_epochs',            '50',
        '--batch_size',            '512',
        '--lr',                    '1e-3',
        '--lr_patience',           '3',
        '--lr_factor',             '0.5',
        '--num_iterations',        '500',
        '--sequence_length_ticks', '64',
    ]
    if DO_TRAIN:
        args.append('--train')
 
    main(standalone_mode=True, args=args)
