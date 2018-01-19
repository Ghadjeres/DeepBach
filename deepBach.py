import argparse
import os

import pickle
from music21 import converter

from DeepBach.data_utils import part_to_inputs, initialization, BACH_DATASET, \
    pickled_dataset_path, SUBDIVISION
from DeepBach.model_manager import generation, load_models, train_models, \
    create_models
from DeepBach.metadata import *


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size used during training phase (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-s', '--steps_per_epoch',
                        help='number of steps per epoch (default: %(default)s)',
                        type=int, default=500)
    parser.add_argument('--validation_steps',
                        help='number of validation steps (default: %(default)s)',
                        type=int, default=20)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help='number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help='size of non recurrent hidden layers (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model name (default: %(default)s)',
                        choices=['deepbach', 'skip'],
                        type=str, default='deepbach')
    parser.add_argument('-i', '--num_iterations',
                        help='number of gibbs iterations (default: %(default)s)',
                        type=int, default=20000)
    parser.add_argument('-t', '--train', nargs='?',
                        help='train models for N epochs (default: 15)',
                        default=0, const=15, type=int)
    parser.add_argument('-p', '--parallel', nargs='?',
                        help='number of parallel updates (default: 16)',
                        type=int, const=16, default=1)
    parser.add_argument('--overwrite',
                        help='overwrite previously computed models',
                        action='store_true')
    parser.add_argument('-m', '--midi_file', nargs='?',
                        help='relative path to midi file',
                        type=str, const='datasets/god_save_the_queen.mid')
    parser.add_argument('-l', '--length',
                        help='length of unconstrained generation',
                        type=int, default=160)
    parser.add_argument('--ext',
                        help='extension of model name',
                        type=str, default='')
    parser.add_argument('-o', '--output_file', nargs='?',
                        help='path to output file',
                        type=str, default='',
                        const='generated_examples/example.mid')
    parser.add_argument('--dataset', nargs='?',
                        help='path to dataset folder',
                        type=str, default='')
    parser.add_argument('-r', '--reharmonization', nargs='?',
                        help='reharmonization of a melody from the corpus identified by its id',
                        type=int)
    args = parser.parse_args()
    print(args)

    # fixed set of metadatas to use when CREATING the dataset
    # Available metadatas:
    # metadatas = [FermataMetadatas(), KeyMetadatas(window_size=1),
    # TickMetadatas(SUBDIVISION), ModeMetadatas()]
    metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas(),
                 KeyMetadatas(window_size=1)]

    if args.ext:
        ext = '_' + args.ext
    else:
        ext = ''

    # datasets
    # set pickled_dataset argument
    if args.dataset:
        dataset_path = args.dataset
        pickled_dataset = pickled_dataset_path(dataset_path)
        print('pickled_dataset', pickled_dataset)
    else:
        dataset_path = None
        pickled_dataset = BACH_DATASET
    if not os.path.exists(pickled_dataset):
        initialization(dataset_path,
                       metadatas=metadatas,
                       voice_ids=[0, 1, 2, 3])

    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset,
             'rb'))
    NUM_VOICES = len(voice_ids
                     )
    num_pitches = list(map(len, index2notes))
    timesteps = args.timesteps
    batch_size = args.batch_size_train
    steps_per_epoch = args.steps_per_epoch
    validation_steps = args.validation_steps
    num_units_lstm = args.num_units_lstm
    model_name = args.name.lower() + ext
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    num_units_lstm = args.num_units_lstm
    num_dense = args.num_dense
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = None

    # when reharmonization
    if args.midi_file:
        melody = converter.parse(args.midi_file)
        melody = part_to_inputs(melody.parts[0], index2note=index2notes[0],
                                note2index=note2indexes[0])
        num_voices = NUM_VOICES - 1
        sequence_length = len(melody)
        # todo find a way to specify metadatas when reharmonizing a given melody
        chorale_metas = [metas.generate(sequence_length) for metas in
                         metadatas]

    elif args.reharmonization:
        melody = X[args.reharmonization][0, :]
        num_voices = NUM_VOICES - 1
        chorale_metas = X_metadatas[args.reharmonization]
    else:
        num_voices = NUM_VOICES
        melody = None
        # todo find a better way to set metadatas

        # chorale_metas = [metas[:sequence_length] for metas in X_metadatas[11]]
        chorale_metas = [metas.generate(sequence_length) for metas in
                         metadatas]

    num_iterations = args.num_iterations // batch_size_per_voice // num_voices
    parallel = batch_size_per_voice > 1
    train = args.train > 0
    num_epochs = args.train
    overwrite = args.overwrite

    if not os.path.exists('models/' + model_name + '_' + str(
                    NUM_VOICES - 1) + '.yaml'):
        create_models(model_name, create_new=overwrite,
                      num_units_lstm=num_units_lstm, num_dense=num_dense,
                      pickled_dataset=pickled_dataset, num_voices=num_voices,
                      metadatas=metadatas, timesteps=timesteps)
    if train:
        models = train_models(model_name=model_name,
                              steps_per_epoch=steps_per_epoch,
                              num_epochs=num_epochs,
                              validation_steps=validation_steps,
                              timesteps=timesteps,
                              pickled_dataset=pickled_dataset,
                              num_voices=NUM_VOICES, metadatas=metadatas,
                              batch_size=batch_size)
    else:
        models = load_models(model_name, num_voices=NUM_VOICES)
        temperature = 1.
        timesteps = int(models[0].input[0]._keras_shape[1])

        seq = generation(model_base_name=model_name, models=models,
                         timesteps=timesteps,
                         melody=melody, initial_seq=None, temperature=temperature,
                         chorale_metas=chorale_metas, parallel=parallel,
                         batch_size_per_voice=batch_size_per_voice,
                         num_iterations=num_iterations,
                         sequence_length=sequence_length,
                         output_file=output_file,
                         pickled_dataset=pickled_dataset)


if __name__ == '__main__':
    main()
