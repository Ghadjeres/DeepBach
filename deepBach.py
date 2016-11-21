"""
Created on 15 mars 2016

@author: Gaetan Hadjeres
"""
import argparse
import os
import pickle

import numpy as np
from keras.engine import Input
from keras.layers import LSTM, Dense, TimeDistributed, merge, Reshape
from keras.layers.core import Dropout
from keras.models import Model
from keras.models import model_from_json, model_from_yaml
from music21 import midi, converter
from tqdm import tqdm

from data_utils import BEAT_SIZE, \
    seq_to_stream, BITS_FERMATA, \
    part_to_list, generator_from_raw_dataset, RAW_DATASET, all_features, \
    F_INDEX, to_fermata, seq_to_stream_slur, fermata_melody_to_fermata, \
    seqs_to_stream, as_ps_to_as_pas, initialization


def generation(model_base_name, models, min_pitches, max_pitches, melody=None,
               initial_seq=None, temperature=1.0,
               fermatas_melody=None, parallel=False, batch_size_per_voice=8, num_iterations=None, sequence_length=160, output_file=None):
    # Test by generating a sequence

    if parallel:
        seq = parallelGibbs(models=models, model_base_name=model_base_name,
                            melody=melody, fermatas_melody=fermatas_melody,
                            num_iterations=num_iterations, sequence_length=sequence_length,
                            min_pitches=min_pitches, max_pitches=max_pitches, temperature=temperature,
                            initial_seq=initial_seq, batch_size_per_voice=batch_size_per_voice,
                            parallel_updates=True)

    else:
        seq = gibbs(models=models, model_base_name=model_base_name,
                    melody=melody, fermatas_melody=fermatas_melody,
                    num_iterations=num_iterations, sequence_length=sequence_length,
                    min_pitches=min_pitches, max_pitches=max_pitches, temperature=temperature,
                    initial_seq=initial_seq)

    # convert
    score = seq_to_stream_slur(np.transpose(seq, axes=(1, 0)),
                               min_pitches, max_pitches
                               )

    # save as MIDI file
    if output_file:
        mf = midi.translate.music21ObjectToMidiFile(score)
        mf.open(output_file, 'wb')
        mf.write()
        mf.close()
        print("File " + output_file + " written")


    score.show()
    return seq


def mlp(num_features_lr, num_features_c, num_pitches, num_hidden=200):
    """
    MLP model
    :param num_hidden:
    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output

    """
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    beat = Input(shape=(BEAT_SIZE,), name='beat')
    beats_right = Input(shape=(timesteps, BEAT_SIZE), name='beats_right')
    beats_left = Input(shape=(timesteps, BEAT_SIZE), name='beats_left')
    fermatas_left = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_left')
    fermatas_right = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_right')
    central_fermata = Input(shape=(BITS_FERMATA,), name='central_fermata')

    predictions_left = (merge((left_features,
                               beats_left,
                               fermatas_left),
                              mode='concat'))
    predictions_right = (merge((right_features,
                                beats_right,
                                fermatas_right),
                               mode='concat'))

    predictions_center = merge((central_features, beat,
                                central_fermata), mode='concat')

    predictions = merge((Reshape(((num_features_lr + BEAT_SIZE + BITS_FERMATA) * timesteps,))(predictions_left),
                         predictions_center,
                         Reshape(((num_features_lr + BEAT_SIZE + BITS_FERMATA) * timesteps,))(predictions_right)),
                        mode='concat')

    predictions = Dense(num_hidden, activation='relu', name='hidden_layer')(predictions)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features,
                         beat, beats_left, beats_right,
                         fermatas_left, fermatas_right, central_fermata],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model


def maxEnt(num_features_lr, num_features_c, num_pitches):
    """
    Returns MaxEnt Model

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    """
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    beat = Input(shape=(BEAT_SIZE,), name='beat')
    beats_right = Input(shape=(timesteps, BEAT_SIZE), name='beats_right')
    beats_left = Input(shape=(timesteps, BEAT_SIZE), name='beats_left')
    fermatas_left = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_left')
    fermatas_right = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_right')
    central_fermata = Input(shape=(BITS_FERMATA,), name='central_fermata')

    predictions_left = (merge((left_features,
                               beats_left,
                               fermatas_left),
                              mode='concat'))
    predictions_right = (merge((right_features,
                                beats_right,
                                fermatas_right),
                               mode='concat'))

    predictions_center = merge((central_features, beat,
                                central_fermata), mode='concat')

    predictions = merge((Reshape(((num_features_lr + BEAT_SIZE + BITS_FERMATA) * timesteps,))(predictions_left),
                         predictions_center,
                         Reshape(((num_features_lr + BEAT_SIZE + BITS_FERMATA) * timesteps,))(predictions_right)),
                        mode='concat')

    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features,
                         beat, beats_left, beats_right,
                         fermatas_left, fermatas_right, central_fermata],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model


def deepBach(num_features_lr, num_features_c, num_pitches, num_units_lstm=[200],
             num_dense=200):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    beat = Input(shape=(BEAT_SIZE,), name='beat')
    beats_right = Input(shape=(timesteps, BEAT_SIZE), name='beats_right')
    beats_left = Input(shape=(timesteps, BEAT_SIZE), name='beats_left')
    fermatas_left = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_left')
    fermatas_right = Input(shape=(timesteps, BITS_FERMATA), name='fermatas_right')
    central_fermata = Input(shape=(BITS_FERMATA,), name='central_fermata')

    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + BEAT_SIZE + BITS_FERMATA,
                           output_dim=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + BEAT_SIZE + BITS_FERMATA,
                            output_dim=num_dense, name='embedding_right')

    predictions_left = TimeDistributed(embedding_left)(merge((left_features,
                                                              beats_left,
                                                              fermatas_left),
                                                             mode='concat'))
    predictions_right = TimeDistributed(embedding_right)(merge((right_features,
                                                                beats_right,
                                                                fermatas_right),
                                                               mode='concat'))

    predictions_center = merge((central_features, beat,
                                central_fermata), mode='concat')

    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)

    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False
        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)
        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)

    predictions = merge((predictions_left, predictions_center, predictions_right),
                        mode='concat')
    predictions = Dense(num_dense, activation='relu')(predictions)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features,
                         beat, beats_left, beats_right,
                         fermatas_left, fermatas_right, central_fermata],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model


def maxent(num_features_lr, num_features_c, num_pitch):
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    beats = Input(shape=(BEAT_SIZE,), name='beats')

    predictions_left = Reshape((timesteps * num_features_lr,))(left_features)
    predictions_right = Reshape((timesteps * num_features_lr,))(right_features)

    predictions = merge((predictions_left, predictions_right, central_features, beats),
                        mode='concat')
    # predictions = Dense(num_dense, activation='relu')(predictions)

    predictions = Dropout(0.2)(predictions)
    pitch_prediction = Dense(num_pitch, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features, beats],
                  output=pitch_prediction)

    model.compile(optimizer='rmsprop', loss={'pitch_prediction': 'categorical_crossentropy'
                                             },
                  metrics=['accuracy'])
    model.summary()
    return model


def gibbs(models=None, melody=None, fermatas_melody=None, sequence_length=50, num_iterations=1000,
          timesteps=16,
          model_base_name='models/raw_dataset/tmp/',
          num_voices=4, temperature=1., min_pitches=None,
          max_pitches=None, initial_seq=None):
    """
    samples from models in model_base_name

    """

    dataset_name = RAW_DATASET
    X, min_pitches, max_pitches, num_voices = pickle.load(open(RAW_DATASET, 'rb'))

    # load models if not
    if models is None:
        for expert_index in range(num_voices):
            model_name = model_base_name + str(expert_index)

            model = load_model(model_name=model_name, yaml=False)
            models.append(model)

    # initialization sequence
    if melody is not None:
        sequence_length = len(melody)

    if fermatas_melody is not None:
        sequence_length = len(fermatas_melody)
        if melody is not None:
            assert len(melody) == len(fermatas_melody)

    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add slur_symbol
        seq[timesteps:-timesteps, expert_index] = np.random.random_integers(min_pitches[expert_index],
                                                                            max_pitches[expert_index] + 1,
                                                                            size=sequence_length)

    if initial_seq is not None:
        seq = initial_seq
        min_voice = 1
        # works only with reharmonization

    # melody = X[-1][0, :, 0]
    # melody is pa !
    if melody is not None:
        seq[timesteps:-timesteps, 0] = melody[:, 0]
        mask = melody[:, 1] == 0
        seq[timesteps:-timesteps, 0][mask] = max_pitches[0] + 1
        min_voice = 1
    else:
        min_voice = 0

    if fermatas_melody is not None:
        fermatas_melody = np.concatenate((np.zeros((timesteps,)),
                                          fermatas_melody,
                                          np.zeros((timesteps,)))
                                         )

    min_temperature = temperature
    temperature = 1.2
    # Main loop
    for iteration in tqdm(range(num_iterations)):

        temperature = max(min_temperature, temperature * 0.99996)  # Recuit

        voice_index = np.random.randint(min_voice, num_voices)
        time_index = np.random.randint(timesteps, sequence_length + timesteps)

        (left_feature,
         central_feature,
         right_feature,
         (beats_left, beat, beats_right),
         label) = all_features(seq, voice_index, time_index, timesteps, min_pitches, max_pitches, num_voices,
                               chorale_as_pas=False)

        input_features = {'left_features': left_feature[None, :, :],
                          'central_features': central_feature[None, :],
                          'right_features': right_feature[None, :, :],
                          'beat': beat[None, :],
                          'beats_left': beats_left[None, :, :],
                          'beats_right': beats_right[None, :, :]}

        # add fermatas evenly spaced
        if fermatas_melody is None:
            (fermatas_left,
             central_fermata,
             fermatas_right) = to_fermata(time_index, timesteps=timesteps)
            input_features.update({'fermatas_left': fermatas_left[None, :, :],
                                   'central_fermata': central_fermata[None, :],
                                   'fermatas_right': fermatas_right[None, :, :]
                                   })
        else:
            (fermatas_left,
             central_fermata,
             fermatas_right) = fermata_melody_to_fermata(time_index, timesteps=timesteps,
                                                         fermatas_melody=fermatas_melody)
            input_features.update({'fermatas_left': fermatas_left[None, :, :],
                                   'central_fermata': central_fermata[None, :],
                                   'fermatas_right': fermatas_right[None, :, :]
                                   })

        probas = models[voice_index].predict(input_features, batch_size=1)

        probas_pitch = probas[0]

        # use temperature
        probas_pitch = np.log(probas_pitch) / temperature
        probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

        # pitch can include slur_symbol
        pitch = np.argmax(np.random.multinomial(1, probas_pitch)) + min_pitches[voice_index]

        seq[time_index, voice_index] = pitch

    return seq[timesteps:-timesteps, :]


def parallelGibbs(models=None, melody=None, fermatas_melody=None, sequence_length=50, num_iterations=1000,
                  timesteps=16,
                  model_base_name='models/raw_dataset/tmp/',
                  num_voices=4, temperature=1., min_pitches=None,
                  max_pitches=None, initial_seq=None, batch_size_per_voice=16, parallel_updates=True):
    """
    samples from models in model_base_name
    """

    X, min_pitches, max_pitches, num_voices = pickle.load(open(RAW_DATASET, 'rb'))

    # load models if not
    if models is None:
        for expert_index in range(num_voices):
            model_name = model_base_name + str(expert_index)

            model = load_model(model_name=model_name, yaml=False)
            models.append(model)

    # initialization sequence
    if melody is not None:
        sequence_length = len(melody)

    if fermatas_melody is not None:
        sequence_length = len(fermatas_melody)
        if melody is not None:
            assert len(melody) == len(fermatas_melody)

    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add slur_symbol
        seq[timesteps:-timesteps, expert_index] = np.random.random_integers(min_pitches[expert_index],
                                                                            max_pitches[expert_index] + 1,
                                                                            size=sequence_length)

    if initial_seq is not None:
        seq = initial_seq
        min_voice = 1
        # works only with reharmonization

    if melody is not None:
        seq[timesteps:-timesteps, 0] = melody[:, 0]
        mask = melody[:, 1] == 0
        seq[timesteps:-timesteps, 0][mask] = max_pitches[0] + 1
        min_voice = 1
    else:
        min_voice = 0

    if fermatas_melody is not None:
        fermatas_melody = np.concatenate((np.zeros((timesteps,)),
                                          fermatas_melody,
                                          np.zeros((timesteps,)))
                                         )

    min_temperature = temperature
    temperature = 1.2
    # Main loop
    for iteration in tqdm(range(num_iterations)):

        temperature = max(min_temperature, temperature * 0.99)  # Recuit

        time_indexes = {}
        probas = {}
        for voice_index in range(min_voice, num_voices):
            batch_input_features = []
            time_indexes[voice_index] = []
            for batch_index in range(batch_size_per_voice):

                time_index = np.random.randint(timesteps, sequence_length + timesteps)
                time_indexes[voice_index].append(time_index)

                (left_feature,
                 central_feature,
                 right_feature,
                 (beats_left, beat, beats_right),
                 label) = all_features(seq, voice_index, time_index, timesteps, min_pitches, max_pitches, num_voices,
                                       chorale_as_pas=False)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'beat': beat[:],
                                  'beats_left': beats_left[:, :],
                                  'beats_right': beats_right[:, :]}

                # add fermatas evenly spaced
                if fermatas_melody is None:
                    (fermatas_left,
                     central_fermata,
                     fermatas_right) = to_fermata(time_index, timesteps=timesteps)
                    input_features.update({'fermatas_left': fermatas_left[:, :],
                                           'central_fermata': central_fermata[:],
                                           'fermatas_right': fermatas_right[:, :]
                                           })
                else:
                    (fermatas_left,
                     central_fermata,
                     fermatas_right) = fermata_melody_to_fermata(time_index, timesteps=timesteps,
                                                                 fermatas_melody=fermatas_melody)
                    input_features.update({'fermatas_left': fermatas_left[:, :],
                                           'central_fermata': central_fermata[:],
                                           'fermatas_right': fermatas_right[:, :]
                                           })
                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)

            # convert input_features
            batch_input_features = {key: np.array([input_features[key] for input_features in batch_input_features])
                                    for key in batch_input_features[0].keys()
                                    }
            # make all estimations
            probas[voice_index] = models[voice_index].predict(batch_input_features,
                                                              batch_size=batch_size_per_voice)
            if not parallel_updates:
                # update
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch)) + min_pitches[voice_index]

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

        if parallel_updates:
            # update
            for voice_index in range(min_voice, num_voices):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch)) + min_pitches[voice_index]

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


# Utils
def load_model(model_name, yaml=True):
    """

    :rtype: object
    """
    if yaml:
        ext = '.yaml'
        model = model_from_yaml(open(model_name + ext).read())
    else:
        ext = '.json'
        model = model_from_json(open(model_name + ext).read())
    model.load_weights(model_name + '_weights.h5')
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    print("model " + model_name + " loaded")
    return model


def save_model(model, model_name, yaml=True, overwrite=False):
    # SAVE MODEL
    if yaml:
        string = model.to_yaml()
        ext = '.yaml'
    else:
        string = model.to_json()
        ext = '.json'
    open(model_name + ext, 'w').write(string)
    model.save_weights(model_name + '_weights.h5', overwrite=overwrite)
    print("model " + model_name + " saved")


def create_models(model_name=None, create_new=False, num_dense=200, num_units_lstm=[200, 200]):
    """
    Choose one model
    :param model_name:
    :return:
    """

    _, min_pitches, max_pitches, num_voices = pickle.load(open(RAW_DATASET, 'rb'))
    for voice_index in range(4):
        # We only need one example for features dimensions
        gen = generator_from_raw_dataset(batch_size=batch_size, timesteps=timesteps,
                                         voice_index=voice_index)

        (left_features,
         central_features,
         right_features,
         beats,
         labels, fermatas) = next(gen)

        if 'deepbach' in model_name:
            model = deepBach(num_features_lr=left_features.shape[-1],
                             num_features_c=central_features.shape[-1],
                             num_pitches=max_pitches[voice_index] - min_pitches[voice_index] + 1
                                         + 1,  # for continuation symbol
                             num_dense=num_dense, num_units_lstm=num_units_lstm)
        elif 'maxent' in model_name:
            model = maxEnt(num_features_lr=left_features.shape[-1],
                           num_features_c=central_features.shape[-1],
                           num_pitches=max_pitches[voice_index] - min_pitches[voice_index] + 1
                                       + 1)  # for continuation symbol
        elif 'mlp' in model_name:
            model = mlp(num_features_lr=left_features.shape[-1],
                        num_features_c=central_features.shape[-1],
                        num_pitches=max_pitches[voice_index] - min_pitches[voice_index] + 1
                                    + 1,
                        num_hidden=num_dense)  # for continuation symbol
        else:
            raise ValueError

        model_path_name = 'models/' + model_name + '_' + str(voice_index)
        if not os.path.exists(model_path_name + '.json') or create_new:
            save_model(model, model_name=model_path_name, overwrite=create_new)


def load_models(model_base_name=None):
    """
    load 4 models whose base name is model_base_name
    models must exist
    :param model_base_name:
    :return: list of 4 models
    """
    models = []
    for voice_index in range(4):
        model_path_name = 'models/' + model_base_name + '_' + str(voice_index)
        model = load_model(model_path_name)
        model.compile(optimizer='adam', loss={'pitch_prediction': 'categorical_crossentropy'
                                              },
                      metrics=['accuracy'])
        models.append(model)
    return models


def train_models(model_name,
                 samples_per_epoch,
                 num_epochs,
                 nb_val_samples):
    """
    Train models

    """
    models = []
    for voice_index in range(4):
        # Load appropriate generators

        generator_train = (({'left_features': left_features,
                             'central_features': central_features,
                             'right_features': right_features,
                             'beat': beat,
                             'beats_left': beats_left,
                             'beats_right': beats_right,
                             'fermatas_left': fermatas_left,
                             'fermatas_right': fermatas_right,
                             'central_fermata': central_fermata
                             },
                            {'pitch_prediction': labels})
                           for (left_features,
                                central_features,
                                right_features,
                                (beats_left, beat, beats_right),
                                labels,
                                (fermatas_left, central_fermata, fermatas_right))
                           in generator_from_raw_dataset(batch_size=batch_size,
                                                         timesteps=timesteps,
                                                         voice_index=voice_index,
                                                         phase='train'
                                                         ))

        generator_val = (({'left_features': left_features,
                           'central_features': central_features,
                           'right_features': right_features,
                           'beat': beat,
                           'beats_left': beats_left,
                           'beats_right': beats_right,
                           'fermatas_left': fermatas_left,
                           'fermatas_right': fermatas_right,
                           'central_fermata': central_fermata
                           },
                          {'pitch_prediction': labels})
                         for (left_features,
                              central_features,
                              right_features,
                              (beats_left, beat, beats_right),
                              labels,
                              (fermatas_left, central_fermata, fermatas_right))
                         in generator_from_raw_dataset(batch_size=batch_size,
                                                       timesteps=timesteps,
                                                       voice_index=voice_index,
                                                       phase='train'
                                                       ))

        model_path_name = 'models/' + model_name + '_' + str(voice_index)

        model = load_model(model_path_name)

        model.compile(optimizer='adam', loss={'pitch_prediction': 'categorical_crossentropy'
                                              },
                      metrics=['accuracy'])

        model.fit_generator(generator_train, samples_per_epoch=samples_per_epoch,
                            nb_epoch=num_epochs, verbose=1, validation_data=generator_val,
                            nb_val_samples=nb_val_samples)

        models.append(model)

        save_model(model, model_path_name, overwrite=True)
    return models


def export_reharmo_turing_test(model_base_name, models, min_pitches, max_pitches, melodies_and_fermatas,
                               num_iterations=10000, temperature=1.0, fermatas=True, export_directory=None,
                               timesteps=None, in_one_file=False
                               ):
    """
    Helper functions to create Bach or Computer extracts
    """
    if in_one_file:
        seqs = []
    for k, (melody, fermatas_melody) in enumerate(melodies_and_fermatas):
        seq = gibbs(models=models, model_base_name=model_base_name,
                    melody=melody, fermatas_melody=fermatas_melody,
                    num_iterations=num_iterations, sequence_length=160,
                    timesteps=timesteps,
                    min_pitches=min_pitches, max_pitches=max_pitches, temperature=temperature,
                    initial_seq=None)
        if in_one_file:
            seqs.append(as_ps_to_as_pas(np.transpose(seq[timesteps:-timesteps, :], axes=(1, 0)),
                                        min_pitches, max_pitches))

        score = seq_to_stream_slur(np.transpose(seq[timesteps:-timesteps, :], axes=(1, 0)),
                                   min_pitches, max_pitches)

        mf = midi.translate.music21ObjectToMidiFile(score)
        # versioning
        version = 1
        midi_file_name = export_directory + str(k) + '_v' + str(version) + ".mid"
        while (os.path.exists(midi_file_name)):
            version += 1
            midi_file_name = export_directory + str(k) + '_v' + str(version) + ".mid"
        mf.open(midi_file_name, 'wb')
        mf.write()
        mf.close()

    if in_one_file:
        score = seqs_to_stream(seqs)
        mf = midi.translate.music21ObjectToMidiFile(score)
        # versioning
        version = 1
        midi_file_name = export_directory + 'all' + '_v' + str(version) + ".mid"
        while (os.path.exists(midi_file_name)):
            version += 1
            midi_file_name = export_directory + 'all' + str(k) + '_v' + str(version) + ".mid"
        mf.open(midi_file_name, 'wb')
        mf.write()
        mf.close()


def export_bach_turing_test(chorale_list, export_directory, in_one_file=False):
    for k, chorale in enumerate(chorale_list):
        score = seq_to_stream(chorale[:, :, :F_INDEX])
        mf = midi.translate.music21ObjectToMidiFile(score)
        midi_file_name = export_directory + str(k) + ".mid"
        mf.open(midi_file_name, 'wb')
        mf.write()
        mf.close()

    if in_one_file:
        seqs = []
        for k, chorale in enumerate(chorale_list):
            seqs.append(chorale[:, :, :F_INDEX])

        score = seqs_to_stream(seqs)

        mf = midi.translate.music21ObjectToMidiFile(score)
        midi_file_name = export_directory + 'all_chorales' + ".mid"
        mf.open(midi_file_name, 'wb')
        mf.write()
        mf.close()


if __name__ == '__main__':
    if not os.path.exists(RAW_DATASET):
        initialization()

    # load dataset
    X, min_pitches, max_pitches, num_voices = pickle.load(open(RAW_DATASET,
                                                               'rb'))

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size used during training phase (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-s', '--samples_per_epoch',
                        help='number of samples per epoch (default: %(default)s)',
                        type=int, default=12800 * 7)
    parser.add_argument('--num_val_samples',
                        help='number of validation samples (default: %(default)s)',
                        type=int, default=1280)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help='number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help='size of non recurrent hidden layers (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model name (default: %(default)s)',
                        choices=['deepbach', 'mlp', 'maxent'],
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
                        type=str, default='', const='generated_examples/example.mid')
    args = parser.parse_args()
    print(args)

    timesteps = args.timesteps
    batch_size = args.batch_size_train
    samples_per_epoch = args.samples_per_epoch
    nb_val_samples = args.num_val_samples
    num_units_lstm = args.num_units_lstm

    if args.ext:
        ext = '_' + args.ext
    else:
        ext = ''

    model_name = args.name.lower() + ext
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    num_units_lstm = args.num_units_lstm
    num_dense = args.num_dense

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = None

    fermatas_melody = None
    # when reharmonization
    if args.midi_file:
        melody = converter.parse(args.midi_file)
        melody = part_to_list(melody.parts[0])
        num_voices = 3
        # add fermatas for god save the queen
        if args.midi_file == 'datasets/god_save_the_queen.mid':
            fermatas_melody = np.concatenate((np.zeros(15 * 4),
                                              np.ones(3 * 4),
                                              np.zeros(21 * 4),
                                              np.ones(3 * 4))
                                             )
    else:
        num_voices = 4
        melody = None

    num_iterations = args.num_iterations // batch_size_per_voice // num_voices
    parallel = batch_size_per_voice > 1
    train = args.train > 0
    num_epochs = args.train

    overwrite = args.overwrite

    # In order to reharmonize bach chorales melodies:
    # melodies_and_fermatas = []
    # for c in X:
    #     melodies_and_fermatas.append((c[0, :, :F_INDEX],
    #                                   c[0, :, F_INDEX]))
    # melody = None

    if not os.path.exists('models/' + model_name + '_3.yaml'):
        create_models(model_name, create_new=overwrite, num_units_lstm=num_units_lstm, num_dense=num_dense)
    if train:
        models = train_models(model_name=model_name,
                              samples_per_epoch=samples_per_epoch,
                              nb_val_samples=nb_val_samples,
                              num_epochs=num_epochs)
    else:
        models = load_models(model_name)

    temperature = 1.

    timesteps = int(models[0].input[0]._shape[1])


    seq = generation(model_base_name=model_name, models=models,
                     min_pitches=min_pitches,
                     max_pitches=max_pitches, melody=melody, initial_seq=None, temperature=temperature,
                     fermatas_melody=fermatas_melody, parallel=parallel, batch_size_per_voice=batch_size_per_voice,
                     num_iterations=num_iterations,
                     sequence_length=sequence_length,
                     output_file=output_file)

