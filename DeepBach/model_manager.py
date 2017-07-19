"""
Created on 15 mars 2016

@author: Gaetan Hadjeres
"""
import os
import pickle

from keras.models import model_from_json, model_from_yaml
from music21 import midi
from tqdm import tqdm

from .data_utils import generator_from_raw_dataset, BACH_DATASET, \
    all_features, \
    indexed_chorale_to_score, START_SYMBOL, END_SYMBOL, all_metadatas, \
    standard_note, SOP, BASS, PACKAGE_DIR
from .metadata import *
from .models_zoo import deepBach, deepbach_skip_connections


def generation(model_base_name, models, timesteps, melody=None,
               chorale_metas=None,
               initial_seq=None, temperature=1.0, parallel=False,
               batch_size_per_voice=8, num_iterations=None,
               sequence_length=160,
               output_file=None, pickled_dataset=BACH_DATASET):
    # Test by generating a sequence

    # todo -p parameter
    parallel = True
    if parallel:
        seq = parallel_gibbs(models=models, model_base_name=model_base_name,
                             melody=melody, chorale_metas=chorale_metas,
                             timesteps=timesteps,
                             num_iterations=num_iterations,
                             sequence_length=sequence_length,
                             temperature=temperature,
                             initial_seq=initial_seq,
                             batch_size_per_voice=batch_size_per_voice,
                             parallel_updates=True,
                             pickled_dataset=pickled_dataset)
    else:
        # todo refactor
        print('gibbs function must be refactored!')
        # seq = gibbs(models=models, model_base_name=model_base_name,
        #             timesteps=timesteps,
        #             melody=melody, fermatas_melody=fermatas_melody,
        #             num_iterations=num_iterations, sequence_length=sequence_length,
        #             temperature=temperature,
        #             initial_seq=initial_seq,
        #             pickled_dataset=pickled_dataset)
        raise NotImplementedError

    # convert
    score = indexed_chorale_to_score(np.transpose(seq, axes=(1, 0)),
                                     pickled_dataset=pickled_dataset
                                     )

    # save as MIDI file
    if output_file:
        mf = midi.translate.music21ObjectToMidiFile(score)
        mf.open(output_file, 'wb')
        mf.write()
        mf.close()
        print("File " + output_file + " written")

    # display in editor
    score.show()
    return seq


# def gibbs(models=None, melody=None, fermatas_melody=None, sequence_length=50, num_iterations=1000,
#           timesteps=16,
#           model_base_name='models/raw_dataset/tmp/',
#           num_voices=4, temperature=1., min_pitches=None,
#           max_pitches=None, initial_seq=None,
#           pickled_dataset=BACH_DATASET):
#     """
#     samples from models in model_base_name
#
#     """
#     X, X_metadatas, min_pitches, max_pitches, num_voices = pickle.load(open(pickled_dataset, 'rb'))
#
#     # load models if not
#     if models is None:
#         for expert_index in range(num_voices):
#             model_name = model_base_name + str(expert_index)
#
#             model = load_model(model_name=model_name, yaml=False)
#             models.append(model)
#
#     # initialization sequence
#     if melody is not None:
#         sequence_length = len(melody)
#
#     if fermatas_melody is not None:
#         sequence_length = len(fermatas_melody)
#         if melody is not None:
#             assert len(melody) == len(fermatas_melody)
#
#     seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
#     for expert_index in range(num_voices):
#         # Add slur_symbol
#         seq[timesteps:-timesteps, expert_index] = np.random.random_integers(min_pitches[expert_index],
#                                                                             max_pitches[expert_index] + 1,
#                                                                             size=sequence_length)
#
#     if initial_seq is not None:
#         seq = initial_seq
#         min_voice = 1
#         # works only with reharmonization
#
#     # melody = X[-1][0, :, 0]
#     # melody is pa !
#     if melody is not None:
#         seq[timesteps:-timesteps, 0] = melody[:, 0]
#         mask = melody[:, 1] == 0
#         seq[timesteps:-timesteps, 0][mask] = max_pitches[0] + 1
#         min_voice = 1
#     else:
#         min_voice = 0
#
#     if fermatas_melody is not None:
#         fermatas_melody = np.concatenate((np.zeros((timesteps,)),
#                                           fermatas_melody,
#                                           np.zeros((timesteps,)))
#                                          )
#
#     min_temperature = temperature
#     temperature = 1.2
#     # Main loop
#     for iteration in tqdm(range(num_iterations)):
#
#         temperature = max(min_temperature, temperature * 0.99996)  # Recuit
#
#         voice_index = np.random.randint(min_voice, num_voices)
#         time_index = np.random.randint(timesteps, sequence_length + timesteps)
#
#         (left_feature,
#          central_feature,
#          right_feature,
#          (beats_left, beat, beats_right),
#          label) = all_features(seq, voice_index, time_index, timesteps, min_pitches, max_pitches, chorale_as_pas=False)
#
#         input_features = {'left_features': left_feature[None, :, :],
#                           'central_features': central_feature[None, :],
#                           'right_features': right_feature[None, :, :],
#                           'beat': beat[None, :],
#                           'beats_left': beats_left[None, :, :],
#                           'beats_right': beats_right[None, :, :]}
#
#         # add fermatas evenly spaced
#         if fermatas_melody is None:
#             (fermatas_left,
#              central_fermata,
#              fermatas_right) = to_fermata(time_index, timesteps=timesteps)
#             input_features.update({'fermatas_left': fermatas_left[None, :, :],
#                                    'central_fermata': central_fermata[None, :],
#                                    'fermatas_right': fermatas_right[None, :, :]
#                                    })
#         else:
#             (fermatas_left,
#              central_fermata,
#              fermatas_right) = fermata_melody_to_fermata(time_index, timesteps=timesteps,
#                                                          fermatas_melody=fermatas_melody)
#             input_features.update({'fermatas_left': fermatas_left[None, :, :],
#                                    'central_fermata': central_fermata[None, :],
#                                    'fermatas_right': fermatas_right[None, :, :]
#                                    })
#
#         probas = models[voice_index].predict(input_features, batch_size=1)
#
#         probas_pitch = probas[0]
#
#         # use temperature
#         probas_pitch = np.log(probas_pitch) / temperature
#         probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7
#
#         # pitch can include slur_symbol
#         pitch = np.argmax(np.random.multinomial(1, probas_pitch)) + min_pitches[voice_index]
#
#         seq[time_index, voice_index] = pitch
#
#     return seq[timesteps:-timesteps, :]


def parallel_gibbs(models=None, melody=None, chorale_metas=None,
                   sequence_length=50, num_iterations=1000,
                   timesteps=16,
                   model_base_name='models/raw_dataset/tmp/',
                   temperature=1., initial_seq=None, batch_size_per_voice=16,
                   parallel_updates=True,
                   pickled_dataset=BACH_DATASET):
    """
    samples from models in model_base_name
    """

    X, X_metadatas, voices_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    num_voices = len(voices_ids)
    # load models if not
    if models is None:
        for expert_index in range(num_voices):
            model_name = model_base_name + str(expert_index)

            model = load_model(model_name=model_name, yaml=False)
            models.append(model)

    # initialization sequence
    if melody is not None:
        sequence_length = len(melody)
        if chorale_metas is not None:
            sequence_length = min(sequence_length, len(chorale_metas[0]))
    elif chorale_metas is not None:
        sequence_length = len(chorale_metas[0])

    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add start and end symbol + random init
        seq[:timesteps, expert_index] = [note2indexes[expert_index][
                                             START_SYMBOL]] * timesteps
        seq[timesteps:-timesteps, expert_index] = np.random.randint(
            num_pitches[expert_index],
            size=sequence_length)

        seq[-timesteps:, expert_index] = [note2indexes[expert_index][
                                              END_SYMBOL]] * timesteps

    if initial_seq is not None:
        seq = initial_seq
        min_voice = 1
        # works only with reharmonization

    if melody is not None:
        seq[timesteps:-timesteps, 0] = melody
        min_voice = 1
    else:
        min_voice = 0

    if chorale_metas is not None:
        # chorale_metas is a list
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]

    else:
        raise NotImplementedError

    min_temperature = temperature
    temperature = 1.5

    # Main loop
    for iteration in tqdm(range(num_iterations)):
        temperature = max(min_temperature, temperature * 0.9992)  # Recuit
        # print(temperature)

        time_indexes = {}
        probas = {}
        for voice_index in range(min_voice, num_voices):
            batch_input_features = []

            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps,
                                               sequence_length + timesteps)
                time_indexes[voice_index].append(time_index)

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps,
                                       num_pitches, num_voices)

                left_metas, central_metas, right_metas = all_metadatas(
                    chorale_metadatas=extended_chorale_metas,
                    metadatas=metadatas,
                    time_index=time_index, timesteps=timesteps)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas}

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)

            # convert input_features
            batch_input_features = {key: np.array(
                [input_features[key] for input_features in
                 batch_input_features])
                for key in batch_input_features[0].keys()
            }
            # make all estimations
            probas[voice_index] = models[voice_index].predict(
                batch_input_features,
                batch_size=batch_size_per_voice)
            if not parallel_updates:
                # update
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][
                            batch_index], voice_index] = pitch

        if parallel_updates:
            # update
            for voice_index in range(min_voice, num_voices):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][
                            batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


def _diatonic_note_names2indexes(index2notes):
    ds = []
    # build diatonic_note_num 2 indexes dict
    for voice_index, index2note in enumerate(index2notes):
        d = {}
        for i in range(len(index2note)):
            n = standard_note(index2note[i])
            if n.isNote:
                diatonic_note_num = n.pitch.diatonicNoteNum
            else:
                diatonic_note_num = -1
            if diatonic_note_num in d:
                d.update({diatonic_note_num: d.get(diatonic_note_num) + [i]})
            else:
                d.update({diatonic_note_num: [i]})
        ds.append(d)
    # transform as numpy arrays
    for d in ds:
        for k in d:
            d.update({k: np.array(d.get(k))})
    return ds


def canon(models=None, chorale_metas=None, sequence_length=50,
          num_iterations=1000,
          timesteps=16,
          model_base_name='models/raw_dataset/tmp/',
          temperature=1., batch_size_per_voice=16,
          pickled_dataset=BACH_DATASET,
          intervals=[7], delays=[32],
          ):
    """
    samples from models in model_base_name
    """
    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))

    # variables
    num_voices = len(voice_ids)
    assert num_voices == 2

    num_pitches = list(map(len, index2notes))
    max_delay = max(delays)
    delays = np.array([0] + delays)
    intervals = np.array([0] + intervals)

    # compute tables
    diatonic_note_names2indexes = _diatonic_note_names2indexes(index2notes)
    print(diatonic_note_names2indexes)
    # load models if not
    if models is None:
        for expert_index in range(num_voices):
            model_name = model_base_name + str(expert_index)

            model = load_model(model_name=model_name, yaml=False)
            models.append(model)

    seq = np.zeros(
        shape=(2 * timesteps + max_delay + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add start and end symbol + random init
        seq[:timesteps, expert_index] = [note2indexes[expert_index][
                                             START_SYMBOL]] * timesteps
        seq[timesteps:-timesteps - max_delay,
        expert_index] = np.random.randint(num_pitches[expert_index],
                                          size=sequence_length)

        seq[-timesteps - max_delay:, expert_index] = [note2indexes[
                                                          expert_index][
                                                          END_SYMBOL]] * (
                                                         timesteps + max_delay)

    if chorale_metas is not None:
        # chorale_metas is a list
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((
                                                      timesteps + max_delay,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]

    else:
        raise NotImplementedError

    min_temperature = temperature
    temperature = 1.5

    # Main loop
    for iteration in tqdm(range(num_iterations)):

        temperature = max(min_temperature, temperature * 0.9995)  # Recuit
        print(temperature)

        time_indexes = {}
        probas = {}

        for voice_index in range(num_voices):
            batch_input_features = []
            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                # soprano based
                if voice_index == 0:
                    time_index = np.random.randint(timesteps,
                                                   sequence_length + timesteps)
                else:
                    # time_index = sequence_length + timesteps * 2 - time_indexes[0][batch_index]
                    time_index = time_indexes[0][batch_index] + delays[
                        voice_index]

                time_indexes[voice_index].append(time_index)

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps,
                                       num_pitches, num_voices)

                left_metas, central_metas, right_metas = all_metadatas(
                    chorale_metadatas=extended_chorale_metas,
                    metadatas=metadatas,
                    time_index=time_index, timesteps=timesteps)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas}

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)

            # convert input_features
            batch_input_features = {key: np.array(
                [input_features[key] for input_features in
                 batch_input_features])
                for key in batch_input_features[0].keys()
            }
            # make all estimations
            probas[voice_index] = models[voice_index].predict(
                batch_input_features,
                batch_size=batch_size_per_voice)

        # parallel updates
        for batch_index in range(batch_size_per_voice):
            # create list of masks for each note name
            proba_sop = probas[SOP][batch_index]
            proba_bass = probas[BASS][batch_index]

            proba_sop_split = _split_proba(proba_sop,
                                           diatonic_note_names2indexes[SOP])
            proba_bass_split = _split_proba(proba_bass,
                                            diatonic_note_names2indexes[BASS])

            interval = intervals[1]

            # multiply probas
            canon_product_probas, index_merge2pitches = _merge_probas_canon(
                proba_sop_split, proba_bass_split,
                interval,
                diatonic_note_names2indexes)

            # draw
            # use temperature
            canon_product_probas /= np.sum(canon_product_probas)
            canon_product_probas = np.log(canon_product_probas) / temperature
            canon_product_probas = np.exp(canon_product_probas) / np.sum(
                np.exp(canon_product_probas)) - 1e-7

            # pitch can include slur_symbol
            index_drawn_pitches = np.argmax(
                np.random.multinomial(1, canon_product_probas))
            pitches = index_merge2pitches[index_drawn_pitches]
            for voice_index, pitch in enumerate(pitches):
                seq[time_indexes[voice_index][
                        batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


def _split_proba(proba_sop, diatonic_note_name2indexes):
    dnn2probas = {}
    for diatonic_note_name in diatonic_note_name2indexes:
        dnn2probas.update({diatonic_note_name: proba_sop[
            diatonic_note_name2indexes[diatonic_note_name]]})
    return dnn2probas


def _merge_probas_canon(proba_sop_split, proba_bass_split, interval,
                        diatonic_note_names2indexes):
    # todo generalize to multiple voices
    merge_probas = []
    index = 0
    index_merge2pitches = {}
    for dnn_sop in proba_sop_split:
        for dnn_bass in proba_bass_split:
            # when identical notes up to transformation
            if dnn_sop == dnn_bass + interval or dnn_sop == dnn_bass == -1:
                # multiply all probas
                for p_sop_index, p_sop in enumerate(proba_sop_split[dnn_sop]):
                    for p_bass_index, p_bass in enumerate(
                            proba_bass_split[dnn_bass]):
                        # todo other combination than multiplication
                        merge_probas.append(p_sop * p_bass)

                        # create table or index to pitches
                        index_merge2pitches.update({index: [
                            diatonic_note_names2indexes[SOP][dnn_sop][
                                p_sop_index],
                            diatonic_note_names2indexes[BASS][dnn_bass][
                                p_bass_index]
                        ]}
                        )
                        index += 1
    return np.array(merge_probas), index_merge2pitches


def _update_pitches_canon(probas, delays, intervals, index2notes, notes2index,
                          diatonic_note_names2indexes,
                          temperature=1.):
    # create list of masks for each note name
    proba_sop = probas[0][0]
    proba_bass = probas[1][0]

    proba_sop_split = _split_proba(proba_sop, diatonic_note_names2indexes[0])
    proba_bass_split = _split_proba(proba_bass, diatonic_note_names2indexes[1])

    interval = intervals[1]

    # multiply probas
    canon_product_probas, index_merge2pitches = _merge_probas_canon(
        proba_sop_split, proba_bass_split, interval,
        diatonic_note_names2indexes)

    # draw
    # use temperature
    canon_product_probas /= np.sum(canon_product_probas)
    canon_product_probas = np.log(canon_product_probas) / temperature
    canon_product_probas = np.exp(canon_product_probas) / np.sum(
        np.exp(canon_product_probas)) - 1e-7

    # pitch can include slur_symbol
    index_drawn_pitches = np.argmax(
        np.random.multinomial(1, canon_product_probas))
    pitches = index_merge2pitches[index_drawn_pitches]

    return pitches


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


def create_models(model_name=None, create_new=False, num_dense=200,
                  num_units_lstm=[200, 200],
                  pickled_dataset=BACH_DATASET, num_voices=4, metadatas=None,
                  timesteps=16):
    """
    Choose one model
    :param model_name:
    :return:
    """

    _, _, _, index2notes, _, _ = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    for voice_index in range(num_voices):
        # We only need one example for features dimensions
        gen = generator_from_raw_dataset(batch_size=1, timesteps=timesteps,
                                         voice_index=voice_index,
                                         pickled_dataset=pickled_dataset)

        (
            (left_features,
             central_features,
             right_features),
            (left_metas, central_metas, right_metas),
            labels) = next(gen)

        if 'deepbach' in model_name:
            model = deepBach(num_features_lr=left_features.shape[-1],
                             num_features_c=central_features.shape[-1],
                             num_pitches=num_pitches[voice_index],
                             num_features_meta=left_metas.shape[-1],
                             num_dense=num_dense,
                             num_units_lstm=num_units_lstm,
                             timesteps=timesteps)
        elif 'skip' in model_name:
            model = deepbach_skip_connections(
                num_features_lr=left_features.shape[-1],
                num_features_c=central_features.shape[-1],
                num_features_meta=left_metas.shape[-1],
                num_pitches=num_pitches[voice_index],
                num_dense=num_dense, num_units_lstm=num_units_lstm,
                timesteps=timesteps)
        else:
            raise ValueError

        model_path_name = os.path.join(PACKAGE_DIR,
                                  'models/' + model_name + '_' + str(
                                      voice_index)
                                  )
        if not os.path.exists(model_path_name + '.json') or create_new:
            save_model(model, model_name=model_path_name, overwrite=create_new)


def load_models(model_base_name=None, num_voices=4):
    """
    load 4 models whose base name is model_base_name
    models must exist
    :param model_base_name:
    :return: list of num_voices models
    """
    models = []
    for voice_index in range(num_voices):
        model_path_name = os.path.join(PACKAGE_DIR,
                                  'models/' + model_base_name + '_' + str(
                                      voice_index)
                                  )
        model = load_model(model_path_name)
        model.compile(optimizer='adam',
                      loss={'pitch_prediction': 'categorical_crossentropy'
                            },
                      metrics=['accuracy'])
        models.append(model)
    return models


def train_models(model_name, steps_per_epoch, num_epochs, validation_steps,
                 timesteps, pickled_dataset=BACH_DATASET,
                 num_voices=4, batch_size=16, metadatas=None):
    """
    Train models
    :param batch_size:
    :param metadatas:

    """
    models = []
    for voice_index in range(num_voices):
        # Load appropriate generators

        generator_train = (({'left_features': left_features,
                             'central_features': central_features,
                             'right_features': right_features,
                             'left_metas': left_metas,
                             'right_metas': right_metas,
                             'central_metas': central_metas,
                             },
                            {'pitch_prediction': labels})
                           for (
                               (left_features, central_features,
                                right_features),
                               (left_metas, central_metas, right_metas),
                               labels)

                           in generator_from_raw_dataset(batch_size=batch_size,
                                                         timesteps=timesteps,
                                                         voice_index=voice_index,
                                                         phase='train',
                                                         pickled_dataset=pickled_dataset
                                                         ))

        generator_val = (({'left_features': left_features,
                           'central_features': central_features,
                           'right_features': right_features,
                           'left_metas': left_metas,
                           'right_metas': right_metas,
                           'central_metas': central_metas,
                           },
                          {'pitch_prediction': labels})
                         for (
                             (left_features, central_features, right_features),
                             (left_metas, central_metas, right_metas),
                             labels)

                         in generator_from_raw_dataset(batch_size=batch_size,
                                                       timesteps=timesteps,
                                                       voice_index=voice_index,
                                                       phase='test',
                                                       pickled_dataset=pickled_dataset
                                                       ))

        model_path_name = os.path.join(PACKAGE_DIR,
                                       'models/' + model_name +
                                       '_' + str(voice_index)
                                       )

        model = load_model(model_path_name)

        model.compile(optimizer='adam',
                      loss={'pitch_prediction': 'categorical_crossentropy'
                            },
                      metrics=['accuracy'])

        model.fit_generator(generator_train, samples_per_epoch=steps_per_epoch,
                            epochs=num_epochs, verbose=1,
                            validation_data=generator_val,
                            validation_steps=validation_steps)

        models.append(model)

        save_model(model, model_path_name, overwrite=True)
    return models
