#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 mars 2016

@author: Gaetan Hadjeres
"""
import pickle
from tqdm import tqdm

import numpy as np
from music21 import corpus, converter, stream, note, duration

NUM_VOICES = 4
SUBDIVISION = 4  # quarter note subdivision
BITS_FERMATA = 2  # number of bits needed to encode fermata
RANGE_FERMATA = 3  # 3 beats before fermatas
SPACING_FERMATAS = 12  # in beats
FERMATAS_LENGTH = 1  # in beats

P_INDEX = 0  # pitch index in representation
A_INDEX = 1  # articulation index in representation
F_INDEX = 2  # fermata index in representation

BEAT_SIZE = 4

OCTAVE = 12

RAW_DATASET = 'datasets/raw_dataset/bach_chorales.pickle'
RAW_DATASET_NO_TRANSPOSITION = 'datasets/' \
                               'raw_dataset/' \
                               'bach_chorales_no_transposition.pickle'

voice_ids = list(range(NUM_VOICES))  # soprano, alto, tenor, bass


def filter_file_list(file_list):
    """
    Only retain 4 voices chorales
    """
    l = []
    for file_name in file_list:
        c = converter.parse(file_name)
        # if len(c.parts) == NUM_VOICES and c.parts[0].id == 'Soprano':
        if len(c.parts) == NUM_VOICES:
            # Retain only chorals with fermatas

            if len(list(filter(lambda n: len(n.expressions) == 1, c.parts[0].flat.notes))) > 0:
                l.append(file_name)
    return l


def compute_min_max_pitches(file_list, voices=[0]):
    """
    Removes wrong chorales
    :param file_list:
    :type voices: list containing voices ids
    :returns: two lists min_p, max_p containing min and max pitches for each voice
    """
    min_p, max_p = [128] * len(voices), [0] * len(voices)
    to_remove = []
    for file_name in file_list:
        choral = converter.parse(file_name)
        for k, voice_id in enumerate(voices):
            try:
                c = choral.parts[voice_id]  # Retain only voice_id voice
                l = list(map(lambda n: n.pitch.midi, c.flat.notes))
                min_p[k] = min(min_p[k], min(l))
                max_p[k] = max(max_p[k], max(l))
            except AttributeError:
                to_remove.append(file_name)
    for file_name in set(to_remove):
        file_list.remove(file_name)
    return np.array(min_p), np.array(max_p)


def part_to_list(part):
    """

    :rtype: np.ndarray
    Returns (part_length, 2) matrix
    t[0] = (pitch, articulation)
    """
    length = int(part.duration.quarterLength * SUBDIVISION)  # in 16th notes
    list_notes = part.flat.notes
    num_notes = len(list_notes)
    j = 0
    i = 0
    t = np.zeros((length, 2))
    is_articulated = True
    while i < length:
        if j < num_notes - 1:
            if list_notes[j + 1].offset > i / SUBDIVISION:
                t[i, :] = [list_notes[j].pitch.midi, is_articulated]
                i += 1
                is_articulated = False
            else:
                j += 1
                is_articulated = True
        else:
            t[i, :] = [list_notes[j].pitch.midi, is_articulated]
            i += 1
            is_articulated = False
    return t


def part_to_list_with_fermata(part):
    """
    :rtype: np.ndarray
    Returns (part_length, 3) matrix
    t[0] = (pitch, articulation, fermata)
    """
    length = int(part.duration.quarterLength * SUBDIVISION)  # in 16th notes
    list_notes = part.flat.notes
    num_notes = len(list_notes)
    j = 0
    i = 0
    t = np.zeros((length, 3))
    is_articulated = True
    fermata = False
    while i < length:
        if j < num_notes - 1:
            if list_notes[j + 1].offset > i / SUBDIVISION:

                if len(list_notes[j].expressions) == 1:
                    fermata = True
                else:
                    fermata = False
                # fermata = fermata and is_articulated

                t[i, :] = [list_notes[j].pitch.midi, is_articulated, fermata]
                i += 1
                is_articulated = False
            else:
                j += 1
                is_articulated = True
        else:
            if len(list_notes[j].expressions) == 1:
                fermata = True
            else:
                fermata = False
            # fermata = fermata and is_articulated

            t[i, :] = [list_notes[j].pitch.midi, is_articulated, fermata]
            i += 1
            is_articulated = False
    return t


def chorale_to_input(chorale_file, voice_id=0):
    """
    Returns a list of [pitch, articulation]

    """
    part = converter.parse(chorale_file).parts[voice_id]
    # assert part.id == 'Soprano'
    return part_to_list(part)


def chorale_to_inputs(chorale_file, num_voices=None):
    """
    Returns a numpy array [voices, time, (pitch, articulation)]
    :param chorale_file:
    :param num_voices:
    :return:
    """
    mat = []
    for voice_id in range(num_voices):
        mat.append(chorale_to_input(chorale_file, voice_id=voice_id))
    return np.array(mat)


def chorale_to_inputs_with_fermata(chorale_file, num_voices=None):
    """
    Returns a numpy array [voices, time, (pitch, articulation, fermata)]
    :param chorale_file:
    :param num_voices:
    :return:
    """
    mat = []
    for voice_id in range(num_voices):
        mat.append(chorale_to_input_with_fermata(chorale_file, voice_id=voice_id))
    return np.array(mat)


def chorale_to_input_with_fermata(chorale_file, voice_id=0):
    """
    Returns a list of [pitch, articulation, fermata]

    """
    choral = converter.parse(chorale_file)
    part = choral.parts[voice_id]
    sop = choral.parts[0]
    assert sop.id == 'Soprano'
    sop = part_to_list_with_fermata(sop)
    part = part_to_list_with_fermata(part)
    # copy fermatas
    for k, e in enumerate(sop):
        part[k] = [part[k][0], part[k][1], e[2]]
    return part


def pitch_column_to_one_hot(col, MIN_PITCH, MAX_PITCH):
    """
    :param col:
    :param MIN_PITCH: scalar !
    :param MAX_PITCH: scalar !
    :return:
    """
    return np.vectorize(lambda x: x in col)(np.arange(MIN_PITCH, MAX_PITCH + 1))


def to_beat(time, timesteps=None):
    """
    time is given in the number of 16th notes

    put timesteps=None to return only current beat

    Returns metrical position one-hot encoded

    IMPORTANT, right_beats is REVERSED
    """
    beat = [0] * 4
    beat[time % 4] = 1

    if timesteps is None:
        return beat
    left_beats = np.array(list(map(lambda x: p_to_onehot(x, 0, 3),
                                   np.arange(time - timesteps, time) % 4)))

    right_beats = np.array(list(map(lambda x: p_to_onehot(x, 0, 3),
                                    np.arange(time + timesteps, time, -1) % 4)))
    return left_beats, np.array(beat), right_beats


def is_fermata(time):
    """
    Returns a boolean

    custom function
    :param time:
    :return:
    """
    # evenly spaced fermatas
    return (time // SUBDIVISION) % SPACING_FERMATAS < FERMATAS_LENGTH

    # for god save the queen
    # must add timesteps when
    # fermatas_god = np.concatenate((np.arange(5 * 12, 6 * 12) + 16,
    #                               np.arange(13 * 12, 14 * 12) + 16))
    # return time in fermatas_god

    # no fermata
    # return 0


def fermata_melody_to_fermata(time, timesteps=None, fermatas_melody=None):
    """
    time is given in 16th notes

    put timesteps=None only returns the current fermata

    one hot encoded
    :param time:
    :param timesteps:
    :return:
    """
    # custom formula for fermatas
    if fermatas_melody is None:
        print('Error in fermata_melody_to_fermata, fermatas_melody is None')
    central_fermata = p_to_onehot(fermatas_melody[time], min_pitch=0, max_pitch=1)
    if timesteps is None:
        return central_fermata
    fermatas_left = np.array(list(map(lambda f: p_to_onehot(f,
                                                            min_pitch=0,
                                                            max_pitch=1),
                                      fermatas_melody[time - timesteps: time])))
    fermatas_right = np.array(list(map(lambda f: p_to_onehot(f,
                                                             min_pitch=0,
                                                             max_pitch=1),
                                       fermatas_melody[time + timesteps: time: -1])))
    return fermatas_left, central_fermata, fermatas_right


def to_fermata(time, timesteps=None):
    """
    time is given in 16th notes

    put timesteps=None only returns the current fermata

    one hot encoded
    :param time:
    :param timesteps:
    :return:
    """
    # custom formula for fermatas
    central_fermata = p_to_onehot(is_fermata(time), min_pitch=0, max_pitch=1)
    if timesteps is None:
        return central_fermata
    fermatas_left = np.array(list(map(lambda f: p_to_onehot(is_fermata(f),
                                                            min_pitch=0,
                                                            max_pitch=1),
                                      np.arange(time - timesteps, time))))
    fermatas_right = np.array(list(map(lambda f: p_to_onehot(is_fermata(f),
                                                             min_pitch=0,
                                                             max_pitch=1),
                                       np.arange(time + timesteps, time, -1))))
    return fermatas_left, central_fermata, fermatas_right


def inputs_to_feature(inputs, voice_id, initial_beat=0):
    """
    Arguments: inputs  list of input
    Returns: features for voice voice_id
    features : previous_pitch * simultaneous_above_pitch * articulation * beat
    :param voice_id: so that a voice depends only on the preceding voices
    """
    beat_length = len(to_beat(0))
    feature = np.zeros((inputs[voice_id].shape[0], inputs[voice_id].shape[1] + beat_length))
    for k, pitch_and_articulation in enumerate(inputs[voice_id]):
        feature[k, :] = np.concatenate((pitch_and_articulation, to_beat(k + initial_beat)))
    return feature


def inputs_to_feature_with_fermata(inputs, voice_index, initial_beat=0):
    """
    Arguments: inputs  list of input containing fermatas
    Returns: features for voice voice_index in inputs
    features : previous_pitch * articulation * beat
    """
    beat_length = len(to_beat(0))
    feature = np.zeros((inputs[voice_index].shape[0],
                        inputs[voice_index].shape[1] - 1 + BITS_FERMATA + beat_length))
    for k, pitch_and_articulation_and_fermata in enumerate(inputs[voice_index]):
        feature[k, :] = np.concatenate((pitch_and_articulation_and_fermata[:2],
                                        next_fermata_within(inputs, voice_index, k),
                                        to_beat(k + initial_beat)))
    return feature


def next_fermata_within(inputs, voice_id, index, range_fermata=RANGE_FERMATA):
    """
    :param range_fermata:
    :param inputs:
    :param voice_id:
    :param index:
    :return:
    """
    # if fermata
    num = 0
    if inputs[voice_id][index][2]:
        num = 0
    else:
        for k in range(index, len(inputs[voice_id])):
            if inputs[voice_id][k][2]:
                num = ((k - k % SUBDIVISION) - (index - index % SUBDIVISION)) // SUBDIVISION
                break
    if num <= range_fermata:
        return np.array([0, 1], dtype=np.int32)
    else:
        return np.array([1, 0], dtype=np.int32)


def next_fermata_in(inputs, voice_id, index):
    # if fermata
    num = 0
    if inputs[voice_id][index][2]:
        num = 0
    else:
        for k in range(index, len(inputs[voice_id])):
            if inputs[voice_id][k][2]:
                num = ((k - k % SUBDIVISION) - (index - index % SUBDIVISION)) / SUBDIVISION
                break
    return np.array(list(map(lambda x: x == num, range(BITS_FERMATA))), dtype=np.int32)


def input_to_feature_with_fermata(input, initial_beat=0):
    return inputs_to_feature_with_fermata([input], voice_index=0, initial_beat=initial_beat)


def input_to_feature(input, initial_beat=0):
    return inputs_to_feature([input], voice_id=0, initial_beat=initial_beat)


def feature_to_onehot_feature(feature, NUM_PITCHES, MIN_PITCH, MAX_PITCH):
    """
    must only apply onehot encoding to first column
    """
    onehot = np.zeros((feature.shape[0], NUM_PITCHES + feature.shape[1] - 1))
    onehot[:, 0:NUM_PITCHES] = np.array(list(map(lambda col:
                                                 pitch_column_to_one_hot(col, MIN_PITCH, MAX_PITCH),
                                                 feature[:, 0][:, None])), dtype=np.int32)
    onehot[:, NUM_PITCHES:] = feature[:, 1:]
    return onehot


def fusion_features(Xs, voice_index, file_index=None):
    """
    Balanced fusion. Covers all cases
    if file_index is None: an element of Xs is given by
    Xs[voice_index] [t, features]
    else by
    Xs[voice_index][file_index] [t, features]
    Returns shorter sequences
    """
    if file_index is not None:
        total_features = 0
        # X[voice_index][file_index] [t, features]
        for X in Xs:
            # WARNING 4 STANDS FOR THE NUMBER OF BITS USED BY THE BEAT REPRESENTATION
            total_features += X[file_index].shape[1] - 4
        total_features += 4  # Because we keep one beat
        fusion = np.zeros((Xs[voice_index][file_index].shape[0] - 1, total_features))

        for k, vect in enumerate(Xs[voice_index][file_index][:-1, :]):
            i = 0
            for var_voice_index, X in enumerate(Xs):
                feature = X[file_index]
                if var_voice_index < voice_index:
                    fusion[k, i:i + feature.shape[1] - 4] = feature[k + 1, :-4]
                    i += feature.shape[1] - 4
                elif var_voice_index > voice_index:
                    fusion[k, i:i + feature.shape[1] - 4] = feature[k, :-4]
                    i += feature.shape[1] - 4
            # original features at the end
            fusion[k, i:] = vect
            assert i + len(vect) == total_features
        # print(fusion.shape, voice_index)
        return fusion
    else:
        total_features = 0
        # X[voice_index][file_index] [t, features]
        for X in Xs:
            ### WARNING 4 STANDS FOR THE NUMBER OF BITS USED BY THE BEAT REPRESENTATION
            total_features += X.shape[1] - 4
            # print(X[file_index].shape[1])
        total_features += 4  # Because we keep one beat
        fusion = np.zeros((Xs[voice_index].shape[0] - 1, total_features))

        for k, vect in enumerate(Xs[voice_index][:-1, :]):
            i = 0
            for var_voice_index, X in enumerate(Xs):
                feature = X
                if var_voice_index < voice_index:
                    fusion[k, i:i + feature.shape[1] - 4] = feature[k + 1, :-4]
                    i += feature.shape[1] - 4
                elif var_voice_index > voice_index:
                    fusion[k, i:i + feature.shape[1] - 4] = feature[k, :-4]
                    i += feature.shape[1] - 4
            # original features at the end
            fusion[k, i:] = vect
            assert i + len(vect) == total_features
        # print(fusion.shape, voice_index)
        return fusion


def fusion_features_with_fermata(Xs, voice_index, file_index=None):
    """

    """
    num_last_bits_removed = BEAT_SIZE + BITS_FERMATA
    if file_index is not None:
        total_features = 0
        # X[voice_index][file_index] [t, features]
        for X in Xs:
            total_features += X[file_index].shape[1] - num_last_bits_removed
        total_features += num_last_bits_removed  # Because we keep one beat
        fusion = np.zeros((Xs[voice_index][file_index].shape[0] - 1, total_features))

        for k, vect in enumerate(Xs[voice_index][file_index][:-1, :]):
            i = 0
            for var_voice_index, X in enumerate(Xs):
                feature = X[file_index]
                if var_voice_index < voice_index:
                    fusion[k, i:i + feature.shape[1] - num_last_bits_removed] = feature[k + 1, :-num_last_bits_removed]
                    i += feature.shape[1] - num_last_bits_removed
                elif var_voice_index > voice_index:
                    fusion[k, i:i + feature.shape[1] - num_last_bits_removed] = feature[k, :-num_last_bits_removed]
                    i += feature.shape[1] - num_last_bits_removed
            # original features at the end
            fusion[k, i:] = vect
            assert i + len(vect) == total_features
        # print(fusion.shape, voice_index)
        return fusion
    else:
        total_features = 0
        # X[voice_index][file_index] [t, features]
        for X in Xs:
            total_features += X.shape[1] - num_last_bits_removed
            # print(X[file_index].shape[1])
        total_features += num_last_bits_removed  # Because we keep one beat
        fusion = np.zeros((Xs[voice_index].shape[0] - 1, total_features))

        for k, vect in enumerate(Xs[voice_index][:-1, :]):
            i = 0
            for var_voice_index, X in enumerate(Xs):
                feature = X
                if var_voice_index < voice_index:
                    fusion[k, i:i + feature.shape[1] - num_last_bits_removed] = feature[k + 1, :-num_last_bits_removed]
                    i += feature.shape[1] - num_last_bits_removed
                elif var_voice_index > voice_index:
                    fusion[k, i:i + feature.shape[1] - num_last_bits_removed] = feature[k, :-num_last_bits_removed]
                    i += feature.shape[1] - num_last_bits_removed
            # original features at the end
            fusion[k, i:] = vect
            assert i + len(vect) == total_features
        # print(fusion.shape, voice_index)
        return fusion


def list_to_array(X):
    return np.concatenate(X).reshape((len(X),) + X[0].shape)


def make_raw_dataset(file_list, dataset_name, num_voices=4, transpose=False, min_pitches=None, max_pitches=None,
                     voice_ids=None):
    """
    :param file_list:
    :param dataset_name:
    :param num_voices:
    :param transpose:
    :return:  tuple (X, min_pitches, max_pitches, num_voices) where X is list of (num_voices, time, 3)
    """
    if min_pitches is None or max_pitches is None:
        raise Exception('min_pitches and max_pitches must be provided as arguments. See compute_min_max_pitches')
    # Choose whether to extract fermatas or not
    to_inputs = chorale_to_inputs_with_fermata

    X = []
    for chorale_file in tqdm(file_list):
        try:
            inputs = to_inputs(chorale_file, num_voices=num_voices)
            if not transpose:
                X.append(inputs)
            else:
                # same transposition for every voice
                max_transposition = 128
                min_transposition = -128
                for k, _ in enumerate(voice_ids):
                    max_transposition = min(max_transposition,
                                            int(max_pitches[k] - max(inputs[k, :, P_INDEX])))

                    min_transposition = max(min_transposition,
                                            min_pitches[k] - int(min(inputs[k, :, P_INDEX])))

                for num_semitones in range(min_transposition, max_transposition + 1):
                    transposed_input = inputs.copy()
                    transposed_input[:, :, P_INDEX] += num_semitones
                    X.append(transposed_input)
        except AttributeError:
            pass
    dataset = (X, min_pitches, max_pitches, num_voices)
    pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def make_datasets(file_list, dataset_names, voice_ids, MIN_PITCH, MAX_PITCH,
                  transpose=False, export_matlab=False):
    """
    MIN_PITCH and MAX_PITCH are arrays !
    pickles a dataset
    simple version, only returns list of chorals
    """

    assert len(dataset_names) == len(voice_ids)
    print("Making datasets " + str(dataset_names))
    inputs = []
    X = []
    y = []
    for _ in range(len(dataset_names)):
        inputs.append([])
        X.append([])
        y.append([])

    for voice_index, voice_id in enumerate(voice_ids):
        for file_index, file_name in enumerate(file_list):
            inputs[voice_index].append(chorale_to_input(file_name, voice_id=voice_id))
    if export_matlab:
        import scipy.io as sio
        sio.savemat('corpus4voices.mat', {'corpus': np.array(inputs)})

    for voice_index, voice_id in enumerate(voice_ids):
        for file_index, _ in enumerate(file_list):
            try:
                if not transpose:
                    y[voice_index].append(inputs[voice_index][file_index][1:, :])
                    X[voice_index].append(feature_to_onehot_feature(
                        input_to_feature(inputs[voice_index][file_index])[:-1, :],
                        MAX_PITCH[voice_index] - MIN_PITCH[voice_index] + 1,
                        MIN_PITCH[voice_index], MAX_PITCH[voice_index]))
                else:
                    # same transposition for every voice
                    max_transposition = 128
                    min_transposition = -128
                    for k, _ in enumerate(voice_ids):
                        max_transposition = min(max_transposition,
                                                int(MAX_PITCH[k] - max(inputs[k][file_index][:, 0])))

                        min_transposition = max(min_transposition,
                                                MIN_PITCH[k] - int(min(inputs[k][file_index][:, 0])))

                    for num_semitones in range(min_transposition, max_transposition + 1):
                        transposed_input = translate_y([inputs[voice_index][file_index]], num_semitones)[0]
                        y[voice_index].append(transposed_input[1:, :])
                        X[voice_index].append(feature_to_onehot_feature(
                            inputs_to_feature([transposed_input], 0)[:-1, :],
                            MAX_PITCH[voice_index] - MIN_PITCH[voice_index] + 1,
                            MIN_PITCH[voice_index], MAX_PITCH[voice_index]))

            except AttributeError:
                print('print')

    # Fusion features
    num_chorals = len(X[voice_index])
    XX = []
    yy = []
    for voice_index in range(len(voice_ids)):
        XX.append([])
        yy.append([])
        for _ in range(num_chorals):
            XX[voice_index].append([])
            yy[voice_index].append([])

    for voice_index, _ in enumerate(voice_ids):
        for choral_index in range(num_chorals):
            XX[voice_index][choral_index] = fusion_features(X, voice_index, choral_index)
            yy[voice_index][choral_index] = y[voice_index][choral_index][:-1]

    for voice_index, dataset_name in enumerate(dataset_names):
        dataset = {'X': XX[voice_index], 'y': yy[voice_index],
                   'min_pitch': MIN_PITCH[voice_index], 'max_pitch': MAX_PITCH[voice_index]
                   }
        pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def make_datasets_with_fermata(file_list, dataset_names, voice_ids, MIN_PITCH, MAX_PITCH,
                               transpose=False, export_matlab=False):
    """
    MIN_PITCH and MAX_PITCH are arrays !
    pickles a dataset
    simple version, only returns list of chorals
    :rtype: object
    """

    assert len(dataset_names) == len(voice_ids)
    print("Making datasets " + str(dataset_names))
    inputs = []
    X = []
    y = []
    for _ in range(len(dataset_names)):
        inputs.append([])
        X.append([])
        y.append([])

    for voice_index, voice_id in enumerate(voice_ids):
        for file_index, file_name in enumerate(file_list):
            inputs[voice_index].append(chorale_to_input_with_fermata(file_name, voice_id=voice_id))
    if export_matlab:
        import scipy.io as sio

        sio.savemat('corpus4voices.mat', {'corpus': np.array(inputs)})

    for voice_index, voice_id in enumerate(voice_ids):
        for file_index, _ in enumerate(file_list):
            try:
                if not transpose:
                    # Discard fermatas for labels ( [: 2])
                    y[voice_index].append(inputs[voice_index][file_index][1:, :2])
                    X[voice_index].append(feature_to_onehot_feature(
                        input_to_feature_with_fermata(inputs[voice_index][file_index], voice_index)[:-1, :],
                        MAX_PITCH[voice_index] - MIN_PITCH[voice_index] + 1,
                        MIN_PITCH[voice_index], MAX_PITCH[voice_index]))
                else:
                    # same transposition for every voice
                    max_transposition = 128
                    min_transposition = -128
                    for k, _ in enumerate(voice_ids):
                        max_transposition = min(max_transposition,
                                                int(MAX_PITCH[k] - max(inputs[k][file_index][:, 0])))

                        min_transposition = max(min_transposition,
                                                MIN_PITCH[k] - int(min(inputs[k][file_index][:, 0])))

                    for num_semitones in range(min_transposition, max_transposition + 1):
                        transposed_input = translate_y([inputs[voice_index][file_index]], num_semitones)[0]
                        # Discard fermatas for labels ( [: 2])
                        y[voice_index].append(transposed_input[1:, :2])
                        X[voice_index].append(feature_to_onehot_feature(
                            inputs_to_feature_with_fermata([transposed_input], 0)[:-1, :],
                            MAX_PITCH[voice_index] - MIN_PITCH[voice_index] + 1,
                            MIN_PITCH[voice_index], MAX_PITCH[voice_index]))

            except AttributeError:
                print('print')

    # Fusion features
    num_chorals = len(X[voice_index])
    XX = []
    yy = []
    for voice_index in range(len(voice_ids)):
        XX.append([])
        yy.append([])
        for _ in range(num_chorals):
            XX[voice_index].append([])
            yy[voice_index].append([])

    for voice_index, _ in enumerate(voice_ids):
        for choral_index in range(num_chorals):
            XX[voice_index][choral_index] = fusion_features_with_fermata(X, voice_index, choral_index)
            yy[voice_index][choral_index] = y[voice_index][choral_index][:-1]

    for voice_index, dataset_name in enumerate(dataset_names):
        dataset = {'X': XX[voice_index], 'y': yy[voice_index],
                   'min_pitch': MIN_PITCH[voice_index], 'max_pitch': MAX_PITCH[voice_index]
                   }
        pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def zero_padding(mat, size=16):
    m = np.array(mat)
    zeros_shape = (size,) + m.shape[1:]
    return np.concatenate((np.zeros(zeros_shape),
                           m,
                           np.zeros(zeros_shape)))


def unpickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        X = save['X']
        y = save['y']
        min_pitch = save['min_pitch']
        max_pitch = save['max_pitch']
        del save  # hint to help gc free up memory
        print('Number of chorales: ', len(X))
        print('X, y:', X[0].shape, y[0].shape)
        return X, y, min_pitch, max_pitch


def convert_list_to_X(pitch_and_articulation_list, NUM_PITCHES, MIN_PITCH, MAX_PITCH, offset=0):
    """
    list of pitches, starting on offset
    returns matrix length * num_features
    """
    return feature_to_onehot_feature(
        input_to_feature(
            pitch_and_articulation_list, initial_beat=offset), NUM_PITCHES, MIN_PITCH, MAX_PITCH)


# does not take articulation into account for the moment !
def create_consecutive_batch_generator(X, y, BATCH_SIZE, TIMESTEPS, NUM_PITCHES, MIN_PITCH, MAX_PITCH):
    """
    X is a list of chorals
    Returns input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, TIMESTEPS, NUM_PITCHES) (one hot encoded)
    """
    input = []
    labels = []
    batch = 0
    time_index = 0
    choral_index = np.random.randint(len(X))
    while True:
        if time_index + TIMESTEPS < y[choral_index].shape[0]:
            # if there is a full batch
            input.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
            labels.append(feature_to_onehot_feature(y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None],
                                                    NUM_PITCHES, MIN_PITCH, MAX_PITCH))  # Retain pitch
            time_index += 1
            batch += 1
        else:
            choral_index = np.random.randint(len(X))
            time_index = 0
        if batch == BATCH_SIZE:
            yield np.array(input, dtype=np.float32), np.array(labels, dtype=np.int32)
            batch = 0
            input = []
            labels = []


def pa_to_onehot(pa, min_pitch, max_pitch):
    """
    pitch and articulation tuple to onehot

    returns a vector with two ones (pitch and articulation)
    """

    # continuation
    if pa[A_INDEX] == 0:
        return np.concatenate((np.zeros((max_pitch + 1 - min_pitch,)),
                               np.array([1]))
                              )
    else:
        return np.concatenate((
            np.array(pa[P_INDEX] == np.arange(min_pitch, max_pitch + 1),
                     dtype=np.float32),
            np.array([0]))
        )


def paf_to_onehot(paf, min_pitch, max_pitch):
    """
    pitch and articulation and fermata tuple to onehot

    returns a vector with possibly three ones (pitch and articulation)
    """
    # continuation
    if paf[A_INDEX] == 0:
        return np.concatenate((np.zeros((max_pitch + 1 - min_pitch,)),
                               np.array([1]), np.array(paf[F_INDEX]))
                              )
    else:
        return np.concatenate((
            np.array(paf[P_INDEX] == np.arange(min_pitch, max_pitch + 1),
                     dtype=np.float32),
            np.array([0]), np.array(paf[F_INDEX]))
        )


def p_to_onehot(p, min_pitch, max_pitch):
    """
    pitch to one hot
    :param p:
    :param min_pitch:
    :param max_pitch: included !
    :return: np.array of shape (max_pitch - min_pitch + 1)
    """
    return np.array(p == np.arange(min_pitch, max_pitch + 1),
                    dtype=np.float32)


def ps_to_onehot(ps, min_pitches, max_pitches):
    """
    list of pitches to one hot representation
    :param ps: list of pitches
    :param min_pitch:
    :param max_pitch:
    :return: np.array of shape (sum_len(ps), max_pitches - min_pitches + 1)
    """
    vects = []
    for k, p in enumerate(ps):
        vects.append(p_to_onehot(p, min_pitch=min_pitches[k], max_pitch=max_pitches[k]))
    return np.concatenate(vects)


def pas_to_onehot(pas, min_pitches, max_pitches):
    """
    list of (pitch, articulation)  tuple (one for each voice) to ONE onehot-encoded vector
    :param pas:
    :param min_pitches:
    :param max_pitches:
    :return:
    """
    vects = []
    for k, pa in enumerate(pas):
        vects.append(pa_to_onehot(pa, min_pitch=min_pitches[k], max_pitch=max_pitches[k]))
    return np.concatenate(vects)


def pafs_to_onehot(pafs, min_pitches, max_pitches):
    """
    list of (pitch, articulation, fermata)  tuple (one for each voice) to ONE onehot-encoded vector
    :param pafs:
    :param min_pitches:
    :param max_pitches:
    :return:
    """
    vects = []
    for k, paf in enumerate(pafs):
        vects.append(paf_to_onehot(paf, min_pitch=min_pitches[k], max_pitch=max_pitches[k]))
    return np.concatenate(vects)


def as_pas_to_as_ps(chorale_as_pas, min_pitches, max_pitches):
    """
    convert chorale (num_voices, time, 2) to chorale (num_voices, time) by adding a slur symbol
    :return:
    """
    chorale_as_ps = np.zeros(chorale_as_pas.shape[:2])
    for voice_index, voice in enumerate(chorale_as_pas):
        for time_index, pa in enumerate(voice):
            # continuation
            if pa[A_INDEX] == 0:
                chorale_as_ps[voice_index, time_index] = max_pitches[voice_index] + 1
            else:
                chorale_as_ps[voice_index, time_index] = pa[P_INDEX]
    return chorale_as_ps


def as_ps_to_as_pas(chorale_as_ps, min_pitches, max_pitches):
    """
    convert chorale (num_voices, time) to chorale (num_voices, time, 2) by removing slur symbol
    max_pitches DO NOT INCLUDE slur_symbol_pitch
    :return:
    """
    chorale_as_pas = np.zeros(chorale_as_ps.shape + (2,))
    previous_pitch = 0
    for voice_index, voice in enumerate(chorale_as_ps):
        SLUR_SYMBOL_PITCH = max_pitches[voice_index] + 1
        for time_index, p in enumerate(voice):
            if not p == SLUR_SYMBOL_PITCH:
                previous_pitch = p
            # continuation
            if p == SLUR_SYMBOL_PITCH:
                chorale_as_pas[voice_index, time_index, :] = [previous_pitch, 0]
            else:
                chorale_as_pas[voice_index, time_index, :] = [previous_pitch, 1]
    return chorale_as_pas


def chorale_to_onehot(chorale, min_pitches, max_pitches, chorale_as_pas=True, fermatas=True):
    """
    chorale must be of shape (time, num_voices, 3) if chorale_as_pas
    else (time, num_voices)

    :param min_pitches: number of different pitches in chorale for each voice

    """
    mat = []
    if chorale_as_pas:
        if fermatas:
            print('must never go there')
            for pafs in chorale:
                mat.append(pafs_to_onehot(pafs, min_pitches, max_pitches))
        else:
            for pas in chorale:
                mat.append(pas_to_onehot(pas, min_pitches, max_pitches))
            return np.array(mat)
    # if chorale is given with only pitches + slur_symbol
    else:
        for ps in chorale:
            # Add slur_symbol
            mat.append(ps_to_onehot(ps, min_pitches, max_pitches))
        return np.array(mat)


def all_features(chorale, voice_index, time_index, timesteps, min_pitches, max_pitches, num_voices,
                 chorale_as_pas=True):
    """
    Returns all features for voice voice_index at time time_index from chorale
    shapes:
    (time, voice, 3) if chorale_as pas
    (time, voice) if not chorale_as_pas
    """
    if chorale_as_pas:

        return all_features_from_pa_chorale_with_fermatas(chorale, voice_index, time_index, timesteps, min_pitches,
                                                          max_pitches, num_voices)
    else:
        # when chorale already contains slur_symbols

        return all_features_from_slur_chorale(chorale, voice_index, time_index, timesteps,
                                              min_pitches, max_pitches,
                                              num_voices)


def all_features_from_slur_chorale(chorale, voice_index, time_index, timesteps,
                                   min_pitches, max_pitches, num_voices):
    """

    :param max_pitches: TRUE max_pitches from chorale_datasets
    :param min_pitches: TRUE min_pitches from chorale_datasets
    :param chorale:  (time, num_voices) numpy array
    :return:
            (left_feature,
            central_feature,
            right_feature,
            beat,
            label)
    """

    mask = np.array(voice_index == np.arange(num_voices), dtype=bool) == False

    left_feature = chorale_to_onehot(chorale[time_index - timesteps:time_index, :], min_pitches=min_pitches,
                                     max_pitches=max_pitches + 1, chorale_as_pas=False)

    right_feature = chorale_to_onehot(chorale[time_index + timesteps: time_index: -1, :], min_pitches=min_pitches,
                                      max_pitches=max_pitches + 1, chorale_as_pas=False)

    central_feature = ps_to_onehot(chorale[time_index, mask],
                                   min_pitches=min_pitches[mask],
                                   max_pitches=max_pitches[mask] + 1)

    # put timesteps=None to only have the current beat
    beat = to_beat(time_index, timesteps=timesteps)

    # if slur_symbol:
    label = p_to_onehot(chorale[time_index, voice_index], min_pitch=min_pitches[voice_index],
                        max_pitch=max_pitches[voice_index] + 1)

    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
            beat,
            np.array(label)
            )


def all_features_from_pa_chorale(chorale, voice_index, time_index, timesteps,
                                 min_pitches, max_pitches, num_voices):
    """
    :param num_voices:
    :param max_pitches:
    :param min_pitches:
    :param timesteps:
    :param time_index:
    :param voice_index:
    :param chorale:  (time, num_voices, 2) numpy array
    :return:(left_feature,
            central_feature,
            right_feature,
            beat,
            label,
            articulation) or
            (left_feature,
            central_feature,
            right_feature,
            beat,
            label)
             if slur_symbol is True
    """
    mask = np.array(voice_index == np.arange(num_voices), dtype=bool) == False
    left_feature = chorale_to_onehot(chorale[time_index - timesteps:time_index, :, :], min_pitches=min_pitches,
                                     max_pitches=max_pitches, fermatas=False)
    # In reverse order !
    right_feature = chorale_to_onehot(chorale[time_index + timesteps: time_index: -1, :, :], min_pitches=min_pitches,
                                      max_pitches=max_pitches, fermatas=False)

    central_feature = pas_to_onehot(chorale[time_index, mask, :], min_pitches=min_pitches[mask],
                                    max_pitches=max_pitches[mask])

    beat = to_beat(time_index, timesteps=timesteps)

    label = pa_to_onehot(chorale[time_index, voice_index, :], min_pitch=min_pitches[voice_index],
                         max_pitch=max_pitches[voice_index])
    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
            np.array(beat),
            np.array(label),
            )


def all_features_from_pa_chorale_with_fermatas(chorale, voice_index, time_index, timesteps, min_pitches, max_pitches,
                                               num_voices):
    """
        This function adds fermatas =(fermata_left, central_fermata, fermata_right)

        :param chorale:  (time, num_voices, 3) numpy array
        :returns
                (left_feature,
                central_feature,
                right_feature,
                beat,
                label)
        """
    mask = np.array(voice_index == np.arange(num_voices), dtype=bool) == False
    left_feature = chorale_to_onehot(chorale[time_index - timesteps:time_index, :, :F_INDEX],
                                     min_pitches=min_pitches,
                                     max_pitches=max_pitches, fermatas=False)

    right_feature = chorale_to_onehot(chorale[time_index + timesteps: time_index: -1, :, :F_INDEX],
                                      min_pitches=min_pitches, max_pitches=max_pitches, fermatas=False)

    central_feature = pas_to_onehot(chorale[time_index, mask, :F_INDEX], min_pitches=min_pitches[mask],
                                    max_pitches=max_pitches[mask])

    # beat is a tuple (left_beats, beat, right_beats) or only beat if timesteps is None
    beat = to_beat(time_index, timesteps=timesteps)

    # only need to retain fermatas from soprano
    fermatas = (list(map(lambda p: p_to_onehot(p, min_pitch=0, max_pitch=1),
                         chorale[time_index - timesteps:time_index, 0, F_INDEX])),
                p_to_onehot(chorale[time_index, 0, F_INDEX], 0, 1),
                list(map(lambda p: p_to_onehot(p, min_pitch=0, max_pitch=1),
                         chorale[time_index + timesteps: time_index: -1, 0, F_INDEX]))
                )

    label = pa_to_onehot(chorale[time_index, voice_index, :], min_pitch=min_pitches[voice_index],
                         max_pitch=max_pitches[voice_index])
    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
            np.array(beat),
            np.array(label),
            np.array(fermatas)
            )


def generator_from_raw_dataset(batch_size, timesteps, voice_index,
                               phase='train', percentage_train=0.8):
    """
     Returns a generator of
            (left_features,
            central_features,
            right_features,
            beats,
            labels,
            fermatas) tuples

            where fermatas = (fermatas_left, central_fermatas, fermatas_right)
    """

    dataset_name = RAW_DATASET
    X, min_pitches, max_pitches, num_voices = pickle.load(open(dataset_name, 'rb'))

    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))

    left_features = []
    right_features = []
    central_features = []
    beats = []
    beats_right = []
    beats_left = []
    labels = []
    fermatas_left = []
    central_fermatas = []
    fermatas_right = []
    batch = 0

    while True:
        chorale_index = np.random.choice(chorale_indices)
        extended_chorale = np.transpose(X[chorale_index], axes=(1, 0, 2))
        padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

        # Add fermatas before start and after ending
        extended_chorale = np.concatenate((np.zeros(padding_dimensions),
                                           extended_chorale,
                                           np.zeros(padding_dimensions)),
                                          axis=0)
        chorale_length = len(extended_chorale)

        time_index = np.random.randint(timesteps, chorale_length - timesteps)

        features = all_features(chorale=extended_chorale, voice_index=voice_index, time_index=time_index,
                                timesteps=timesteps, min_pitches=min_pitches, max_pitches=max_pitches,
                                num_voices=num_voices, chorale_as_pas=True)

        (left_feature, central_feature, right_feature,
         (beat_left, beat, beat_right),
         label,
         (fermata_left, central_fermata, fermata_right)
         ) = features

        fermatas_left.append(fermata_left)
        fermatas_right.append(fermata_right)
        central_fermatas.append(central_fermata)

        left_features.append(left_feature)
        right_features.append(right_feature)
        central_features.append(central_feature)
        beats.append(beat)
        beats_right.append(beat_right)
        beats_left.append(beat_left)
        labels.append(label)
        batch += 1

        # if there is a full batch
        if batch == batch_size:
            next_element = (np.array(left_features, dtype=np.float32),
                            np.array(central_features, dtype=np.float32),
                            np.array(right_features, dtype=np.float32),
                            (np.array(beats_left, dtype=np.float32),
                             np.array(beats, dtype=np.float32),
                             np.array(beats_right, dtype=np.float32)
                             ),
                            np.array(labels, dtype=np.float32))
            # add fermatas
            next_element += ((np.array(fermatas_left, dtype=np.float32),
                              np.array(central_fermatas, dtype=np.float32),
                              np.array(fermatas_right, dtype=np.float32))
                             ,)

            yield next_element

            batch = 0

            left_features = []
            central_features = []
            right_features = []
            beats = []
            beats_left = []
            beats_right = []
            labels = []
            fermatas_left = []
            fermatas_right = []
            central_fermatas = []


def create_batch_generator(X, y, BATCH_SIZE, TIMESTEPS,
                           NUM_PITCHES, MIN_PITCH, MAX_PITCH,
                           y_onehot=True, phase='train', percentage_train=0.8):
    """
    X is a list of chorals
    Returns (input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, TIMESTEPS, num_pitches),
            replayed (BATCH_SIZE, TIMESTEPS, 2))

    :param y_onehot: True for onehot encoded output
    :param phase: 'train' or 'test'
    """
    inputs = []
    labels = []
    articulations = []
    batch = 0
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))

    while True:
        choral_index = np.random.choice(chorale_indices)
        time_index = np.random.randint(y[choral_index].shape[0] - TIMESTEPS)

        inputs.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
        # Retain pitch
        label = y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None]
        # Retain articulation
        articulation = y[choral_index][time_index:time_index + TIMESTEPS, 1][:, None]

        if y_onehot:
            label = feature_to_onehot_feature(label, NUM_PITCHES, MIN_PITCH, MAX_PITCH)
            articulation = feature_to_onehot_feature(articulation, 2, 0, 1)

        labels.append(label)
        articulations.append(articulation)

        batch += 1

        # if there is a full batch
        if batch == BATCH_SIZE:
            yield (np.array(inputs, dtype=np.float32),
                   np.array(labels, dtype=np.int32),
                   np.array(articulations, dtype=np.int32))
            batch = 0
            inputs = []
            labels = []
            articulations = []


def create_batch_generator_reharmo(X, y, voice_index,
                                   BATCH_SIZE, TIMESTEPS,
                                   MIN_PITCH, MAX_PITCH, NUM_PITCHES, given_voices,
                                   y_onehot=True, phase='train', percentage_train=0.8):
    """
    X is a list of chorals
    Returns (input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, num_pitches),
            replayed (BATCH_SIZE, 2))

    WARNING: sequences in X,y are padded with 16 zeros at the beginning and at the end
    given_voices MUST be at the beginning of voice_ids when building dataset

    :param voice_index: index of voice in voice_ids
    :param y_onehot: True for onehot encoded output
    :param phase: 'train' or 'test'
    """
    inputs = []
    labels = []
    articulations = []
    imposed_voices = []
    num_bits_imposed_voices = 0
    for k, voice in enumerate(given_voices):
        num_bits_imposed_voices += NUM_PITCHES[k]
    batch = 0
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    elif phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
    else:
        raise ValueError
    while True:
        choral_index = np.random.choice(chorale_indices)
        time_index = np.random.randint(y[choral_index].shape[0] - 2 * TIMESTEPS)

        inputs.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
        # Retain pitch (one for each timestep)
        label = y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None]
        # Retain articulation (one for each timestep)
        articulation = y[choral_index][time_index:time_index + TIMESTEPS, 1][:, None]
        # Add right part of imposed melodies in reversed order
        imposed_voice = (X[choral_index]
                         [time_index + 2 * TIMESTEPS - 1:
                         time_index + TIMESTEPS - 1: -1, :num_bits_imposed_voices])

        if y_onehot:
            label = feature_to_onehot_feature(label, NUM_PITCHES[voice_index], MIN_PITCH[voice_index],
                                              MAX_PITCH[voice_index])
            articulation = feature_to_onehot_feature(articulation, 2, 0, 1)

        labels.append(label[-1])
        articulations.append(articulation[-1])
        imposed_voices.append(imposed_voice)

        batch += 1

        # if there is a full batch
        if batch == BATCH_SIZE:
            yield (np.array(inputs, dtype=np.float32),
                   np.array(imposed_voices, dtype=np.float32),
                   np.array(labels, dtype=np.int32),
                   np.array(articulations, dtype=np.int32))
            batch = 0
            inputs = []
            labels = []
            articulations = []
            imposed_voices = []


def create_batch_generator_full(X, y, BATCH_SIZE, TIMESTEPS):
    """
    X is a list of chorals
    Returns input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, TIMESTEPS, 2) #retains pitch and articulation
    """
    input = []
    labels = []
    batch = 0
    while True:
        choral_index = np.random.randint(len(X))
        time_index = np.random.randint(y[choral_index].shape[0] - TIMESTEPS)
        # if there is a full batch
        input.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
        labels.append(y[choral_index][time_index:time_index + TIMESTEPS, :])  # Retain pitch and articulation
        batch += 1
        if batch == BATCH_SIZE:
            yield np.array(input, dtype=np.float32), np.array(labels, dtype=np.int32)
            batch = 0
            input = []
            labels = []


def create_batch_generator_one_hot(X, y, BATCH_SIZE, TIMESTEPS, NUM_PITCHES, MIN_PITCH, MAX_PITCH):
    """
    X is a list of chorals
    Returns input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, TIMESTEPS, num_pitches)

    :param X:
    :param y:
    :param BATCH_SIZE:
    :param TIMESTEPS:
    :param NUM_PITCHES:
    :param MIN_PITCH:
    :param MAX_PITCH:
    """
    input = []
    labels = []
    batch = 0
    while True:
        choral_index = np.random.randint(len(X))
        time_index = np.random.randint(y[choral_index].shape[0] - TIMESTEPS)
        # if there is a full batch
        input.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
        labels.append(feature_to_onehot_feature(y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None],
                                                NUM_PITCHES, MIN_PITCH, MAX_PITCH))  # Retain pitch
        batch += 1
        if batch == BATCH_SIZE:
            yield np.array(input, dtype=np.float32), np.array(labels, dtype=np.int32)
            batch = 0
            input = []
            labels = []


def create_batch_generator_one_hot_full(X, y, BATCH_SIZE, TIMESTEPS, NUM_PITCHES, MIN_PITCH, MAX_PITCH):
    '''
    X is a list of chorals
    Returns (input (BATCH_SIZE, TIMESTEPS, NUM_FEATURES),
            labels (BATCH_SIZE, TIMESTEPS, num_pitches),
            replayed (BATCH_SIZE, TIMESTEPS, 2))

    :param X:
    :param y:
    :param BATCH_SIZE:
    :param TIMESTEPS:
    :param NUM_PITCHES:
    :param MIN_PITCH:
    :param MAX_PITCH:
    '''
    input = []
    labels = []
    replayed = []
    batch = 0
    while True:
        choral_index = np.random.randint(len(X))
        time_index = np.random.randint(y[choral_index].shape[0] - TIMESTEPS)
        input.append(X[choral_index][time_index:time_index + TIMESTEPS, :])
        labels.append(feature_to_onehot_feature(y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None],
                                                NUM_PITCHES, MIN_PITCH, MAX_PITCH))  # Retain pitch
        replayed.append(feature_to_onehot_feature(y[choral_index][time_index:time_index + TIMESTEPS, 1][:, None],
                                                  2, 0, 1))  # Retain articulation
        batch += 1
        # if there is a full batch
        if batch == BATCH_SIZE:
            yield (np.array(input, dtype=np.float32),
                   np.array(labels, dtype=np.int32),
                   np.array(replayed, dtype=np.int32))
            batch = 0
            input = []
            labels = []
            replayed = []


def create_generator_one_hot(X, y, TIMESTEPS, NUM_PITCHES, MIN_PITCH, MAX_PITCH):
    """
    X is a list of chorals
    Returns input (TIMESTEPS, NUM_FEATURES),
            labels (TIMESTEPS, num_pitches)
    """
    input = []
    labels = []
    batch = 0
    while True:
        choral_index = np.random.randint(len(X))
        time_index = np.random.randint(y[choral_index].shape[0] - TIMESTEPS)
        # if there is a full batch
        yield (X[choral_index][time_index:time_index + TIMESTEPS, :],
               feature_to_onehot_feature(y[choral_index][time_index:time_index + TIMESTEPS, 0][:, None],
                                         NUM_PITCHES, MIN_PITCH, MAX_PITCH))


def translate_y(y, num_semitones):
    """
    Translates only the first column of each member of the list
    """
    yy = []
    for chorale in y:
        new_chorale = np.array(chorale)
        new_chorale[:, 1:] = chorale[:, 1:]
        new_chorale[:, 0] = chorale[:, 0] + num_semitones
        yy.append(new_chorale)
    return yy


def seq_to_stream(seq):
    """
    :param seq: list (one for each voice) of list of (pitch, articulation)
    :return:
    """
    score = stream.Score()
    for voice, v in enumerate(seq):
        part = stream.Part(id='part' + str(voice))
        dur = 0
        f = note.Rest()
        for k, n in enumerate(v):
            if n[1] == 1:
                # add previous note
                if not f.name == 'rest':
                    f.duration = duration.Duration(dur / SUBDIVISION)
                    part.append(f)

                dur = 1
                f = note.Note()
                f.pitch.midi = n[0]
            else:
                dur += 1
        # add last note
        f.duration = duration.Duration(dur / SUBDIVISION)
        part.append(f)
        score.insert(part)
    return score


def seqs_to_stream(seqs):
    """
    :param seqs: list of sequences
    a sequence is a list (one for each voice) of list of (pitch, articulation)
    add rests between sequences
    :return:
    """
    score = stream.Score()
    for voice_index in range(len(seqs[0])):
        part = stream.Part(id='part' + str(voice_index))
        for s_index, seq in enumerate(seqs):
            # print(voice_index, s_index)
            voice = seq[voice_index]
            dur = 0
            f = note.Rest()
            for k, n in enumerate(voice):
                if n[1] == 1:
                    # add previous note
                    if not f.name == 'rest':
                        f.duration = duration.Duration(dur / SUBDIVISION)
                        part.append(f)

                    dur = 1
                    f = note.Note()
                    f.pitch.midi = n[0]
                else:
                    dur += 1
            # add last note
            f.duration = duration.Duration(dur / SUBDIVISION)
            part.append(f)
            # add rests (8 beats)
            f = note.Rest()
            f.duration = duration.Duration(SUBDIVISION * 8)
            part.append(f)

        score.insert(part)
    return score


def seq_to_stream_slur(seq, min_pitches=None, max_pitches=None):
    """
    :param seq: list of pitches where max_pitch + 1 stands for the slur_symbol
    :return:
    """
    score = stream.Score()
    for voice_index, v in enumerate(seq):
        part = stream.Part(id='part' + str(voice_index))
        dur = 0
        f = note.Rest()
        for k, n in enumerate(v):
            # if it is a played note
            if n <= max_pitches[voice_index]:
                # add previous note
                if dur > 0:
                    f.duration = duration.Duration(dur / SUBDIVISION)
                    part.append(f)

                dur = 1
                if n == 0:
                    f = note.Rest()
                else:
                    f = note.Note()
                    f.pitch.midi = n
            else:
                dur += 1
        # add last note
        f.duration = duration.Duration(dur / SUBDIVISION)
        part.append(f)
        score.insert(part)
    return score


def initialization(dataset_path=None):
    from glob import glob
    print('Creating dataset')
    if dataset_path:
        chorale_list = filter_file_list(glob(dataset_path + '/*.mid') + glob(dataset_path + '/*.xml'))
        pickled_dataset = 'datasets/custom_dataset/' + dataset_path.split('/')[-1] + '.pickle'
    else:
        chorale_list = filter_file_list(corpus.getBachChorales(fileExtensions='xml'))
        pickled_dataset = RAW_DATASET

    min_pitches, max_pitches = compute_min_max_pitches(chorale_list, voices=voice_ids)

    make_raw_dataset(chorale_list, pickled_dataset,
                     num_voices=len(voice_ids),
                     transpose=True,
                     min_pitches=min_pitches, max_pitches=max_pitches,
                     voice_ids=voice_ids)


if __name__ == '__main__':
    chorale_list = filter_file_list(corpus.getBachChorales(fileExtensions='xml'))

    min_pitches, max_pitches = compute_min_max_pitches(chorale_list, voices=voice_ids)

    # make_raw_dataset(chorale_list, RAW_DATASET,
    #                  num_voices=len(voice_ids),
    #                  transpose=True,
    #                  min_pitches=min_pitches, max_pitches=max_pitches,
    #                  voice_ids=voice_ids)
    make_raw_dataset(chorale_list, RAW_DATASET_NO_TRANSPOSITION, num_voices=4, transpose=False,
                     min_pitches=min_pitches, max_pitches=max_pitches)
    exit()
