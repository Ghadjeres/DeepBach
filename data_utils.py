#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 mars 2016

@author: Gaetan Hadjeres
"""
import pickle
from tqdm import tqdm

import numpy as np
from music21 import corpus, converter, stream, note, duration, analysis

NUM_VOICES = 4

SUBDIVISION = 4  # quarter note subdivision
BEAT_SIZE = 4

BITS_FERMATA = 2  # number of bits needed to encode fermata
RANGE_FERMATA = 3  # 3 beats before fermatas
SPACING_FERMATAS = 12  # in beats
FERMATAS_LENGTH = 2  # in beats

P_INDEX = 0  # pitch index in representation
A_INDEX = 1  # articulation index in representation
F_INDEX = 2  # fermata index in representation

OCTAVE = 12

BACH_DATASET = 'datasets/raw_dataset/bach_dataset.pickle'

voice_ids = list(range(NUM_VOICES))  # soprano, alto, tenor, bass

SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'




def standard_name(note_or_rest):
    if isinstance(note_or_rest, note.Note):
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name
    if isinstance(note_or_rest, str):
        return note_or_rest


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    if note_or_rest_string == START_SYMBOL or note_or_rest_string == END_SYMBOL:
        return note.Rest()
    if note_or_rest_string == SLUR_SYMBOL:
        print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


def filter_file_list(file_list, num_voices=4):
    """
    Only retain num_voices voices chorales
    """
    l = []
    for k, file_name in enumerate(file_list):
        c = converter.parse(file_name)
        # print(k, file_name)
        if len(c.parts) == num_voices:
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


def to_beat(time, timesteps=None):
    """
    time is given in the number of 16th notes

    put timesteps=None to return only current beat

    Returns metrical position one-hot encoded

    IMPORTANT, right_beats is REVERSED
    """
    beat = [0] * BEAT_SIZE
    beat[time % BEAT_SIZE] = 1

    if timesteps is None:
        return beat
    left_beats = np.array(list(map(lambda x: to_onehot(x, BEAT_SIZE),
                                   np.arange(time - timesteps, time) % BEAT_SIZE)))

    right_beats = np.array(list(map(lambda x: to_onehot(x, BEAT_SIZE),
                                    np.arange(time + timesteps, time, -1) % BEAT_SIZE)))
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
    central_fermata = to_onehot(fermatas_melody[time], 2)
    if timesteps is None:
        return central_fermata
    fermatas_left = np.array(list(map(lambda f: to_onehot(f, 2),
                                      fermatas_melody[time - timesteps: time])))
    fermatas_right = np.array(list(map(lambda f: to_onehot(f, 2),
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
    central_fermata = to_onehot(is_fermata(time), 2)
    if timesteps is None:
        return central_fermata
    fermatas_left = np.array(list(map(lambda f: to_onehot((is_fermata(time), 2)),
                                      np.arange(time - timesteps, time))))
    fermatas_right = np.array(list(map(lambda f: to_onehot((is_fermata(time), 2)),
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


def chorale_to_inputs(chorale, num_voices, index2notes, note2indexes):
    """
    :param chorale: music21 chorale
    :param num_voices:
    :param index2notes:
    :param note2indexes:
    :return: (num_voices, time) matrix of indexes
    """
    inputs = []
    for voice_index in range(num_voices):
        inputs.append(part_to_inputs(chorale.parts[voice_index], index2notes[voice_index], note2indexes[voice_index]))
    return np.array(inputs)


def part_to_inputs(part, index2note, note2index):
    """
    Can modify note2index and index2note!
    :param part:
    :param note2index:
    :param index2note:
    :return:
    """
    length = int(part.duration.quarterLength * SUBDIVISION)  # in 16th notes
    list_notes = part.flat.notes
    list_note_strings = [n.nameWithOctave for n in list_notes]
    num_notes = len(list_notes)
    # add entries to dictionaries if not present
    # should only be called by make_dataset when transposing
    for note_name in list_note_strings:
        if note_name not in index2note.values():
            new_index = len(index2note)
            index2note.update({new_index: note_name})
            note2index.update({note_name: new_index})
            print('Warning: Entry ' + str({new_index: note_name}) + ' added to dictionaries')

    j = 0
    i = 0
    t = np.zeros((length, 2))
    is_articulated = True
    while i < length:
        if j < num_notes - 1:
            if list_notes[j + 1].offset > i / SUBDIVISION:
                t[i, :] = [note2index[standard_name(list_notes[j])], is_articulated]
                i += 1
                is_articulated = False
            else:
                j += 1
                is_articulated = True
        else:
            t[i, :] = [note2index[standard_name(list_notes[j])], is_articulated]
            i += 1
            is_articulated = False
    return list(map(lambda pa: pa[0] if pa[1] else note2index[SLUR_SYMBOL], t))


def _min_max_midi_pitch(note_strings):
    """

    :param note_strings:
    :return:
    """
    all_notes = list(map(lambda note_string: standard_note(note_string),
                         note_strings))
    min_pitch = min(list(
        map(lambda n: n.pitch.midi if n.isNote else 128,
            all_notes
            )
    )
    )
    max_pitch = max(list(
        map(lambda n: n.pitch.midi if n.isNote else 0,
            all_notes
            )
    )
    )
    return min_pitch, max_pitch


def make_dataset(chorale_list, dataset_name, num_voices=4, transpose=False, metadatas=None):
    # todo transposition
    X = []
    X_metadatas = []
    index2notes, note2indexes = create_index_dicts(chorale_list, num_voices=num_voices)

    # todo clean this part
    min_max_midi_pitches = np.array(list(map(lambda d: _min_max_midi_pitch(d.values()), index2notes)))
    min_midi_pitches = min_max_midi_pitches[:, 0]
    max_midi_pitches = min_max_midi_pitches[:, 1]
    for chorale_file in tqdm(chorale_list):
        try:
            chorale = converter.parse(chorale_file)
            if transpose:
                midi_pitches = [[n.pitch.midi for n in part.flat.notes] for part in chorale.parts]
                min_midi_pitches_current = np.array([min(l) for l in midi_pitches])
                max_midi_pitches_current = np.array([max(l) for l in midi_pitches])
                min_transposition = max(min_midi_pitches - min_midi_pitches_current)
                max_transposition = min(max_midi_pitches - max_midi_pitches_current)
                for t in range(min_transposition, max_transposition + 1):
                    try:
                        chorale_tranposed = chorale.transpose(t)
                        inputs = chorale_to_inputs(chorale_tranposed, num_voices=num_voices, index2notes=index2notes,
                                                   note2indexes=note2indexes
                                                   )
                        md = []
                        if metadatas:
                            for metadata in metadatas:
                                # todo add this
                                if metadata.is_global:
                                    pass
                                else:
                                    md.append(metadata.evaluate(chorale_tranposed))
                        X.append(inputs)
                        X_metadatas.append(md)
                    except KeyError:
                        pass
            else:
                print("Warning: no transposition! shouldn't be used!")
                inputs = chorale_to_inputs(chorale, num_voices=num_voices,
                                           index2notes=index2notes,
                                           note2indexes=note2indexes)
                X.append(inputs)

        except (AttributeError, IndexError):
            pass

    # todo save metadatas objects in pickle file
    dataset = (X, X_metadatas, num_voices, index2notes, note2indexes)
    pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(str(len(X)) + ' files written in ' + dataset_name)


#
# def p_to_onehot(p, min_pitch, max_pitch):
#     """
#     pitch to one hot
#     :param p:
#     :param min_pitch:
#     :param max_pitch: included !
#     :return: np.array of shape (max_pitch - min_pitch + 1)
#     """
#     return np.array(p == np.arange(min_pitch, max_pitch + 1),
#                     dtype=np.float32)


def to_onehot(index, num_indexes):
    return np.array(index == np.arange(0, num_indexes),
                    dtype=np.float32)


def chorale_to_onehot(chorale, num_pitches):
    """
    chorale is time major
    :param chorale:
    :param num_pitches:
    :return:
    """
    return np.array(list(map(lambda time_slice: time_slice_to_onehot(time_slice, num_pitches), chorale)))


def time_slice_to_onehot(time_slice, num_pitches):
    l = []
    for voice_index, voice in enumerate(time_slice):
        l.append(to_onehot(voice, num_pitches[voice_index]))
    return np.concatenate(l)


def all_features(chorale, voice_index, time_index, timesteps, num_pitches, num_voices):
    """
    chorale with time major
    :param chorale:
    :param voice_index:
    :param time_index:
    :param timesteps:
    :param num_pitches:
    :param num_voices:
    :return:
    """
    mask = np.array(voice_index == np.arange(num_voices), dtype=bool) == False
    num_pitches = np.array(num_pitches)

    left_feature = chorale_to_onehot(chorale[time_index - timesteps:time_index, :], num_pitches=num_pitches)

    right_feature = chorale_to_onehot(chorale[time_index + timesteps: time_index: -1, :], num_pitches=num_pitches)

    central_feature = time_slice_to_onehot(chorale[time_index, mask],
                                           num_pitches[mask])

    # put timesteps=None to only have the current beat
    beat = to_beat(time_index, timesteps=timesteps)
    label = to_onehot(chorale[time_index, voice_index], num_indexes=num_pitches[voice_index])

    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
            beat,
            np.array(label)
            )


def all_metadatas(chorale_metadatas, time_index=None, timesteps=None, metadatas=None):
    left = []
    right = []
    center = []
    for metadata_index, metadata in enumerate(metadatas):
        left.append(list(map(lambda value: to_onehot(value, num_indexes=metadata.num_values),
                             chorale_metadatas[metadata_index][time_index - timesteps:time_index])))
        right.append(list(map(lambda value: to_onehot(value, num_indexes=metadata.num_values),
                              chorale_metadatas[metadata_index][time_index + timesteps: time_index: -1])))
        center.append(to_onehot(chorale_metadatas[metadata_index][time_index],
                                num_indexes=metadata.num_values))
    left = np.concatenate(left, axis=1)
    right = np.concatenate(right, axis=1)
    center = np.concatenate(center)
    return left, center, right


def generator_from_raw_dataset(batch_size, timesteps, voice_index,
                               phase='train', percentage_train=0.8, pickled_dataset=BACH_DATASET,
                               transpose=True, metadatas=None):
    """
     Returns a generator of
            (left_features,
            central_features,
            right_features,
            beats,
            metas,
            labels,
            fermatas) tuples

            where fermatas = (fermatas_left, central_fermatas, fermatas_right)
    """

    X, X_metadatas, num_voices, index2notes, note2indexes = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))

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
    left_metas = []
    right_metas = []
    metas = []

    labels = []
    batch = 0

    while True:
        chorale_index = np.random.choice(chorale_indices)
        extended_chorale = np.transpose(X[chorale_index])
        chorale_metas = X_metadatas[chorale_index]
        padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

        start_symbols = np.array(list(map(lambda note2index: note2index[START_SYMBOL], note2indexes)))
        end_symbols = np.array(list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))

        extended_chorale = np.concatenate((np.full(padding_dimensions, start_symbols),
                                           extended_chorale,
                                           np.full(padding_dimensions, end_symbols)),
                                          axis=0)
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]
        chorale_length = len(extended_chorale)

        time_index = np.random.randint(timesteps, chorale_length - timesteps)

        features = all_features(chorale=extended_chorale, voice_index=voice_index, time_index=time_index,
                                timesteps=timesteps, num_pitches=num_pitches,
                                num_voices=num_voices)
        left_meta, meta, right_meta = all_metadatas(chorale_metadatas=extended_chorale_metas, metadatas=metadatas,
                                                    time_index=time_index, timesteps=timesteps)

        (left_feature, central_feature, right_feature,
         (beat_left, beat, beat_right),
         label
         ) = features

        left_features.append(left_feature)
        right_features.append(right_feature)
        central_features.append(central_feature)

        beats.append(beat)
        beats_right.append(beat_right)
        beats_left.append(beat_left)

        left_metas.append(left_meta)
        right_metas.append(right_meta)
        metas.append(meta)
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
                            (np.array(left_metas, dtype=np.float32),
                             np.array(metas, dtype=np.float32),
                             np.array(right_metas, dtype=np.float32)
                             ),
                            np.array(labels, dtype=np.float32))

            yield next_element

            batch = 0

            left_features = []
            central_features = []
            right_features = []
            beats = []
            beats_left = []
            beats_right = []
            left_metas = []
            right_metas = []
            metas = []
            labels = []


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


def indexed_chorale_to_score(seq, pickled_dataset):
    _, _, _, index2notes, note2indexes = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    slur_indexes = list(map(lambda d: d[SLUR_SYMBOL], note2indexes))

    score = stream.Score()
    for voice_index, v in enumerate(seq):
        part = stream.Part(id='part' + str(voice_index))
        dur = 0
        f = note.Rest()
        for k, n in enumerate(v):
            # if it is a played note
            if not n == slur_indexes[voice_index]:
                # add previous note
                if dur > 0:
                    f.duration = duration.Duration(dur / SUBDIVISION)
                    part.append(f)

                dur = 1
                f = standard_note(index2notes[voice_index][n])
            else:
                dur += 1
        # add last note
        f.duration = duration.Duration(dur / SUBDIVISION)
        part.append(f)
        score.insert(part)
    return score


def create_index_dicts(chorale_list, num_voices=4):
    """
    Returns two lists (index2notes, note2indexes) of size num_voices containing dictionaries
    :param chorale_list:
    :param num_voices:
    :param min_pitches:
    :param max_pitches:
    :return:
    """
    # store all notes
    voice_ranges = []
    for voice_index in range(num_voices):
        voice_range = set()
        for chorale_path in chorale_list:
            # todo transposition
            chorale = converter.parse(chorale_path)
            part = chorale.parts[voice_index].flat
            for n in part.notesAndRests:
                voice_range.add(standard_name(n))
        # add additional symbols
        voice_range.add(SLUR_SYMBOL)
        voice_range.add(START_SYMBOL)
        voice_range.add(END_SYMBOL)
        voice_ranges.append(voice_range)
    # create tables
    index2notes = []
    note2indexes = []
    for voice_index in range(num_voices):
        l = list(voice_ranges[voice_index])
        index2note = {}
        note2index = {}
        for k, n in enumerate(l):
            index2note.update({k: n})
            note2index.update({n: k})
        index2notes.append(index2note)
        note2indexes.append(note2index)
    return index2notes, note2indexes


def initialization(dataset_path=None, metadatas=None):
    from glob import glob
    print('Creating dataset')
    if dataset_path:
        chorale_list = filter_file_list(glob(dataset_path + '/*.mid') + glob(dataset_path + '/*.xml'),
                                        num_voices=NUM_VOICES)
        pickled_dataset = 'datasets/custom_dataset/' + dataset_path.split('/')[-1] + '.pickle'
    else:
        chorale_list = filter_file_list(corpus.getBachChorales(fileExtensions='xml'))
        pickled_dataset = BACH_DATASET

    min_pitches, max_pitches = compute_min_max_pitches(chorale_list, voices=voice_ids)

    make_dataset(chorale_list, pickled_dataset,
                 num_voices=len(voice_ids),
                 transpose=True,
                 metadatas=metadatas)


if __name__ == '__main__':
    num_voices = 4
    make_dataset(None, BACH_DATASET, num_voices=4, transpose=False)
    exit()
