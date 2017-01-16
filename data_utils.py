#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 mars 2016

@author: Gaetan Hadjeres
"""
import pickle

from music21.analysis.floatingKey import FloatingKeyException
from tqdm import tqdm

import numpy as np
from music21 import corpus, converter, stream, note, duration, analysis, interval

NUM_VOICES = 4

SUBDIVISION = 4  # quarter note subdivision
BEAT_SIZE = 4

SOP = 0
BASS = 1

OCTAVE = 12

BACH_DATASET = 'datasets/raw_dataset/bach_dataset.pickle'

voice_ids_default = list(range(NUM_VOICES))  # soprano, alto, tenor, bass

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


def chorale_to_inputs(chorale, voice_ids, index2notes, note2indexes):
    """
    :param chorale: music21 chorale
    :param voice_ids:
    :param index2notes:
    :param note2indexes:
    :return: (num_voices, time) matrix of indexes
    """
    inputs = []
    for voice_index, voice_id in enumerate(voice_ids):
        inputs.append(part_to_inputs(chorale.parts[voice_id], index2notes[voice_index], note2indexes[voice_index]))
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


def make_dataset(chorale_list, dataset_name, voice_ids=voice_ids_default, transpose=False, metadatas=None):
    X = []
    X_metadatas = []
    index2notes, note2indexes = create_index_dicts(chorale_list, voice_ids=voice_ids)

    # todo clean this part
    min_max_midi_pitches = np.array(list(map(lambda d: _min_max_midi_pitch(d.values()), index2notes)))
    min_midi_pitches = min_max_midi_pitches[:, 0]
    max_midi_pitches = min_max_midi_pitches[:, 1]
    for chorale_file in tqdm(chorale_list):
        try:
            chorale = converter.parse(chorale_file)
            if transpose:
                midi_pitches = [[n.pitch.midi for n in chorale.parts[voice_id].flat.notes] for voice_id in voice_ids]
                min_midi_pitches_current = np.array([min(l) for l in midi_pitches])
                max_midi_pitches_current = np.array([max(l) for l in midi_pitches])
                min_transposition = max(min_midi_pitches - min_midi_pitches_current)
                max_transposition = min(max_midi_pitches - max_midi_pitches_current)
                for semi_tone in range(min_transposition, max_transposition + 1):
                    try:
                        # necessary, won't transpose correctly otherwise
                        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(semi_tone)
                        transposition_interval = interval.Interval(str(interval_nature) + interval_type)
                        chorale_tranposed = chorale.transpose(transposition_interval)
                        inputs = chorale_to_inputs(chorale_tranposed, voice_ids=voice_ids, index2notes=index2notes,
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
                        print('KeyError: File ' + chorale_file + ' skipped')
                    except FloatingKeyException:
                        print('FloatingKeyException: File ' + chorale_file + ' skipped')
            else:
                print("Warning: no transposition! shouldn't be used!")
                inputs = chorale_to_inputs(chorale, voice_ids=voice_ids,
                                           index2notes=index2notes,
                                           note2indexes=note2indexes)
                X.append(inputs)

        except (AttributeError, IndexError):
            pass

    dataset = (X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas)
    pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(str(len(X)) + ' files written in ' + dataset_name)


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

    if num_voices > 1:
        central_feature = time_slice_to_onehot(chorale[time_index, mask],
                                               num_pitches[mask])
    else:
        central_feature = []

    # put timesteps=None to only have the current beat
    # beat is now considered as a metadata
    # beat = to_beat(time_index, timesteps=timesteps)
    label = to_onehot(chorale[time_index, voice_index], num_indexes=num_pitches[voice_index])

    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
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
                               transpose=True):
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

    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)
    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
    if phase == 'all':
        chorale_indices = np.arange(int(len(X)))

    left_features = []
    right_features = []
    central_features = []
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
         label
         ) = features

        left_features.append(left_feature)
        right_features.append(right_feature)
        central_features.append(central_feature)

        left_metas.append(left_meta)
        right_metas.append(right_meta)
        metas.append(meta)
        labels.append(label)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            next_element = (
                (np.array(left_features, dtype=np.float32),
                 np.array(central_features, dtype=np.float32),
                 np.array(right_features, dtype=np.float32)
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
    """

    :param seq: voice major
    :param pickled_dataset:
    :return:
    """
    _, _, _, index2notes, note2indexes, _ = pickle.load(open(pickled_dataset, 'rb'))
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


def create_index_dicts(chorale_list, voice_ids=voice_ids_default):
    """
    Returns two lists (index2notes, note2indexes) of size num_voices containing dictionaries
    :param chorale_list:
    :param voice_ids:
    :param min_pitches:
    :param max_pitches:
    :return:
    """
    # store all notes
    voice_ranges = []
    for voice_id in voice_ids:
        voice_range = set()
        for chorale_path in chorale_list:
            # todo transposition
            chorale = converter.parse(chorale_path)
            part = chorale.parts[voice_id].flat
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
    for voice_index, _ in enumerate(voice_ids):
        l = list(voice_ranges[voice_index])
        index2note = {}
        note2index = {}
        for k, n in enumerate(l):
            index2note.update({k: n})
            note2index.update({n: k})
        index2notes.append(index2note)
        note2indexes.append(note2index)
    return index2notes, note2indexes


def initialization(dataset_path=None, metadatas=None, voice_ids=voice_ids_default, BACH_DATASET=BACH_DATASET):
    from glob import glob
    print('Creating dataset')
    if dataset_path:
        chorale_list = filter_file_list(glob(dataset_path + '/*.mid') + glob(dataset_path + '/*.xml'),
                                        num_voices=NUM_VOICES)
        pickled_dataset = 'datasets/custom_dataset/' + dataset_path.split('/')[-1] + '.pickle'
    else:
        chorale_list = filter_file_list(corpus.getBachChorales(fileExtensions='xml'))
        pickled_dataset = BACH_DATASET

    # remove wrong chorales:
    min_pitches, max_pitches = compute_min_max_pitches(chorale_list, voices=voice_ids)

    make_dataset(chorale_list, pickled_dataset,
                 voice_ids=voice_ids,
                 transpose=True,
                 metadatas=metadatas)


# specific methods when number of rests
def split_note(n, max_length):
    """
    :param n:
    :param max_length: in quarter length
    :return:
    """
    if n.duration.quarterLength > max_length:
        l = []
        o = n.offset
        start = n.offset
        end = n.offset + n.duration.quarterLength

        while o < n.offset + n.duration.quarterLength:
            # new note
            f = standard_note(standard_name(n))
            if o + max_length:
                # todo tout est faux !
                new_length = max_length - o % max_length
            f.duration.quarterLength = (new_length)
            l.append(f)
            o += new_length
    else:
        return [n]


def split_part(part, max_length, part_index=-1):
    new_part = stream.Part(id='part' + str(part_index))
    for n in part.notesAndRests:
        for new_note in split_note(n, max_length):
            new_part.append(new_note)
    return new_part


if __name__ == '__main__':
    make_dataset(None, BACH_DATASET, voice_ids=4, transpose=False)
    exit()
