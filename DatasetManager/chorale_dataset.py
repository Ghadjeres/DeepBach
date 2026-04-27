import music21
import torch
import numpy as np

from music21 import interval, stream
from torch.utils.data import TensorDataset
from tqdm import tqdm

from DatasetManager.helpers import standard_name, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, \
    standard_note, OUT_OF_RANGE, REST_SYMBOL
from DatasetManager.metadata import FermataMetadata
from DatasetManager.music_dataset import MusicDataset


class ChoraleDataset(MusicDataset):
    """
    Class for all chorale-like datasets
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 voice_ids,
                 metadatas=None,
                 sequences_size=8,
                 subdivision=4,
                 cache_dir=None):
        super(ChoraleDataset, self).__init__(cache_dir=cache_dir)
        self.voice_ids = voice_ids
        self.num_voices = len(voice_ids)
        self.name = name
        self.sequences_size = sequences_size
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.corpus_it_gen = corpus_it_gen
        self.voice_ranges = None
        self.metadatas = metadatas
        self.subdivision = subdivision

    def __repr__(self):
        return (f'ChoraleDataset('
                f'{self.voice_ids},'
                f'{self.name},'
                f'{[metadata.name for metadata in self.metadatas]},'
                f'{self.sequences_size},'
                f'{self.subdivision})')

    def iterator_gen(self):
        return (chorale for chorale in self.corpus_it_gen() if self.is_valid(chorale))

    def make_tensor_dataset(self):
        """
        Build the TensorDataset from the corpus.
        Called only when no cached version exists.
        """
        print('Making tensor dataset')
        self.compute_index_dicts()
        self.compute_voice_ranges()
        one_tick = 1 / self.subdivision
        chorale_tensor_dataset = []
        metadata_tensor_dataset = []

        for chorale_id, chorale in tqdm(enumerate(self.iterator_gen())):
            chorale_transpositions = {}
            metadatas_transpositions = {}

            # MODERNIZE: .flat is deprecated in music21 7+; use .flatten()
            for offsetStart in np.arange(
                    chorale.flatten().lowestOffset - (self.sequences_size - one_tick),
                    chorale.flatten().highestOffset,
                    one_tick):
                offsetEnd = offsetStart + self.sequences_size
                current_subseq_ranges = self.voice_range_in_subsequence(
                    chorale, offsetStart=offsetStart, offsetEnd=offsetEnd)
                transposition = self.min_max_transposition(current_subseq_ranges)
                min_t, max_t = transposition

                for semi_tone in range(min_t, max_t + 1):
                    start_tick = int(offsetStart * self.subdivision)
                    end_tick = int(offsetEnd * self.subdivision)
                    try:
                        if semi_tone not in chorale_transpositions:
                            (ct, mt) = self.transposed_score_and_metadata_tensors(
                                chorale, semi_tone=semi_tone)
                            chorale_transpositions[semi_tone] = ct
                            metadatas_transpositions[semi_tone] = mt
                        else:
                            ct = chorale_transpositions[semi_tone]
                            mt = metadatas_transpositions[semi_tone]

                        local_ct = self.extract_score_tensor_with_padding(ct, start_tick, end_tick)
                        local_mt = self.extract_metadata_with_padding(mt, start_tick, end_tick)

                        chorale_tensor_dataset.append(local_ct[None, :, :].int())
                        metadata_tensor_dataset.append(local_mt[None, :, :, :].int())
                    except KeyError:
                        print(f'KeyError with chorale {chorale_id}')

        chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        dataset = TensorDataset(chorale_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        return dataset

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(semi_tone)
        transposition_interval = interval.Interval(str(interval_nature) + str(interval_type))
        chorale_tranposed = score.transpose(transposition_interval)
        chorale_tensor = self.get_score_tensor(
            # MODERNIZE: .flat deprecated → .flatten()
            chorale_tranposed, offsetStart=0., offsetEnd=chorale_tranposed.flatten().highestTime)
        metadatas_transposed = self.get_metadata_tensor(chorale_tranposed)
        return chorale_tensor, metadatas_transposed

    def get_metadata_tensor(self, score):
        """
        Returns tensor (num_voices, chorale_length, len(self.metadatas) + 1)
        """
        md = []
        if self.metadatas:
            for metadata in self.metadatas:
                sequence_metadata = torch.from_numpy(
                    metadata.evaluate(score, self.subdivision)).long().clone()
                square_metadata = sequence_metadata.repeat(self.num_voices, 1)
                md.append(square_metadata[:, :, None])
        chorale_length = int(score.duration.quarterLength * self.subdivision)
        voice_id_metadata = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        square_metadata = torch.transpose(voice_id_metadata.repeat(chorale_length, 1), 0, 1)
        md.append(square_metadata[:, :, None])
        return torch.cat(md, 2)

    def set_fermatas(self, metadata_tensor, fermata_tensor):
        if self.metadatas:
            for metadata_index, metadata in enumerate(self.metadatas):
                if isinstance(metadata, FermataMetadata):
                    metadata_tensor[:, :, metadata_index] = fermata_tensor
                    break
        return metadata_tensor

    def add_fermata(self, metadata_tensor, time_index_start, time_index_stop):
        fermata_tensor = torch.zeros(self.sequences_size)
        fermata_tensor[time_index_start:time_index_stop] = 1
        return self.set_fermatas(metadata_tensor, fermata_tensor)

    def min_max_transposition(self, current_subseq_ranges):
        if current_subseq_ranges is None:
            return (0, 0)
        transpositions = [
            (min_pitch_corpus - min_pitch_current,
             max_pitch_corpus - max_pitch_current)
            for ((min_pitch_corpus, max_pitch_corpus),
                 (min_pitch_current, max_pitch_current))
            in zip(self.voice_ranges, current_subseq_ranges)
        ]
        transpositions = list(zip(*transpositions))
        return [max(transpositions[0]), min(transpositions[1])]

    def get_score_tensor(self, score, offsetStart, offsetEnd):
        chorale_tensor = []
        for part_id, part in enumerate(score.parts[:self.num_voices]):
            part_tensor = self.part_to_tensor(part, part_id,
                                              offsetStart=offsetStart,
                                              offsetEnd=offsetEnd)
            chorale_tensor.append(part_tensor)
        return torch.cat(chorale_tensor, 0)

    def part_to_tensor(self, part, part_id, offsetStart, offsetEnd):
        """
        :return: torch LongTensor (1, length)
        """
        # MODERNIZE: .flat deprecated → .flatten()
        list_notes_and_rests = list(part.flatten().getElementsByOffset(
            offsetStart=offsetStart,
            offsetEnd=offsetEnd,
            classList=[music21.note.Note, music21.note.Rest]))
        list_note_strings_and_pitches = [
            (n.nameWithOctave, n.pitch.midi) for n in list_notes_and_rests if n.isNote]
        length = int((offsetEnd - offsetStart) * self.subdivision)

        note2index = self.note2index_dicts[part_id]
        index2note = self.index2note_dicts[part_id]
        voice_range = self.voice_ranges[part_id]
        min_pitch, max_pitch = voice_range

        for note_name, pitch in list_note_strings_and_pitches:
            if pitch < min_pitch or pitch > max_pitch:
                note_name = OUT_OF_RANGE

            if note_name not in note2index:
                new_index = len(note2index)
                index2note[new_index] = note_name
                note2index[note_name] = new_index
                print(f'Warning: Entry {{{new_index}: {note_name!r}}} added to dictionaries')

        j = 0
        i = 0
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(list_notes_and_rests)
        while i < length:
            if j < num_notes - 1:
                if list_notes_and_rests[j + 1].offset > i / self.subdivision + offsetStart:
                    t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
                                                        voice_range=voice_range)],
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
                                                    voice_range=voice_range)],
                           is_articulated]
                i += 1
                is_articulated = False

        seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        return torch.from_numpy(seq).long()[None, :]

    def voice_range_in_subsequence(self, chorale, offsetStart, offsetEnd):
        """Returns None if any voice has no notes in the window (prevents transposition)."""
        voice_ranges = []
        for part in chorale.parts[:self.num_voices]:
            vr = self.voice_range_in_part(part, offsetStart=offsetStart, offsetEnd=offsetEnd)
            if vr is None:
                return None
            voice_ranges.append(vr)
        return voice_ranges

    def voice_range_in_part(self, part, offsetStart, offsetEnd):
        # MODERNIZE: .flat deprecated → .flatten()
        notes_in_subsequence = part.flatten().getElementsByOffset(
            offsetStart, offsetEnd,
            includeEndBoundary=False,
            mustBeginInSpan=True,
            mustFinishInSpan=False,
            classList=[music21.note.Note, music21.note.Rest])
        midi_pitches_part = [n.pitch.midi for n in notes_in_subsequence if n.isNote]
        if midi_pitches_part:
            return min(midi_pitches_part), max(midi_pitches_part)
        return None

    def compute_index_dicts(self):
        """
        Build note <-> index mappings for each voice.

        BUG FIX: original code used set() whose iteration order is
        non-deterministic.  Enumerating an unsorted set produced different
        index assignments every run, silently corrupting pretrained embeddings
        even when vocab sizes matched.

        Fix: sort the collected note strings before enumerating so the mapping
        is identical across runs, Python versions, and platforms.
        """
        print('Computing index dicts')
        self.index2note_dicts = [{} for _ in range(self.num_voices)]
        self.note2index_dicts = [{} for _ in range(self.num_voices)]

        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.update([SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL])

        for chorale in tqdm(self.iterator_gen()):
            for part_id, part in enumerate(chorale.parts[:self.num_voices]):
                # MODERNIZE: .flat deprecated → .flatten()
                for n in part.flatten().notesAndRests:
                    note_sets[part_id].add(standard_name(n))

        for note_set, index2note, note2index in zip(
                note_sets, self.index2note_dicts, self.note2index_dicts):
            # sorted() gives deterministic, reproducible index assignments
            for note_index, note in enumerate(sorted(note_set)):
                index2note[note_index] = note
                note2index[note] = note_index

    def is_valid(self, chorale):
        return len(chorale.parts) == 4

    def compute_voice_ranges(self):
        assert self.index2note_dicts is not None
        assert self.note2index_dicts is not None
        self.voice_ranges = []
        print('Computing voice ranges')
        for voice_index, note2index in tqdm(enumerate(self.note2index_dicts)):
            notes = [standard_note(ns) for ns in note2index]
            midi_pitches = [n.pitch.midi for n in notes if n.isNote]
            self.voice_ranges.append((min(midi_pitches), max(midi_pitches)))

    def extract_score_tensor_with_padding(self, tensor_score, start_tick, end_tick):
        """
        Returns tensor_score[:, start_tick:end_tick] with START/END padding as needed.
        """
        assert start_tick < end_tick
        assert end_tick > 0
        length = tensor_score.size()[1]
        padded = []

        if start_tick < 0:
            s = np.array([n2i[START_SYMBOL] for n2i in self.note2index_dicts])
            s = torch.from_numpy(s).long().repeat(-start_tick, 1).transpose(0, 1)
            padded.append(s)

        slice_start = max(start_tick, 0)
        slice_end = min(end_tick, length)
        padded.append(tensor_score[:, slice_start:slice_end])

        if end_tick > length:
            e = np.array([n2i[END_SYMBOL] for n2i in self.note2index_dicts])
            e = torch.from_numpy(e).long().repeat(end_tick - length, 1).transpose(0, 1)
            padded.append(e)

        return torch.cat(padded, 1)

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        """
        :param tensor_metadata: (num_voices, length, num_metadatas)
        """
        assert start_tick < end_tick
        assert end_tick > 0
        num_voices, length, num_metadatas = tensor_metadata.size()
        padded = []

        if start_tick < 0:
            s = np.zeros((self.num_voices, -start_tick, num_metadatas))
            padded.append(torch.from_numpy(s).long())

        slice_start = max(start_tick, 0)
        slice_end = min(end_tick, length)
        padded.append(tensor_metadata[:, slice_start:slice_end, :])

        if end_tick > length:
            e = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            padded.append(torch.from_numpy(e).long())

        return torch.cat(padded, 1)

    def empty_score_tensor(self, score_length):
        s = np.array([n2i[START_SYMBOL] for n2i in self.note2index_dicts])
        return torch.from_numpy(s).long().repeat(score_length, 1).transpose(0, 1)

    def random_score_tensor(self, score_length):
        t = np.array([np.random.randint(len(n2i), size=score_length)
                      for n2i in self.note2index_dicts])
        return torch.from_numpy(t).long()

    def tensor_to_score(self, tensor_score, fermata_tensor=None):
        """
        :param tensor_score: (num_voices, length)
        :return: music21 Score
        """
        slur_indexes = [n2i[SLUR_SYMBOL] for n2i in self.note2index_dicts]
        score = music21.stream.Score()
        num_voices = tensor_score.size(0)
        name_parts = (num_voices == 4)
        part_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

        for voice_index, (voice, index2note, slur_index) in enumerate(
                zip(tensor_score, self.index2note_dicts, slur_indexes)):
            add_fermata = False
            if name_parts:
                part = stream.Part(
                    id=part_names[voice_index],
                    partName=part_names[voice_index],
                    partAbbreviation=part_names[voice_index],
                    instrumentName=part_names[voice_index])
            else:
                part = stream.Part(id='part' + str(voice_index))

            dur = 0
            total_duration = 0
            f = music21.note.Rest()
            for note_index in [n.item() for n in voice]:
                if note_index != slur_indexes[voice_index]:
                    if dur > 0:
                        f.duration = music21.duration.Duration(dur / self.subdivision)
                        if add_fermata:
                            f.expressions.append(music21.expressions.Fermata())
                            add_fermata = False
                        part.append(f)
                    dur = 1
                    f = standard_note(index2note[note_index])
                    if fermata_tensor is not None and voice_index == 0:
                        add_fermata = (fermata_tensor[0, total_duration] == 1)
                    total_duration += 1
                else:
                    dur += 1
                    total_duration += 1

            f.duration = music21.duration.Duration(dur / self.subdivision)
            if add_fermata:
                f.expressions.append(music21.expressions.Fermata())
            part.append(f)
            score.insert(part)

        return score


class ChoraleBeatsDataset(ChoraleDataset):
    """Beat-level variant of ChoraleDataset (inherits the sorted dict fix)."""

    def __repr__(self):
        return (f'ChoraleBeatsDataset('
                f'{self.voice_ids},'
                f'{self.name},'
                f'{[metadata.name for metadata in self.metadatas]},'
                f'{self.sequences_size},'
                f'{self.subdivision})')

    def make_tensor_dataset(self):
        print('Making tensor dataset')
        self.compute_index_dicts()
        self.compute_voice_ranges()
        one_beat = 1.
        chorale_tensor_dataset = []
        metadata_tensor_dataset = []

        for chorale_id, chorale in tqdm(enumerate(self.iterator_gen())):
            chorale_transpositions = {}
            metadatas_transpositions = {}

            # MODERNIZE: .flat deprecated → .flatten()
            for offsetStart in np.arange(
                    chorale.flatten().lowestOffset - (self.sequences_size - one_beat),
                    chorale.flatten().highestOffset,
                    one_beat):
                offsetEnd = offsetStart + self.sequences_size
                current_subseq_ranges = self.voice_range_in_subsequence(
                    chorale, offsetStart=offsetStart, offsetEnd=offsetEnd)
                transposition = self.min_max_transposition(current_subseq_ranges)
                min_t, max_t = transposition

                for semi_tone in range(min_t, max_t + 1):
                    start_tick = int(offsetStart * self.subdivision)
                    end_tick = int(offsetEnd * self.subdivision)
                    try:
                        if semi_tone not in chorale_transpositions:
                            (ct, mt) = self.transposed_score_and_metadata_tensors(
                                chorale, semi_tone=semi_tone)
                            chorale_transpositions[semi_tone] = ct
                            metadatas_transpositions[semi_tone] = mt
                        else:
                            ct = chorale_transpositions[semi_tone]
                            mt = metadatas_transpositions[semi_tone]

                        local_ct = self.extract_score_tensor_with_padding(ct, start_tick, end_tick)
                        local_mt = self.extract_metadata_with_padding(mt, start_tick, end_tick)

                        chorale_tensor_dataset.append(local_ct[None, :, :].int())
                        metadata_tensor_dataset.append(local_mt[None, :, :, :].int())
                    except KeyError:
                        print(f'KeyError with chorale {chorale_id}')

        chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        dataset = TensorDataset(chorale_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        return dataset