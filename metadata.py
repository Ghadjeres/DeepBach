"""
Metadata classes
"""
from data_utils import SUBDIVISION
from music21 import analysis, note
import numpy as np


class Metadata:
    def __init__(self):
        self.num_values = None
        raise NotImplementedError

    def get_index(self, value):
        # trick with the 0 value
        raise NotImplementedError

    def get_value(self, index):
        raise NotImplementedError

    def evaluate(self, chorale):
        """
        takes a music21 chorale as input
        """
        raise NotImplementedError

    def generate(self, length):
        raise NotImplementedError


# todo add strong/weak beat metadata
# todo add minor/major metadata

# todo BeatMetadata class
class BeatMetadata(Metadata):
    def __init__(self):
        raise NotImplementedError

    def get_index(self, value):
        # trick with the 0 value
        raise NotImplementedError

    def get_value(self, index):
        raise NotImplementedError

    def evaluate(self, chorale):
        """
        takes a music21 chorale as input
        """
        raise NotImplementedError

    def generate(self, length):
        raise NotImplementedError


# todo add voice_i_playing metadata
class VoiceIPlaying(Metadata):
    def __init__(self, window_size):
        """ Initiate the VoiceIPlaying metadata.
        Voice I is considered to be muted if more than 'window_size' contiguous subdivisions that contains a rest.

        :param window_size: (int) number of subdivision
        """
        self.window_size = window_size
        self.is_global = False
        self.num_values = 2

    def get_index(self, value):
        # trick with the 0 value
        if value == 'Rest':
            return 0
        elif value == 'NotRest':
            return 1
        else:
            raise NotImplementedError

    def get_value(self, index):
        if index == 0:
            return 'Rest'
        elif index == 1:
            return 'NotRest'
        else:
            raise ValueError

    def evaluate(self, chorale):
        """
        takes a music21 chorale as input
        """
        num_voices = len(chorale.parts)
        length = int(chorale.duration.quarterLength * SUBDIVISION)
        is_playing = np.ones(shape=(length, num_voices))
        for voice_index in range(num_voices):
            current_part = chorale[voice_index].flat.getElementsByClass(note.GeneralNote).stream()
            curr_status = 'Rest'
            deb_time = 0.0
            fin_time = 0.0
            if not current_part:
                pass
            else:
                fin_time = current_part[0].offset
                for event in current_part:
                    if curr_status == 'Rest':
                        if event.isRest:
                            fin_time = event.offset + event.quarterLength
                        else:
                            deb_index = int(deb_time * SUBDIVISION)
                            fin_index = int(fin_time * SUBDIVISION)
                            if fin_index - deb_index > self.window_size:
                                is_playing[deb_index:fin_index, voice_index] = 0
                            else:
                                pass
                            deb_time = event.offset
                            fin_time = deb_time + event.quarterLength
                            curr_status = 'NotRest'
                    elif curr_status == 'NotRest':
                        deb_time = event.offset
                        fin_time = deb_time + event.quarterLength
                        if event.isRest:
                            curr_status = 'Rest'
                        else:
                            pass
                    else:
                        raise ValueError
            if curr_status == 'NotRest':
                deb_time = fin_time
            else:
                pass
            deb_index = int(deb_time * SUBDIVISION)
            fin_index = length
            is_playing[deb_index:fin_index, voice_index] = 0
        return is_playing

    def generate(self, length):
        return np.zeros(shape=(length,))


class TickMetadatas(Metadata):
    def __init__(self, num_subdivisions):
        self.is_global = False
        self.num_values = num_subdivisions

    def get_index(self, value):
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale):
        # suppose all pieces start on a beat
        length = int(chorale.duration.quarterLength * SUBDIVISION)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))

    def generate(self, length):
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))


class ModeMetadatas(Metadata):
    def __init__(self):
        self.is_global = False
        self.num_values = 3  # major, minor or other

    def get_index(self, value):
        if value == 'major':
            return 1
        if value == 'minor':
            return 2
        return 0

    def get_value(self, index):
        if index == 1:
            return 'major'
        if index == 2:
            return 'minor'
        return 'other'

    def evaluate(self, chorale):
        # init key analyzer
        ka = analysis.floatingKey.KeyAnalyzer(chorale)
        res = ka.run()

        measure_offset_map = chorale.parts[0].measureOffsetMap()
        length = int(chorale.duration.quarterLength * SUBDIVISION)  # in 16th notes

        modes = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / SUBDIVISION
            if beat_index in measure_offset_map:
                measure_index += 1
                modes[time_index] = self.get_index(res[measure_index].mode)

        return np.array(modes, dtype=np.int32)

    def generate(self, length):
        return np.full((length,), self.get_index('major'))


class KeyMetadatas(Metadata):
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.is_global = False
        self.num_max_sharps = 7
        self.num_values = 16

    def get_index(self, value):
        """

        :param value: number of sharps (between -7 and +7)
        :return: index in the representation
        """
        return value + self.num_max_sharps + 1

    def get_value(self, index):
        """

        :param index:  index (between 0 and self.num_values); 0 is unused (no constraint)
        :return: true number of sharps (between -7 and 7)
        """
        return index - 1 - self.num_max_sharps

    # todo check if this method is correct for windowSize > 1
    def evaluate(self, chorale):
        # init key analyzer
        ka = analysis.floatingKey.KeyAnalyzer(chorale)
        ka.windowSize = self.window_size
        res = ka.run()

        measure_offset_map = chorale.parts[0].measureOffsetMap()
        length = int(chorale.duration.quarterLength * SUBDIVISION)  # in 16th notes

        key_signatures = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / SUBDIVISION
            if beat_index in measure_offset_map:
                measure_index += 1
            key_signatures[time_index] = self.get_index(res[measure_index].sharps)
        return np.array(key_signatures, dtype=np.int32)

    def generate(self, length):
        return np.full((length,), self.get_index(0))


class FermataMetadatas(Metadata):
    def __init__(self):
        self.is_global = False
        self.num_values = 2

    def get_index(self, value):
        # values are 1 and 0
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale):
        part = chorale.parts[0]
        length = int(part.duration.quarterLength * SUBDIVISION)  # in 16th notes
        list_notes = part.flat.notes
        num_notes = len(list_notes)
        j = 0
        i = 0
        fermatas = np.zeros((length,))
        fermata = False
        while i < length:
            if j < num_notes - 1:
                if list_notes[j + 1].offset > i / SUBDIVISION:

                    if len(list_notes[j].expressions) == 1:
                        fermata = True
                    else:
                        fermata = False
                    fermatas[i] = fermata
                    i += 1
                else:
                    j += 1
            else:
                if len(list_notes[j].expressions) == 1:
                    fermata = True
                else:
                    fermata = False

                fermatas[i] = fermata
                i += 1
        return np.array(fermatas, dtype=np.int32)

    def generate(self, length):
        # fermata every 2 bars
        return np.array([1 if i % 32 > 28 else 0
                         for i in range(length)])
