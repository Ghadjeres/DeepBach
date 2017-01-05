"""
Metadata classes
"""
from data_utils import SUBDIVISION
from music21 import analysis
import numpy as np


class Metadata:
    def __init__(self):
        self.num_values = None
        raise NotImplementedError

    def get_index(self, value):
        # trick with the 0 value
        raise NotImplementedError

    def evaluate(self, chorale):
        """
        takes a music21 chorale as input
        """
        raise NotImplementedError


# todo BeatMetadata class
# todo add strong/weak beat metadata
# todo add minor/major metadata
# todo add voice_i_playing metadata


class TickMetadatas(Metadata):
    def __init__(self, num_subdivisions):
        self.is_global = False
        self.num_values = num_subdivisions

    def evaluate(self, chorale):
        # suppose all pieces start on a beat
        length = int(chorale.duration.quarterLength * SUBDIVISION)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))


class KeyMetadatas(Metadata):
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.is_global = False
        self.num_max_sharps = 7
        self.num_values = 16

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
            key_signatures[time_index] = res[measure_index].sharps + self.num_max_sharps + 1
        return np.array(key_signatures, dtype=np.int32)


class FermataMetadatas(Metadata):
    def __init__(self, ):
        self.is_global = False
        self.num_values = 2

    def get_index(self, value):
        # values are 1 and 0
        return value

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
