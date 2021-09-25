import json
import os
import subprocess
import tempfile
from pathlib import Path

import click
import cog
import music21
from midi2audio import FluidSynth

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, KeyMetadata, TickMetadata
from DeepBach.model_manager import DeepBach


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""

        # music21.environment.set("musicxmlPath", "/bin/true")

        note_embedding_dim = 20
        meta_embedding_dim = 20
        num_layers = 2
        lstm_hidden_size = 256
        dropout_lstm = 0.5
        linear_hidden_size = 256
        batch_size = 256
        num_epochs = 5
        train = False
        num_iterations = 500
        sequence_length_ticks = 64

        dataset_manager = DatasetManager()

        metadatas = [FermataMetadata(), TickMetadata(subdivision=4), KeyMetadata()]
        chorale_dataset_kwargs = {
            "voice_ids": [0, 1, 2, 3],
            "metadatas": metadatas,
            "sequences_size": 8,
            "subdivision": 4,
        }
        bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
            name="bach_chorales", **chorale_dataset_kwargs
        )
        dataset = bach_chorales_dataset

        self.deepbach = DeepBach(
            dataset=dataset,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            linear_hidden_size=linear_hidden_size,
        )

        self.deepbach.load()

        # load fluidsynth fo rmidi 2 audio conversion
        self.fs = FluidSynth()

        # self.converter = music21.converter.parse('path_to_musicxml.xml')

    @cog.input(
        "num_iterations",
        type=int,
        help="Number of parallel pseudo-Gibbs sampling iterations",
        default=500,
    )
    @cog.input(
        "sequence_length_ticks",
        type=int,
        min=16,
        help="Length of the generated chorale (in ticks)",
        default=64,
    )
    @cog.input(
        "output_type",
        type=str,
        help="Output representation type: can be audio or midi",
        default="audio",
        options=["midi", "audio"],
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, num_iterations, sequence_length_ticks, output_type, seed):
        """Score Generation"""
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

        score, tensor_chorale, tensor_metadata = self.deepbach.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
        )

        if output_type == "audio":
            output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
            output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"

            midi_score = score.write("midi")
            self.fs.midi_to_audio(midi_score, str(output_path_wav))

            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(output_path_wav),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(output_path_mp3),
                ],
            )

            return output_path_mp3

        elif output_type == "midi":
            output_path_midi = Path(tempfile.mkdtemp()) / "output.mid"
            score.write("midi", fp=output_path_midi)
            return output_path_midi
