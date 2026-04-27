"""
@author: Gaetan Hadjeres
Modified: train() now forwards lr / lr_patience / lr_factor to VoiceModel.train_model()
"""

from DatasetManager.metadata import FermataMetadata
import numpy as np
import torch
from DeepBach.helpers import cuda_variable, to_numpy

from torch import optim, nn
from tqdm import tqdm

from DeepBach.voice_model import VoiceModel

# GPU SPEEDUP: allow cuDNN to benchmark and select the fastest kernel for the
# fixed-size inputs DeepBach uses.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class DeepBach:
    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size,
                 models_dir=None,
                 ):
        self.dataset = dataset
        self.num_voices = self.dataset.num_voices
        self.num_metas = len(self.dataset.metadatas) + 1
        self.activate_cuda = torch.cuda.is_available()

        self.voice_models = [VoiceModel(
            dataset=self.dataset,
            main_voice_index=main_voice_index,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            hidden_size_linear=linear_hidden_size,
            models_dir=models_dir,
        )
            for main_voice_index in range(self.num_voices)
        ]

    def cuda(self, main_voice_index=None):
        if self.activate_cuda:
            if main_voice_index is None:
                for voice_index in range(self.num_voices):
                    self.cuda(voice_index)
            else:
                self.voice_models[main_voice_index].cuda()

    def load(self, main_voice_index=None):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.load(main_voice_index=voice_index)
        else:
            self.voice_models[main_voice_index].load()

    def save(self, main_voice_index=None):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.save(main_voice_index=voice_index)
        else:
            self.voice_models[main_voice_index].save()

    def train(self,
              main_voice_index=None,
              lr: float = 1e-3,
              lr_patience: int = 3,
              lr_factor: float = 0.5,
              **kwargs):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.train(
                    main_voice_index=voice_index,
                    lr=lr,
                    lr_patience=lr_patience,
                    lr_factor=lr_factor,
                    **kwargs,
                )
        else:
            voice_model = self.voice_models[main_voice_index]
            if self.activate_cuda:
                voice_model.cuda()
            optimizer = optim.Adam(voice_model.parameters(), lr=lr)
            voice_model.train_model(
                optimizer=optimizer,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                **kwargs,
            )

    def eval_phase(self):
        for voice_model in self.voice_models:
            voice_model.eval()

    def train_phase(self):
        for voice_model in self.voice_models:
            voice_model.train()

    def generation(self,
                   temperature=1.0,
                   batch_size_per_voice=8,
                   num_iterations=None,
                   sequence_length_ticks=160,
                   tensor_chorale=None,
                   tensor_metadata=None,
                   time_index_range_ticks=None,
                   voice_index_range=None,
                   fermatas=None,
                   random_init=True,
                   ):
        self.eval_phase()

        if tensor_chorale is None:
            tensor_chorale = self.dataset.random_score_tensor(sequence_length_ticks)
        else:
            sequence_length_ticks = tensor_chorale.size(1)

        if tensor_metadata is None:
            test_chorale = next(self.dataset.corpus_it_gen().__iter__())
            tensor_metadata = self.dataset.get_metadata_tensor(test_chorale)
            if tensor_metadata.size(1) < sequence_length_ticks:
                tensor_metadata = tensor_metadata.repeat(
                    1, sequence_length_ticks // tensor_metadata.size(1) + 1, 1)
            tensor_metadata = tensor_metadata[:, :sequence_length_ticks, :]
        else:
            assert tensor_metadata.size(1) == sequence_length_ticks

        if fermatas is not None:
            tensor_metadata = self.dataset.set_fermatas(tensor_metadata, fermatas)

        timesteps_ticks = self.dataset.sequences_size * self.dataset.subdivision // 2

        if time_index_range_ticks is None:
            time_index_range_ticks = [timesteps_ticks,
                                      sequence_length_ticks + timesteps_ticks]
        else:
            a_ticks, b_ticks = time_index_range_ticks
            assert 0 <= a_ticks < b_ticks <= sequence_length_ticks
            time_index_range_ticks = [a_ticks + timesteps_ticks,
                                      b_ticks + timesteps_ticks]

        if voice_index_range is None:
            voice_index_range = [0, self.dataset.num_voices]

        tensor_chorale_padded = self.dataset.extract_score_tensor_with_padding(
            tensor_score=tensor_chorale,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks)
        tensor_metadata_padded = self.dataset.extract_metadata_with_padding(
            tensor_metadata=tensor_metadata,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks)

        if random_init:
            a, b = time_index_range_ticks
            v_start, v_end = voice_index_range
            random_chunk = self.dataset.random_score_tensor(b - a)
            tensor_chorale_padded[v_start:v_end, a:b] = random_chunk[v_start:v_end, :]

        tensor_chorale_final = self.parallel_gibbs(
            tensor_chorale=tensor_chorale_padded,
            tensor_metadata=tensor_metadata_padded,
            num_iterations=num_iterations,
            timesteps_ticks=timesteps_ticks,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice,
            time_index_range_ticks=time_index_range_ticks,
            voice_index_range=voice_index_range,
        )

        metadata_index = 0
        for i, metadata in enumerate(self.dataset.metadatas):
            if isinstance(metadata, FermataMetadata):
                metadata_index = i
                break

        score = self.dataset.tensor_to_score(
            tensor_score=tensor_chorale_final,
            fermata_tensor=tensor_metadata[:, :, metadata_index])

        return score, tensor_chorale_final, tensor_metadata

    def parallel_gibbs(self,
                       tensor_chorale,
                       tensor_metadata,
                       timesteps_ticks,
                       num_iterations=1000,
                       batch_size_per_voice=16,
                       temperature=1.,
                       time_index_range_ticks=None,
                       voice_index_range=None,
                       ):
        start_voice, end_voice = voice_index_range
        tensor_chorale = tensor_chorale.unsqueeze(0)
        tensor_chorale_no_cuda = tensor_chorale.clone()
        tensor_metadata = tensor_metadata.unsqueeze(0)
        tensor_metadata_cuda = cuda_variable(tensor_metadata)

        min_temperature = temperature
        temperature_sa = max(min_temperature, 1.1)

        for iteration in tqdm(range(num_iterations)):
            temperature_sa = max(min_temperature, temperature_sa * 0.9993)

            time_indexes_ticks = {}
            probas = {}

            with torch.no_grad():
                tensor_chorale_cuda = cuda_variable(tensor_chorale_no_cuda)

                for voice_index in range(start_voice, end_voice):
                    batch_notes = []
                    batch_metas = []
                    time_indexes_ticks[voice_index] = []

                    for batch_index in range(batch_size_per_voice):
                        time_index_ticks = np.random.randint(*time_index_range_ticks)
                        time_indexes_ticks[voice_index].append(time_index_ticks)

                        notes, _ = self.voice_models[voice_index].preprocess_notes(
                            tensor_chorale=tensor_chorale_cuda[
                                :, :,
                                time_index_ticks - timesteps_ticks:
                                time_index_ticks + timesteps_ticks],
                            time_index_ticks=timesteps_ticks)
                        metas = self.voice_models[voice_index].preprocess_metas(
                            tensor_metadata=tensor_metadata_cuda[
                                :, :,
                                time_index_ticks - timesteps_ticks:
                                time_index_ticks + timesteps_ticks, :],
                            time_index_ticks=timesteps_ticks)

                        batch_notes.append(notes)
                        batch_metas.append(metas)

                    batch_notes = list(map(list, zip(*batch_notes)))
                    batch_notes = [torch.cat(lcr) if lcr[0] is not None else None
                                   for lcr in batch_notes]
                    batch_metas = list(map(list, zip(*batch_metas)))
                    batch_metas = [torch.cat(lcr) for lcr in batch_metas]

                    output = self.voice_models[voice_index].forward(batch_notes, batch_metas)
                    probas[voice_index] = nn.Softmax(dim=1)(output)

                for voice_index in range(start_voice, end_voice):
                    for batch_index in range(batch_size_per_voice):
                        probas_pitch = to_numpy(probas[voice_index][batch_index])
                        probas_pitch = np.log(probas_pitch + 1e-12) / temperature_sa
                        probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7
                        probas_pitch[probas_pitch < 0] = 0
                        pitch = np.argmax(np.random.multinomial(1, probas_pitch))
                        tensor_chorale_no_cuda[
                            0, voice_index,
                            time_indexes_ticks[voice_index][batch_index]
                        ] = int(pitch)

        return tensor_chorale_no_cuda[0, :, timesteps_ticks:-timesteps_ticks]