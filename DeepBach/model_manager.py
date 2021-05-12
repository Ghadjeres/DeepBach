"""
@author: Gaetan Hadjeres
"""

from DatasetManager.metadata import FermataMetadata
import numpy as np
import torch
from DeepBach.helpers import cuda_variable, to_numpy

from torch import optim, nn
from tqdm import tqdm

from DeepBach.voice_model import VoiceModel


class DeepBach:
    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size,
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

    # Utils
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

    def train(self, main_voice_index=None,
              **kwargs):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.train(main_voice_index=voice_index, **kwargs)
        else:
            voice_model = self.voice_models[main_voice_index]
            if self.activate_cuda:
                voice_model.cuda()
            optimizer = optim.Adam(voice_model.parameters())
            voice_model.train_model(optimizer=optimizer, **kwargs)

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
                   random_init=True
                   ):
        """

        :param temperature:
        :param batch_size_per_voice:
        :param num_iterations:
        :param sequence_length_ticks:
        :param tensor_chorale:
        :param tensor_metadata:
        :param time_index_range_ticks: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between timesteps a and b
        :param voice_index_range: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between voice_index a and b
        :param fermatas: list[Fermata]
        :param random_init: boolean, whether or not to randomly initialize
        the portion of the score on which we apply the pseudo-Gibbs algorithm
        :return: tuple (
        generated_score [music21 Stream object],
        tensor_chorale (num_voices, chorale_length) torch.IntTensor,
        tensor_metadata (num_voices, chorale_length, num_metadata) torch.IntTensor
        )
        """
        self.eval_phase()

        # --Process arguments
        # initialize generated chorale
        # tensor_chorale = self.dataset.empty_chorale(sequence_length_ticks)
        if tensor_chorale is None:
            tensor_chorale = self.dataset.random_score_tensor(
                sequence_length_ticks)
        else:
            sequence_length_ticks = tensor_chorale.size(1)

        # initialize metadata
        if tensor_metadata is None:
            test_chorale = next(self.dataset.corpus_it_gen().__iter__())
            tensor_metadata = self.dataset.get_metadata_tensor(test_chorale)

            if tensor_metadata.size(1) < sequence_length_ticks:
                tensor_metadata = tensor_metadata.repeat(1, sequence_length_ticks // tensor_metadata.size(1) + 1, 1)

            # todo do not work if metadata_length_ticks > sequence_length_ticks
            tensor_metadata = tensor_metadata[:, :sequence_length_ticks, :]
        else:
            tensor_metadata_length = tensor_metadata.size(1)
            assert tensor_metadata_length == sequence_length_ticks

        if fermatas is not None:
            tensor_metadata = self.dataset.set_fermatas(tensor_metadata,
                                                        fermatas)

        # timesteps_ticks is the number of ticks on which we unroll the LSTMs
        # it is also the padding size
        timesteps_ticks = self.dataset.sequences_size * self.dataset.subdivision // 2
        if time_index_range_ticks is None:
            time_index_range_ticks = [timesteps_ticks, sequence_length_ticks + timesteps_ticks]
        else:
            a_ticks, b_ticks = time_index_range_ticks
            assert 0 <= a_ticks < b_ticks <= sequence_length_ticks
            time_index_range_ticks = [a_ticks + timesteps_ticks, b_ticks + timesteps_ticks]

        if voice_index_range is None:
            voice_index_range = [0, self.dataset.num_voices]

        tensor_chorale = self.dataset.extract_score_tensor_with_padding(
            tensor_score=tensor_chorale,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks
        )

        tensor_metadata_padded = self.dataset.extract_metadata_with_padding(
            tensor_metadata=tensor_metadata,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks
        )

        # randomize regenerated part
        if random_init:
            a, b = time_index_range_ticks
            tensor_chorale[voice_index_range[0]:voice_index_range[1], a:b] = self.dataset.random_score_tensor(
                b - a)[voice_index_range[0]:voice_index_range[1], :]

        tensor_chorale = self.parallel_gibbs(
            tensor_chorale=tensor_chorale,
            tensor_metadata=tensor_metadata_padded,
            num_iterations=num_iterations,
            timesteps_ticks=timesteps_ticks,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice,
            time_index_range_ticks=time_index_range_ticks,
            voice_index_range=voice_index_range,
        )

        # get fermata tensor
        for metadata_index, metadata in enumerate(self.dataset.metadatas):
            if isinstance(metadata, FermataMetadata):
                break


        score = self.dataset.tensor_to_score(
            tensor_score=tensor_chorale,
            fermata_tensor=tensor_metadata[:, :, metadata_index])

        return score, tensor_chorale, tensor_metadata

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
        """
        Parallel pseudo-Gibbs sampling
        tensor_chorale and tensor_metadata are padded with
        timesteps_ticks START_SYMBOLS before,
        timesteps_ticks END_SYMBOLS after
        :param tensor_chorale: (num_voices, chorale_length) tensor
        :param tensor_metadata: (num_voices, chorale_length) tensor
        :param timesteps_ticks:
        :param num_iterations: number of Gibbs sampling iterations
        :param batch_size_per_voice: number of simultaneous parallel updates
        :param temperature: final temperature after simulated annealing
        :param time_index_range_ticks: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between timesteps a and b
        :param voice_index_range: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between voice_index a and b
        :return: (num_voices, chorale_length) tensor
        """
        start_voice, end_voice = voice_index_range
        # add batch_dimension
        tensor_chorale = tensor_chorale.unsqueeze(0)
        tensor_chorale_no_cuda = tensor_chorale.clone()
        tensor_metadata = tensor_metadata.unsqueeze(0)

        # to variable
        tensor_chorale = cuda_variable(tensor_chorale, volatile=True)
        tensor_metadata = cuda_variable(tensor_metadata, volatile=True)

        min_temperature = temperature
        temperature = 1.1

        # Main loop
        for iteration in tqdm(range(num_iterations)):
            # annealing
            temperature = max(min_temperature, temperature * 0.9993)
            # print(temperature)
            time_indexes_ticks = {}
            probas = {}

            for voice_index in range(start_voice, end_voice):
                batch_notes = []
                batch_metas = []

                time_indexes_ticks[voice_index] = []

                # create batches of inputs
                for batch_index in range(batch_size_per_voice):
                    time_index_ticks = np.random.randint(
                        *time_index_range_ticks)
                    time_indexes_ticks[voice_index].append(time_index_ticks)

                    notes, label = (self.voice_models[voice_index]
                                    .preprocess_notes(
                            tensor_chorale=tensor_chorale[
                                           :, :,
                                           time_index_ticks - timesteps_ticks:
                                           time_index_ticks + timesteps_ticks],
                            time_index_ticks=timesteps_ticks
                        )
                    )
                    metas = self.voice_models[voice_index].preprocess_metas(
                        tensor_metadata=tensor_metadata[
                                        :, :,
                                        time_index_ticks - timesteps_ticks:
                                        time_index_ticks + timesteps_ticks,
                                        :],
                        time_index_ticks=timesteps_ticks
                    )

                    batch_notes.append(notes)
                    batch_metas.append(metas)

                # reshape batches
                batch_notes = list(map(list, zip(*batch_notes)))
                batch_notes = [torch.cat(lcr) if lcr[0] is not None else None
                               for lcr in batch_notes]
                batch_metas = list(map(list, zip(*batch_metas)))
                batch_metas = [torch.cat(lcr)
                               for lcr in batch_metas]

                # make all estimations
                probas[voice_index] = (self.voice_models[voice_index]
                                       .forward(batch_notes, batch_metas)
                                       )
                probas[voice_index] = nn.Softmax(dim=1)(probas[voice_index])

            # update all predictions
            for voice_index in range(start_voice, end_voice):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    probas_pitch = to_numpy(probas_pitch)

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # avoid non-probabilities
                    probas_pitch[probas_pitch < 0] = 0

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    tensor_chorale_no_cuda[
                        0,
                        voice_index,
                        time_indexes_ticks[voice_index][batch_index]
                    ] = int(pitch)

            tensor_chorale = cuda_variable(tensor_chorale_no_cuda.clone(),
                                           volatile=True)

        return tensor_chorale_no_cuda[0, :, timesteps_ticks:-timesteps_ticks]
