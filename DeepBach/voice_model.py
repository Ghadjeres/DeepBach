"""
@author: Gaetan Hadjeres
Modified:
  - train_model() accepts lr_patience / lr_factor for ReduceLROnPlateau
  - best-val-loss checkpointing (only saves when val loss improves)
  - forward() and embed() are exactly the original — do NOT modify these
  - MODERNIZE: ReduceLROnPlateau verbose=True removed (deprecated in PyTorch 2.x);
    LR changes are printed manually instead
"""

import os
import random

import torch
from torch import nn

try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE = None

from DatasetManager.chorale_dataset import ChoraleDataset
from DeepBach.helpers import cuda_variable, init_hidden
from DeepBach.data_utils import reverse_tensor, mask_entry

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_MODELS_DIR = os.path.join(_PACKAGE_ROOT, 'models')


class VoiceModel(nn.Module):
    def __init__(self,
                 dataset: ChoraleDataset,
                 main_voice_index: int,
                 note_embedding_dim: int,
                 meta_embedding_dim: int,
                 num_layers: int,
                 lstm_hidden_size: int,
                 dropout_lstm: float,
                 hidden_size_linear=200,
                 models_dir: str = None,
                 ):
        super(VoiceModel, self).__init__()
        self.dataset = dataset
        self.main_voice_index = main_voice_index
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_notes_per_voice = [len(d) for d in dataset.note2index_dicts]
        self.num_voices = self.dataset.num_voices
        self.num_metas_per_voice = [
            metadata.num_values for metadata in dataset.metadatas
        ] + [self.num_voices]
        self.num_metas = len(self.dataset.metadatas) + 1
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.hidden_size_linear = hidden_size_linear

        self.models_dir = models_dir or _DEFAULT_MODELS_DIR

        self.other_voices_indexes = [i for i in range(self.num_voices)
                                     if i != main_voice_index]

        self.note_embeddings = nn.ModuleList(
            [nn.Embedding(num_notes, note_embedding_dim)
             for num_notes in self.num_notes_per_voice]
        )
        self.meta_embeddings = nn.ModuleList(
            [nn.Embedding(num_metas, meta_embedding_dim)
             for num_metas in self.num_metas_per_voice]
        )
        self.lstm_left = nn.LSTM(
            input_size=note_embedding_dim * self.num_voices + meta_embedding_dim * self.num_metas,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True)
        self.lstm_right = nn.LSTM(
            input_size=note_embedding_dim * self.num_voices + meta_embedding_dim * self.num_metas,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True)
        self.mlp_center = nn.Sequential(
            nn.Linear(note_embedding_dim * (self.num_voices - 1)
                      + meta_embedding_dim * self.num_metas,
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, lstm_hidden_size))
        self.mlp_predictions = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 3, hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, self.num_notes_per_voice[main_voice_index]))

        if torch.cuda.is_available():
            self._scaler = (GradScaler(_AMP_DEVICE) if _AMP_DEVICE
                            else GradScaler())
        else:
            self._scaler = None

    # ---------------------------------------------------------------------------
    # Dynamic embedding resize
    # ---------------------------------------------------------------------------

    def resize_embeddings_to_dataset(self):
        changed = False

        for voice_idx in range(self.num_voices):
            actual = len(self.dataset.note2index_dicts[voice_idx])
            emb = self.note_embeddings[voice_idx]
            current = emb.num_embeddings
            dim = emb.embedding_dim

            if actual == current:
                continue
            if actual < current:
                print(f'  [resize_embeddings] WARNING: voice {voice_idx} vocab '
                      f'shrank {current} -> {actual}; skipping.')
                continue

            print(f'  [resize_embeddings] Voice {voice_idx}: expanding embedding '
                  f'{current} -> {actual} (+{actual - current} new tokens)')
            new_emb = nn.Embedding(actual, dim)
            with torch.no_grad():
                new_emb.weight[:current] = emb.weight
            new_emb = new_emb.to(emb.weight.device)
            self.note_embeddings[voice_idx] = new_emb
            self.num_notes_per_voice[voice_idx] = actual
            changed = True

        main_vocab = len(self.dataset.note2index_dicts[self.main_voice_index])
        out_layer = self.mlp_predictions[-1]
        if out_layer.out_features != main_vocab:
            print(f'  [resize_embeddings] Resizing output projection: '
                  f'{out_layer.out_features} -> {main_vocab}')
            new_linear = nn.Linear(out_layer.in_features, main_vocab)
            new_linear = new_linear.to(out_layer.weight.device)
            with torch.no_grad():
                keep = min(out_layer.out_features, main_vocab)
                new_linear.weight[:keep] = out_layer.weight[:keep]
                new_linear.bias[:keep] = out_layer.bias[:keep]
            layers = list(self.mlp_predictions.children())
            layers[-1] = new_linear
            self.mlp_predictions = nn.Sequential(*layers)
            changed = True

        if changed:
            print(f'  [resize_embeddings] Voice {self.main_voice_index}: '
                  f'embeddings updated and ready for training.')
        return changed

    # ---------------------------------------------------------------------------
    # Forward pass — original implementation, do not modify
    # ---------------------------------------------------------------------------

    def forward(self, *input):
        notes, metas = input
        batch_size, num_voices, timesteps_ticks = notes[0].size()

        ln, cn, rn = notes
        ln, rn = ln.transpose(1, 2), rn.transpose(1, 2)
        notes = ln, cn, rn

        notes_embedded = self.embed(notes, type='note')
        metas_embedded = self.embed(metas, type='meta')

        input_embedded = [
            torch.cat([n, m], 2) if n is not None else None
            for n, m in zip(notes_embedded, metas_embedded)
        ]
        left, center, right = input_embedded

        hidden = init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        left, _ = self.lstm_left(left, hidden)
        left = left[:, -1, :]

        if self.num_voices == 1:
            center = cuda_variable(torch.zeros(batch_size, self.lstm_hidden_size))
        else:
            center = self.mlp_center(center[:, 0, :])

        hidden = init_hidden(self.num_layers, batch_size, self.lstm_hidden_size)
        right, _ = self.lstm_right(right, hidden)
        right = right[:, -1, :]

        predictions = self.mlp_predictions(torch.cat([left, center, right], 1))
        return predictions

    def embed(self, notes_or_metas, type):
        if type == 'note':
            embeddings = self.note_embeddings
            embedding_dim = self.note_embedding_dim
            other_voices_indexes = self.other_voices_indexes
        else:
            embeddings = self.meta_embeddings
            embedding_dim = self.meta_embedding_dim
            other_voices_indexes = range(self.num_metas)

        batch_size, timesteps_left_ticks, num_voices = notes_or_metas[0].size()
        batch_size, timesteps_right_ticks, _ = notes_or_metas[2].size()
        left, center, right = notes_or_metas

        left_embedded = torch.cat([
            embeddings[vid](left[:, :, vid])[:, :, None, :]
            for vid in range(num_voices)
        ], 2).view(batch_size, timesteps_left_ticks, num_voices * embedding_dim)

        right_embedded = torch.cat([
            embeddings[vid](right[:, :, vid])[:, :, None, :]
            for vid in range(num_voices)
        ], 2).view(batch_size, timesteps_right_ticks, num_voices * embedding_dim)

        if self.num_voices == 1 and type == 'note':
            center_embedded = None
        else:
            center_embedded = torch.cat([
                embeddings[vid](center[:, k].unsqueeze(1))
                for k, vid in enumerate(other_voices_indexes)
            ], 1).view(batch_size, 1, len(list(other_voices_indexes)) * embedding_dim)

        return left_embedded, center_embedded, right_embedded

    # ---------------------------------------------------------------------------
    # Save / load
    # ---------------------------------------------------------------------------

    def save(self, models_dir=None):
        target_dir = models_dir or self.models_dir
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, self.__repr__())
        torch.save(self.state_dict(), save_path)
        print(f'Model {self.__repr__()} saved to {target_dir}')

    def load(self, models_dir=None):
        target_dir = models_dir or self.models_dir
        load_path = os.path.join(target_dir, self.__repr__())
        state_dict = torch.load(load_path,
                                map_location=lambda storage, loc: storage,
                                weights_only=True)
        print(f'Loading {self.__repr__()} from {target_dir}')
        self.load_state_dict(state_dict)

    def __repr__(self):
        return (f'VoiceModel('
                f'{self.dataset.__repr__()},'
                f'{self.main_voice_index},'
                f'{self.note_embedding_dim},'
                f'{self.meta_embedding_dim},'
                f'{self.num_layers},'
                f'{self.lstm_hidden_size},'
                f'{self.dropout_lstm},'
                f'{self.hidden_size_linear})')

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------

    def train_model(self,
                    batch_size=16,
                    num_epochs=10,
                    optimizer=None,
                    lr_patience: int = 3,
                    lr_factor: float = 0.5,
                    ):
        # MODERNIZE: verbose=True is deprecated in PyTorch 2.x; print LR changes manually
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=lr_patience,
            factor=lr_factor,
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'=== Voice {self.main_voice_index} — Epoch {epoch} ===')
            (dataloader_train,
             dataloader_val,
             _) = self.dataset.data_loaders(batch_size=batch_size)

            print('  Checking embedding sizes after dataset build...')
            resized = self.resize_embeddings_to_dataset()
            if resized:
                print('  Embeddings resized — rebuilding optimizer & scheduler.')
                optimizer = torch.optim.Adam(self.parameters(),
                                             lr=optimizer.param_groups[0]['lr'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=lr_patience, factor=lr_factor)
            else:
                print('  Embedding sizes OK.')

            loss, acc = self.loss_and_acc(dataloader_train,
                                          optimizer=optimizer,
                                          phase='train')
            print(f'  Train  loss: {loss:.4f}  acc: {acc:.2f}%')

            val_loss, val_acc = self.loss_and_acc(dataloader_val,
                                                  optimizer=None,
                                                  phase='test')
            print(f'  Val    loss: {val_loss:.4f}  acc: {val_acc:.2f}%')

            # Step scheduler and print if LR changed
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f'  LR reduced: {old_lr:.2e} → {new_lr:.2e}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save()
                print(f'  ✓ Best model saved (val_loss={val_loss:.4f})')

        print(f'Voice {self.main_voice_index} done. Best val loss: {best_val_loss:.4f}')

    def loss_and_acc(self, dataloader, optimizer=None, phase='train'):
        average_loss = 0
        average_acc = 0

        if phase == 'train':
            self.train()
        elif phase in ('eval', 'test'):
            self.eval()
        else:
            raise NotImplementedError(f'Unknown phase: {phase}')

        loss_fn = nn.CrossEntropyLoss()

        for tensor_chorale, tensor_metadata in dataloader:
            tensor_chorale = cuda_variable(tensor_chorale).long()
            tensor_metadata = cuda_variable(tensor_metadata).long()

            for v in range(self.num_voices):
                max_idx = self.note_embeddings[v].num_embeddings - 1
                tensor_chorale[:, v, :] = tensor_chorale[:, v, :].clamp(0, max_idx)

            notes, metas, label = self.preprocess_input(tensor_chorale, tensor_metadata)

            if self._scaler is not None and phase == 'train':
                with autocast(_AMP_DEVICE or 'cuda'):
                    weights = self.forward(notes, metas)
                    loss = loss_fn(weights, label)
                optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                weights = self.forward(notes, metas)
                loss = loss_fn(weights, label)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            average_loss += loss.item()
            average_acc += self.accuracy(weights, label).item()

        n = len(dataloader)
        return average_loss / n, average_acc / n

    def accuracy(self, weights, target):
        batch_size, = target.size()
        pred = nn.Softmax(dim=1)(weights).max(1)[1].type_as(target)
        return (pred == target).float().sum() / batch_size * 100

    def preprocess_input(self, tensor_chorale, tensor_metadata):
        batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()
        offset = random.randint(0, self.dataset.subdivision)
        time_index_ticks = chorale_length_ticks // 2 + offset
        notes, label = self.preprocess_notes(tensor_chorale, time_index_ticks)
        metas = self.preprocess_metas(tensor_metadata, time_index_ticks)
        return notes, metas, label

    def preprocess_notes(self, tensor_chorale, time_index_ticks):
        left_notes = tensor_chorale[:, :, :time_index_ticks]
        right_notes = reverse_tensor(tensor_chorale[:, :, time_index_ticks + 1:], dim=2)
        if self.num_voices == 1:
            central_notes = None
        else:
            central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                       entry_index=self.main_voice_index,
                                       dim=1)
        label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
        return (left_notes, central_notes, right_notes), label

    def preprocess_metas(self, tensor_metadata, time_index_ticks):
        left_metas = tensor_metadata[:, self.main_voice_index, :time_index_ticks, :]
        right_metas = reverse_tensor(
            tensor_metadata[:, self.main_voice_index, time_index_ticks + 1:, :], dim=1)
        central_metas = tensor_metadata[:, self.main_voice_index, time_index_ticks, :]
        return left_metas, central_metas, right_metas