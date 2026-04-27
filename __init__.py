import os
import sys
import shutil

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import torch
import music21
from tqdm import tqdm

from .DeepBach.model_manager import DeepBach
from .DatasetManager.dataset_manager import DatasetManager
from .DatasetManager.chorale_dataset import ChoraleDataset
from .DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata


# ---------------------------------------------------------------------------
# Custom MIDI corpus support
# ---------------------------------------------------------------------------

def _make_custom_corpus_iterator(midi_dir):
    """
    Return a callable that yields music21 Score objects parsed from a directory.

    Supported file types: .mid .midi .xml .mxl .musicxml
    Files are yielded in sorted order (deterministic across runs).
    Unparseable files are skipped with a warning.
    """
    midi_dir = os.path.abspath(midi_dir)
    _SUPPORTED = ('.mid', '.midi', '.xml', '.mxl', '.musicxml')

    def corpus_it_gen():
        files = sorted(f for f in os.listdir(midi_dir)
                       if f.lower().endswith(_SUPPORTED))
        if not files:
            raise FileNotFoundError(
                f"No supported music files found in: {midi_dir}\n"
                f"Supported extensions: {_SUPPORTED}")
        print(f"  Custom corpus: {len(files)} files found in {midi_dir}")
        for fname in files:
            fpath = os.path.join(midi_dir, fname)
            try:
                score = music21.converter.parse(fpath)
                yield score
            except Exception as e:
                print(f"  Warning: skipping {fname}: {e}")

    return corpus_it_gen


def _load_or_create_chorale_dataset(corpus_it_gen, name, cache_dir, **kwargs):
    """
    Replicates the DatasetManager caching pattern for custom datasets.
    """
    dataset = ChoraleDataset(
        corpus_it_gen=corpus_it_gen,
        name=name,
        cache_dir=cache_dir,
        **kwargs)

    filepath = dataset.filepath
    if os.path.exists(filepath):
        print(f'Loading {dataset.__repr__()} from {filepath}')
        try:
            loaded = torch.load(filepath, weights_only=False)
            loaded.corpus_it_gen = corpus_it_gen
            print('(the corresponding TensorDataset is not loaded)')
            return loaded
        except Exception as e:
            print(f'Warning: could not load cached dataset ({e}); rebuilding.')

    print(f'Saving dataset object to {filepath}')
    torch.save(dataset, filepath)
    return dataset


def _get_default_dataset(custom_midi_dir=None):
    """
    Build or load the ChoraleDataset.

    Args:
        custom_midi_dir: optional path to a folder of MIDI/XML files to use
            instead of the built-in Bach chorales corpus.

    Returns:
        ChoraleDataset instance (may be freshly created or loaded from cache)
    """
    metadatas = [
        FermataMetadata(),
        TickMetadata(subdivision=4),
        KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4,
    }

    if custom_midi_dir is not None:
        custom_midi_dir = os.path.abspath(custom_midi_dir)
        if not os.path.isdir(custom_midi_dir):
            raise ValueError(
                f"custom_midi_dir is not a valid directory: {custom_midi_dir}")

        corpus_it_gen = _make_custom_corpus_iterator(custom_midi_dir)
        dataset_name  = 'custom_' + os.path.basename(custom_midi_dir)
        cache_dir     = os.path.join(PACKAGE_ROOT, 'DatasetManager', 'dataset_cache')
        os.makedirs(cache_dir, exist_ok=True)

        dataset = _load_or_create_chorale_dataset(
            corpus_it_gen=corpus_it_gen,
            name=dataset_name,
            cache_dir=cache_dir,
            **chorale_dataset_kwargs)
    else:
        dataset_manager = DatasetManager()
        dataset = dataset_manager.get_dataset(
            name='bach_chorales',
            **chorale_dataset_kwargs)

    return dataset


def _get_default_model(dataset=None, models_dir=None):
    """Instantiate a DeepBach model with standard hyperparameters."""
    if dataset is None:
        dataset = _get_default_dataset()
    if models_dir is None:
        models_dir = os.path.join(PACKAGE_ROOT, 'models')

    model = DeepBach(
        dataset=dataset,
        note_embedding_dim=20,
        meta_embedding_dim=20,
        num_layers=2,
        lstm_hidden_size=256,
        dropout_lstm=0.5,
        linear_hidden_size=256,
        models_dir=models_dir,
    )
    return model


def _ensure_models_dir(models_dir=None):
    if models_dir is None:
        models_dir = os.path.join(PACKAGE_ROOT, 'models')
    else:
        models_dir = os.path.abspath(models_dir)
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"Model directory not found: {models_dir}\n"
            f"Please ensure pretrained weights are downloaded to this directory.")
    return models_dir


def _load_pretrained_weights(model, models_dir=None):
    """
    Load pretrained weights with shape-mismatch handling.

    Handles vocab-size growth in note_embeddings and mlp_predictions by
    zero-padding the checkpoint tensors to match the current model.
    """
    models_dir = _ensure_models_dir(models_dir)
    print(f">>> Loading pretrained weights from {models_dir}...")

    device_map = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(4):
        found = False
        target_suffix = f",{i},20,20,2,256,0.5,256)"

        for fname in os.listdir(models_dir):
            if not fname.endswith(target_suffix):
                continue

            model_path = os.path.join(models_dir, fname)
            print(f"\n  Loading voice {i}...")

            try:
                loaded_state  = torch.load(model_path,
                                           map_location=device_map,
                                           weights_only=True)
                current_state = model.voice_models[i].state_dict()
                had_mismatch  = False

                for key in list(loaded_state.keys()):
                    if key not in current_state:
                        continue
                    lt = loaded_state[key]
                    ct = current_state[key]

                    if lt.shape == ct.shape:
                        continue

                    had_mismatch = True
                    print(f"    WARNING: shape mismatch '{key}'")
                    print(f"      Checkpoint : {lt.shape}")
                    print(f"      Model      : {ct.shape}")

                    if 'note_embeddings' in key and lt.dim() == 2:
                        ckpt_v, edim = lt.shape
                        curr_v = ct.shape[0]
                        if curr_v > ckpt_v:
                            print(f"      Strategy : zero-pad {ckpt_v} -> {curr_v} rows")
                            padded = torch.zeros(curr_v, edim, dtype=lt.dtype)
                            padded[:ckpt_v] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip (vocab shrank)")
                            del loaded_state[key]

                    elif 'mlp_predictions' in key and lt.dim() == 2:
                        ckpt_o, in_f = lt.shape
                        curr_o = ct.shape[0]
                        if curr_o > ckpt_o:
                            print(f"      Strategy : zero-pad output weight "
                                  f"{ckpt_o} -> {curr_o}")
                            padded = torch.zeros(curr_o, in_f, dtype=lt.dtype)
                            padded[:ckpt_o] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip output weight (shrank)")
                            del loaded_state[key]

                    elif 'mlp_predictions' in key and lt.dim() == 1:
                        ckpt_o = lt.shape[0]
                        curr_o = ct.shape[0]
                        if curr_o > ckpt_o:
                            print(f"      Strategy : zero-pad output bias "
                                  f"{ckpt_o} -> {curr_o}")
                            padded = torch.zeros(curr_o, dtype=lt.dtype)
                            padded[:ckpt_o] = lt
                            loaded_state[key] = padded
                        else:
                            print(f"      Strategy : skip output bias (shrank)")
                            del loaded_state[key]

                    else:
                        print(f"      Strategy : skip (unhandled mismatch)")
                        del loaded_state[key]

                model.voice_models[i].load_state_dict(loaded_state, strict=False)
                status = "with mismatch handling" if had_mismatch else "successfully"
                print(f"  OK: Voice {i} weights loaded {status}")
                found = True

            except RuntimeError as e:
                print(f"  ERROR: Voice {i} weight loading failed: {e}")
                import traceback; traceback.print_exc()
                raise

            break

        if not found:
            raise FileNotFoundError(
                f"Weights for voice {i} not found.\n"
                f"Expected file ending with: {target_suffix}\n"
                f"Search directory: {models_dir}")

    print("\nOK: All weights loaded successfully\n")


def _to_cuda_if_available(model):
    if torch.cuda.is_available():
        model.cuda()
        print(f"OK: Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not detected, using CPU mode (slower)")
    return model


def harmonize(input_file, output_path="output.xml", num_iterations=500,
              temperature=1.0, batch_size_per_voice=8, voice_index_range=None,
              random_init=None, melody_voice=0, keep_melody=True,
              models_dir=None, custom_midi_dir=None):
    """
    Harmonise a melody file using DeepBach.

    Reads a single-voice (or multi-voice) MIDI / MusicXML file, keeps the
    melody in *melody_voice* fixed, and generates the remaining three voices
    via Gibbs sampling.

    Args:
        input_file:           Path to a .mid / .xml / .mxl melody file.
        output_path:          Where to write the harmonised MusicXML.
        num_iterations:       Gibbs sampling iterations (higher = better quality,
                              slower).  Typical range: 200–1000.
        temperature:          Sampling temperature.  1.0 = standard DeepBach;
                              <1 more conservative, >1 more varied.
        batch_size_per_voice: Parallel Gibbs proposals per step (8 is default).
        voice_index_range:    [start, end] of voice indices to regenerate.
                              None lets keep_melody / melody_voice decide.
        random_init:          Randomly initialise harmony voices before sampling.
                              None → True.
        melody_voice:         Index of the voice that contains the input melody
                              (0 = soprano, 1 = alto, 2 = tenor, 3 = bass).
        keep_melody:          If True only regenerate voices other than
                              melody_voice; if False regenerate all four.
        models_dir:           Directory holding VoiceModel weight files.
                              None → <package>/models/.
        custom_midi_dir:      If training was done on a custom corpus, pass the
                              same directory here so the dataset vocab matches.

    Returns:
        music21.stream.Score — the harmonised score (also written to output_path).
    """
    print("\n" + "=" * 70)
    print("DeepBach Harmony Generation")
    print("=" * 70)

    input_file  = os.path.abspath(input_file)
    output_path = os.path.abspath(output_path)

    print("\n>>> Initializing model...")
    dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model   = _get_default_model(dataset, models_dir=models_dir)

    _load_pretrained_weights(model, models_dir)
    _to_cuda_if_available(model)
    model.eval_phase()

    print(f"\n>>> Reading input melody: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    try:
        user_score = music21.converter.parse(input_file)
    except Exception as e:
        raise ValueError(
            f"Failed to parse input file: {input_file}\n"
            f"Supported formats: .mid .midi .xml .mxl\n"
            f"Error: {e}")

    print(">>> Converting melody to tensor...")
    try:
        score_tensor = dataset.get_score_tensor(
            user_score,
            offsetStart=0.,
            offsetEnd=user_score.flatten().highestTime)
    except Exception as e:
        raise ValueError(f"Failed to convert melody to tensor: {e}")

    metadata_tensor       = dataset.get_metadata_tensor(user_score)
    sequence_length_ticks = score_tensor.shape[1]
    print(f"  Melody length: {sequence_length_ticks} ticks "
          f"({sequence_length_ticks / 16:.1f} measures)")

    if melody_voice not in [0, 1, 2, 3]:
        raise ValueError(f"melody_voice must be 0-3, got: {melody_voice}")

    actual_voice_range = voice_index_range
    if actual_voice_range is None:
        if keep_melody:
            others = [i for i in range(4) if i != melody_voice]
            actual_voice_range = [min(others), max(others) + 1]
        else:
            actual_voice_range = [0, 4]

    actual_random_init = random_init if random_init is not None else True

    print(f"\n>>> Generation Parameters:")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Temperature: {temperature}")
    print(f"  Keep melody: {keep_melody}  (voice range: {actual_voice_range})")

    print(f"\n>>> Generating harmony (this may take a while)...")
    try:
        final_score, _, _ = model.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
            tensor_chorale=score_tensor,
            tensor_metadata=metadata_tensor,
            time_index_range_ticks=None,
            voice_index_range=actual_voice_range,
            random_init=actual_random_init,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice)
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")

    print(f"\n>>> Saving results...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        final_score.write('musicxml', fp=output_path)
    except Exception as e:
        raise RuntimeError(f"Could not save output file: {e}")

    print("=" * 70)
    print(f"OK: Saved to: {output_path}")
    print("=" * 70 + "\n")
    return final_score


def generate_from_scratch(sequence_length_ticks=64, num_iterations=500,
                          temperature=1.0, batch_size_per_voice=8,
                          voice_index_range=None, random_init=True,
                          models_dir=None, custom_midi_dir=None):
    """
    Generate a four-voice Bach-style chorale from scratch (no melody input).

    Args:
        sequence_length_ticks: Length in 16th-note ticks.  64 = 4 measures,
                               128 = 8 measures.
        num_iterations:        Gibbs sampling iterations.
        temperature:           Sampling diversity.  1.0 = default.
        batch_size_per_voice:  Parallel Gibbs proposals per step.
        voice_index_range:     [start, end] voices to generate.  None = all four.
        random_init:           Randomly initialise before sampling.
        models_dir:            Directory holding VoiceModel weight files.
        custom_midi_dir:       Custom corpus directory (must match training).

    Returns:
        music21.stream.Score
    """
    print("\n" + "=" * 70)
    print("DeepBach Free Composition - Generating from scratch")
    print("=" * 70)

    print("\n>>> Initializing model...")
    dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model   = _get_default_model(dataset, models_dir=models_dir)

    _load_pretrained_weights(model, models_dir)
    _to_cuda_if_available(model)
    model.eval_phase()

    actual_voice_range = voice_index_range if voice_index_range is not None else [0, 4]

    print(f"\n>>> Generation Parameters:")
    print(f"  Length:      {sequence_length_ticks} ticks "
          f"({sequence_length_ticks / 16:.1f} measures)")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Temperature: {temperature}")
    print(f"  Voice range: {actual_voice_range}")

    print(f"\n>>> Generating composition...")
    try:
        score, _, _ = model.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
            tensor_chorale=None,
            tensor_metadata=None,
            time_index_range_ticks=None,
            voice_index_range=actual_voice_range,
            random_init=random_init,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice)
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")

    print("=" * 70)
    print("OK: Generation complete!")
    print("=" * 70 + "\n")
    return score


def train_from_scratch(num_epochs=50, batch_size=32, lr=1e-3,
                       lr_patience=3, lr_factor=0.5,
                       models_dir=None, custom_midi_dir=None):
    """
    Train DeepBach from scratch on the Bach chorales (or a custom corpus).

    All four VoiceModels are trained sequentially.  Each voice saves only its
    best-validation-loss checkpoint.

    Args:
        num_epochs:      Maximum training epochs per voice.  Convergence
                         typically occurs at 15–30 epochs; 50 is a safe ceiling.
        batch_size:      Training batch size.  512 for RTX 4050; 256 for 4 GB VRAM.
        lr:              Initial Adam learning rate.
        lr_patience:     ReduceLROnPlateau patience (epochs with no val improvement
                         before the LR is halved).
        lr_factor:       LR reduction multiplier (new_lr = lr * lr_factor).
        models_dir:      Directory where weight files will be saved.
                         None → <cwd>/models/.
        custom_midi_dir: Optional path to a folder of MIDI/XML files to use
                         instead of the built-in Bach corpus.

    Returns:
        DeepBach model with best-epoch weights loaded.
    """
    print("\n" + "=" * 70)
    print("DeepBach Training from Scratch")
    print("=" * 70)

    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')
    else:
        models_dir = os.path.abspath(models_dir)

    if os.path.exists(models_dir):
        print(f"\n>>> WARNING: models directory already exists; "
              f"weights may be overwritten: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    print("\n>>> Initializing dataset and model...")
    dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model   = _get_default_model(dataset, models_dir=models_dir)

    _to_cuda_if_available(model)

    corpus_desc = custom_midi_dir or 'Bach Chorales (371 pieces)'
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}")
    print(f"Epochs:     {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR:         {lr}  (patience={lr_patience}, factor={lr_factor})")
    print(f"Corpus:     {corpus_desc}")
    print(f"Save path:  {models_dir}")
    print(f"{'='*70}\n")

    try:
        model.train(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
        )
        print(f"\n{'='*70}")
        print(f"OK: Training complete!  Models saved to: {models_dir}")
        print(f"{'='*70}\n")
        return model
    except Exception as e:
        raise RuntimeError(f"Error during training: {e}")


def finetune(num_epochs=5, batch_size=32, lr=1e-4,
             lr_patience=2, lr_factor=0.5,
             voice_indices=None, models_dir=None, custom_midi_dir=None):
    """
    Fine-tune pretrained DeepBach weights on the Bach corpus or a custom one.

    Loads existing weights first, then continues training.  Lower default lr
    (1e-4) than train_from_scratch to avoid destroying pretrained features.

    Args:
        num_epochs:      Fine-tuning epochs per voice.  5 is the standard recipe.
        batch_size:      Training batch size.
        lr:              Initial Adam learning rate for fine-tuning.
        lr_patience:     ReduceLROnPlateau patience.
        lr_factor:       LR reduction multiplier.
        voice_indices:   Which voices to fine-tune.  None = all four [0,1,2,3].
        models_dir:      Directory containing pretrained weights.  Weights are
                         saved back to the same directory after fine-tuning.
        custom_midi_dir: Optional custom corpus for fine-tuning data.

    Returns:
        DeepBach model with fine-tuned weights.
    """
    print("\n" + "=" * 70)
    print("DeepBach Fine-tuning")
    print("=" * 70)

    if models_dir is None:
        models_dir = os.path.join(PACKAGE_ROOT, 'models')

    print("\n>>> Initializing model...")
    dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model   = _get_default_model(dataset, models_dir=models_dir)

    _load_pretrained_weights(model, models_dir)
    _to_cuda_if_available(model)

    voices_to_finetune = voice_indices if voice_indices is not None else [0, 1, 2, 3]
    for idx in voices_to_finetune:
        if idx not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid voice index: {idx}")

    corpus_desc = custom_midi_dir or 'Bach Chorales'
    print(f"\n{'='*70}")
    print(f"Starting Fine-tuning on Pretrained Model")
    print(f"{'='*70}")
    print(f"Epochs:     {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR:         {lr}  (patience={lr_patience}, factor={lr_factor})")
    print(f"Voices:     {voices_to_finetune}")
    print(f"Corpus:     {corpus_desc}")
    print(f"{'='*70}\n")

    try:
        for voice_idx in voices_to_finetune:
            print(f"\n>>> Fine-tuning voice {voice_idx} ...")
            model.train(
                main_voice_index=voice_idx,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
            )

        print(f"\n{'='*70}")
        print(f"OK: Fine-tuning complete!  Weights saved to: {models_dir}")
        print(f"{'='*70}\n")
        return model
    except Exception as e:
        raise RuntimeError(f"Error during fine-tuning: {e}")


def build_dataset(custom_midi_dir=None):
    """
    Build (or load from cache) the ChoraleDataset without constructing a model.

    Useful when you want to inspect the dataset or call lower-level APIs
    directly (e.g. dataset.tensor_to_score, dataset.get_score_tensor).

    Args:
        custom_midi_dir: optional custom corpus directory.

    Returns:
        ChoraleDataset
    """
    return _get_default_dataset(custom_midi_dir=custom_midi_dir)


def build_model(dataset=None, models_dir=None, load_weights=True,
                custom_midi_dir=None):
    """
    Construct a DeepBach model and optionally load weights.

    This is the lower-level alternative to harmonize() / generate_from_scratch()
    for callers that need direct access to the DeepBach object (e.g. to call
    model.generation() with custom tensor_chorale / tensor_metadata).

    Args:
        dataset:      ChoraleDataset (from build_dataset()).  None → build one.
        models_dir:   Weight directory.  None → <package>/models/.
        load_weights: If True, load weights from models_dir.
        custom_midi_dir: Used when dataset is None.

    Returns:
        DeepBach model (on GPU if available, weights loaded if load_weights=True).
    """
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model = _get_default_model(dataset, models_dir=models_dir)
    if load_weights:
        _load_pretrained_weights(model, models_dir)
    _to_cuda_if_available(model)
    return model


def create_model(dataset=None, models_dir=None):
    """
    Instantiate a DeepBach model without loading weights.

    Equivalent to build_model(load_weights=False).  Useful before training
    from scratch (weights do not exist yet).

    Args:
        dataset:    ChoraleDataset.  None → build default Bach chorales dataset.
        models_dir: Directory where weights will be saved during training.

    Returns:
        DeepBach model (no weights, not moved to GPU).
    """
    return _get_default_model(dataset, models_dir=models_dir)


def load_model(dataset=None, models_dir=None, custom_midi_dir=None):
    """
    Load a fully ready DeepBach model (weights loaded, on GPU if available).

    Shorthand for build_model(load_weights=True).

    Args:
        dataset:         ChoraleDataset.  None → build default.
        models_dir:      Weight directory.  None → <package>/models/.
        custom_midi_dir: Used when dataset is None.

    Returns:
        DeepBach model ready for inference.
    """
    if dataset is None:
        dataset = _get_default_dataset(custom_midi_dir=custom_midi_dir)
    model = _get_default_model(dataset, models_dir=models_dir)
    _load_pretrained_weights(model, models_dir)
    _to_cuda_if_available(model)
    return model


def check_pretrained_weights(models_dir=None):
    """
    Check whether all four VoiceModel weight files are present.

    Args:
        models_dir: directory to check.  None → <package>/models/.

    Returns:
        dict with keys:
            complete        (bool)  — True if all four voices are found
            models_dir      (str)   — the directory that was checked
            found_weights   (int)   — how many voice files were found
            required_weights(int)   — always 4
            missing_voices  (list)  — indices of missing voice files
            error           (str)   — present only if models_dir does not exist
    """
    if models_dir is None:
        models_dir = os.path.join(PACKAGE_ROOT, 'models')

    if not os.path.exists(models_dir):
        return {
            'complete':         False,
            'models_dir':       models_dir,
            'found_weights':    0,
            'required_weights': 4,
            'missing_voices':   [0, 1, 2, 3],
            'error':            f'Model directory does not exist: {models_dir}',
        }

    found_weights = 0
    missing_voices = []
    for i in range(4):
        target_suffix = f",{i},20,20,2,256,0.5,256)"
        found = any(f.endswith(target_suffix) for f in os.listdir(models_dir))
        if found:
            found_weights += 1
        else:
            missing_voices.append(i)

    return {
        'complete':         found_weights == 4,
        'models_dir':       models_dir,
        'found_weights':    found_weights,
        'required_weights': 4,
        'missing_voices':   missing_voices,
    }


def get_device_info():
    """
    Return information about the compute device that will be used.

    Returns:
        dict with keys:
            device_type  (str)  — 'cuda' or 'cpu'
            device_name  (str)  — GPU name or 'CPU'
            cuda_available (bool)
    """
    if torch.cuda.is_available():
        return {
            'device_type':    'cuda',
            'device_name':    torch.cuda.get_device_name(0),
            'cuda_available': True,
        }
    return {
        'device_type':    'cpu',
        'device_name':    'CPU',
        'cuda_available': False,
    }



__all__ = [
    'harmonize',
    'generate_from_scratch',
    'train_from_scratch',
    'finetune',
    'build_dataset',       # build / load ChoraleDataset
    'build_model',         # build model + optionally load weights + move to GPU
    'create_model',        # build model only, no weights, no GPU move
    'load_model',          # build model + load weights + move to GPU
    'check_pretrained_weights',
    'get_device_info',
    'DeepBach',
    'DatasetManager',
    'ChoraleDataset',
    'FermataMetadata',
    'TickMetadata',
    'KeyMetadata',
    'PACKAGE_ROOT',
]


if __name__ == '__main__':
    print(f"DeepBach PyTorch — package root: {PACKAGE_ROOT}")
    info = get_device_info()
    print(f"Compute device : {info['device_name']}")
    status = check_pretrained_weights()
    if status['complete']:
        print("Pretrained weights : OK (all 4 voices found)")
    else:
        print(f"Pretrained weights : MISSING voices {status['missing_voices']}")
        print(f"  Expected in      : {status['models_dir']}")
