import os
import pickle
import tempfile
import sys
from glob import glob

from data_utils import START_SYMBOL, END_SYMBOL, all_features, \
    all_metadatas, indexed_chorale_to_score, chorale_to_inputs, BACH_DATASET
from deepBach import load_models
from flask import Flask, request, make_response, jsonify

from music21 import musicxml, converter
from tqdm import tqdm

import numpy as np

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'xml', 'mxl', 'mid', 'midi'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parallel_gibbs_server(models=None,
                          start_tick=None, end_tick=None,
                          start_voice_index=None, end_voice_index=None,
                          chorale_metas=None,
                          num_iterations=1000,
                          timesteps=16,
                          num_voices=None,
                          temperature=1.,
                          input_chorale=None,
                          batch_size_per_voice=16,
                          parallel_updates=True,
                          metadatas=None):
    """
    input_chorale is time major
    Returns (time, num_voices) matrix of indexes

    """
    assert models is not None
    assert input_chorale is not None

    print(models)
    print(type(models))

    sequence_length = len(input_chorale[:, 0])
    # init
    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    seq[timesteps: -timesteps, :] = input_chorale

    for expert_index in range(num_voices):
        # Add start and end symbol
        seq[:timesteps, expert_index] = [note2indexes[expert_index][START_SYMBOL]] * timesteps
        seq[-timesteps:, expert_index] = [note2indexes[expert_index][END_SYMBOL]] * timesteps
    for expert_index in range(start_voice_index, end_voice_index + 1):
        # Randomize selected zone
        seq[timesteps + start_tick: timesteps + end_tick, expert_index] = np.random.randint(num_pitches[expert_index],
                                                                                            size=end_tick - start_tick)

    if chorale_metas is not None:
        # chorale_metas is a list
        # todo how to specify chorale_metas from musescore
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]

    else:
        raise NotImplementedError

    min_temperature = temperature
    temperature = 1.3
    discount_factor = np.power(1. / temperature, 3 / 2 / num_iterations)
    # Main loop
    for iteration in tqdm(range(num_iterations)):

        temperature = max(min_temperature, temperature * discount_factor)  # Simulated annealing

        time_indexes = {}
        probas = {}
        for voice_index in range(start_voice_index, end_voice_index + 1):

            batch_input_features = []

            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps + start_tick, timesteps + end_tick)
                time_indexes[voice_index].append(time_index)

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps, num_pitches, num_voices)

                left_metas, central_metas, right_metas = all_metadatas(chorale_metadatas=extended_chorale_metas,
                                                                       metadatas=metadatas,
                                                                       time_index=time_index, timesteps=timesteps)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],

                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas}

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)

            # convert input_features
            batch_input_features = {key: np.array([input_features[key] for input_features in batch_input_features])
                                    for key in batch_input_features[0].keys()
                                    }
            # make all estimations
            probas[voice_index] = models[voice_index].predict(batch_input_features,
                                                              batch_size=batch_size_per_voice)
            if not parallel_updates:
                # update
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

        if parallel_updates:
            # update
            for voice_index in range(start_voice_index, end_voice_index + 1):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


# INITIALIZATION
response_headers = {"Content-Type": "text/html",
                    "charset": "utf-8"
                    }

# datasets only Bach for the moment
pickled_dataset = BACH_DATASET
if not os.path.exists(pickled_dataset):
    print('Warning: no dataset')
    raise NotImplementedError

# load dataset
X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(pickled_dataset, 'rb'))

num_voices = len(voice_ids)
num_pitches = list(map(len, index2notes))

# get model names present in folder models/
models_list = glob('models/*.yaml')
models_list = list(set(map(lambda name: '_'.join(name.split('_')[:-1]).split('/')[-1], models_list)))

model_name = 'deepbach'
assert os.path.exists('models/' + model_name + '_' + str(num_voices - 1) + '.yaml')

# load models
models = load_models(model_name, num_voices=num_voices)

temperature = 1.
timesteps = int(models[0].input[0]._keras_shape[1])


@app.route('/compose', methods=['POST'])
def compose():
    # global models
    # --- Parse request---
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml') as file:
        print(file.name)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        xml_string = request.form['xml_string']
        file.write(xml_string)

        # load chorale with music21
        input_chorale = converter.parse(file.name)
        input_chorale = chorale_to_inputs(input_chorale,
                                          voice_ids=voice_ids,
                                          index2notes=index2notes,
                                          note2indexes=note2indexes
                                          )

        sequence_length = input_chorale.shape[-1]
        # generate metadata:
        # todo find a way to set metadata from musescore
        # you may choose a given chorale:
        # chorale_metas = X_metadatas[11]
        # or just generate them
        chorale_metas = [metas.generate(sequence_length) for metas in metadatas]

        # make chorale time major
        input_chorale = np.transpose(input_chorale, axes=(1, 0))
        NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE = 120
        start_tick_selection = float(request.form['start_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE
        end_tick_selection = float(request.form['end_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE
        start_voice_index = int(request.form['start_staff'])
        end_voice_index = int(request.form['end_staff'])
        # if no selection
        if start_tick_selection == 0 and end_tick_selection == 0:
            chorale_length = input_chorale.shape[0]
            end_tick_selection = chorale_length

        diff = end_tick_selection - start_tick_selection + 1
        num_iterations = 100 * diff

        if diff < 16:
            batch_size_per_voice = 4
        else:
            batch_size_per_voice = 16

        num_iterations = max(int(num_iterations // batch_size_per_voice // num_voices), 5)

        # --- Generate---
        output_chorale = parallel_gibbs_server(models=models,
                                               start_tick=start_tick_selection,
                                               end_tick=end_tick_selection,
                                               start_voice_index=start_voice_index,
                                               end_voice_index=end_voice_index,
                                               input_chorale=input_chorale,
                                               chorale_metas=chorale_metas,
                                               num_iterations=num_iterations,
                                               num_voices=num_voices,
                                               timesteps=timesteps,
                                               temperature=temperature,
                                               batch_size_per_voice=batch_size_per_voice,
                                               parallel_updates=True,
                                               metadatas=metadatas)

        # convert back to music21
        output_chorale = indexed_chorale_to_score(np.transpose(output_chorale, axes=(1, 0)),
                                                  pickled_dataset=pickled_dataset
                                                  )

        # convert chorale to xml
        goe = musicxml.m21ToXml.GeneralObjectExporter(output_chorale)
        xml_chorale_string = goe.parse()

        response = make_response((xml_chorale_string, response_headers))
    return response


@app.route('/test', methods=['POST', 'GET'])
def test_generation():
    response = make_response(('TEST', response_headers))

    if request.method == 'POST':
        print(request)

    return response


@app.route('/models', methods=['GET'])
def models():
    global models_list
    # recompute model names present in folder models/
    models_list = glob('models/*.yaml')
    models_list = list(set(map(lambda name: '_'.join(name.split('_')[:-1]).split('/')[-1], models_list)))
    return jsonify(models_list)


@app.route('/current_model', methods=['POST', 'PUT'])
def current_model_update():
    global model_name
    global models
    model_name = request.form['model_name']
    # todo to remove this statement
    if model_name == 'undefined':
        return ''
    models = load_models(model_base_name=model_name, num_voices=num_voices)
    return 'Model ' + model_name + ' loaded'


@app.route('/current_model', methods=['GET'])
def current_model_get():
    global model_name
    return model_name
