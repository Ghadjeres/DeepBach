from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from DeepBach.model_manager import DeepBach

from music21 import musicxml, metadata
import music21

import flask
from flask import Flask, request, make_response
from flask_cors import CORS

import logging
from logging import handlers as logging_handlers
import sys

import torch
import math
from typing import List, Optional
import click
import os

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'midi'}

# INITIALIZATION
xml_response_headers = {"Content-Type": "text/xml",
                        "charset":      "utf-8"
                        }
mp3_response_headers = {"Content-Type": "audio/mpeg3"
                        }

deepbach = None
_num_iterations = None
_sequence_length_ticks = None
_ticks_per_quarter = None

# TODO use this parameter or extract it from the metadata somehow
timesignature = music21.meter.TimeSignature('4/4')

# generation parameters
# todo put in click?
batch_size_per_voice = 8

metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=_ticks_per_quarter),
    KeyMetadata()
]


def get_fermatas_tensor(metadata_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the fermatas tensor from a metadata tensor
    """
    fermatas_index = [m.__class__ for m in metadatas].index(
        FermataMetadata().__class__)
    # fermatas are shared across all voices so we only consider the first voice
    soprano_voice_metadata = metadata_tensor[0]

    # `soprano_voice_metadata` has shape
    # `(sequence_duration, len(metadatas + 1))`  (accouting for the voice
    # index metadata)
    # Extract fermatas for all steps
    return soprano_voice_metadata[:, fermatas_index]


@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--num_iterations', default=50,
              help='number of parallel pseudo-Gibbs sampling iterations (for a single update)')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
@click.option('--ticks_per_quarter', default=4,
              help='number of ticks per quarter note')
@click.option('--port', default=5000,
              help='port to serve on')
def init_app(note_embedding_dim,
             meta_embedding_dim,
             num_layers,
             lstm_hidden_size,
             dropout_lstm,
             linear_hidden_size,
             num_iterations,
             sequence_length_ticks,
             ticks_per_quarter,
             port
             ):
    global metadatas
    global _sequence_length_ticks
    global _num_iterations
    global _ticks_per_quarter

    _ticks_per_quarter = ticks_per_quarter
    _sequence_length_ticks = sequence_length_ticks
    _num_iterations = num_iterations

    dataset_manager = DatasetManager()
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }

    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
    )
    assert sequence_length_ticks % bach_chorales_dataset.subdivision == 0

    global deepbach
    deepbach = DeepBach(
        dataset=bach_chorales_dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )
    deepbach.load()
    deepbach.cuda()

    # launch the script
    # use threaded=True to fix Chrome/Chromium engine hanging on requests
    # [https://stackoverflow.com/a/30670626]
    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        app.run(host='0.0.0.0', port=port, threaded=True)


@app.route('/generate', methods=['GET', 'POST'])
def compose():
    """
    Return a new, generated sheet
    Usage:
        - Request: empty, generation is done in an unconstrained fashion
        - Response: a sheet, MusicXML
    """
    global deepbach
    global _sequence_length_ticks
    global _num_iterations

    # Use more iterations for the initial generation step
    # FIXME hardcoded 4/4 time-signature
    num_measures_generation = math.floor(_sequence_length_ticks /
                                         deepbach.dataset.subdivision)
    initial_num_iterations = math.floor(_num_iterations * num_measures_generation
                                        / 3)  # HACK hardcoded reduction

    (generated_sheet, _, generated_metadata_tensor) = (
        deepbach.generation(num_iterations=initial_num_iterations,
                            sequence_length_ticks=_sequence_length_ticks)
    )

    generated_fermatas_tensor = get_fermatas_tensor(generated_metadata_tensor)

    # convert sheet to xml
    response = sheet_and_fermatas_to_json_response(
        generated_sheet, generated_fermatas_tensor)
    return response


@app.route('/test-generate', methods=['GET'])
def ex():
    _current_sheet = next(music21.corpus.chorales.Iterator())
    return sheet_to_xml_response(_current_sheet)


@app.route('/musicxml-to-midi', methods=['POST'])
def get_midi():
    """
    Convert the provided MusicXML sheet to MIDI and return it
    Usage:
        POST -d @sheet.mxml /musicxml-to-midi
        - Request: the payload is expected to contain the sheet to convert, in
        MusicXML format
        - Response: a MIDI file
    """
    sheetString = request.data
    sheet = music21.converter.parseData(sheetString, format="musicxml")
    insert_musicxml_metadata(sheet)

    return sheet_to_midi_response(sheet)


@app.route('/timerange-change', methods=['POST'])
def timerange_change():
    """
    Perform local re-generation on a sheet and return the updated sheet
    Usage:
        POST /timerange-change?time_range_start_beat=XXX&time_range_end_beat=XXX
        - Request:
            The payload is expected to be a JSON with the following keys:
                * 'sheet': a string containing the sheet to modify, in MusicXML
                  format
                * 'fermatas': a list of integers describing the positions of
                  fermatas in the sheet
                  TODO: could store the fermatas in the MusicXML client-side
            The start and end positions (in beats) of the portion to regenerate
            are passed as arguments in the URL:
                * time_range_start_quarter, integer:
        - Response:
            A JSON document with same schema as the request containing the
            updated sheet and fermatas
    """
    global deepbach
    global _num_iterations
    global _sequence_length_ticks
    request_parameters = parse_timerange_request(request)
    time_range_start_quarter = request_parameters['time_range_start_quarter']
    time_range_end_quarter = request_parameters['time_range_end_quarter']
    fermatas_tensor = request_parameters['fermatas_tensor']

    input_sheet = request_parameters['sheet']

    time_index_range_ticks = [
        time_range_start_quarter * deepbach.dataset.subdivision,
        time_range_end_quarter * deepbach.dataset.subdivision]

    input_tensor_sheet, input_tensor_metadata = (
        deepbach.dataset.transposed_score_and_metadata_tensors(
            input_sheet, 0)
    )

    (output_sheet,
     output_tensor_sheet,
     output_tensor_metadata) = deepbach.generation(
        tensor_chorale=input_tensor_sheet,
        tensor_metadata=input_tensor_metadata,
        temperature=1.,
        batch_size_per_voice=batch_size_per_voice,
        num_iterations=_num_iterations,
        sequence_length_ticks=_sequence_length_ticks,
        time_index_range_ticks=time_index_range_ticks,
        fermatas=fermatas_tensor
    )

    output_fermatas_tensor = get_fermatas_tensor(output_tensor_metadata)

    # create JSON response
    response = sheet_and_fermatas_to_json_response(
        output_sheet, output_fermatas_tensor)
    return response


@app.route('/analyze-notes', methods=['POST'])
def dummy_read_audio_file():
    global deepbach
    import wave
    print(request.args)
    print(request.files)
    chunk = 1024
    audio_fp = wave.open(request.files['audio'], 'rb')
    data = audio_fp.readframes(chunk)
    print(data)
    notes = ['C', 'D', 'Toto', 'Tata']

    return flask.jsonify({'success': True, 'notes': notes})


def insert_musicxml_metadata(sheet: music21.stream.Stream):
    """
    Insert various metadata into the provided XML document
    The timesignature in particular is required for proper MIDI conversion
    """
    global timesignature

    from music21.clef import TrebleClef, BassClef, Treble8vbClef
    for part, name, clef in zip(
            sheet.parts,
            ['soprano', 'alto', 'tenor', 'bass'],
            [TrebleClef(), TrebleClef(), Treble8vbClef(), BassClef()]
    ):
        # empty_part = part.template()
        part.insert(0, timesignature)
        part.insert(0, clef)
        part.id = name
        part.partName = name

    md = metadata.Metadata()
    sheet.insert(0, md)

    # required for proper musicXML formatting
    sheet.metadata.title = 'DeepBach'
    sheet.metadata.composer = 'DeepBach'


def parse_fermatas(fermatas_list: List[int]) -> Optional[torch.Tensor]:
    """
    Parses fermata GET option, given at the quarter note level
    """
    global _sequence_length_ticks
    # the data is expected to be provided as a list in the request
    return fermatas_to_tensor(fermatas_list)


def fermatas_to_tensor(fermatas: List[int]) -> torch.Tensor:
    """
    Convert a list of fermata positions (in beats) into a subdivion-rate tensor
    """
    global _sequence_length_ticks
    global deepbach
    subdivision = deepbach.dataset.subdivision
    sequence_length_quarterNotes = math.floor(_sequence_length_ticks / subdivision)

    fermatas_tensor_quarterNotes = torch.zeros(sequence_length_quarterNotes)
    fermatas_tensor_quarterNotes[fermatas] = 1
    # expand the tensor to the subdivision level
    fermatas_tensor = (fermatas_tensor_quarterNotes
                       .repeat((subdivision, 1))
                       .t()
                       .contiguous())
    return fermatas_tensor.view(_sequence_length_ticks)


def fermatas_tensor_to_list(fermatas_tensor: torch.Tensor) -> List[int]:
    """
    Convert a binary fermatas tensor into a list of positions (in beats)
    """
    global _sequence_length_ticks
    global deepbach

    subdivision = deepbach.dataset.subdivision

    # subsample fermatas to beat rate
    beat_rate_fermatas_tensor = fermatas_tensor[::subdivision]

    # pick positions of active fermatas
    fermatas_positions_tensor = beat_rate_fermatas_tensor.nonzero().squeeze()
    fermatas = fermatas_positions_tensor.int().tolist()

    return fermatas


def parse_timerange_request(request):
    """
    must cast
    :param req:
    :return:
    """
    json_data = request.get_json(force=True)
    time_range_start_quarter = int(request.args.get('time_range_start_quarter'))
    time_range_end_quarter = int(request.args.get('time_range_end_quarter'))
    fermatas_tensor = parse_fermatas(json_data['fermatas'])

    sheet = music21.converter.parseData(json_data['sheet'], format="musicxml")

    return {
        'sheet':                    sheet,
        'time_range_start_quarter': time_range_start_quarter,
        'time_range_end_quarter':   time_range_end_quarter,
        'fermatas_tensor':          fermatas_tensor
    }


def sheet_to_xml_bytes(sheet: music21.stream.Stream):
    """Convert a music21 sheet to a MusicXML document"""
    # first insert necessary MusicXML metadata
    insert_musicxml_metadata(sheet)

    sheet_to_xml_bytes = musicxml.m21ToXml.GeneralObjectExporter(sheet).parse()

    return sheet_to_xml_bytes


def sheet_to_xml_response(sheet: music21.stream.Stream):
    """Generate and send XML sheet"""
    xml_sheet_bytes = sheet_to_xml_bytes(sheet)

    response = flask.make_response((xml_sheet_bytes, xml_response_headers))
    return response


def sheet_and_fermatas_to_json_response(sheet: music21.stream.Stream,
                                        fermatas_tensor: torch.Tensor):
    sheet_xml_string = sheet_to_xml_bytes(sheet).decode('utf-8')
    fermatas_list = fermatas_tensor_to_list(fermatas_tensor)

    print(fermatas_list)

    return flask.jsonify({
        'sheet':    sheet_xml_string,
        'fermatas': fermatas_list
    })


def sheet_to_midi_response(sheet):
    """
    Convert the provided sheet to midi and send it as a file
    """
    midiFile = sheet.write('midi')
    return flask.send_file(midiFile, mimetype="audio/midi",
                           cache_timeout=-1  # disable cache
                           )


def sheet_to_mp3_response(sheet):
    """Generate and send MP3 file
    Uses server-side `timidity`
    """
    sheet.write('midi', fp='./uploads/midi.mid')
    os.system(f'rm uploads/midi.mp3')
    os.system(f'timidity uploads/midi.mid -Ow -o - | '
              f'ffmpeg -i - -acodec libmp3lame -ab 64k '
              f'uploads/midi.mp3')
    return flask.send_file('uploads/midi.mp3')


if __name__ == '__main__':
    file_handler = logging_handlers.RotatingFileHandler(
        'app.log', maxBytes=10000, backupCount=5)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
init_app()
