# DeepBach
Version 1.0 of this repository contains the implementation of the DeepBach model described in

*DeepBach: a Steerable Model for Bach chorales generation*<br/>
Gaëtan Hadjeres, François Pachet, Frank Nielsen<br/>
*arXiv preprint [arXiv:1612.01010](https://arxiv.org/abs/1612.01010)*

The code uses python 3.6 together with [Keras](https://keras.io/) and [music21](http://web.mit.edu/music21/) libraries.

Version 2.0 (this one) elaborates on this approach. Results will be presented in an upcoming paper.
This version contains a Python Flask server and a MuseScore plugin providing an interactive use of DeepBach. 

## Installation

You can download and install DeepBach's dependencies using Anaconda with the following commands:

```
git clone git@github.com:SonyCSL-Paris/DeepBach.git
cd DeepBach
conda env create -f environment.yml
```

Make sure either  [Theano](<https://github.com/Theano/Theano>) or [Tensorflow](https://www.tensorflow.org/) is installed.
You also need to [configure properly the music editor called by music21](http://web.mit.edu/music21/doc/moduleReference/moduleEnvironment.html). 

## Usage

```
usage: deepBach.py [-h] [--timesteps TIMESTEPS] [-b BATCH_SIZE_TRAIN]
                   [-s SAMPLES_PER_EPOCH] [--num_val_samples NUM_VAL_SAMPLES]
                   [-u NUM_UNITS_LSTM [NUM_UNITS_LSTM ...]] [-d NUM_DENSE]
                   [-n {deepbach,skip}] [-i NUM_ITERATIONS] [-t [TRAIN]]
                   [-p [PARALLEL]] [--overwrite] [-m [MIDI_FILE]] [-l LENGTH]
                   [--ext EXT] [-o [OUTPUT_FILE]] [--dataset [DATASET]]
                   [-r [REHARMONIZATION]]

optional arguments:
  -h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        model's range (default: 16)
  -b BATCH_SIZE_TRAIN, --batch_size_train BATCH_SIZE_TRAIN
                        batch size used during training phase (default: 128)
  -s SAMPLES_PER_EPOCH, --samples_per_epoch SAMPLES_PER_EPOCH
                        number of samples per epoch (default: 89600)
  --num_val_samples NUM_VAL_SAMPLES
                        number of validation samples (default: 1280)
  -u NUM_UNITS_LSTM [NUM_UNITS_LSTM ...], --num_units_lstm NUM_UNITS_LSTM [NUM_UNITS_LSTM ...]
                        number of lstm units (default: [200, 200])
  -d NUM_DENSE, --num_dense NUM_DENSE
                        size of non recurrent hidden layers (default: 200)
  -n {deepbach,skip}, --name {deepbach,skip}
                        model name (default: deepbach)
  -i NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                        number of gibbs iterations (default: 20000)
  -t [TRAIN], --train [TRAIN]
                        train models for N epochs (default: 15)
  -p [PARALLEL], --parallel [PARALLEL]
                        number of parallel updates (default: 16)
  --overwrite           overwrite previously computed models
  -m [MIDI_FILE], --midi_file [MIDI_FILE]
                        relative path to midi file
  -l LENGTH, --length LENGTH
                        length of unconstrained generation
  --ext EXT             extension of model name
  -o [OUTPUT_FILE], --output_file [OUTPUT_FILE]
                        path to output file
  --dataset [DATASET]   path to dataset folder
  -r [REHARMONIZATION], --reharmonization [REHARMONIZATION]
                        reharmonization of a melody from the corpus identified
                        by its id


```

## Examples
Generate a chorale of length 100:
```
python3 deepBach.py -l 100
```
Create a DeepBach model with three stacked lstm layers of size 200, hidden layers of size 500 and train it for 10 epochs before sampling:
```
python3 deepBach.py --ext big -u 200 200 200 -d 500 -t 10
```

Generate chorale harmonization with soprano extracted from midi/file/path.mid using parallel Gibbs sampling with 10000 updates (total number of updates)
```
python3 deepBach.py -m midi/file/path.mid -p -i 20000
```


Use another model with custom parameters:
```
python3 deepBach.py --ext  big  -t 30 --timesteps 32 -u 512 256 -d 256 -b 16
```

Use another database, your dataset folder must contain .xml or .mid files with the same number of voices:
```
python3 deepBach.py --dataset /path/to/dataset/folder/ --ext dowland -t 30 --timesteps 32 -u 256 256 -d 256 -b 32
```

Reharmonization of a melody from the training or testing set:
```
python3 deepBach.py  -p -i 40000 -r 25
```

Default values load pre-trained DeepBach model and generate a chorale using sequential Gibbs sampling with 20000 iterations


# MuseScore plugin and Flask server
Put  ``deepBachMuseScore.qml`` file in your ``MuseScore2/Plugins`` directory.

Run local Flask server:
```
export FLASK_APP=plugin_flask_server.py
flask run
```
or a public server (only one connection is supported for the moment).
```
export FLASK_APP=plugin_flask_server.py
flask run --host 0.0.0.0
```

Open MuseScore and activate deepBachMuseScore plugin using the Plugin manager.
Open a four-part chorale.
Press enter on the server address, a list of computed models should appear.
Select and (re)load a model.
Select a zone in the chorale and click on the compose button.


This plugin only generates C major/A minor chorales with cadences every two bars. This is a limitation of the plugin, not the model itself.


Please consider citing this work or emailing me if you use DeepBach in musical projects. 

### Pretrained model
A pretrained model is available [here](https://www.dropbox.com/sh/qlcxv3dzj5zpcu5/AAB0PD55W3DCTJxQIRCNSbW1a?dl=0).
Extract the archive contents in the DeepBach project root folder.

### Issues
ImportError issues: Make sure DeepBach project is in your PYTHONPATH
```
export PYTHONPATH=/Path/to/DeepBach/Project
```
