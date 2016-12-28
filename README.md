# DeepBach
This repository contains the implementation of the DeepBach model described in 

*DeepBach: a Steerable Model for Bach chorales generation*<br/>
Gaëtan Hadjeres, François Pachet<br/>
*arXiv preprint [arXiv:1612.01010](https://arxiv.org/abs/1612.01010)*

The code uses python 3.5 together with [Keras](https://keras.io/) and [music21](http://web.mit.edu/music21/) libraries.

# Installation

You can download and install DeepBach's dependencies with the following commands:

```
git clone git@github.com:SonyCSL-Paris/DeepBach.git
cd DeepBach
sudo pip3 install -r requirements.txt
```

Make sure either  [Theano](<https://github.com/Theano/Theano>) or [Tensorflow](https://www.tensorflow.org/) is installed.
You also need to [configure properly the music editor called by music21](http://web.mit.edu/music21/doc/moduleReference/moduleEnvironment.html). 

# Usage

```
 usage: deepBach.py [-h] [--timesteps TIMESTEPS] [-b BATCH_SIZE_TRAIN]
                   [-s SAMPLES_PER_EPOCH] [--num_val_samples NUM_VAL_SAMPLES]
                   [-u NUM_UNITS_LSTM [NUM_UNITS_LSTM ...]] [-d NUM_DENSE]
                   [-n {deepbach,mlp,maxent}] [-i NUM_ITERATIONS] [-t [TRAIN]]
                   [-p [PARALLEL]] [--overwrite] [-m [MIDI_FILE]] [-l LENGTH]
                   [--ext EXT]

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
  -n {deepbach,mlp,maxent}, --name {deepbach,mlp,maxent}
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

```

# Examples
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
python3 deepBach.py -m midi/file/path.mid -p -i 10000
```


Use another model with custom parameters:
```
python3 deepBach.py -n fastbach --ext  big  -t 30 --timesteps 32 -u 512 256 -d 256 -b 16
```

Use another database:
```
python3 deepBach.py --dataset /home/gaetan/data/Dowland --ext dowland -t 30 --timesteps 32 -u 256 256 -d 256 -b 32
```

Reharmonization of a melody from the training or testing set:
```
python3 deepBach.py -n skip   -p -i 40000 -r 25
```

Ravenscroft: (BEAT_SIZE and SUBDIVISION constants set to 2)
```

python3 deepBach.py -n skip  --ext ravenscroches  --dataset /home/gaetan/data/RavenscroftMidiMt -p -l 300 -i 50000
```



Default values load pre-trained DeepBach model and generate a chorale using sequential Gibbs sampling with 20000 iterations
