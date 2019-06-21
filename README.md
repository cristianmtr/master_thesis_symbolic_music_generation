# OK COMPOSER: SYMBOLIC MUSIC GENERATION WITH RECURRENT NEURAL NETWORKS

## Author: Cristian Mitroi

June, 2019

Code instructions

Requires **Python 3.6**

**Modules:**

```
keras_self_attention
keras
music21
tensorflow
sklearn
keras_tqdm
pypianoroll
tqdm
librosa
pretty_midi
matplotlib
numpy
```

### PREPROCESSING A DATASET

**NOTE:** There is a small subset of the folk dataset available here, in the `folkdataset_small` folder.

#### MAKING YOUR OWN

You need to install the Magenta fork from this github: https://github.com/cristianmtr/magenta

Place it a folder and edit `preprocessing\procedure.py` (and other files where the magenta folder is added to the path):

from

```
sys.path.append("D:\\data\\magenta-1.0.2\\magenta\\music")
```

to 

```
sys.path.append("YOUR PATH")
```

Download the dataset and place the MIDI files in a folder called `1orig` in an parent empty folder.

Create a script `(your dataset)_main.py` in the folder `preprocessing`.

Put the code below in that script

```python
import subprocess
from glob import glob
import os
import filter_four_four
import encode
import transpose

import procedure

if __name__ == "__main__":
    max_sequences = 100000
    nr_bars = 4
    bar_len = 16  # based on steps_per_quarter=4 in midi_file_to_melody in D:\data\magenta-1.0.2\magenta\music\melodies_lib.py
    max_seq_len = nr_bars * bar_len

    # top dir
    top_dir = "(PATH TO TOP DIR)"
    procedure.main(top_dir, max_sequences, nr_bars, bar_len, max_seq_len)
```

Run the script

It will generate two datasets, one in pianoroll and one in magenta melody format.

Edit the `config.py` file to point to these.

### TRAINING A MODEL:

**NOTE**: A pre-trained simple model is included in `model_folksmall_pianoroll_1lstm10_noattention.zip`

```
python main.py --new --layers (nr of layers) --cells (nr of cells per layer) --dataset (which dataset)
```

add `--bi` if you want to use bidirectional wrapper for each layer
add `--att` if you want to add an attention layer on top

list of datasets can be found in `config.py`

model will be saved in a folder named `model_(dataset)_(encoding)_(bidir.)(nr layers)lstm(nr cells)_(attention)` (e.g. `model_folk100k_melody_1lstm10_noattention` or `model_folk100k_melody_bi1lstm10_attention`)


#### CONTINUE TRAINING OF A MODEL:

```
python main.py --id (id of model)
```

### SAMPLING 

```
python sampling.py (MODEL FOLDER/ID) (PATH TO MIDI FILE) --nr=(NR OF SAMPLES) --temp=(TEMP)
```

e.g.
```
python sampling.py model_folk100k_melody_1lstm10_noattention d:\data\thesis_model2\lamb.mid --nr=20 --temp=1.0
```

Prompt will indicate where files are stored

### EVALUATION 

_NOTE: You need at least 100 samples in the `7_comparison` file and 100 generated samples._

You need a separate **python 2.7** environment to run the mgeval comparison tool.

Modules required:

```
seaborn
matplotlib
pypianoroll
numpy
```

```
cd mgeval2
```

Comparison with training set

```
python compare.py (PATH TO DATASET)\7_comparison training_set (PATH TO SAMPLES DIR) model (NAME OF COMPARISON)
```

e.g.

```
python compare.py ..\folkdataset_small\7_comparison training_set ..\model_folksmall_pianoroll_1lstm10_noattention\samples\sessiontune102_transposed\ model debug_comparison
```

### PLAGIARISM CHECKER

You need to install Java

open cmd prompt in `plagiarism_checker`

```
java -jar melodyshape-1.4.jar -q (PATH TO MODEL SAMPLES) -c (PATH TO MIDI DATASET TOP DIR)\4_mono -a 2015-shapeh -k 2 -t 4 | paste -s -d '\n' - > match.tsv
```

This will output scores in the match.tsv file

To generate a Latex table with stats:

```
python scripts\extract_plagiarism.py plagiarism_checker\match.tsv model
```

