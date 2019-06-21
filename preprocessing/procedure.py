"""
see https://www.evernote.com/l/APNuf7PAz8hK_JxUjDO-mOHx0CRkRNroiTY/
"""
import subprocess
from glob import glob
import os
import filter_four_four
import numpy as np
import encode
import transpose
import monophonize
import sys
import encode_pianoroll
import comparisons

sys.path.append("D:\\data\\magenta-1.0.2\\magenta\\music")

from melody_encoder_decoder import MelodyOneHotEncoding


def main(top_dir, max_sequences, nr_bars, bar_len, max_seq_len):
        # original midi files
    orig_dir = os.path.join(top_dir, '1orig')
    orig_dir_glob = os.path.join(orig_dir, "*.mid")
    
    # contains 4/4 only midis
    four_four_dir = os.path.join(top_dir, "2_fourfour")
    if not os.path.exists(four_four_dir):
        os.mkdir(four_four_dir)
    
    # contains the transposed midi files
    transposed_dir = os.path.join(
        top_dir, "3_transposed")
    if not os.path.exists(transposed_dir):
        os.mkdir(transposed_dir)

    # contains the transposed, monophonic melodies files
    monophonic_dir = os.path.join(
        top_dir, "4_mono"
    )
    if not os.path.exists(monophonic_dir):
        os.mkdir(monophonic_dir)

    # 100 random files to be used in evaluation
    comparison_dir = os.path.join(top_dir, "7_comparison")
    if not os.path.exists(comparison_dir):
        os.mkdir(comparison_dir)

    # contains the transposed, split, magenta one hot encoded dataset
    # as a big mmap file
    magenta_dir = os.path.join(top_dir, "5_encoded")
    if not os.path.exists(magenta_dir):
        os.mkdir(magenta_dir)

    pianoroll_dir = os.path.join(top_dir, "6_pianoroll")
    if not os.path.exists(pianoroll_dir):
        os.mkdir(pianoroll_dir)

    magenta_dataset_file = os.path.join(top_dir, "dataset.dat")

    pianoroll_dataset_file = os.path.join(top_dir, "pianoroll.dat")

    # FILTER TO 4/4 ONLY
    four_four_dir_files = glob(os.path.join(four_four_dir, "*.mid"))
    if len(four_four_dir_files) == 0:
        orig_files = glob(orig_dir_glob)
        filter_four_four.main(orig_files, four_four_dir)
    else:
        print('skipping filtering to 4/4 as directory %s is not empty' %
              four_four_dir)

    # TRANSPOSE THE 4/4 MIDI FILES
    transposed_files = glob(os.path.join(transposed_dir, "*.mid"))
    if len(transposed_files) == 0:
        transpose.main(
            glob(os.path.join(four_four_dir, "*.mid")),
            transposed_dir
                )
    else:
        print("skipping transposing as %s is not empty" % transposed_dir)
    
    # monophonize
    mono_files = glob(os.path.join(monophonic_dir, "*.mid"))
    if len(mono_files) == 0:
        monophonize.main(
            glob(os.path.join(transposed_dir, "*.mid")),
            monophonic_dir
                )
    else:
        print("skipping monophonize as %s is not empty" % monophonic_dir)

    # choose 100 random samples, take first 4 bars
    mono_files = glob(os.path.join(monophonic_dir, "*.mid"))
    comparisons.main(
        mono_files,
        comparison_dir
    )

    encoder = None
    min_note = None
    max_note = None
    # ENCODING AND SPLITTING THE TRANSPOSED MIDIs INTO MAGENTA FORMAT AND THEN
    # CREATING ONE BIG MMAP FILE OF ONE-HOT ENCODED SEQUENCES
    if len(glob(os.path.join(magenta_dir, '*.npy'))) == 0:
        encoder = encode.main(
            glob(os.path.join(monophonic_dir, "*.mid")),
            magenta_dir,
            nr_bars,
            max_seq_len,
            bar_len
        )
    else:
        print('skipping encoding into melody as directory %s was not empty' %magenta_dir)
    min_note, max_note = np.load(os.path.join(top_dir, 'min_max.npy'))
    encoder = MelodyOneHotEncoding(min_note, max_note+1)

    if not os.path.exists(magenta_dataset_file): # dat file doesnt exist
        encode.dat_file(
            glob(os.path.join(magenta_dir, "*.npy")),
            max_sequences,
            magenta_dataset_file,
            max_seq_len,
            encoder
        )
        print('dataset at ', magenta_dataset_file)
    else:
        print('skipping creating dataset file as %s exists' %(magenta_dataset_file))

    ## encode into pianoroll
    if len(glob(os.path.join(pianoroll_dir, "*.npy"))) == 0:
        encode_pianoroll.main(
            glob(os.path.join(monophonic_dir, "*.mid")),
            pianoroll_dir,
            nr_bars,
            max_seq_len,
            bar_len,
            min_note,
            max_note
        )
    else:
        print('skipping encoding into pianoroll as directory %s was not empty' %pianoroll_dir)

    if not os.path.exists(pianoroll_dataset_file): # dat file doesnt exist
        encode_pianoroll.dat_file(
            glob(os.path.join(pianoroll_dir, "*.npy")),
            max_sequences,
            pianoroll_dataset_file,
            max_seq_len,
            min_note,
            max_note
        )
        print('dataset at ', pianoroll_dataset_file)
    else:
        print('skipping creating dataset file as %s exists' %(pianoroll_dataset_file))
