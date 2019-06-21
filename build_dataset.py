"""
produce pre-processed dataset

1. trim to min, max pitches
2. section into n-bar sequences
"""
import os
from glob import glob
import config
import numpy as np
import pypianoroll
from utils import load_pianoroll_fromfile, compute_sequences_available
from tqdm import tqdm

NR_FILES = 10000


def min_max_from_folder(directory):
    fname = os.path.join(directory, '*.txt')
    fname = glob(fname)[0]
    min_max = fname.split(os.path.sep)[-1].split(".txt")[0]
    pitch_min, pitch_max = min_max.split("_")
    return int(pitch_min), int(pitch_max)


def reconstruct_full_sequence(section, min_pitch, max_pitch):
    zeros_down = np.zeros((section.shape[0], min_pitch))
    zeros_above = np.zeros((section.shape[0], 128-max_pitch))

    reconstruction = np.hstack(
        [
            zeros_down,
            section,
            zeros_above
        ]
    )
    return reconstruction


def min_max_pitch(files):
    the_min = None
    the_max = None
    print('searching for min and max pitch')
    for fname in tqdm(files):
        this_min, this_max = pypianoroll.Multitrack(
            fname).tracks[0].get_active_pitch_range()
        if the_min is None or this_min < the_min:
            the_min = this_min
        if the_max is None or this_max > the_max:
            the_max = this_max
    print('')  # handle tqdm spacing
    return the_min, the_max


def produce_dataset(files, out_dir, min_pitch, max_pitch, nr_bars, bar_len):
    seq_len = nr_bars * bar_len
    """
    Trim to min, max
    Slide window and produce npy matrices
    """
    print('storing dataset')
    for fname in tqdm(files):
        full_roll = load_pianoroll_fromfile(fname)
        sequences_available = compute_sequences_available(
            full_roll.shape[0], seq_len, bar_len)

        for seq_i in range(sequences_available):
            seq_i_start = seq_i * bar_len
            seq_i_end = (seq_i + nr_bars) * bar_len
            sequence = full_roll[seq_i_start:seq_i_end]

            assert sequence.shape == (seq_len, 128)
            # trim
            sequence = sequence[:, min_pitch:max_pitch]

            # save sequence
            split_name = fname.split(os.path.sep)[-1]
            out_name = os.path.join(out_dir, '%s_%s.npy' % (split_name, seq_i))
            np.save(out_name, sequence)


def main(files, out_dir, nr_bars, bar_len):
    print('found %s files...' % len(files))

    # min_pitch, max_pitch = min_max_pitch(files)
    min_pitch, max_pitch = 48, 95
    min_max_file = os.path.join(out_dir, "%s_%s.txt" %(min_pitch, max_pitch))
    with open(min_max_file, "w") as f:
        f.write('')
    print('wrote %s' %min_max_file)
    produce_dataset(files, out_dir, min_pitch, max_pitch, nr_bars, bar_len)


if __name__ == "__main__":
    dataset = config.FOLK_DATASET_TRIM_4
    files = glob(dataset['glob'])
    files = files[:NR_FILES]
    out_dir = dataset['out']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main(files, out_dir, dataset)
