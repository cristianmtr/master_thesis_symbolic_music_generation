import keras
from tqdm import tqdm
import pypianoroll
import os
import numpy as np
import sys

sys.path.append("D:\\data\\thesis_model2")
from utils import compute_sequences_available


def main(files, dst_dir, nr_bars, max_seq_len, bar_len, min_note, max_note):
    print('encoding into pianoroll and saving files...')
    nr_seqs_available = 0
    failed = 0

    for file in tqdm(files):
        try:
            multitrack = pypianoroll.Multitrack(file,beat_resolution=4)
            sequence_full = multitrack.tracks[0]
            sequence_full.binarize()
            sequence_full = sequence_full.pianoroll
            sequence = sequence_full[:,min_note:max_note+1]

            # HOW TO RECONSTRUCT
            # sequence_zeros = np.zeros((sequence_full.shape))
            # sequence_zeros[:,min_note:max_note+1] = sequence
            # reconstructed = pypianoroll.Multitrack(tracks=[pypianoroll.Track(sequence_zeros)],beat_resolution=4)

            file_id = file.split(os.path.sep)[-1]

            if len(sequence) > max_seq_len:
                nr_windows = compute_sequences_available(
                    len(sequence), max_seq_len, bar_len)
                nr_seqs_available += nr_windows

                for window_index in range(nr_windows - 1):  # drop the last
                    seq_start_index = window_index * bar_len
                    seq_end_index = (window_index + nr_bars) * bar_len
                    seq_window = sequence[seq_start_index:seq_end_index]
                    dst_file = os.path.join(dst_dir,
                                            "%s_%s.npy" % (file_id, window_index))
                    np.save(dst_file, seq_window)
            else:
                nr_seqs_available += 1
                new_file_path_magenta = os.path.join(dst_dir, file_id + ".npy")
                sequence_zeros = np.zeros((max_seq_len, sequence.shape[1]))
                sequence_zeros[max_seq_len - len(sequence):] = sequence
                np.save(new_file_path_magenta, sequence_zeros)

        except Exception as _:
            failed += 1

    print('failed = %s' %failed)
    return None


def dat_file(
        files,
        max_sequences,
        pianoroll_dataset_file,
        max_seq_len,
        min_note,
        max_note
):
    print('creating memmap file for pianoroll')

    if len(files) > max_sequences:
        files = files[:max_sequences]

    print('we have %s sequences' % len(files))

    vocabsize = max_note-min_note+1
    dataset = np.memmap(
        pianoroll_dataset_file,
        shape=(len(files), max_seq_len, vocabsize),
        mode='w+',
        dtype="uint8")
    print('dataset shape', len(files), max_seq_len, vocabsize)

    for seq_index, seq_file in enumerate(tqdm(files)):
        seq = np.load(seq_file)
        dataset[seq_index] = seq

    dataset.flush()
    return pianoroll_dataset_file