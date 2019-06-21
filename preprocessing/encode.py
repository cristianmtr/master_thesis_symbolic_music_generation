"""
Encodes each track into numpy sequences of a max length, with prefix padding of 0s
"""
import os
import sys
from glob import glob

import keras
import pretty_midi

import music21
import numpy as np
from tqdm import tqdm

sys.path.append("D:\\data\\magenta-1.0.2\\magenta\\music")

import melodies_lib
import midi_io
from melody_encoder_decoder import MelodyOneHotEncoding

sys.path.append("D:\\data\\thesis_model2")
from utils import compute_sequences_available


def main(files, dst_dir, nr_bars, max_seq_len, bar_len):
    print('encoding and saving files...')
    nr_seqs_available = 0
    vocab = set() # keep track of the unique vocab tokens

    for file_index, file in enumerate(tqdm(files)):
        sequence = list(melodies_lib.midi_file_to_melody(file))
        sequence_vocab = set(sequence)
        vocab = vocab.union(sequence_vocab)

        file_id = file.split(os.path.sep)[-1]
    
        if len(sequence) > max_seq_len:
            nr_windows = compute_sequences_available(
                len(sequence), max_seq_len, bar_len)
            nr_seqs_available += nr_windows

            for window_index in range(nr_windows-1):  # drop the last
                seq_start_index = window_index * bar_len
                seq_end_index = (window_index+nr_bars) * bar_len
                seq_window = sequence[seq_start_index:seq_end_index]
                dst_file = os.path.join(dst_dir, "%s_%s.npy" %(file_id, window_index))
                np.save(dst_file, seq_window)
        else:
            nr_seqs_available += 1
            new_file_path_magenta = os.path.join(dst_dir, file_id + ".npy")
            sequence_zeros = np.repeat(-2, max_seq_len)
            sequence_zeros[max_seq_len-len(sequence):] = sequence
            np.save(new_file_path_magenta, sequence_zeros)

    notes_vocab = vocab.difference(set([-2,-1]))
    encoder = MelodyOneHotEncoding(min(notes_vocab), max(notes_vocab)+1)
    # store min,max notes
    path_to_min_max = os.path.join(dst_dir, '..', "min_max.npy")
    print("VOCABULARY: min note = %s, max note = %s" %(min(notes_vocab), max(notes_vocab)))
    print("saved to %s" %path_to_min_max)
    np.save(path_to_min_max, [min(notes_vocab), max(notes_vocab)])
    return encoder


def dat_file(files,max_sequences,magenta_one_hot_filename,max_seq_len,encoder):
    print('creating memmap file and loading dataset')

    if len(files) > max_sequences:
        files = files[:max_sequences]
    
    print('we have %s sequences' %len(files))

    dataset = np.memmap(magenta_one_hot_filename, shape=(
        len(files), max_seq_len, encoder.num_classes), mode='w+', dtype="uint8")
    print('dataset shape' , len(files), max_seq_len, encoder.num_classes)

    for seq_index, seq_file in enumerate(tqdm(files)):
        seq = np.load(seq_file)
        seq = [encoder.encode_event(ev) for ev in seq]
        seq = keras.utils.to_categorical(
            seq, num_classes=encoder.num_classes, dtype='uint8')
        dataset[seq_index] = seq

    dataset.flush()
    return magenta_one_hot_filename
