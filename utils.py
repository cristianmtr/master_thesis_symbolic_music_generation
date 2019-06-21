import tqdm
import os
import numpy as np
from librosa import display
import pypianoroll
import pretty_midi

PIANOROLL_FS = 52


def compute_sequences_available(roll_len, seq_len, step_size):
    return roll_len//step_size - (seq_len//step_size - 1)


def notes_in_sequence(sequence):
    return np.argmax(sequence, axis=1)


def notes_density(sequence):
    notes = notes_in_sequence(sequence)
    counts = np.unique(notes, return_counts=True)
    return np.around(1 - np.around(counts[1][0]/len(sequence), 1), 1)


def parse_generated_piano_roll_to_midi(sequence):
    pm = pypianoroll.Track(pianoroll=sequence, program=0,
                           is_drum=False, name="generated")
    return pm


def sample(model, sequence, temperature=1.0, withatt=False):
    # helper function to sample an index from a probability array
    preds = None
    att = None
    softmax_preds = None
    if withatt:
        preds,att = model.predict(np.array([sequence]), verbose=0)
        att = att[0]
    else:
        preds = model.predict(np.array([sequence]), verbose=0)[0]
    preds = np.asarray(preds).astype('float64')
    softmax_preds = preds
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    if len(preds.shape) == 2:
        preds = preds[0]
    probas = np.random.multinomial(1, preds, 1)
    if withatt:
        return probas, att, softmax_preds
    else:
        return probas, softmax_preds


def load_pianoroll_fromfile(file):
    if file.split(".")[-1] not in ["mid", "midi"]:
        raise Exception("Only supports midi")
    tracks = pypianoroll.Multitrack(file)
    if len(tracks.tracks) > 1:
        raise Exception("No support for multi-track midi")
    else:
        # if using folk dataset
        track = tracks.tracks[0]
        track.binarize()
        return track.pianoroll


def is_colab():
    return os.path.exists("/content")


def remove_zeros_from_onehot(seq):
    flatseq = np.argmax(seq,axis=1)
    nozeros = np.where(flatseq != 0)
    return flatseq[nozeros]


def is_cluster():
    return os.path.exists("/zhome")


def is_abacus():
    return os.path.exists("/scratch/kumuge")
