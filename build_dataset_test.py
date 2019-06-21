import pypianoroll
import numpy as np
import os
from build_dataset import main, reconstruct_full_sequence, min_max_from_folder
from glob import glob

dataset = "d:/data/thesis_model2/MIDI_tests/session*.mid"
files = glob(dataset)
out_dir = "d:/data/thesis_model2/MIDI_tests/trims/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

dataset = {
    "NR_BARS": 4,
    "BAR_LEN": 96
}

main(files, out_dir, dataset)

# check first seq. is the same as the one from loading the file
min_pitch, max_pitch = min_max_from_folder(out_dir)

first_sequence = np.load(glob(os.path.join(out_dir, '*.npy'))[0])
reconstructed_seq = reconstruct_full_sequence(first_sequence, min_pitch, max_pitch)

orig_first_sequence = pypianoroll.Multitrack(files[0]).tracks[0]
orig_first_sequence.binarize()
orig_first_sequence = orig_first_sequence.pianoroll[:first_sequence.shape[0]]

assert False not in np.unique(orig_first_sequence == reconstructed_seq)
print('ALL GOOD')
