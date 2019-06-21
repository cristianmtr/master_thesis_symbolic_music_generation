from utils import notes_in_sequence, notes_density, compute_sequences_available

import numpy as np
import os

sequence = np.load(os.path.join("tests", "test_sequence_utils_presence1.npy"))

assert notes_density(sequence) == 0.2

sequence = np.load(os.path.join("tests", "test_sequence_utils_presence2.npy"))

assert notes_density(sequence) == 0.4

# test def compute_sequences_available(roll_len, seq_len, step_size):

assert compute_sequences_available(6, 2, 1) == 5
assert compute_sequences_available(6, 3, 1) == 4
assert compute_sequences_available(12, 4, 2) == 5