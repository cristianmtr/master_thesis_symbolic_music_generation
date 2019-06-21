"""
read into pypianoroll with beat_res.
get 4 bars
write as midi in dir
"""
import os
import pypianoroll
from tqdm import tqdm
import numpy as np


def main(mono_files, comparison_dir):
    np.random.shuffle(mono_files)

    found = 0
    with tqdm(total=100) as progress:
        for f in mono_files:
            midi = pypianoroll.Multitrack(f, beat_resolution=4)
            roll = midi.tracks[0].pianoroll
            # 64 = 4 bars, each bar at 16 timesteps each (4 * 4 (beat_resolution))
            roll = roll[:64]
            notes_in_roll = np.argmax(roll, axis=1)
            notes_non_zero = len(notes_in_roll[np.where(notes_in_roll != 0)])
            # minimum 30%
            rate = notes_non_zero / 64
            if rate >= .3:
                new_midi = pypianoroll.Multitrack(
                    tracks=[pypianoroll.Track(roll)], beat_resolution=4)
                unique_name = f.split(os.path.sep)[-1]
                new_midi.write(os.path.join(comparison_dir, unique_name))
                found += 1
                progress.update(1)
            if found == 100:
                return