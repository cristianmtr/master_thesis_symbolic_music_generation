"""
see https://www.evernote.com/l/APNuf7PAz8hK_JxUjDO-mOHx0CRkRNroiTY/
"""
import os
import sys
from glob import glob

import music21
import numpy as np
from tqdm import tqdm


majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])


def main(files, dst_dir):
    print('transposing...')
    for file in tqdm(files):
        # transpose
        score = music21.converter.parse(file)
        key = score.analyze('key')
        if key.mode == "major":
            halfSteps = majors[key.tonic.name]

        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]

        newscore = score.transpose(halfSteps)

        file = os.path.abspath(file)
        unique_name = ''.join(file.split(os.path.sep)[-1].split(".mid")[:-1])
        new_file_path = os.path.join(dst_dir, unique_name + "_transposed.mid")
        newscore.write("midi", new_file_path)
