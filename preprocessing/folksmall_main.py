"""
see https://www.evernote.com/l/APNuf7PAz8hK_JxUjDO-mOHx0CRkRNroiTY/
"""
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
    top_dir = "D:\\data\\thesis_model2_dist\\folkdataset_small"
    procedure.main(top_dir, max_sequences, nr_bars, bar_len, max_seq_len)
