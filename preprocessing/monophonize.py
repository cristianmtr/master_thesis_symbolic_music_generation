import os
from tqdm import tqdm
import sys

sys.path.append("D:\\data\\magenta-1.0.2\\magenta\\music")

import midi_io
import melodies_lib

def main(files, monophonic_dir):
    print("converting to monophonic melodies...")
    for file in tqdm(files):
        melody = melodies_lib.midi_file_to_melody(file)
        filename = file.split(os.path.sep)[-1]
        dst_file = os.path.join(monophonic_dir, filename)
        midi_io.note_sequence_to_midi_file(melody.to_sequence(), dst_file)
