from generator import data_generation
import utils
from music21 import midi
import pypianoroll
import pretty_midi
from parse_pianoroll import piano_roll_to_pretty_midi
import numpy as np
import os

files = ["D:\\data\\thesis_model2\\MIDI_tests\\generator_test.mid"]

b_size = 1
vocab_size = 128
bar_len = 96
nr_bars = 2
expected_sequences = 3
generator = data_generation(files, b_size, vocab_size, bar_len, nr_bars)

full = None

for _ in range(expected_sequences):
    full = next(generator)
    y = full[:,-1]
    x = full[:,:-1]
    assert x.shape == (b_size, bar_len*nr_bars-1, vocab_size)
    assert y.shape == (b_size, vocab_size)

sequence = full[0]
sequence[sequence == 1] = 120.
tracks = pypianoroll.Multitrack(tracks=[pypianoroll.Track(sequence)])
tracks.write(os.path.join("MIDI_tests", "parsed_and_wrote.mid"))

try:
    x = next(generator)
except Exception as e:
    print('caught exception:')
    print(type(e))
    assert type(e) == StopIteration

# mf = midi.MidiFile()
# mf.open("generator_test_generated.mid")
# mf.read()
# mf.close()
# stream = midi.translate.midiFileToStream(mf)
# stream.write('lily.png', "generator_test_generated.pdf")
print('ALL OK')
