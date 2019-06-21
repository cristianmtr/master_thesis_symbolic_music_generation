# move files from child dirs into top level
# the orig.mid, with filename "artist_song"
# and make single track melody MIDI
from glob import glob
import os
from tqdm import tqdm
import pretty_midi
import music21

all_glob_str = "D:\\data\\hooktheory_dataset\\1orig\\**\\*_key.mid"
all_glob = glob(all_glob_str, recursive=True)
dstfolder = os.path.abspath("D:\\data\\hooktheory_dataset\\1orig")

problematic = []

for fpath in tqdm(all_glob):
    artist_name = fpath.split(os.path.sep)[5]
    song_name = fpath.split(os.path.sep)[6]
    section_name = fpath.split(os.path.sep)[-1]

    midi = pretty_midi.PrettyMIDI(fpath)
    if len(midi.instruments) == 2:
        del midi.instruments[1] # we assume first track is always melody
        fname = "%s_%s_%s" %(artist_name, song_name, section_name)
        dst = os.path.join(dstfolder, fname)
        midi.write(dst)
    else:
        problematic.append(fpath) # most likely we only have chords


print(problematic)