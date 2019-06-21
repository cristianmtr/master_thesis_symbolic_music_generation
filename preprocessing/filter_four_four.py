import os
import tqdm
import pretty_midi
import glob

def filter_midis(files, dst):
    """for each file, check if timing is only 4/4 and there is only one track
    Then save the ones in dst folder"""
    print('filtering to 4/4...')
    for fpath in tqdm.tqdm(files):
        pm = pretty_midi.PrettyMIDI(fpath)
        if len(pm.time_signature_changes) == 1:
            ts = pm.time_signature_changes[0]
            if ts.numerator == 4 and ts.denominator == 4:
                # single track
                if len(pm.instruments) == 1:
                    fname = fpath.split(os.path.sep)[-1]
                    pm.write(os.path.join(dst, fname))


def main(files, dst):
    filter_midis(files, dst)
