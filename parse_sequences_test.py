import pypianoroll
import pretty_midi
import matplotlib.pyplot as plt

roll = pypianoroll.parse("D:/data/thesis_model2/MIDI_tests/1.mid")

roll = roll.tracks[0].pianoroll

BAR_LEN = 96
NR_BARS = 2
SEQ_LEN = NR_BARS * BAR_LEN
print('sequence_length = ', SEQ_LEN)

dataset = []

sequences_available = int(np.floor(roll.shape[0]/(SEQ_LEN)))

notes_played_all = np.argmax(roll, axis=1)
print('all notes ', notes_played_all)
print(len(notes_played_all))

for seq_i in [0,1]:
    seq_i_start = seq_i * BAR_LEN
    seq_i_end = (seq_i + NR_BARS )* BAR_LEN
    print(seq_i_start, seq_i_end)
    sequence = roll[seq_i_start:seq_i_end]
    
    assert sequence.shape == (192, 128)
    
    notes_played = np.argmax(sequence, axis=1)
    if i == 0:
        assert notes_played = notes_played_all[:192]
    elif i == 1:
        assert notes_played = notes_played_all[96:288]
    print('%s notes: ' %len(notes_played), notes_played)

    pm = piano_roll_to_pretty_midi(sequence.T, fs=48)

#     plt.figure(figsize=(12, 8))
#     plot_piano_roll(pm.instruments[0], 55, 70)
#     plt.show()
    

