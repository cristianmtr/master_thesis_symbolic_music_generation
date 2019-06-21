import tensorflow as tf
import argparse
import os
import sys
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import pypianoroll
from keras_self_attention import *
from keras_self_attention import SeqWeightedAttention
from tqdm import tqdm

import config
import keras.backend as K

from config import datasets
from generator import *
from utils import *

sys.path.append("d:/data/magenta-1.0.2/magenta/models/score2perf/")
sys.path.append("preprocessing/")
sys.path.append("D:\\data\\magenta-1.0.2\\magenta\\music")

import melodies_lib
import midi_io
import transpose

from melody_encoder_decoder import MelodyOneHotEncoding

def attention_loss(factor=1e-6):
    def attention_regularizer(y, y_pred):
        input_len = K.shape(y_pred)[-1]
        return factor * K.square(K.batch_dot(y_pred, K.permute_dimensions(y_pred, (1, 0)))
                                 - tf.eye(input_len))
    return attention_regularizer

def att_model(cells, bi, layers, att):
    """
    hardcoded model for vis. attention
    """
    # cells = 64
    vocab_size=58
    # bi = True
    # layers=3
    # att=True
    inputs = keras.layers.Input(
        shape=(63, 58,), name='Input')

    prev = inputs
    for i in range(layers):
        ret_seq = True
        if i == layers-1 and att == False:
            ret_seq = False

        this_layer = keras.layers.LSTM(
            cells,
            dropout=0.4,
            name='LSTM_%s' %i,
            return_sequences=ret_seq
        )
        if bi:
            this_layer = keras.layers.Bidirectional(
                this_layer,
                name='bi_%s' %i
            )
        prev = this_layer(prev)

    attention = SeqWeightedAttention(
        return_attention=True,
        name='Attention'
    )
    attention_layer = attention(prev)
    attention_layer, attention = attention_layer

    dense = keras.layers.Dense(
        vocab_size, activation='softmax', name="dense_outputs")(attention_layer)

    outputs = [dense, attention]
    model = keras.Model(inputs=inputs, outputs=outputs)   
    model.compile(
        optimizer='adam',
        loss={
            'dense_outputs':'categorical_crossentropy',
            'Attention': attention_loss(1e-4)
        }
    )
    return model



def get_model(args):
    model = None
    modelname = args.model_id
    # workaround for getting vis for attention
    if modelname == "model_folk100k_melody_2lstm32_attention":
        # (100000, 64, 58)
        model = att_model(32, False, 2, True)
    elif modelname == "model_folk100k_melody_bi3lstm64_attention":
        model = att_model(64, True, 3, True)
    else:
        json_model = open(os.path.join(modelname, "model.json"), "r").read()
        model = keras.models.model_from_json(
            json_model, custom_objects=SeqWeightedAttention.get_custom_objects())
    model.load_weights(os.path.join(modelname, "model.h5"))
    print(model.summary(line_length=100))
    return model


def get_dataset(args):
    dataset = None
    # if "folk" in args.model_id and "pianoroll" in args.model_id:
    #     dataset = config.datasets["folk100k_pianoroll"]
    # if "folk" in args.model_id and "melody" in args.model_id:
    #     dataset = config.datasets["folk100k_melody"]
    # if "hook" in args.model_id and "pianoroll" in args.model_id:
    #     dataset = config.datasets["hook100k_pianoroll"]
    # if "hook" in args.model_id and "melody" in args.model_id:
    #     dataset = config.datasets["hook100k_melody"]
    # if "both" in args.model_id and "pianoroll" in args.model_id:
    #     dataset = config.datasets["both_pianoroll"]
    # for n in ["30k", "45k", "60k", "75k"]:
    #     if n in args.model_id:
    #         dataset = config.datasets["folk%s" %n]
    for dset in config.datasets.keys():
        if dset in args.model_id:
            dataset = config.datasets[dset]
            break
    dshape = dataset['shape']
    print('dataset : %s' % dataset)
    print(dshape)
    input_seq_len = dshape[1] - 1
    print(input_seq_len)
    min_note, max_note = np.load(
        os.path.abspath(os.path.join(dataset['path'], '..', 'min_max.npy')))
    return input_seq_len, dshape, min_note, max_note


def transpose_seed(args):
    file = args.seed
    unique_name = ''.join(file.split(os.path.sep)[-1].split(".mid")[:-1])

    if not os.path.exists('midi_seeds_transposed'):
        os.mkdir('midi_seeds_transposed')

    transpose.main([file], os.path.abspath("midi_seeds_transposed/"))

    transposed_seed = os.path.abspath(
        glob(os.path.join("midi_seeds_transposed", unique_name) + "*")[0])
    print(transposed_seed)
    return transposed_seed


def from_trim_pianoroll_to_full(seq, min_note, max_note):
    zero_sequence = np.zeros((seq.shape[0], 128))
    zero_sequence[:, min_note:max_note + 1] = seq
    zero_sequence[zero_sequence == 1] = 127
    return zero_sequence


def save_trim_pianoroll_seq(seq, min_note, max_note, thepath):
    pypianoroll.Multitrack(
        tracks=[
            pypianoroll.Track(
                from_trim_pianoroll_to_full(
                    seq,
                    min_note,
                    max_note,
                ))
        ],
        beat_resolution=4).write(thepath)


def read_encode_pad_sequence_melody(filepath, min_note, max_note, input_seq_len):
    print("loading encoder...")
    encoder = MelodyOneHotEncoding(min_note, max_note+1)
    seed_melody = melodies_lib.midi_file_to_melody(filepath)
    seed_melody.squash(min_note, max_note)

    seed_sequence = [encoder.encode_event(ev) for ev in list(seed_melody)]

    print("padding...")
    if len(seed_sequence) > input_seq_len:
        seed_sequence = np.array(seed_sequence[:input_seq_len])
    else:
        zero_padded_seq = np.repeat(0, input_seq_len)
        zero_padded_seq[input_seq_len - len(seed_sequence):] = seed_sequence
        seed_sequence = zero_padded_seq
    print("size after padding: ", seed_sequence.shape)

    seed_sequence = keras.utils.to_categorical(seed_sequence, num_classes=encoder.num_classes, dtype='uint8')
    print('shape of seed sequence after 1h encoding: ', seed_sequence.shape)
    return seed_sequence, encoder


def read_encode_pad_sequence_pianoroll(filepath, min_note, max_note, input_seq_len):
    multitrack = pypianoroll.Multitrack(filepath, beat_resolution=4)
    sequence_full = multitrack.tracks[0]
    sequence_full.binarize()
    sequence_full = sequence_full.pianoroll
    seed_sequence = sequence_full[:, min_note:max_note + 1]
    print(seed_sequence.shape)

    print("padding...")
    if len(seed_sequence) > input_seq_len:
        seed_sequence = np.array(seed_sequence[:input_seq_len])
    else:
        zero_padded_seq = np.repeat(0, input_seq_len)
        zero_padded_seq[input_seq_len - len(seed_sequence):] = seed_sequence
        seed_sequence = zero_padded_seq
    print("size after padding: ", seed_sequence.shape)
    return seed_sequence


def build_template_for_generated_pianoroll(dshape, seed_sequence, model, min_note, max_note, input_seq_len, seedfilename, model_dir):
    generated = np.zeros((2*dshape[1], seed_sequence.shape[1]))
    print('shape of generated ', generated.shape)
    generated[:input_seq_len] = seed_sequence
    seed_filename = seedfilename.split(os.path.sep)[-1].split(".mid")[0]

    if not os.path.exists(os.path.join(model_dir, "samples")):
        os.mkdir(os.path.join(model_dir, "samples"))

    samples_dir = os.path.abspath(os.path.join(model_dir, "samples", seed_filename))
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)

    seed_dir = os.path.join(samples_dir, "seed")
    if not os.path.exists(seed_dir):
        os.mkdir(seed_dir)

    seedpath = os.path.join(seed_dir, "1seed.mid")

    print("saving seed...")
    save_trim_pianoroll_seq(seed_sequence,min_note,max_note,seedpath)
    print('seed saved at ', seedpath)
    return generated, samples_dir


def save_trim_melody_seq(seed_sequence,encoder,seedpath):
    midi_io.note_sequence_to_midi_file(melodies_lib.Melody(
    [
        encoder.decode_event(ev) for ev in np.trim_zeros(np.argmax(seed_sequence,axis=1), 'f')
    ]
    ).to_sequence(), seedpath)


def build_template_for_generated_melody(dshape, seed_sequence, model, min_note, max_note, input_seq_len, seedfilename, model_dir, encoder):
    generated = np.zeros((2*dshape[1], seed_sequence.shape[1]))
    print('shape of generated ', generated.shape)
    generated[:input_seq_len] = seed_sequence
    seed_filename = seedfilename.split(os.path.sep)[-1].split(".mid")[0]

    if not os.path.exists(os.path.join(model_dir, "samples")):
        os.mkdir(os.path.join(model_dir, "samples"))

    samples_dir = os.path.abspath(os.path.join(model_dir, "samples", seed_filename))
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)

    seedpath = os.path.join(samples_dir, "1seed.mid")

    print("saving seed...")
    save_trim_melody_seq(seed_sequence,encoder,seedpath)
    print('seed saved at ', seedpath)
    return generated, samples_dir


def plot_midifile(filepath, samples_dir, name):
    roll = None
    try:
        roll = pypianoroll.Multitrack(filepath,beat_resolution=4).tracks[0].pianoroll
    except Exception as _:
        return None
    plt.figure(figsize=(14,8))
    ax = plt.gca()
    pypianoroll.plot_pianoroll(ax, roll)
    plt.title(name)
    pathtopng = os.path.join(samples_dir, name)
    print('plotting pianoroll to %s' %pathtopng)
    plt.savefig(pathtopng, bbox_inches='tight')
    return True


def generate_pianoroll(args, input_seq_ln, model, generated, samples_dir, min_note, max_note,):
    temperature = float(args.temp)

    nr_samples = int(args.nr)

    for i in tqdm.tqdm(list(range(nr_samples))):
        
        for timestep in range(input_seq_ln, len(generated)):
            start_index = timestep - (input_seq_ln)
            sequence_for_prediction = generated[start_index:timestep]
    #         next_step, att = sample(model, sequence_for_prediction, temperature, withatt=True)
            next_step, _ = sample(model, sequence_for_prediction, temperature, withatt=args.att)
    #         print(att.argsort()[-10:][::-1])
            generated[timestep] = next_step

        generated_noseed = generated[input_seq_ln:]
        
        new_path = os.path.join(samples_dir, "temp_%s_%s.mid" %(temperature, i))
        save_trim_pianoroll_seq(generated_noseed,min_note,max_note,new_path)
        plot_midifile(new_path,samples_dir,"temp_%s_%s.png" %(temperature, i))



def pianoroll_sampling(filepath, min_note, max_note, model,
                       input_seq_len, dshape, model_dir):
    print('shape of sequence from pypianoroll...')
    seed_sequence = read_encode_pad_sequence_pianoroll(filepath, min_note, max_note, input_seq_len)
    generated, samples_dir = build_template_for_generated_pianoroll(dshape, seed_sequence, model, min_note, max_note, input_seq_len, filepath, model_dir)

    # plot seed and save in folder
    plot_midifile(filepath, samples_dir, '1seed.png')
    generate_pianoroll(args, input_seq_len, model, generated, samples_dir, min_note, max_note)


def generate_melody(args, input_seq_len, model, generated, samples_dir, min_note, max_note, encoder):
    temperature = float(args.temp)
    to_generate = int(args.nr)
    nr_empty = 0
    nr_generated = 0

    progress = tqdm.tqdm(total=to_generate)
    atts = []
    softmax_es = []
    tokens_low = []
    tokens_high = []
    while nr_generated != to_generate:
        for timestep in range(input_seq_len, len(generated)):
            start_index = timestep - (input_seq_len)
            sequence_for_prediction = generated[start_index:timestep]
    #         next_step, att = sample(model, sequence_for_prediction, temperature, withatt=True)
            next_step = None
            if args.att:
                next_step, att, softmax_preds = sample(model, sequence_for_prediction, temperature, withatt=args.att)
                if args.no_zero:
                    input_tokens = np.argmax(sequence_for_prediction,axis=1)
                    mask = np.where(input_tokens==0)
                    att[mask] = 0
                if np.argmax(att) < 6:
                    # print('focusing on token', np.argmax(sequence_for_prediction[np.argmax(att)]), 'at time step index', np.argmax(att))
                    tokens_low.append(
                        np.argmax(sequence_for_prediction[np.argmax(att)])
                    )
                    if args.debug_print:
                        print(
                            'window around focused token ', 
                            np.argmax(
                                sequence_for_prediction[0:np.argmax(att)+3], 
                            axis=1)
                        )
                        print('softmax pointing at ', np.argmax(softmax_preds), ' actual prediction is ', np.argmax(next_step))
                elif np.argmax(att) > 30:
                    tokens_high.append(
                        np.argmax(sequence_for_prediction[np.argmax(att)])
                    )
                atts.append(att)
            else:
                next_step, softmax_preds = sample(model, sequence_for_prediction, temperature, withatt=args.att)
            softmax_es.append(softmax_preds)
    #         print(att.argsort()[-10:][::-1])
            generated[timestep] = next_step


        generated_noseed = generated[input_seq_len:]
        unique_pitches = np.unique(np.argmax(generated_noseed,axis=1))
        if len(unique_pitches) == 1 and unique_pitches[0] == 0:
            nr_empty += 1
        else:
            new_path = os.path.join(samples_dir, "temp_%s_%s.mid" %(temperature, nr_generated))

            save_trim_melody_seq(generated_noseed, encoder, new_path)
            if not plot_midifile(new_path,samples_dir,"temp_%s_%s.png" %(temperature, nr_generated)):
                nr_empty += 1
            else:
                nr_generated += 1
                progress.update(1)

    print('generated %s empty rolls' %nr_empty)
    if args.att:
        atts = np.array(atts)
        atts = atts.reshape(to_generate, -1, atts.shape[-1])
        np.save(os.path.join(samples_dir, 'atts.npy'), atts)

    softmax_es = np.array(softmax_es)
    softmax_es = softmax_es.reshape(to_generate, -1, softmax_es.shape[-1])
    np.save(os.path.join(samples_dir, 'softmax.npy'), softmax_es)
    with open(os.path.join(samples_dir, '%s empty.txt' %nr_empty), 'w') as f:
        f.writelines('\n')
                


def melody_sampling(filepath, min_note, max_note, model,
                           input_seq_len, dshape, model_dir):
    print('shape of sequence from pypianoroll...')
    seed_sequence, encoder = read_encode_pad_sequence_melody(filepath, min_note, max_note, input_seq_len)
    generated, samples_dir = build_template_for_generated_melody(dshape, seed_sequence, model, min_note, max_note, input_seq_len, filepath, model_dir, encoder)

    # plot seed and save in folder
    plot_midifile(filepath, samples_dir, '1seed.png')
    generate_melody(args, input_seq_len, model, generated, samples_dir, min_note, max_note, encoder)


def main(args):
    input_seq_len, dshape, min_note, max_note = get_dataset(args)
    model = get_model(args)
    transposed_seed = transpose_seed(args)
    print("min, max:")
    print(min_note, max_note)
    model_dir = os.path.abspath(args.model_id)

    if "pianoroll" in args.model_id:
        # pianoroll encoding
        pianoroll_sampling(transposed_seed, min_note, max_note, model,
                           input_seq_len, dshape, model_dir)

    elif "melody" in args.model_id or args.melody:
        # melody encoding
        # melody_sampling(transposed_seed, min_note, max_note, model,
        #                 input_seq_len, dshape)
        melody_sampling(transposed_seed, min_note, max_note, model,
                           input_seq_len, dshape, model_dir)
    else:
        print("unknown encoding in model name : %s" % args.model_id)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sample from model using a specific seed")
    parser.add_argument('model_id', metavar='id', type=str, help='model id')
    parser.add_argument(
        'seed', metavar='seed', type=str, help='path to seed midi file')
    parser.add_argument(
        '--nr',
        type=str,
        default="10",
        help='how many samples to generate. default = 10')
    parser.add_argument(
        '--temp',
        type=float,
        default="1.0",
        help='temperature for sampling. default = 1.0')
    parser.add_argument(
        '--att',
        action='store_true'
    )
    parser.add_argument(
        '--melody',
        action='store_true',
    )
    parser.add_argument(
        '--debug_print',
        action='store_true',
        help='whether to print info about attention tokens'
    )
    parser.add_argument(
        '--no_zero',
        action='store_true',
        help='in plotting attention remove all zeros'
    )

    args = parser.parse_args()

    print('generating %s samples, at %s temperature, using %s, from seed %s' %(args.nr, args.temp, args.model_id, args.seed))
    main(args)
