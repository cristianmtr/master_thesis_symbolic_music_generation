import sys
import pickle
import argparse
from utils import *
from callbacks import get_callbacks, delete_epoch_counters
from glob import glob
import shutil
from generator import BatchGenerator
from keras_self_attention import SeqWeightedAttention
from sklearn.model_selection import train_test_split
from build_dataset import min_max_from_folder
import keras
import os
import config
from architecture import new_architecture


def get_data(dataset):
    dpath = dataset['path']
    dataset = np.memmap(dpath, mode="r",
                        dtype="uint8", shape=dataset['shape'])
    x = dataset[:, :-1]
    y = dataset[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True)
    print('we have %s training files and %s validation files' %
          (len(y_train), len(y_val)))

    return X_train, X_val, y_train, y_val


def get_model_id(args):
    # model_folk100k_pianoroll_bi2lstm32_attention
    model_id = None
    if args.new:
        model_id = "model_"
        model_id += args.dataset
        if args.bi:
            model_id += "_bi"
        else:
            model_id += "_"
        
        model_id += "%slstm%s_" %(args.layers, args.cells)
        if args.att:
            model_id += "attention"
        else:
            model_id += "noattention"
        print("generated model id from args: %s" %model_id)
    else:
        model_id = args.id
        print("using existing model id %s" %model_id)
    return model_id


def get_model_dir(args):
    model_id = get_model_id(args)
    model_dir = os.path.abspath(os.path.join('.', model_id))
    if os.path.exists("/gdrive/"):
        model_dir = os.path.join("/gdrive", "My Drive", "THESIS", model_id)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    else:
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    print('model id: ', model_id)
    print('model dir: ', model_dir)
    return model_dir


def get_model(args, dshape):
    model_dir = get_model_dir(args)

    model = None
    loss = 'categorical_crossentropy'
    optimizer = keras.optimizers.Adam(lr=0.005)

    if args.new:
        print('generating NEW model...')
        model = new_architecture(
            dshape[1]-1, 
            dshape[2],
            args.layers,
            args.bi,
            args.att,
            args.cells
        )
        # copy arch to folder
        shutil.copy('architecture.py', model_dir)
        model_json = model.to_json()
        model_json_path = os.path.join(model_dir, "model.json")
        print('storing model json in %s' % model_json_path)
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        # delete epoch counters
        delete_epoch_counters(model_dir)
        model.compile(
            loss=loss,
            optimizer=optimizer
        )

    else:
        print('using existing model...')
        model_json_path = os.path.join(model_dir, "model.json")
        model = keras.models.model_from_json(open(model_json_path, "r").read(
        ), custom_objects=SeqWeightedAttention.get_custom_objects())

        model_weights_path = os.path.join(model_dir, "model.h5")
        print('loading existing weights from %s...' % model_weights_path)
        model.load_weights(model_weights_path)
        model.compile(
            loss=loss,
            optimizer=optimizer
        )

    print(model.summary())

    return model, model_dir


def get_dataset_name(args):
    if args.dataset:
        return config.datasets[args.dataset] 
    elif args.id:
        for name in config.datasets.keys():
            if name in args.id:
                print("found name of dataset in model id : %s" %name)
                return config.datasets[name]
    else:
        print("Dataset could not be deduced...")
        sys.exit(1)


def main(args):
    dataset = get_dataset_name(args)
    X_train, X_val, y_train, y_val = get_data(dataset)
    model, model_dir = get_model(args, dataset['shape'])

    verbosity, callbacks = get_callbacks(model_dir, args, model)

    model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        verbose=verbosity
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train LSTM model for music generation")
    parser.add_argument('--id', metavar='id', type=str,
                        help='model id to load weights if continuing training')
    parser.add_argument('--new', action='store_true', default=False,
                        help='whether to load existing weights for model or \
                             create new one')
    parser.add_argument('--tqdm', action='store_true', default=False,
                        help='whether this is running in a Jupyter environment')
    parser.add_argument('--dataset', type=str,
                        default="folk100k_melody",
                        help='what dataset to use. Check "config.py" for options')
    parser.add_argument('--layers', type=int, help="nr of layers")
    parser.add_argument('--bi', action="store_true", help="include bidirectionality wrapper for each layer")
    parser.add_argument('--att', action="store_true", help="add attention mechanism on top of last layer")
    parser.add_argument('--cells', type=int, help="nr of cells in each layer")


    args = parser.parse_args()

    if not args.new and not args.id:
        print('either continue training a model by using "--id" or train a new one by using "--new"')
        sys.exit(1)

    if args.id and args.new:
        print('either continue training a model by using "--id" or train a new one by using "--new"')
        sys.exit(1)

    if args.id and not args.new:
        print('continue training of model %s' %args.id)

    if args.new and not args.id:
        if not args.layers or not args.cells:
            print('need to specify nr of layers and nr of cells per layer')
            sys.exit(1)
        else:
            print('training a new model...')

    main(args)
