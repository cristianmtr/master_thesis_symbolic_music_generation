"""
callbacks for model
"""
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LambdaCallback, CSVLogger
from glob import glob
from utils import is_cluster, is_abacus
from keras_tqdm import TQDMNotebookCallback
from keras import backend as K
import pickle
import numpy as np
import time


start_time = None


def write_elapsed_time(model_dir):
    global start_time
    # your code
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H %M %S", time.gmtime(elapsed_time))
    timefiles = glob(os.path.join(model_dir, 'TIME*'))
    for timefile in timefiles:
        os.remove(timefile)
    with open(os.path.join(model_dir, 'TIME %s.txt' %elapsed_time_str), 'w') as f:
        f.writelines('\n')


def start_timer():
    global start_time
    toprint = False
    if start_time is None:
        toprint = True
    start_time = time.time()
    if toprint:
        start_time_str = time.strftime("%H:%M:%S", time.gmtime(start_time))
        print('Started timer at %s' %start_time_str)


def write_logs(model_dir, logs):
    path_to_logs = os.path.join(model_dir, 'logs.npy')
    np.save(path_to_logs, logs)
    print('wrote logs to %s' % path_to_logs)
    return


def delete_epoch_counters(model_dir):
    epoch_files = glob(os.path.join(model_dir, "EPOCH*"))
    for f in epoch_files:
        print('removing %s' % f)
        os.remove(f)
    return


def highest_epoch(epoch_files):
    maxnr = None
    for f in epoch_files:
        epoch = int(f.split("EPOCH ")[-1])
        if maxnr is None or epoch > maxnr:
            maxnr = epoch
    return maxnr


def write_epoch_in_folder(epoch, logs, model_dir):
    """
    Read for 'EPOCH (NR)'
    Get highest nr
    Add +1
    Write 'EPOCH (NEW NR)'
    """
    epoch_files = glob(os.path.join(model_dir, "EPOCH*"))
    if len(epoch_files) == 0:
        print('no epoch file found')
        # write epoch file
        epoch_file = os.path.join(model_dir, "EPOCH %s" % epoch)
        with open(epoch_file, 'w') as f:
            f.write('\n')
        print('wrote epoch file %s' % epoch_file)
    else:
        last_epoch = highest_epoch(epoch_files)
        new_epoch = last_epoch + 1
        epoch_file = os.path.join(model_dir, "EPOCH %s" % new_epoch)
        with open(epoch_file, 'w') as f:
            f.write('\n')
        print('wrote new epoch: %s' % epoch_file)
    return


def get_callbacks(model_dir, args, model, checkpoint_monitor='val_loss'):
    callbacks = []

    # save model checkpoints
    filepath = os.path.join(model_dir,
                            'model.h5')

    callbacks.append(ModelCheckpoint(filepath,
                                     monitor=checkpoint_monitor,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       patience=3,
                                       verbose=1,
                                       mode='auto',
                                       epsilon=0.0005,
                                       cooldown=0,
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(model_dir, 'tensorboard-logs'),
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=False))

    WriteEpochNumber = LambdaCallback(
        on_epoch_end=lambda epoch, logs: write_epoch_in_folder(
            epoch, logs, model_dir)
    )

    callbacks.append(WriteEpochNumber)

    StartTimerCallback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: start_timer()
    )

    callbacks.append(StartTimerCallback)

    WriteElapsedTime = LambdaCallback(
        on_epoch_end=lambda epoch, logs: write_elapsed_time(model_dir)
    )

    callbacks.append(WriteElapsedTime)

    logfile = os.path.join(model_dir, 'log.csv')
    csv_logger = CSVLogger(logfile, append=True, separator=';')
    callbacks.append(csv_logger)
    print('will be logging into %s' % logfile)

    verbosity = 1

    if args.tqdm:
        callbacks.append(TQDMNotebookCallback())
        verbosity = 0

    if is_cluster() or is_abacus():
        verbosity = 0

    return verbosity, callbacks
