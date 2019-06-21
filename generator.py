import numpy as np
import tqdm
import utils
from keras.utils import Sequence


class BatchGenerator(Sequence):

    def __init__(self, filelist, batch_size, x_len, vocab_size):
        self.filelist = filelist
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.x_len = x_len
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)//self.batch_size

    def __getitem__(self, idx):
        batch_start_index = idx * self.batch_size
        batch_end_index = (idx + 1) * self.batch_size
        files = self.filelist[batch_start_index:batch_end_index]
        x, y = self.read_return(files)
        return x, y

    def read_return(self, files):
        x_batch = np.zeros((self.batch_size, self.x_len, self.vocab_size))
        y_batch = np.zeros((self.batch_size, self.vocab_size))
        for b_i, fname in enumerate(files):
            full_seq = np.load(fname).astype('uint8')
            x_batch[b_i] = full_seq[:-1]
            y_batch[b_i] = full_seq[-1]
        return x_batch, y_batch
