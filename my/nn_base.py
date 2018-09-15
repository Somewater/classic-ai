from keras import backend, Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Embedding, InputLayer
from keras.models import load_model
from typing import Callable
import numpy as np

from my import *

class NNBase:
    """
    Neural net base class
    """
    nn: Sequential

    def __init__(self, reader: DataReader, w2v: CorpusW2v):
        self.reader = reader
        self.nn = None
        self.w2v = w2v
        self.pretrained_weights = self.w2v.model.wv.syn0
        self.vocab_size = self.pretrained_weights.shape[0] # number of words
        self.emdedding_size = self.pretrained_weights.shape[1] # hidden layer size
        self.max_sentence_length = 5
        self.name = self.__class__.__name__

    def create_model(self):
        raise RuntimeError("Not implemented")

    def prepare_data(self, lines: Iterator[List[str]], lines_len: int):
        raise RuntimeError("Not implemented")

    def generate_line(self, words = 'помнить'):
        raise RuntimeError("Not implemented")

    def word2idx(self, word):
        return self.w2v.model.wv.vocab[word].index

    def idx2word(self, idx):
        return self.w2v.model.wv.index2word[idx]

    def lines(self, limit = None) -> Iterator[List[str]]:
        i = 0
        for cb in self.reader.read_best_164443_lemms():
            lines = cb.get_sentence_lemms()
            for line in lines:
                words = []
                for w in line:
                    if w in self.w2v.model.wv:
                        words.append(w)
                if len(words) > 1:
                    yield words
                    if not limit is None:
                        i += 1
                        if i >= limit:
                            return

    def train(self):
        limit = 1000
        if self.nn is None:
            nn = self.create_model()
            if self.nn is None:
                self.nn = nn
        X, y = self.prepare_data(self.lines(limit), limit)
        self.nn.fit(X, y, batch_size=128, epochs=1)

    def save(self):
        self.nn.save('tmp/%s.hdf5' % self.name)

    def load(self):
        self.nn = load_model('tmp/%s.hdf5' % self.name)
