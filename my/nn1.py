from keras import backend, Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Embedding
from keras.models import load_model
from typing import Callable
import numpy as np

from my import *

class NN1:
    nn: Sequential

    def __init__(self, reader: DataReader, w2v: CorpusW2v):
        self.reader = reader
        self.nn = None
        self.w2v = w2v

    def create_model(self, pretrained_weights):
        vocab_size, emdedding_size = pretrained_weights.shape
        nn = Sequential()
        nn.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        nn.add(LSTM(units=emdedding_size))
        nn.add(Dense(units=vocab_size))
        nn.add(Activation('softmax'))
        nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.nn = nn

    def word2idx(self, word):
        return self.w2v.model.wv.vocab[word].index

    def idx2word(self, idx):
        return self.w2v.model.wv.index2word[idx]

    def prepare_data(self, max_sentence_length: int, sentence_len: int, lines: Iterator[List[str]]):
        train_x = np.zeros([sentence_len, max_sentence_length], dtype=np.int32)
        train_y = np.zeros([sentence_len], dtype=np.int32)
        for i, line in enumerate(lines):
            for t, word in enumerate(line[:-1]):
                train_x[i, t] = self.word2idx(word)
            train_y[i] = self.word2idx(line[-1])
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)
        return train_x, train_y

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
                        if i > limit:
                            break

    def train(self):
        all_lines_count = 0
        max_line_length = 0
        for line in self.lines(10000):
            l = len(line)
            if l > max_line_length:
                max_line_length = l
            all_lines_count += 1
        if self.nn is None:
            self.create_model(self.w2v.model.wv.syn0)
        X, y = self.prepare_data(max_line_length, all_lines_count, self.lines(10000))
        self.nn.fit(X, y, batch_size=128, epochs=1)

    def save(self):
        self.nn.save('tmp/nn1.hdf5')

    def load(self):
        self.nn = load_model('tmp/nn1.hdf5')

    def generate_line(self):
        pass