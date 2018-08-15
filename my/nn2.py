from keras import backend, Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Embedding, InputLayer
from keras.models import load_model
from typing import Callable
import numpy as np

from my import *

class NN2:
    nn: Sequential

    def __init__(self, reader: DataReader, w2v: CorpusW2v):
        self.reader = reader
        self.nn = None
        self.w2v = w2v
        self.vocab_size = self.w2v.model.wv.syn0.shape[0] # number of words
        self.emdedding_size = self.w2v.model.wv.syn0.shape[1] # hidden layer size
        self.max_sentence_length = 5

    def create_model(self):
        HIDDEN_DIM = 500
        LAYER_NUM = 2
        model = Sequential()
        model.add(LSTM(HIDDEN_DIM, input_shape=(None, self.emdedding_size), return_sequences=True))
        for i in range(LAYER_NUM - 1):
            model.add(LSTM(HIDDEN_DIM, return_sequences=True))
        model.add(TimeDistributed(Dense(self.emdedding_size)))
        #model.add(Dense(self.emdedding_size))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
        self.nn = model

    def prepare_data(self, lines: Iterator[List[str]], lines_len: int):
        train_x = np.zeros([lines_len, self.max_sentence_length, self.emdedding_size], dtype=np.float32)
        train_y = np.zeros([lines_len, self.max_sentence_length, self.emdedding_size], dtype=np.float32)
        i = 0
        for line in lines:
            #line = ['помнить', 'чудный']
            j = 0
            line = line[:self.max_sentence_length]
            padding = 1 + self.max_sentence_length - len(line)
            for word in line[:-1]:
                # TODO: handle words out of model
                train_x[i, j + padding, :] = self.w2v.model.wv[word]
                j += 1
            train_y[i, 0, :] = self.w2v.model.wv[line[-1]]
            i += 1
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
                        if i >= limit:
                            return

    def train(self):
        limit = 1000
        if self.nn is None:
            self.create_model()
        X, y = self.prepare_data(self.lines(limit), limit)
        self.nn.fit(X, y, batch_size=128, epochs=1)

    def save(self):
        self.nn.save('tmp/nn2.hdf5')

    def load(self):
        self.nn = load_model('tmp/nn2.hdf5')

    def generate_line(self, words = 'помнить'):
        if isinstance(words, str):
            words = words.split(' ')
        words = words[:self.max_sentence_length]
        x = np.zeros([len(words), self.max_sentence_length, self.emdedding_size,], dtype=np.float32)
        i = 0
        padding = self.max_sentence_length - len(words)
        for word in words:
            # TODO: handle words out of model
            x[0, i + padding, :] = self.w2v.model.wv[word]
            i += 1
        preds = self.nn.predict(x)
        v = preds[0][0]
        return self.w2v.model.similar_by_vector(v), preds