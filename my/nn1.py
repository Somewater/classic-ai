from keras import backend, Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Embedding, Dropout
from keras.models import load_model
from keras.utils import to_categorical
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
        self.vocab_size = vocab_size
        return_sequences = False
        nn = Sequential()
        nn.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        #nn.add(self.w2v.model.wv.get_keras_embedding())
        nn.add(LSTM(units=emdedding_size, return_sequences=False))
        #nn.add(LSTM(units=emdedding_size, return_sequences=return_sequences))
        if return_sequences:
            nn.add(TimeDistributed(Dense(vocab_size)))
        else:
            nn.add(Dense(vocab_size))
        #nn.add(Dropout(0.5))
        nn.add(Activation('softmax'))
        nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])
        self.nn = nn

    def word2idx(self, word):
        return self.w2v.model.wv.vocab[word].index

    def idx2word(self, idx):
        return self.w2v.model.wv.index2word[idx]

    def prepare_data(self, max_sentence_length: int, sentence_count: int, lines: Iterator[List[str]]):
        train_x = np.zeros([sentence_count, max_sentence_length], dtype=np.int32)
        train_y = np.zeros([sentence_count], dtype=np.int32)
        for i, line in enumerate(lines):
            line = line[:max_sentence_length]
            #line = ['помнить', 'чудный']
            padding = 0 # 1 + max_sentence_length - len(line)
            for t, word in enumerate(line[:-1]):
                train_x[i, t + padding] = self.word2idx(word)
            train_y[i] = self.word2idx(line[-1])
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
        max_line_length = 10
        self.max_sentence_length = max_line_length
        if self.nn is None:
            self.create_model(self.w2v.model.wv.syn0)
        X, y = self.prepare_data(max_line_length, limit, self.lines(limit))
        self.nn.fit(X, y, batch_size=128, epochs=1)

    def save(self):
        self.nn.save('tmp/nn1.hdf5')

    def load(self):
        self.nn = load_model('tmp/nn1.hdf5')

    def generate_line(self, text = 'помнить', num_generated=10, temperature=0.0):
        def sample(preds, temperature=1.0):
            if temperature <= 0:
                return np.argmax(preds)
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.nn.predict(x=np.array(word_idxs))
            idx = sample(prediction[-1], temperature)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)