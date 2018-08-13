# https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b

from __future__ import print_function
from my import *

__author__ = 'maxim'

import numpy as np
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

# https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b
class NN3(NNBase):
    def __init__(self, reader: DataReader, w2v: CorpusW2v):
        super().__init__(reader, w2v)

    def create_model(self):
        nn = Sequential()
        nn.add(Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights]))
        nn.add(LSTM(units=self.emdedding_size))
        nn.add(Dense(units=self.vocab_size))
        nn.add(Activation('softmax'))
        nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.nn = nn

    def prepare_data(self, lines: Iterator[List[str]], lines_len: int):
        print('\nPreparing the data for LSTM...')
        max_sentence_len = 40
        sentences = [sent[:max_sentence_len] for sent in lines if len(sent) > 1]
        for i, _ in enumerate(sentences):
            sentences[i] = ['помнить', 'чудный']
        print('Num sentences:', len(sentences))
        train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
        train_y = np.zeros([len(sentences)], dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence[:-1]):
                train_x[i, t] = self.word2idx(word)
            train_y[i] = self.word2idx(sentence[-1])
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)
        return train_x, train_y

    def train(self):
        limit = 1000
        self.create_model()
        X, y = self.prepare_data(self.lines(limit), limit)
        self.nn.fit(X, y, batch_size=128, epochs=1)
        def on_epoch_end(epoch, _):
            print('\nGenerating text after epoch: %d' % epoch)
            texts = [
                'пленник собственный',
                'красавец мужчина', # конь
                'честь красный армия оркестр медь' # грянуть
            ]
            for text in texts:
                sample = self.generate_line(text)
                print('%s... -> %s' % (text, sample))

        self.nn.fit(X, y,
                  batch_size=128,
                  epochs=20,
                  callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_line(self, text, num_generated=10, temperature=0.7):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.nn.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)

