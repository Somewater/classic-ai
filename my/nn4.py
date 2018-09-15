from my import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

# https://github.com/ChunML/text-generator
class NN4(NNBase):
    def create_model(self):
        nn = Sequential()
        nn.add(LSTM(self.HIDDEN_DIM, input_shape=(None, self.VOCAB_SIZE), return_sequences=True))
        for i in range(self.LAYER_NUM - 1):
            nn.add(LSTM(self.HIDDEN_DIM, return_sequences=True))
        nn.add(TimeDistributed(Dense(self.VOCAB_SIZE)))
        nn.add(Activation('softmax'))
        nn.compile(loss="categorical_crossentropy", optimizer="rmsprop")
        self.nn = nn

    # method for generating text
    def generate_line(self, length):
        # starting with random character
        ix = [np.random.randint(self.VOCAB_SIZE)]
        y_char = [self.ix_to_char[ix[-1]]]
        X = np.zeros((1, length, self.VOCAB_SIZE))
        for i in range(length):
            # appending the last predicted character to sequence
            X[0, i, :][ix[-1]] = 1
            ix = np.argmax(self.nn.predict(X[:, :i+1, :])[0], 1)
            y_char.append(self.ix_to_char[ix[-1]])
        return ('').join(y_char)

    def prepare_data(self, lines: Iterator[List[str]], lines_len: int):
        data = ''
        for line in lines:
            for w in line:
                data += ' ' + w
        chars = sorted(set(data))
        VOCAB_SIZE = len(chars)

        print('Data length: {} characters'.format(len(data)))
        print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

        ix_to_char = {ix:char for ix, char in enumerate(chars)}
        char_to_ix = {char:ix for ix, char in enumerate(chars)}

        X = np.zeros((len(data)//self.SEQ_LENGTH, self.SEQ_LENGTH, VOCAB_SIZE))
        y = np.zeros((len(data)//self.SEQ_LENGTH, self.SEQ_LENGTH, VOCAB_SIZE))
        for i in range(0, len(data)//self.SEQ_LENGTH):
            X_sequence = data[i*self.SEQ_LENGTH:(i+1)*self.SEQ_LENGTH]
            X_sequence_ix = [char_to_ix[value] for value in X_sequence]
            input_sequence = np.zeros((self.SEQ_LENGTH, VOCAB_SIZE))
            for j in range(self.SEQ_LENGTH):
                input_sequence[j][X_sequence_ix[j]] = 1.
                X[i] = input_sequence

            y_sequence = data[i*self.SEQ_LENGTH+1:(i+1)*self.SEQ_LENGTH+1]
            y_sequence_ix = [char_to_ix[value] for value in y_sequence]
            target_sequence = np.zeros((self.SEQ_LENGTH, VOCAB_SIZE))
            for j in range(self.SEQ_LENGTH):
                target_sequence[j][y_sequence_ix[j]] = 1.
                y[i] = target_sequence
        return X, y, VOCAB_SIZE, ix_to_char

    def train(self):
        self.BATCH_SIZE = 50
        self.HIDDEN_DIM = 500
        self.SEQ_LENGTH = 50
        self.WEIGHTS = ''
        self.GENERATE_LENGTH = 50
        self.LAYER_NUM = 2

        # Creating training data
        X, y, self.VOCAB_SIZE, self.ix_to_char = self.prepare_data(self.lines(1000), 1000)

        # Creating and compiling the Network
        self.create_model()

        if not self.WEIGHTS == '':
            self.nn.load_weights(self.WEIGHTS)
            nb_epoch = int(self.WEIGHTS[self.WEIGHTS.rfind('_') + 1:self.WEIGHTS.find('.')])
        else:
            nb_epoch = 0

        # Training if there is no trained weights specified
        if self.WEIGHTS == '2':
            while True:
                print('\n\nEpoch: {}\n'.format(nb_epoch))
                self.nn.fit(X, y, batch_size=self.BATCH_SIZE, verbose=1, nb_epoch=1)
                nb_epoch += 1
                if nb_epoch % 100 == 0:
                    print(self.generate_line(self.GENERATE_LENGTH))
                    self.nn.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(self.LAYER_NUM, self.HIDDEN_DIM, nb_epoch))

        # Else, loading the trained weights and performing generation only
        elif self.WEIGHTS == '':
            # Loading the trained weights
            self.nn.load_weights(self.WEIGHTS)
            print(self.generate_line(self.GENERATE_LENGTH))
            print('\n\n')
        else:
            print('\n\nNothing to do!')