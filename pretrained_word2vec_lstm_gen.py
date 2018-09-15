#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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

print('\nFetching the text...')
url = 'https://raw.githubusercontent.com/maxim5/stanford-tensorflow-tutorials/master/data/arxiv_abstracts.txt'

print('\nPreparing the sentences...')
max_sentence_len = 40
#docs = [" ".join(words) for words in nn1.lines(1000)]

docs = []
i = 0
for p in reader.read_best_164443():
  for l in  p.get_cyrillic_lines():
    l = l.strip()
    if l:
      docs.append(l.lower())


sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:max_sentence_len]] for doc in docs]
print('Num sentences:', len(sentences))

print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(sentences, size=100, min_count=3, window=5, iter=100)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['жить', 'петь', 'душа']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))

def word2idx(word):
  if word in word_model.wv.vocab:
    return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10, temperature=0.0):
  word_idxs = [word2idx(word) for word in text.lower().split() if word2idx(word)]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'жить петь душа',
    'ребеночек жить петь'
  ]
  for text in texts:
    sample = generate_next(text, temperature=0.3)
    print('%s... -> %s' % (text, sample))

i = 0
while True:
  model.fit(train_x, train_y,
            batch_size=128,
            epochs=20,
            callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
  i += 1
  print('fitting %d completed' % i)
