import logging

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging

from gensim.models.callbacks import CallbackAny2Vec
class MyCallback(CallbackAny2Vec):
    pass


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

generator2 = Generator2()
reader = generator2.reader
ortho = generator2.ortho
freq = generator2.freq
preparator = WikiPagePreparator()

corpus = WikiCorpus(reader, type='lemm')
corpusw2v = CorpusW2v(corpus, reader)

corpusw2v.load()
nn1 = NN1(reader, corpusw2v)
nn2 = NN2(reader, corpusw2v)
nn3 = NN3(reader, corpusw2v)
nn4 = NN4(reader, corpusw2v)

poems = generator2.poems