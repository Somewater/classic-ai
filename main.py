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

reader = DataReader()
generator = Generator1(logging.getLogger('generator'), reader)
preparator = WikiPagePreparator()
freq = Frequency(reader)

corpus = WikiCorpus(reader, type='lemm')
corpusw2v = CorpusW2v(corpus, reader)

corpusw2v.load()
nn1 = NN1(reader, corpusw2v)
nn2 = NN2(reader, corpusw2v)
nn3 = NN3(reader, corpusw2v)
nn4 = NN4(reader, corpusw2v)

ortho = OrthoDict()