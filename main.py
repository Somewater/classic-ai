import logging

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec

generator = Generator1(logging.getLogger('generator'))
generator.start()
reader = generator.reader
preparator = WikiPagePreparator()

corpus = OpCorpus(reader)
corpusw2v = CorpusW2v(corpus, reader)