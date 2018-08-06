import logging

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

reader = DataReader()
generator = Generator1(logging.getLogger('generator'), reader)
generator.start()
preparator = WikiPagePreparator()

corpus = OpCorpus(reader)
corpusw2v = CorpusW2v(corpus, reader)