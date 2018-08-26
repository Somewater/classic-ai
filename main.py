import logging

import my
from my import *
from collections import Counter
from gensim.models import Word2Vec
import logging
import random

from gensim.models.callbacks import CallbackAny2Vec
class MyCallback(CallbackAny2Vec):
    pass


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

generator2 = Generator2(1)
reader = generator2.reader
ortho = generator2.ortho
freq = generator2.freq
poems = generator2.poems
preparator = WikiPagePreparator()
corpusw2v = generator2.corpusw2v
corpus = corpusw2v.corpus

nn1 = NN1(reader, corpusw2v)
nn2 = NN2(reader, corpusw2v)
nn3 = NN3(reader, corpusw2v)
nn4 = NN4(reader, corpusw2v)

print("Initialize all generators...")
generator2.start()
generator2.random = random.Random(1)
generator2.poems.random = generator2.random

# generator3 = Generator2()
# generator3.start()
# generator3.corpusw2v.model_filepath = 'weights/5-2-500-5e-05-0-1/wiki_w2v.bin'
# generator3.corpusw2v.load()
#
# generator4 = Generator2()
# generator4.start()
# generator4.corpusw2v.model_filepath = 'weights/5-2-500-5e-05-1-1/wiki_w2v.bin'
# generator4.corpusw2v.load()
#
# generator2.random = random.Random(1)
# generator3.random = random.Random(1)
# generator4.random = random.Random(1)
# generator2.poems.random = generator2.random
# generator3.poems.random = generator3.random
# generator4.poems.random = generator4.random

# print(requests.post('http://localhost:8000/generate/p', json={'seed': 'регрессиозный'}).json())

"""
import pandas as pd
types = ['min', 'mean', 'min_idf', 'mean_vector'] # W/O min_freq!
pd.options.display.width = 200
pd.options.display.max_columns = 100
results = []
for t in types:
    generator2.w2v_distance_type = t
    generator2.random = random.Random(3)
    generator2.poems.random = generator2.random
    r = generator2.generate('p', 'проспект осветил луч солнца')
    results.append((t,r))
print(r)
data = [types[:]]
for i in range(8):
    data.append([])
    for t, r in results:
        if len(r.lines) > i:
            data[i+1].append(r.lines[i])
print(pd.DataFrame(data))
"""
