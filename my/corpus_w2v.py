from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Callable, Tuple, Set, Dict

from my import DataHelper
from my.model import *
from gensim.models import Word2Vec, FastText
from my import DataReader, profiler
from my.utils import stem, lemma, MakeIter
import os
from gensim.corpora import WikiCorpus
#from scipy.spatial.distance import cosine
import numpy as np
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils, matutils
from gensim.similarities.index import AnnoyIndexer

class CorpusW2v(object):
    def __init__(self, corpus: Corpus, reader: DataReader, vector_size: int = 100):
        self.corpus = corpus
        self.reader = reader
        self.stop_words = reader.read_stop_words()
        self.model_filepath = self.reader.get_tmp_filepath((self.corpus and self.corpus.name() or 'wiki_corpus') + '_w2v.bin')
        self.model = Word2Vec(size=vector_size, window=5, min_count=5, workers=multiprocessing.cpu_count(),
                              max_final_vocab=100000, hs=1)
        self.helper = DataHelper(reader)
        self.indexer = None

    def sentences(self, stemm: bool = False, lemmatize: bool = False) -> Iterator[Iterator[str]]:
        i = 0
        for topic in self.corpus.get_topics():
            tokens = [w for w in topic.get_cyrillic_words() if len(w) > 2 and not w in self.stop_words]
            if lemmatize:
                tokens = [lemma(w) for w in tokens]
            if stemm:
                tokens = [stem(w) for w in tokens]
            i += 1
            yield tokens

    def train(self, alpha, min_alpha = None, epochs=1):
        if self.model.corpus_count == 0:
            self.model.build_vocab(self.sentences())
        self.model.alpha = alpha
        self.model.min_alpha = min_alpha
        self.model.train(sentences=MakeIter(self.sentences),
                         total_examples=self.model.corpus_count,
                         epochs=epochs,
                         start_alpha=alpha,
                         end_alpha=min_alpha,
                         report_delay=60.0)

    def build_save_indexer(self):
        self.indexer = AnnoyIndexer(self.model, 300)
        self.indexer.save(self.model_filepath + '.index', protocol=3)

    def save(self):
        self.model.save(self.model_filepath)

    def load(self):
        self.model = Word2Vec.load(self.model_filepath)
        self.model.init_sims(replace=True)
        if os.path.exists(self.model_filepath + '.index'):
            self.indexer = AnnoyIndexer()
            self.indexer.load(self.model_filepath + '.index')
        return self

    def find_similar_words(self, words: List[str], stemmer: Callable[[str], str] = None, topn=1000) -> Iterator[str]:
        word_in_corpus = []
        for word in words:
            if stemmer:
                word = stemmer(word)
            if word in self.model.wv:
                word_in_corpus.append(word)
        if word_in_corpus:
            for w, score in self.model.wv.most_similar(positive=word_in_corpus, topn=topn, indexer=self.indexer):
                yield w

    def accuracy(self) -> Tuple[float, Dict[str, Tuple[int, int]]]:
        topics = self.reader.read_check_topics()
        result_data = dict()
        result_acc = []
        for topic, lemms in topics.items():
            lemms = set(lemms)
            result_data[topic] = []
            for n in [100, 1000]:
                res = self.model.wv.most_similar(topic, topn=n)
                union = len(set([w for w, _ in res]) & lemms)
                result_data[topic].append(union)
                result_acc.append(union / n)
        return sum(result_acc) / len(result_acc), result_data

    def analogy_accuracy(self):
        section = self.model.accuracy('data/ru_analogy.txt')[-1]
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            return correct / (correct + incorrect)

    def word_vector(self, lemm: str):
        r =  self.model.wv.word_vec(lemm) if lemm in self.model.wv else None
        return r

    def text_vector(self, text: str, stemmer: Callable[[str], str] = None):
        """Вектор текста, получается путем усреднения векторов всех слов в тексте"""
        word_vectors = [
            self.word_vector(lemm)
            for lemm in [stemmer(token) for token in get_cyrillic_words(text.lower())]
            if len(lemm) > 2 and not (lemm in self.stop_words)
        ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return np.mean(word_vectors, axis=0)

    def mean_vector(self, text: str):
        vectors = self.vectors(text)
        if vectors:
            return matutils.unitvec(np.array(vectors).mean(axis=0)).astype('float32')
        else:
            print("Can't build mean vector: %s" % text)
            return np.zeros((self.model.vector_size,))

    def vectors(self, text: str):
        lemmas = [lemma(w) for w in get_cyrillic_words(text)]
        vectors = [self.model.wv.word_vec(w, use_norm=True) for w in lemmas if w in self.model.wv]
        if vectors:
            return vectors
        else:
            print("Can't build mean vector: %s" % text)
            return [np.zeros((self.model.vector_size,))]

    def distance(self, vec1, vec2, strategy = 'min_freq'):
        if vec1 is None or vec2 is None:
            return 2
        if isinstance(vec1, Seed):
            seed: Seed = vec1
            if strategy == 'min':
                scores = [cosine(vec2, v) for v in seed.vectors if v is not None]
                if scores:
                    return min(scores)
                else:
                    return 2
            elif strategy == 'mean':
                scores = [cosine(vec2, v) for v in seed.vectors if v is not None]
                if scores:
                    return sum(scores) / len(scores)
                else:
                    return 2
            elif strategy == 'min_idf':
                if seed.weighted_vectors:
                    vector, idf, freq = min(seed.weighted_vectors, key=lambda pair: cosine(vec2, pair[0]))
                    return cosine(vec2, vector) / idf * 22 # max idf
                else:
                    return 2
            elif strategy == 'min_freq': # WORST in the worstests!!!!
                if seed.weighted_vectors:
                    vector, idf, freq = min(seed.weighted_vectors, key=lambda pair: cosine(vec2, pair[0]))
                    return cosine(vec2, vector) * freq / 42329 # max freq
                else:
                    return 2
            elif strategy == 'mean_vector': # fastest
                return cosine(vec2, seed.mean_vector)
        else:
            r = cosine(vec1, vec2)
            return r

    @staticmethod
    def create_fasttext_model(self):
        return FastText.load_fasttext_format(os.path.join('data', 'fasttext', 'ru'))


def cosine(u, v):
    return 1.0 - np.average(u * v) / np.sqrt(np.average(np.square(u)) * np.average(np.square(v)))