from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Callable, Tuple, Set, Dict

from my import DataHelper
from my.model import *
from gensim.models import Word2Vec, FastText
from my import DataReader
from my.utils import stem, lemma
import os
from gensim.corpora import WikiCorpus
from scipy.spatial.distance import cosine
import numpy as np
import multiprocessing

class CorpusW2v(object):
    def __init__(self, corpus: Corpus, reader: DataReader, vector_size: int = 100):
        self.corpus = corpus
        self.reader = reader
        self.stop_words = reader.read_stop_words()
        self.model_filepath = self.reader.get_tmp_filepath(self.corpus.name() + '_w2v.bin')
        self.model = Word2Vec(size=vector_size, window=5, min_count=5, workers=multiprocessing.cpu_count())
        self.helper = DataHelper(reader)

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
        self.model.train(sentences=iter(self.sentences()),
                         total_examples=self.model.corpus_count,
                         epochs=epochs,
                         start_alpha=alpha,
                         end_alpha=min_alpha,
                         report_delay=60.0)

    def save(self):
        self.model.save(self.model_filepath)

    def load(self):
        self.model = Word2Vec.load(self.model_filepath)
        self.model.init_sims(replace=True)
        #self.lemma2word = {word.split('_')[0]: word for word in self.model.wv.index2word}

    def find_similar_words(self, words: List[str], stemmer: Callable[[str], str] = None) -> Iterator[str]:
        word_in_corpus = []
        for word in words:
            if stemmer:
                word = stemmer(word)
            if word in self.model.wv:
                word_in_corpus.append(word)
        if word_in_corpus:
            for w, score in self.model.wv.most_similar(positive=word_in_corpus, topn=1000):
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

    def word_vector(self, word):
        word = lemma(word)
        #word = self.lemma2word.get(lemma)
        return self.model[word] if word in self.model else None

    def text_vector(self, text):
        """Вектор текста, получается путем усреднения векторов всех слов в тексте"""
        word_vectors = [
            self.word_vector(token)
            for token in get_cyrillic_words(text.lower())
            if len(token) > 2 and not (token in self.stop_words)
        ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return np.mean(word_vectors, axis=0)

    def distance(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 2
        return cosine(vec1, vec2)

    @staticmethod
    def create_fasttext_model(self):
        return FastText.load_fasttext_format(os.path.join('data', 'fasttext', 'ru'))