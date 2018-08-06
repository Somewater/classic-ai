from typing import Callable

from my.model import *
from gensim.models import Word2Vec, FastText
from my import DataReader
from my.utils import stem, lemma
import os

class CorpusW2v(object):
    def __init__(self, corpus: Corpus, reader: DataReader):
        self.corpus = corpus
        self.reader = reader
        self.stop_words = reader.read_stop_words()
        self.model_filepath = self.reader.get_tmp_filepath(self.corpus.name() + '_w2v.bin')

    def sentences(self, stemm: bool = False, lemmatize: bool = False) -> Iterator[Iterator[str]]:
        i = 0
        for topic in self.corpus.get_topics():
            tokens = [w for w in topic.get_cyrillic_words() if len(w) > 2 and not w in self.stop_words]
            if lemmatize:
                tokens = [lemma(w) for w in tokens]
            if stemm:
                tokens = [stem(w) for w in tokens]
            i += 1
            #if i % 100000 == 0: print(i, 'topics iterated')
            yield tokens

    def train(self):
        self.model = Word2Vec(size=100, window=5, min_count=5, workers=4)
        self.model.build_vocab(self.sentences())
        self.model.train(self.sentences(), total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model.save(self.model_filepath)

    def load(self):
        self.model = Word2Vec.load(self.model_filepath)

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
        #return .most_similar(positive=[])

    @staticmethod
    def create_fasttext_model(self):
        return FastText.load_fasttext_format(os.path.join('data', 'fasttext', 'ru'))