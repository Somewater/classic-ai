from typing import List

from my import DataReader, Topic
from my.utils import *
import math

class DataHelper:
    def __init__(self, reader: DataReader):
        self.reader = reader
        self.stop_words = reader.read_stop_words()

    def get_lemms(self, any) -> List[str]:
        if isinstance(any, str):
            words = get_cyrillic_words(any)
        elif isinstance(any, list):
            words = any
        elif isinstance(any, Topic):
            words = any.get_cyrillic_words()
        else:
            raise RuntimeError("Undefined type %s" % type(any))

        filtered_lemms = []
        for w in words:
            w = w.lower()
            if len(w) > 2 and not (w in self.stop_words):
                filtered_lemms.append(lemma(w))

        return filtered_lemms

    @staticmethod
    def tf(word: str, blob: List[str]):
        return blob.count(word) / len(blob)

    @staticmethod
    def n_containing(word, bloblist: Iterator[List[str]]):
        return sum(1 for blob in bloblist if word in blob)

    @staticmethod
    def idf(word, bloblist: Iterator[List[str]]):
        return math.log(len(bloblist) / (1 + DataHelper.n_containing(word, bloblist)))

    @staticmethod
    def tfidf(word, blob, bloblist):
        return DataHelper.tf(word, blob) * DataHelper.idf(word, bloblist)