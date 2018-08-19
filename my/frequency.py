from typing import Dict, Union

from my import DataReader, Word
import pickle
from collections import defaultdict, Counter
from my.utils import lemma


class Frequency(object):
    def __init__(self, reader: DataReader):
        self.reader = reader
        self.word_count = None
        self._max_freq = None

    def load(self):
        with open('data/frequency.pickle', 'rb') as f:
            self.word_count = pickle.load(f)

    def load_from_dictionary(self):
        wc_lists = defaultdict(list)
        for need_lemmatization, generator in [(False, self.reader.read_freq_2011), (False, self.reader.read_freq_hagen), (True, self.reader.read_freq_litc_win)]:
            wc = generator()
            if need_lemmatization:
                wc2 = Counter()
                for w, ipm in wc.items():
                    wc2[lemma(w.lower())] += ipm
                wc = wc2
            else:
                wc2 = Counter()
                for w, ipm in wc.items():
                    wc2[w.lower()] += ipm
                wc = wc2
            for w, ipm in wc.items():
                wc_lists[w].append(ipm)
        wc = dict()
        for w, cs in wc_lists.items():
            wc[w] = sum(cs) / len(cs)
        self.word_count = wc

    def save(self):
        if self.word_count is None:
            raise RuntimeError("Dictionaries not loaded yet")
        with open('data/frequency.pickle', 'wb') as f:
            pickle.dump(self.word_count, f)

    def freq(self, word: Union[str, Word]) -> float:
        if self.word_count is None:
            print("Frequency loading...")
            self.word_count = self.reader.load_word_count()
        if isinstance(word, Word):
            word = word.text
        return self.word_count.get(word.lower()) or 0.0

    def max_freq(self):
        if self._max_freq is None:
            self._max_freq = max(self.word_count.values())
        return self._max_freq