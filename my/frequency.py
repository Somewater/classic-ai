from typing import Dict, Union

from my import DataReader, Word
import pickle


class Frequency(object):
    def __init__(self, reader: DataReader):
        self.reader = reader
        self.word_count = None
        self._max_freq = None

    def load(self):
        with open('data/frequency.pickle', 'rb') as f:
            self.word_count = pickle.load(f)

    def load_from_dictionary(self):
        self.word_count = self.reader.load_word_count()

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