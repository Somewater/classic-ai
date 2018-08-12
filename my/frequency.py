from typing import Dict

from my import DataReader


class Frequency(object):
    def __init__(self, reader: DataReader):
        self.reader = reader
        self.word_count = None

    def freq(self, word: str) -> float:
        if self.word_count is None:
            print("Frequency loading...")
            self.word_count = self.reader.load_word_count()
        return self.word_count.get(word.lower()) or 0.0