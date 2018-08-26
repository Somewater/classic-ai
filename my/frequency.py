from typing import Dict, Union, Iterator, Tuple

from my import DataReader, Word
import pickle
from collections import defaultdict, Counter
from my.utils import lemma
import marisa_trie


class Frequency(object):
    def __init__(self, reader: DataReader, lemmatize=False):
        self.reader = reader
        self.word_count = None
        self._max_freq = None
        self.lemmatize = lemmatize
        self.filepath_tree = 'data/frequency_tree_lemms.bin' if lemmatize else 'data/frequency_tree.bin'
        self.filepath_pickle = 'data/frequency_lemms.pickle' if lemmatize else 'data/frequency.pickle'

    def load(self):
        self.tree = marisa_trie.RecordTrie('<f')
        self.tree.load(self.filepath_tree)
        with open(self.filepath_pickle, 'rb') as f:
            self._max_freq = pickle.load(f)

    def load_from_dictionary(self):
        wc_lists = defaultdict(list)
        for need_lemmatization, generator, source_name in [(False, self.reader.read_freq_2011, '2011'),
                                              (False, self.reader.read_freq_hagen, 'hagen'),
                                              (True, self.reader.read_freq_litc_win, 'litc_win'),
                                              (False, self.reader.read_freq_wikipedia, 'wikipedia'),
                                              (True, self.reader.read_freq_flibusta, 'flibusta'),
                                              (True, self.reader.read_freq_puhlyi, 'puhlyi')]:
            print('read %s' % source_name)
            wc = generator()
            if self.lemmatize and need_lemmatization:
                wc2 = Counter()
                for w, ipm in wc.items():
                    wc2[lemma(w.lower())] += ipm
                wc = wc2
            else:
                wc2 = Counter()
                for w, ipm in wc.items():
                    wc2[w.lower()] += ipm
                wc = wc2
            wc2 = None
            for w, ipm in wc.items():
                wc_lists[w].append(ipm)
        wc = dict()
        for w, cs in wc_lists.items():
            wc[w] = sum(cs) / len(cs)
        self.word_count = wc
        self._max_freq = max(self.word_count.values())
        self.tree = marisa_trie.RecordTrie('<f', [(w, (ipm,)) for w, ipm in wc.items()])

    def save(self):
        if self.word_count is None:
            raise RuntimeError("Dictionaries not loaded yet")
        self.tree.save(self.filepath_tree)
        with open(self.filepath_pickle, 'wb') as f:
            pickle.dump(self._max_freq, f)

    def freq(self, word: Union[str, Word]) -> float:
        if self.tree is None:
            print("Frequency loading...")
            self.load()
        if isinstance(word, Word):
            word = word.text
        text = word.lower().replace('ё', 'е')
        if text in self.tree:
            return self.tree[text][0][0]
        else:
            return 0.0

    def max_freq(self):
        return self._max_freq

    def iterate_words(self, min_ipm=0.01) -> Iterator[Tuple[str, float]]:
        for w, (ipm, ) in self.tree.iteritems():
            if ipm >= min_ipm:
                yield w, ipm
