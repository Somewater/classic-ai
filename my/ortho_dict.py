from typing import Optional, Dict, Union
from my import Phonetic, Word
import os
from functools import lru_cache
from typing import List, Set, Union, Optional

import os
from threading import Lock
import logging
import time


class Hagen:
    STRESSED_CHAR = "'"

    def __init__(self):
        self.name = 'hagen'
        self.words = []

    def load(self):
        with open(os.path.join('data', 'ortho', 'hagen-orf.txt')) as file:
            normal_form = None
            already_added_hashes = set()
            for line in file:
                line = line.strip()
                if line:
                    if line[0] != ' ':
                        normal_form = None
                    not_used = False
                    if line.startswith('*'):
                        not_used = True
                        line = line[1:]
                    (text, stressed, number) = line.split(' | ')
                    if Word.allow_word(text):
                        stressed = stressed.replace('`', '')
                        word = Word.from_stressed(stressed, self.name, stressed_char=Hagen.STRESSED_CHAR,
                                                  stress_char_after=True)
                        key = word.__hash__()
                        if word and not key in already_added_hashes:
                            already_added_hashes.add(key)
                            if normal_form:
                                word.normal_form = normal_form
                                word.is_normal_form = False
                            else:
                                word.is_normal_form = True
                                normal_form = word
                            self.words.append(word)
                else:
                    normal_form = None

class Lopatin:
    STRESSED_CHAR = "`"

    def __init__(self):
        self.name = 'lopatin'
        self.words = []

    def load(self):
        with open(os.path.join('data', 'ortho', 'lop1v2.txt')) as file:
            for line in file:
                sharp_index = line.index('#')
                percent_index = line.index('%', sharp_index)
                text = line[0:sharp_index]
                if Word.allow_word(text):
                    stressed = line[(sharp_index + 1):percent_index].strip()
                    description = line[(percent_index + 1):].strip()
                    word = Word.from_stressed(stressed, self.name, stressed_char=Lopatin.STRESSED_CHAR,
                                              stress_char_after=False)
                    if word:
                        word.is_normal_form = True
                        self.words.append(word)

class Zalizniak:
    STRESSED_CHAR = "'"

    def __init__(self):
        self.name = 'zalizniak'
        self.words = []

    def load(self):
        with open(os.path.join('data', 'ortho', 'zalizniak.txt')) as file:
            already_added_hashes = set()
            for line in file:
                line = line.strip()
                sharp_index = line.index('#')
                text = line[:sharp_index]
                if Word.allow_word(text):
                    stressed_forms = line[(sharp_index + 1):].split(',')
                    normal_form = None
                    for stressed_form in stressed_forms:
                        stressed_form = stressed_form.replace('`', '')
                        word_text = stressed_form.replace(Zalizniak.STRESSED_CHAR, '')
                        if Word.allow_word(word_text):
                            word = Word.from_stressed(stressed_form, self.name, stressed_char=Zalizniak.STRESSED_CHAR,
                                                      stress_char_after=True)
                            key = word.__hash__()
                            if word and not key in already_added_hashes:
                                already_added_hashes.add(key)
                                if normal_form:
                                    word.normal_form = normal_form
                                    word.is_normal_form = False
                                else:
                                    normal_form = word
                                    word.is_normal_form = True
                                self.words.append(word)

class WordResult(object):
    def __init__(self, word: Word):
        self.word = word
        self.fuzzy = None

    def with_fuzzy(self, fuzzy: int) -> 'WordResult':
        self.fuzzy = fuzzy
        return self

    def __str__(self):
        return "<WordResult %s, fuzzy=%s>" % (self.word.stressed(), repr(self.fuzzy))

    def __repr__(self):
        return "<WordResult %s, fuzzy=%s>" % (self.word.stressed(), repr(self.fuzzy))

class OrthoDict:
    """Thread-safe implementation"""

    MAX_RESULTS_IN_QUERY = 100
    FUZZIES = [0, Phonetic.FUZZY_CONS_STUNING, Phonetic.FUZZY_CUTOFF_FINAL_CONS,
               Phonetic.FUZZY_CONS_STUNING + Phonetic.FUZZY_CUTOFF_FINAL_CONS]

    def __init__(self):
        self.dictionaries = [Lopatin(), Hagen(), Zalizniak()]
        self.frequency = None
        self.name = 'compound'

    def load(self):
        logging.info("Dictionaries loading started")
        already_added_words = dict()
        for dictionary in self.dictionaries:
            dictionary.load()
            logging.info("Dictionary %s loaded" % dictionary.__class__)
            for word in dictionary.words:
                key = word.__hash__()
                word0: Word = already_added_words.get(key)
                if word0:
                    # replace to more qualitative information
                    if word.normal_form and not word0.normal_form:
                        already_added_words[key] = word
                else:
                    already_added_words[key] = word
            logging.info("Dictionary %s words added" % dictionary.__class__)
        self.words = list(already_added_words.values())
        logging.info("Dictionaries loaded")
        logging.info("Phonetic hash generation started")
        for w in self.words:
            w.phonetic()
        logging.info("Phonetic hash generated")
        self._rhymes_by_phonetic_after_stress = dict()
        for fuzzy in OrthoDict.FUZZIES:
            self._rhymes_by_phonetic_after_stress[fuzzy] = self._prepare_phonetic_after_stress(fuzzy=fuzzy)
        self._text_to_words = dict()
        for w in self.words:
            words = self._text_to_words.get(w.text.lower())
            if words is None:
                words = []
                self._text_to_words[w.text.lower()] = words
            words.append(w)
        logging.info("Stress finding map prepared")

    def set_loaded(self):
        with self._globalLock:
            self._loaded = True
            logging.info("Dictionary ready")

    def loaded(self):
        with self._globalLock:
            return self._loaded

    def find_all_rhymes(self, text: Union[str, Word]) -> List[WordResult]:
        if isinstance(text, Word):
            word = text
        else:
            word = self.__create_word__(text)
        all_results = []
        added_words = dict()
        for fuzzy in OrthoDict.FUZZIES:
            postfix = word.phonetic_after_stress(fuzzy=fuzzy)
            results: List[Word] = self._rhymes_by_phonetic_after_stress[fuzzy].get(postfix) or []
            for result in results:
                if added_words.get(result) is None:
                    all_results.append(WordResult(result).with_fuzzy(fuzzy))
                    added_words[result] = fuzzy
            if len(all_results) >= 10:
                break
        return [w for w in all_results if w.word.phonetic() != word.phonetic()]

    def find_word(self, text: str) -> Optional[Word]:
        variants = self._text_to_words.get(text.lower())
        if variants:
            if len(variants) == 1:
                return variants[0]
            elif len(variants) == 2 and self._high_frequency_ratio(variants[0], variants[1]):
                if self.frequency.get_frequency(variants[0]) > self.frequency.get_frequency(variants[1]):
                    return variants[0]
                else:
                    return variants[1]
            else:
                logging.warning("Many (%d) variants for word %s: %s" %
                                (len(variants), text, ", ".join([repr(w) for w in variants])))
                return None

    def find_stressed_index(self, text: str) -> Optional[int]:
        word = self.find_word(text)
        if word:
            return word.stressed_index

    def _prepare_phonetic_after_stress(self, fuzzy: int):
        start = time.time()
        d = dict()
        for w in self.words:
            postfix = w.phonetic_after_stress(fuzzy=fuzzy)
            if not postfix in d:
                d[postfix] = []
            d[postfix].append(w)

        for key in d.keys():
            d[key].sort(key=lambda w: (-w.frequency, w.text.lower()))

        # remove doubles like "Ганимед", "ганимед"
        for key in list(d.keys()):
            words_for_key = []
            added_words: Dict[(str, int), Word] = dict()
            for word in d[key]:
                mirror_key = (word.text.lower(), word.stressed_index)
                mirror_word = added_words.get(mirror_key)
                if mirror_word:
                    if mirror_word.frequency < word.frequency:
                        added_words[mirror_key] = word
                        words_for_key.remove(mirror_word)
                        words_for_key.append(word)
                        print("%s -> %s" % (repr((mirror_word, mirror_word.frequency)), repr((word, word.frequency))))
                else:
                    added_words[(word.text.lower(), word.stressed_index)] = word
                    words_for_key.append(word)
            d[key] = words_for_key
        logging.info("Rhyme map with fuzzy=%d generated in %.1f seconds" % (fuzzy, time.time() - start))
        return d

    # check word pairs like  "Ганимед", "ганимед"
    def _mirror_words(self, word1, word2):
        if len(word1) != len(word2):
            return False
        elif word1[0] != word2[0] and word1[0].lower() == word2[0].lower() and word1[1:] == word2[1:]:
            return True
        else:
            return False

    def _high_frequency_ratio(self, w1: Word, w2: Word) -> bool:
        f1 = self.frequency.get_frequency(w1)
        f2 = self.frequency.get_frequency(w2)
        if f1 == 0 and f2 == 0:
            return False
        elif f1 == 0 or f2 == 0:
            return True
        ratio = f1 / f2
        if ratio < 1:
            ratio = 1 / ratio
        return ratio >= 2