from typing import Optional, Dict, Union, Iterator
from my import Phonetic, Word, lemma
from my.word import WordTag
import os
from functools import lru_cache
from typing import List, Set, Union, Optional

import os
from threading import Lock
import logging
import time
import pickle
import pymorphy2
import gc

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
                        if word:
                            key = (word.text, word.stressed_index)
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
                            print(line)
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
                            if word:
                                key = (word.text, word.stressed_index)
                                if word and not key in already_added_hashes:
                                    already_added_hashes.add(key)
                                    if normal_form:
                                        word.normal_form = normal_form
                                        word.is_normal_form = False
                                    else:
                                        normal_form = word
                                        word.is_normal_form = True
                                    self.words.append(word)
                            else:
                                print(line)

class WordResult(object):
    def __init__(self, word: Word, fuzzy = None, freq = None):
        self.word = word
        self.fuzzy = fuzzy
        self.freq = freq

    def with_fuzzy(self, fuzzy: int) -> 'WordResult':
        self.fuzzy = fuzzy
        return self

    def __str__(self):
        return "<WordResult %s, fuzzy=%s, freq=%d>" % (self.word.stressed(), repr(self.fuzzy), self.freq or -1)

    def __repr__(self):
        return "<WordResult %s, fuzzy=%s, freq=%d>" % (self.word.stressed(), repr(self.fuzzy), self.freq or -1)

class OrthoDict:
    """Thread-safe implementation"""

    MAX_RESULTS_IN_QUERY = 100
    FUZZIES = [0, Phonetic.FUZZY_CONS_STUNING, Phonetic.FUZZY_CUTOFF_FINAL_CONS,
               Phonetic.FUZZY_CONS_STUNING + Phonetic.FUZZY_CUTOFF_FINAL_CONS]

    def __init__(self, frequency):
        self.frequency = frequency
        self.name = 'compound'
        self.words = None

    def load_from_dictionaries(self):
        self.dictionaries = [Lopatin(), Hagen(), Zalizniak()]
        logging.info("Dictionaries loading started")
        already_added_words = dict()
        for dictionary in self.dictionaries:
            dictionary.load()
            logging.info("Dictionary %s loaded" % dictionary.__class__)
            for word in dictionary.words:
                if len(word.text) < 3:
                    continue
                key = (word.text, word.stressed_index)
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
        for w in self.words:
            w.lemma = lemma(w.text.lower())
        logging.info('Lemmas prepared')
        morph = pymorphy2.MorphAnalyzer()
        def str_or_none(s):
            if s is None:
                return None
            else:
                return str(s)
        for w in self.words:
            tag = morph.tag(w.text)[0]
            w.tag = WordTag(str_or_none(tag.POS), str_or_none(tag.case), str_or_none(tag.tense),
                            str_or_none(tag.number), str_or_none(tag.person), str_or_none(tag.gender))
        logging.info('Tags prepared')
        for w in self.words:
            freq = self.frequency.freq(w.text)
            if freq == 0:
                freq = self.frequency.freq(w.lemma) * 0.95
            w.frequency = freq
        logging.info('Frequency prepared')
        for w in self.words:
            w.normal_form = None
            w.is_normal_form = None
            w.dictionary_name = None

    def load(self):
        gc_enabled = gc.isenabled()
        if gc_enabled: gc.disable()
        with open('data/ortho.pickle', 'rb') as f:
            _rhymes_by_phonetic_after_stress, _text_to_words, words = pickle.load(f)
            self._rhymes_by_phonetic_after_stress = _rhymes_by_phonetic_after_stress
            self._text_to_words = _text_to_words
            self.words = words
        if gc_enabled: gc.enable()

    def loaded(self):
        return self.words is not None

    def save(self):
        if self._rhymes_by_phonetic_after_stress is None:
            raise RuntimeError("Dictionaries not loaded yet")
        with open('data/ortho.pickle', 'wb') as f:
            pickle.dump([self._rhymes_by_phonetic_after_stress, self._text_to_words, self.words], f)

    def rhymes(self, text: str, limit: int = 10000) -> List[WordResult]:
        word = self.find_word(text)
        if word:
            results = self.find_all_rhymes(word, limit)
            return sorted(results, key=lambda wr: (wr.fuzzy, -wr.freq))
        else:
            return []

    def find_all_rhymes(self, text: Union[str, Word], limit: int = None) -> List[WordResult]:
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
                if added_words.get(result) is None and result.phonetic() != word.phonetic():
                    all_results.append(WordResult(result, fuzzy, result.frequency))
                    added_words[result] = fuzzy
                    if limit and len(all_results) >= limit:
                        break
            if limit and len(all_results) >= limit:
                break
        return all_results

    def find_word(self, text: str) -> Optional[Word]:
        variants = self._text_to_words.get(text)
        if variants:
            if len(variants) == 1:
                return variants[0]
            elif len(variants) == 2 and self._high_frequency_ratio(variants[0], variants[1]):
                if self.frequency.freq(variants[0]) > self.frequency.freq(variants[1]):
                    return variants[0]
                else:
                    return variants[1]
            else:
                return variants[0] # TODO: just select first
        else:
            yo_index = text.find('ё')
            if yo_index != -1:
                return Word(text, stressed_index=yo_index, dictionary_name='none')
            else:
                first_vowel_index = 0
                vowel_count = 0
                for idx, c in enumerate(text):
                    if c in Phonetic.VOWELS:
                        first_vowel_index = idx
                        vowel_count += 1
                if vowel_count == 0:
                    if len(text) > 1:
                        return self._create_no_vowel_words(text)
                elif vowel_count == 1:
                    return Word(text, stressed_index=first_vowel_index, dictionary_name='none')

    VowelizeChar = {
        'б': ('бэ', 1), 'в': ('вэ', 1), 'г': ('гэ', 1), 'д': ('дэ', 1), 'ж': ('жэ', 1), 'з': ('зэ', 1), 'й': ('йи', 1),
        'к': ('кэ', 1), 'л': ('эл', 0), 'м': ('эм', 0), 'н': ('эн', 0), 'п': ('пэ', 1), 'р': ('эр', 0), 'с': ('эс', 0),
        'т': ('тэ', 1), 'ф': ('эф', 0), 'х': ('хэ', 1), 'ц': ('цэ', 1), 'ч': ('че', 1), 'ш': ('шэ', 1), 'щ': ('ще', 1)}

    def _create_no_vowel_words(self, text: str) -> Word:
        last_char = text[-1]
        replace, stress_index = OrthoDict.VowelizeChar[last_char]
        stress_index += len(text) - 1
        new_text = text[:-1] + replace
        return Word(new_text, stressed_index=stress_index, dictionary_name='none')

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

    def _high_frequency_ratio(self, w1: Word, w2: Word) -> bool:
        f1 = self.frequency.freq(w1)
        f2 = self.frequency.freq(w2)
        if f1 == 0 and f2 == 0:
            return False
        elif f1 == 0 or f2 == 0:
            return True
        ratio = f1 / f2
        if ratio < 1:
            ratio = 1 / ratio
        return ratio >= 2

    def __create_word__(self, text: str) -> Word:
        """Create word from text"""
        word = None
        if "'" in text:
            word = Word.from_stressed(text, self.name, stressed_char="'", stress_char_after=True)

        if not word:
            text_lower = text.lower()
            first_vowel_index = text_lower.find('ё')
            if first_vowel_index < 0:
                first_vowel_index = 0
                for idx, c in enumerate(text_lower):
                    if c in Phonetic.VOWELS:
                        first_vowel_index = idx
                        break
            word = Word(text_lower, stressed_index=first_vowel_index, dictionary_name='none')
        return word