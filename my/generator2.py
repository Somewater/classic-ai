from logging import Logger
from typing import Dict, List, Iterator, Callable, Tuple, Union

from my import *
from my.utils import levenshtein_distance
from collections import defaultdict, namedtuple
import random
import json
import os
import copy

import bz2
import csv
import collections
import itertools

import numpy as np
from scipy.spatial.distance import cosine

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import pymorphy2
from math import *

from my.ortho_dict import WordResult
import time

from my.word import WordTag


class Phonetic0_2(object):
    """Объект для работы с фонетическими формами слова"""

    def __init__(self, ortho: OrthoDict):
        self.ortho = ortho

    def get_form(self, text: Union[str, Word]):
        profiler.enter('get_form')
        if isinstance(text, Word):
            word = text
        else:
            word = self.ortho.find_word(text)
        if word:
            profiler.exit()
            return (word.vowels_count(), word.stressed_vowel_index())
        else:
            word_syllables = sum((ch in Phonetic.VOWELS) for ch in text)
            word_accent = (word_syllables + 1) // 2
            profiler.exit()
            return (word_syllables, word_accent)

    def sound_distance(self, word1, word2):
        """Фонетическое растояние на основе расстояния Левенштейна по окончаниям
        (число несовпадающих символов на соответствующих позициях)"""

        profiler.enter('sound_distance')
        suffix_len = 3
        suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
        suffix2 = (' ' * suffix_len + word2)[-suffix_len:]

        distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
        profiler.exit()
        return distance

class PoemTemplate(namedtuple('PoemTemplate', ['poem']), ContentBase):
    poem: Poem

    def get_template(self) -> Iterator[List[str]]:
        return [get_cyrillic_words_and_punctuations(line.lower()) for line in get_cyrillic_lines(self.poem.content)]

class PoemTemplateLoader2(object):
    def __init__(self, poems: Iterator[Poem], min_lines=3):
        self.poet_templates = collections.defaultdict(list)
        self.min_lines = min_lines

        for poem in poems:
            template = PoemTemplate(poem)
            enough_lnes = False
            for i, _ in enumerate(template.get_template()):
                if i + 1 >= min_lines:
                    enough_lnes = True
            if enough_lnes:
                self.poet_templates[poem.poet].append(template)

    def get_random_template(self, poet: Poet) -> Tuple[PoemTemplate]:
        """Возвращает случайный шаблон выбранного поэта"""
        if not self.poet_templates[poet]:
            raise KeyError('Unknown poet "%s"' % poet)
        return random.choice(self.poet_templates[poet])

# seed -> List[word]
# word -> List[word2] (collocations)
# word2 in poet lyric -> poet lines
# lines -> poem
class Generator2:
    """
    Алгоритм генерации стихотворения на основе фонетических шаблонов
    """
    poems_by_poet: Dict[Poet, List[Poem]]

    def __init__(self, log: Logger, reader: DataReader, ortho: OrthoDict, freq: Frequency):
        self.log = log
        self.reader = reader
        self.ortho = ortho
        self.freq = freq
        self.morph = None
        self.started = False

    def start(self):
        self.freq.load()
        self.log.info('Frequency ready')
        # Шаблоны стихов: строим их на основе собраний сочинений от организаторов
        self.template_loader = PoemTemplateLoader2(self.reader.read_classic_poems())
        self.log.info('Templates ready')
        # Словарь ударений: берется из локального файла, который идет вместе с решением
        self.phonetic = Phonetic0_2(self.ortho)
        self.log.info('Phonetic ready')
        if not self.ortho.loaded():
            self.ortho.load()
            self.log.info('Ortho dictionary ready')
        self.corpusw2v = CorpusW2v(WikiCorpus(self.reader, 'lemm'), self.reader)
        self.corpusw2v.load()
        self.log.info('Word2Vec ready')
        for w in self.ortho.words:
            w.vector = self.corpusw2v.word_vector(w.lemma)
        self.log.info('Word2Vec vectors applied to words')
        # Словарь слов-кандидатов по фонетическим формам: строится из набора данных SDSJ 2017
        self.stop_words = self.reader.read_stop_words()
        self._generate_word_by_form()
        self.log.info('Word by form ready')
        self.morph = pymorphy2.MorphAnalyzer()
        self.started = True

    def _generate_word_by_form(self):
        self.word_by_form = defaultdict(set)
        self.word_by_form_by_pos = defaultdict(set)
        self.word_by_form_by_pos_by_case = defaultdict(set)
        for word in self.ortho.words:
            if len(word.text) > 2 and not word.text in self.stop_words and not word.lemma in self.stop_words:
                form = self.phonetic.get_form(word)
                self.word_by_form[form].add(word)
                pos = word.tag.POS
                if pos:
                    self.word_by_form_by_pos[(form, pos)].add(word)
                    case = word.tag.case
                    if case:
                        self.word_by_form_by_pos_by_case[(form, pos, case)].add(word)

    def generate(self, poet_id: str, seed: str) -> PoemResult:
        if not self.started:
            self.start()
        profiler.enter('generate')
        start_time = time.time()
        poet_id = Poet.recover(poet_id)
        request = PoemRequest(Poet.by_poet_id(poet_id), seed)

        # выбираем шаблон на основе случайного стихотворения из корпуса
        profiler.enter('header')
        poem_template = self.template_loader.get_random_template(request.poet)
        template = poem_template.get_template()
        diff8 = len(template) - 8
        offset = 0
        if diff8 >= 2:
            offset = random.randint(0, int(diff8 / 2)) * 2
            template = template[offset: (offset + 8)]

        # оцениваем word2vec-вектор темы
        seed_mean_vector = self.corpusw2v.mean_vector(request.seed)
        used_replacement_lemms = set()
        profiler.exit()

        # заменяем слова в шаблоне на более релевантные теме
        for li, line in enumerate(template):
            last_word_idx = self._last_cyrillic_word_idx(line)
            for ti, token in enumerate(line):
                word = token.lower()
                last_word = ti == last_word_idx
                if len(word) > 2 and not (word in self.stop_words) and is_cyrillic_word(word):
                    word_tag = self._word_tag(self.morph.tag(word)[0])
                    if last_word:
                        profiler.enter('rhymes')
                        replacements = self.ortho.rhymes(word)
                        profiler.exit()
                        profiler.enter('replacements_params.rhymes')
                        replacements_params: List[Tuple[Word, float, float, WordResult]] = [
                            (
                                wr.word,
                                self.corpusw2v.distance(seed_mean_vector, wr.word.vector),
                                self.phonetic.sound_distance(wr.word.text, word),
                                wr
                            )
                            for wr in replacements
                            if self._filter_candidates_by_params(wr.word, word_tag, word) and wr.word.lemma not in used_replacement_lemms
                        ]
                        profiler.exit()
                    else:
                        profiler.enter('replacements_params')
                        replacements_params: List[Tuple[Word, float, float, WordResult]] = [
                            (
                                wrd,
                                self.corpusw2v.distance(seed_mean_vector, wrd.vector),
                                self.phonetic.sound_distance(wrd.text, word),
                                None
                            )
                            for wrd in self._find_by_form(word, word_tag)
                            if self._filter_candidates_by_params(wrd, word_tag, word) and wrd.lemma not in used_replacement_lemms
                        ]
                        profiler.exit()

                    if replacements_params:
                        new_wrd = min(replacements_params, key=self._sort_candidates_by_params)[0]
                        used_replacement_lemms.add(new_wrd.lemma)
                        new_word = new_wrd.text
                        #replacements_params = sorted(replacements_params, key=self._sort_candidates_by_params) #  TODO: remove mE!!!
                        #import ipdb; ipdb.set_trace()
                    else:
                        new_word = token
                else:
                    new_word = token
                print('%s -> %s' % (word, new_word))
                template[li][ti] = new_word
        profiler.exit()

        return PoemResult(request, poem_template.poem, self._lines_from_template(template),
                          round(time.time() - start_time, 3),
                          offset)

    def _find_by_form(self, word: str, word_tag: WordTag):
        form = self.phonetic.get_form(word)
        if word_tag.POS:
            if word_tag.case:
                return self.word_by_form_by_pos_by_case[(form, word_tag.POS, word_tag.case)]
            else:
                return self.word_by_form_by_pos[(form, word_tag.POS)]
        else:
            return self.word_by_form[form]

    # less is BETTER
    def _sort_candidates_by_params(self, tuple: Tuple[Word, float, float, WordResult]):
        profiler.enter('sort_candidates_by_params')
        word, w2v_distance, sound_distance, word_result = tuple
        # normalize
        w2v_distance = w2v_distance
        sound_distance = sound_distance / 2.5
        freq = word.frequency
        freq = log(self.freq.max_freq()) / log(freq) if freq >= 2 else log(self.freq.max_freq()) / 0.5
        if word_result:
            profiler.exit()
            return w2v_distance  + word_result.fuzzy * 0.1 + freq * 0.05
        else:
            profiler.exit()
            return w2v_distance  + sound_distance * 0.1 + freq * 0.05

    def _filter_candidates_by_params(self, word: Word, orig_word_tag: object, orig_word: str):
        profiler.enter('filter_candidates_by_params')
        if word.text == orig_word or word.lemma == orig_word:
            profiler.exit()
            return False
        tag = word.tag
        if (orig_word_tag.POS and tag.POS != orig_word_tag.POS) or \
                (orig_word_tag.case and tag.case != orig_word_tag.case) or \
                (orig_word_tag.tense and tag.tense != orig_word_tag.tense) or \
                (orig_word_tag.number and tag.number != orig_word_tag.number) or \
                (orig_word_tag.gender and tag.gender != orig_word_tag.gender) or \
                (orig_word_tag.person and tag.person != orig_word_tag.person):
            profiler.exit()
            return False
        profiler.exit()
        return True

    def _last_cyrillic_word_idx(self, line: List[str]):
        i = 0
        result = None
        for w in line:
            if is_cyrillic_word(w):
                result = i
            i += 1
        return result

    def _lines_from_template(self, template: List[List[str]]) -> List[str]:
        result = []
        for line in template:
            generated_line = ''
            for i, word in enumerate(line):
                if is_cyrillic_word(word):
                    if i == 0:
                        word = word[0].upper() + word[1:]
                    generated_line += ' ' + word
                else:
                    generated_line += word
            result.append(generated_line.strip())
        return result

    # WordTag = namedtuple('WordTag', ['POS', 'case', 'tense', 'number', 'person', 'gender'])
    def _word_tag(self, tag) -> WordTag:
        return WordTag(str_or_none(tag.POS), str_or_none(tag.case), str_or_none(tag.tense), str_or_none(tag.number),
                       str_or_none(tag.person), str_or_none(tag.gender))

def str_or_none(s):
    if s is None:
        return None
    else:
        return str(s)