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

class Phonetic0_2(object):
    """Объект для работы с фонетическими формами слова"""

    def __init__(self, ortho: OrthoDict):
        self.ortho = ortho

    def get_form(self, text: Union[str, Word]):
        if isinstance(text, Word):
            word = text
        else:
            word = self.ortho.find_word(text)
        if word:
            return (word.vowels_count(), word.stressed_vowel_index())
        else:
            word_syllables = sum((ch in Phonetic.VOWELS) for ch in text)
            word_accent = (word_syllables + 1) // 2
            return (word_syllables, word_accent)

    def sound_distance(self, word1, word2):
        """Фонетическое растояние на основе расстояния Левенштейна по окончаниям
        (число несовпадающих символов на соответствующих позициях)"""
        suffix_len = 3
        suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
        suffix2 = (' ' * suffix_len + word2)[-suffix_len:]

        distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
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
        self.w2v_window = 2.0
        self.started = False

    def start(self):
        # Шаблоны стихов: строим их на основе собраний сочинений от организаторов
        self.template_loader = PoemTemplateLoader2(self.reader.read_classic_poems())
        self.log.info('Templates ready')
        # Словарь ударений: берется из локального файла, который идет вместе с решением
        self.phonetic = Phonetic0_2(self.ortho)
        self.log.info('Phonetic ready')
        if not self.ortho.loaded():
            self.ortho.load()
            self.log.info('Ortho dictionary ready')
        # Словарь слов-кандидатов по фонетическим формам: строится из набора данных SDSJ 2017
        self._generate_word_by_form()
        self.log.info('Word by form ready')
        self.corpusw2v = CorpusW2v(WikiCorpus(self.reader, 'lemm'), self.reader)
        self.corpusw2v.load()
        self.log.info('Word2Vec ready')
        self.stop_words = self.reader.read_stop_words()
        self.morph = pymorphy2.MorphAnalyzer()
        self.started = True

    def _generate_word_by_form(self):
        self.word_by_form = defaultdict(set)
        for word in self.ortho.words:
            form = self.phonetic.get_form(word)
            self.word_by_form[form].add(word.text.lower())

    def generate(self, poet_id: str, seed: str) -> PoemResult:
        if not self.started:
            self.start()
        poet_id = Poet.recover(poet_id)
        request = PoemRequest(Poet.by_poet_id(poet_id), seed)
        # poet = Poet.by_poet_id(poet_id)
        # poems = self.poems_by_poet[poet]
        # poem = choice(poems)
        # words = self.corpusw2v.find_similar_words(request.get_cyrillic_words(), lemma)
        # lines = generate_text_lines(text2template(poem.content, lemma), build_corpus(words))
        # return PoemResult(request, poem, lines)

        # выбираем шаблон на основе случайного стихотворения из корпуса
        poem_template = self.template_loader.get_random_template(request.poet)
        template = poem_template.get_template()

        # оцениваем word2vec-вектор темы
        seed_mean_vector = self.corpusw2v.mean_vector(request.seed)

        # заменяем слова в шаблоне на более релевантные теме
        for li, line in enumerate(template):
            if li >= 8:
                break
            line_len = len(line)
            for ti, token in enumerate(line):
                word = token.lower()
                word_tag = self.morph.tag(word)[0]
                last_word = ti == line_len - 1
                if len(word) > 2 and not (word in self.stop_words) and is_cyrillic_word(word):
                    if last_word:
                        # TODO: подбор на основе рифмы
                        new_word = word
                    else:
                        form = self.phonetic.get_form(token)
                        replacements_dist_sound = [
                            (w, lemm, distance, sound, freq, tag)
                            for w, lemm, distance, sound, freq, tag in [
                                (w,
                                 lemm,
                                 self.corpusw2v.distance(seed_mean_vector, self.corpusw2v.word_vector(lemm)),
                                 self.phonetic.sound_distance(w, word),
                                 self.freq.freq(lemm),
                                 self.morph.tag(w)[0])
                                for w, lemm in [(w, lemma(w)) for w in self.word_by_form[form]]
                            ]
                            if self._filter_candidates_by_params(w, lemm, distance, sound, freq, tag, word_tag, word)
                        ]
                        if replacements_dist_sound:
                            new_word = min(replacements_dist_sound,
                                           key=self._sort_candidates_by_params)[0]
                            replacements_dist_sound = sorted(replacements_dist_sound, key=self._sort_candidates_by_params) #  TODO: remove mE!!!
                            import ipdb; ipdb.set_trace()
                        else:
                            new_word = word
                else:
                    new_word = word
                template[li][ti] = new_word

        # собираем получившееся стихотворение из слов
        generated_poem_lines = [' '.join([token for token in line]) for line in template]
        return PoemResult(request, poem_template.poem, generated_poem_lines)

    # less is BETTER
    def _sort_candidates_by_params(self, tuple: Tuple[str, str, float, float, float, object, object, str]):
        word, lemm, w2v_distance, sound_distance, freq, tag = tuple
        # normalize
        w2v_distance = w2v_distance / self.w2v_window
        sound_distance = sound_distance / 2.5
        freq = self.freq.max_freq() / freq if freq > 0 else 1000
        return w2v_distance  + sound_distance + freq

    def _filter_candidates_by_params(self, word: str, lemm: str, w2v_distance: float, sound_distance: float, freq: int,
                                     tag: object, orig_word_tag: object, orig_word: str):
        if len(word) < 3:
            return False
        if word in self.stop_words or lemm in self.stop_words:
            return False
        if word == orig_word or lemm == orig_word:
            return False
        if (orig_word_tag.POS and tag.POS != orig_word_tag.POS) or \
                (orig_word_tag.case and tag.case != orig_word_tag.case):
            return False
        return True