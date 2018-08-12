from logging import Logger
from typing import Dict, List, Iterator, Callable, Tuple

from my import DataReader, Poem, Poet, PoemResult, PoemRequest, CorpusW2v, WikiCorpus,\
    get_cyrillic_lines, get_cyrillic_words, lemma
from my.utils import levenshtein_distance
from collections import defaultdict
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

class Phonetic(object):
    """Объект для работы с фонетическими формами слова"""

    def __init__(self, accent_file, vowels='уеыаоэёяию'):
        self.vowels = vowels
        with bz2.BZ2File(accent_file) as fin:
            self.accents_dict = json.load(fin)

    def syllables_count(self, word):
        """Количество гласных букв (слогов) в слове"""
        return sum((ch in self.vowels) for ch in word)

    def accent_syllable(self, word):
        """Номер ударного слога в слове"""
        default_accent = (self.syllables_count(word) + 1) // 2
        return self.accents_dict.get(word, default_accent)

    def get_form(self, word):
        word_syllables = self.syllables_count(word)
        word_accent = self.accent_syllable(word)
        return (word_syllables, word_accent)

    def sound_distance(self, word1, word2):
        """Фонетическое растояние на основе расстояния Левенштейна по окончаниям
        (число несовпадающих символов на соответствующих позициях)"""
        suffix_len = 3
        suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
        suffix2 = (' ' * suffix_len + word2)[-suffix_len:]

        distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
        return distance

class PoemTemplateLoader(object):
    """
    Хранит шаблоны стихотворений, полученные из собрания сочинений.
    Шаблон — обработанное и обрезанное стихотворение в виде набора отдельных токенов (слов).
    """

    def __init__(self, poems: Iterator[Poem], min_lines=3, max_lines=8):
        self.poet_templates = collections.defaultdict(list)
        self.min_lines = min_lines
        self.max_lines = max_lines

        for poem in poems:
            template = self.poem_to_template(poem.content)
            if len(template) >= self.min_lines:
                self.poet_templates[poem.poet].append((template, poem))

    def poem_to_template(self, poem_text):
        poem_lines = poem_text.split('\n')[:self.max_lines]
        poem_template = []
        for line in poem_lines:
            line_tokens = [token for token in word_tokenize(line) if token.isalpha()]
            poem_template.append(line_tokens)
        return poem_template

    def get_random_template(self, poet: Poet) -> Tuple[str, Poem]:
        """Возвращает случайный шаблон выбранного поэта"""
        if not self.poet_templates[poet]:
            raise KeyError('Unknown poet "%s"' % poet)
        return random.choice(self.poet_templates[poet])


class Word2vecProcessor(object):
    """Объект для работы с моделью word2vec сходства слов"""

    def __init__(self, w2v_model_file):
        self.mystem = Mystem()
        self.word2vec = KeyedVectors.load_word2vec_format(w2v_model_file, binary=True)
        self.lemma2word = {word.split('_')[0]: word for word in self.word2vec.index2word}

    def word_vector(self, word):
        lemma = self.mystem.lemmatize(word)[0]
        word = self.lemma2word.get(lemma)
        return self.word2vec[word] if word in self.word2vec else None

    def text_vector(self, text):
        """Вектор текста, получается путем усреднения векторов всех слов в тексте"""
        word_vectors = [
            self.word_vector(token)
            for token in word_tokenize(text.lower())
            if token.isalpha()
        ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return np.mean(word_vectors, axis=0)

    def distance(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 2
        return cosine(vec1, vec2)

# seed -> List[word]
# word -> List[word2] (collocations)
# word2 in poet lyric -> poet lines
# lines -> poem
class Generator1:
    """
    Алгоритм генерации стихотворения на основе фонетических шаблонов
    """
    poems_by_poet: Dict[Poet, List[Poem]]

    def __init__(self, log: Logger, reader: DataReader):
        self.log = log
        self.reader = reader

    def start(self):
        # Шаблоны стихов: строим их на основе собраний сочинений от организаторов
        self.template_loader = PoemTemplateLoader(self.reader.read_classic_poems())
        # Словарь ударений: берется из локального файла, который идет вместе с решением
        self.phonetic = Phonetic('data/words_accent.json.bz2')
        # Словарь слов-кандидатов по фонетическим формам: строится из набора данных SDSJ 2017
        self.word_by_form = self.reader.form_dictionary_from_csv(self.phonetic)
        self.corpusw2v = CorpusW2v(WikiCorpus(self.reader, 'lemm'), self.reader)
        self.corpusw2v.load()

    def generate(self, poet_id: str, seed: str) -> PoemResult:
        poet_id = Poet.recover(poet_id)
        request = PoemRequest(Poet.by_poet_id(poet_id), seed)
        # poet = Poet.by_poet_id(poet_id)
        # poems = self.poems_by_poet[poet]
        # poem = choice(poems)
        # words = self.corpusw2v.find_similar_words(request.get_cyrillic_words(), lemma)
        # lines = generate_text_lines(text2template(poem.content, lemma), build_corpus(words))
        # return PoemResult(request, poem, lines)

        # выбираем шаблон на основе случайного стихотворения из корпуса
        template, poem = self.template_loader.get_random_template(request.poet)
        template = copy.deepcopy(template)

        # оцениваем word2vec-вектор темы
        seed_vec = self.corpusw2v.text_vector(request.seed)

        # заменяем слова в шаблоне на более релевантные теме
        for li, line in enumerate(template):
            for ti, token in enumerate(line):
                if not token.isalpha():
                    continue

                word = token.lower()

                # выбираем слова - кандидаты на замену: максимально похожие фонетически на исходное слово
                form = self.phonetic.get_form(token)
                candidate_phonetic_distances = [
                    (replacement_word, self.phonetic.sound_distance(replacement_word, word))
                    for replacement_word in self.word_by_form[form]
                ]
                if not candidate_phonetic_distances or form == (0, 0):
                    continue
                min_phonetic_distance = min(d for w, d in candidate_phonetic_distances)
                replacement_candidates = [w for w, d in candidate_phonetic_distances if d == min_phonetic_distance]

                # из кандидатов берем максимально близкое теме слово
                word2vec_distances = [
                    (replacement_word, self.corpusw2v.distance(seed_vec, self.corpusw2v.word_vector(replacement_word)))
                    for replacement_word in replacement_candidates
                ]
                word2vec_distances.sort(key=lambda pair: pair[1])
                new_word, _ = word2vec_distances[0]

                template[li][ti] = new_word

        # собираем получившееся стихотворение из слов
        generated_poem_lines = [' '.join([token for token in line]) for line in template]
        return PoemResult(request, poem, generated_poem_lines)