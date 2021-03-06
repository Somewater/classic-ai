from logging import Logger
from typing import Dict, List, Iterator, Callable, Tuple, Union, Set

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
import logging

from my.word import WordTag
from multiprocessing import Process, SimpleQueue, cpu_count, Pool
import gc


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

# seed -> List[word]
# word -> List[word2] (collocations)
# word2 in poet lyric -> poet lines
# lines -> poem
class Generator2:
    """
    Алгоритм генерации стихотворения на основе фонетических шаблонов
    """
    BeforeSpaceChars = set(['«', "'", '"', '№'])
    NoSpaceChars = set(['’', "'"])
    EndSentenceChars = set(['.', '!', '?', '…'])

    def __init__(self, rand_seed: int = None):
        self.random = random.Random(rand_seed)
        self.log = logging.getLogger('generator')
        self.reader = DataReader()
        self.freq = Frequency(self.reader)
        self.ortho = OrthoDict(self.freq)
        self.poems = ClassicPoems(self.reader, self.random)
        self.corpusw2v = CorpusW2v(WikiCorpus(self.reader, 'lemm'), self.reader)
        self.corpusw2v.model_filepath = '5-2-5-4e-05-1-1_1M/wiki_w2v.bin'
        self.tree = TrieNode()
        self.morph = None
        self.started = False
        self.tasks_queue = SimpleQueue()
        self.results_queue = SimpleQueue()
        self.cpu_count = 2 # max(cpu_count(), 4)
        self.debug = False
        self.w2v_distance_type = 'min_idf'

    def start(self):
        self.poems.load()
        self.log.info('Classic poems ready')
        self.freq.load()
        self.log.info('Frequency ready')
        self.phonetic = Phonetic0_2(self.ortho)
        self.log.info('Phonetic ready')
        if not self.ortho.loaded():
            self.ortho.load()
            self.log.info('Ortho dictionary ready')
        self.corpusw2v.load()
        self.log.info('Word2Vec ready')
        for w in self.ortho.words:
            vector, vector_index = self.corpusw2v.word_vector_index(w.lemma)
            w.vector = vector
            w.vector_index = vector_index
        self.log.info('Word2Vec vectors applied to words')
        # Словарь слов-кандидатов по фонетическим формам: строится из набора данных SDSJ 2017
        self.stop_words = self.reader.read_stop_words()
        self._generate_word_by_form()
        self.log.info('Word by form ready')
        added_word_text = set()
        # for w in self.ortho.words:
        #     self.tree.insert(w)
        #     added_word_text.add(w.text)
        #for word_text in self.freq.iterate_words():
        #    if not word_text in added_word_text:
        #        self.tree.insert(word_text)
        self.log.info('Tree-levenshtein ready')
        self.morph = pymorphy2.MorphAnalyzer()
        self.started = True

    def start_workers(self):
        self.workers = []
        for i in range(self.cpu_count):
            p = Process(target=self.run_worker, args=(i, self.tasks_queue, self.results_queue))
            self.workers.append(p)
            p.start()

    def stop_workers(self):
        for w in self.workers:
            w.terminate()
        self.workers.clear()

    def run_worker(self, worker_id: int, tasks_queue: SimpleQueue, results_queue: SimpleQueue):
        gc.disable()
        while True:
            template_line, line_idx, seed = tasks_queue.get()
            generated_line = self.generate_line(template_line, line_idx, seed)
            results_queue.put(((generated_line, line_idx), worker_id))

    def run_once(self,
                 template_lines: Iterator[Tuple[List[str], int]],
                 seed: Seed,
                 start_time: float,
                 results_queue: SimpleQueue):
        result = []
        for template_line, line_idx in template_lines:
            generated_line = self.generate_line(template_line, line_idx, seed)
            result.append((generated_line, line_idx))
        results_queue.put(result)

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

    def process_in_parallel_with_workers(self, template, seed, start_time):
        if not self.workers:
            self.log.warning('Workers started amoung generation')
            self.start_workers()

        task_counter = 0
        for li, line in enumerate(template):
            self.tasks_queue.put((line, li, seed))
            task_counter += 1

        while task_counter > 0:
            (generated_line, line_idx), worker_id = self.results_queue.get()
            template[line_idx] = generated_line
            task_counter -= 1

            seconds = time.time() - start_time
            if seconds > 4.0:
                self.log.warning('premature exit after %f seconds, %d tasks result remaining' % (seconds, task_counter))
                break
        return template

    def process_in_parallel(self, template, seed, start_time):
        workers = []
        line_per_worker = ceil(len(template) / self.cpu_count)
        lines_count = len(template)
        i = 0
        for worker_id in range(self.cpu_count):
            lines = []
            for _ in range(line_per_worker):
                if i >= lines_count:
                    break
                lines.append((template[i], i))
                i += 1
            if lines:
                p = Process(target=self.run_once, args=(lines, seed, start_time, self.results_queue))
                workers.append(p)
                p.start()

        for _ in workers:
            generated_lines = self.results_queue.get()
            for generated_line, line_idx in generated_lines:
                template[line_idx] = generated_line
        return template

    def process_sequence(self, template, seed: Seed, start_time):
        used_replacement_lemms = set()
        return [self.generate_line(template_line, line_idx, seed, used_replacement_lemms)
                for line_idx, template_line in enumerate(template)]

    def build_seed(self, text: str) -> Seed:
        seed_words = get_cyrillic_words(text)
        seed_lemms = [lemma(w) for w in seed_words]
        return Seed(text,
                    seed_words,
                    seed_lemms,
                    [self.corpusw2v.word_vector_index(l) for l in seed_lemms],
                    self.corpusw2v.mean_vector(text),
                    [self.freq.freq(w) for w in seed_words])

    def generate(self, poet_id: str, seed_text: str, process_type: str = 's') -> PoemResult:
        if not self.started:
            self.log.warning('Method start() invoked amoung generation')
            self.start()
        start_time = time.time()
        poet_id = Poet.recover(poet_id)
        request = PoemRequest(Poet.by_poet_id(poet_id), seed_text.lower())

        # выбираем шаблон на основе случайного стихотворения из корпуса
        poem_template: PoemTemplate = self.poems.get_random_template(request.poet)
        template = poem_template.get_template()
        diff8 = poem_template.lines_count - 8
        offset = 0
        if diff8 >= 2:
            offset = self.random.randint(0, int(diff8 / 2)) * 2
            template = template[offset: (offset + 8)]
        else:
            template = template[:8]
        template = copy.deepcopy(template)
        self.replaces = dict()

        # оцениваем word2vec-вектор темы
        seed = self.build_seed(request.seed)
        if process_type == 'w':
            template = self.process_in_parallel_with_workers(template, seed, start_time)
        elif process_type == 'p':
            template = self.process_in_parallel(template, seed, start_time)
        elif process_type == 's':
            template = self.process_sequence(template, seed, start_time)

        return PoemResult(request, poem_template, self._lines_from_template(template),
                          round(time.time() - start_time, 3),
                          offset,
                          self.replaces)

    def generate_line(self,
                      template_line: List[str],
                      line_idx: int,
                      seed: Seed,
                      used_replacement_lemms: Set[str] = None) -> List[str]:
        if used_replacement_lemms is None:
            used_replacement_lemms = set()
        last_word_idx = self._last_cyrillic_word_idx(template_line)
        for ti, token in enumerate(template_line):
            word = token.lower()
            last_word = ti == last_word_idx
            if len(word) > 2 and not (word in self.stop_words) and is_cyrillic_word(word):
                word_tag = self._word_tag(self.morph.tag(word)[0])
                if last_word:
                    replacements = self.ortho.rhymes(word)
                    replacements_params: List[Tuple[Word, float, float, WordResult]] = [
                        (
                            wr.word,
                            self.corpusw2v.distance(seed, wr.word.vector, self.w2v_distance_type),
                            self.phonetic.sound_distance(wr.word.text, word),
                            wr
                        )
                        for wr in replacements
                        if self._filter_candidates_by_params(wr.word, word_tag, word) and wr.word.lemma not in used_replacement_lemms
                    ]
                else:
                    replacements_with_levenshtein_dist = None # self.tree.search(word, 3)
                    if replacements_with_levenshtein_dist:
                        replacements_params: List[Tuple[Word, float, float, WordResult]] = [
                            (
                                wrd,
                                self.corpusw2v.distance(seed, wrd.vector, self.w2v_distance_type),
                                levenshtein_distance,# self.phonetic.sound_distance(wrd.text, word),
                                None
                            )
                            # for wrd in self._find_by_form(word, word_tag)
                            for wrd, levenshtein_distance in replacements_with_levenshtein_dist # TODO: not sorted!
                            if self._filter_candidates_by_params(wrd, word_tag, word) and wrd.lemma not in used_replacement_lemms
                        ]
                    else:
                        replacements_by_form =  self._find_by_form(word, word_tag)
                        replacements_params: List[Tuple[Word, float, float, WordResult]] = [
                            (
                                wrd,
                                self.corpusw2v.distance(seed, wrd.vector, self.w2v_distance_type),
                                self.phonetic.sound_distance(wrd.text, word),
                                None
                            )
                            for wrd in replacements_by_form
                            if self._filter_candidates_by_params(wrd, word_tag, word) and wrd.lemma not in used_replacement_lemms
                        ]

                if replacements_params:
                    if self.debug:
                        replacements_params = sorted(replacements_params, key=lambda x: self._sort_candidates_by_params(x, debug=False))
                        for t in reversed(replacements_params): self._sort_candidates_by_params(t, debug=True)
                        new_wrd = replacements_params[0][0] # min(replacements_params, key=self._sort_candidates_by_params)[0]
                    else:
                        new_wrd = min(replacements_params, key=self._sort_candidates_by_params)[0]
                    used_replacement_lemms.add(new_wrd.lemma)
                    new_word = new_wrd.text
                    if self.debug:
                        self.replaces[token] = replacements_params
                        import ipdb; ipdb.set_trace()
                else:
                    new_word = token
                    #print('Not found')
                    #import ipdb; ipdb.set_trace()
            else:
                new_word = token
            template_line[ti] = new_word
        return template_line

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
    def _sort_candidates_by_params(self, tuple: Tuple[Word, float, float, WordResult], debug=False):
        word, w2v_distance, sound_distance, word_result = tuple
        # normalize
        w2v_distance = w2v_distance
        sound_distance = sound_distance / 2.5
        freq = word.frequency
        freq = log(self.freq.max_freq()) / log(freq) if freq >= 2 else log(self.freq.max_freq()) / 0.5
        if word_result:
            sm = w2v_distance * 1.1  + word_result.fuzzy * 0.1 + freq * 0.01
            if debug: print('%s: %.5f (distnance=%.5f, fuzzy=%.5f, freq=%.5f)' % (word, sm, w2v_distance*1.1, word_result.fuzzy * 0.1, freq * 0.01))
            return sm
        else:
            sm = w2v_distance * 1.3  + sound_distance * 0.1 + freq * 0.01
            if debug: print('%s: %.5f (distnance=%.5f, sound=%.5f, freq=%.5f)' % (word, sm, w2v_distance*1.3, sound_distance * 0.1, freq * 0.01))
            return sm

    def _filter_candidates_by_params(self, word: Word, orig_word_tag: object, orig_word: str):
        if word.text == orig_word or word.lemma == orig_word:
            return False
        tag = word.tag
        if (orig_word_tag.POS and tag.POS != orig_word_tag.POS) or \
                (orig_word_tag.case and tag.case != orig_word_tag.case) or \
                (orig_word_tag.tense and tag.tense != orig_word_tag.tense) or \
                (orig_word_tag.number and tag.number != orig_word_tag.number) or \
                (orig_word_tag.gender and tag.gender != orig_word_tag.gender) or \
                (orig_word_tag.person and tag.person != orig_word_tag.person):
            return False
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
            no_space_char = False
            before_space_char = False
            big_letter_after = False
            for i, word in enumerate(line):
                if is_cyrillic_word(word):
                    if i == 0 or big_letter_after:
                        word = word[0].upper() + word[1:]
                    if before_space_char or no_space_char:
                        generated_line += word
                    else:
                        generated_line += ' ' + word
                    before_space_char = False
                    no_space_char = False
                    big_letter_after = False
                else:
                    before_space_char = word in Generator2.BeforeSpaceChars
                    if before_space_char:
                        generated_line += ' ' + word
                    else:
                        no_space_char = word in Generator2.NoSpaceChars
                        big_letter_after = word in Generator2.EndSentenceChars
                        generated_line += word
            result.append(generated_line.strip()[:120])
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