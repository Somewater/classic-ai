from logging import Logger
from typing import Dict, List, Iterator, Callable

from my import DataReader, Poem, Poet, PoemResult, PoemRequest, CorpusW2v, WikiCorpus,\
    get_cyrillic_lines, get_cyrillic_words, lemma
from my.utils import levenshtein_distance
from collections import defaultdict
from random import choice
import json
import os

word_accents_dict = json.loads(open(os.path.join('data', "words_accent.json")).read())

def get_vowel_count(word):
    vowels = "уеыаоэёяию"
    vowel_count = 0

    for ch in word:
        if ch in vowels:
            vowel_count += 1

    return vowel_count


def get_accent(word):
    if word in word_accents_dict:
        return word_accents_dict[word]

    vowel_count = get_vowel_count(word)
    return (vowel_count + 1) // 2


def get_phoneme(word):
    word = word.lower()

    word_end = word[-3:]
    vowel_count = get_vowel_count(word)
    accent = get_accent(word)

    return word_end, vowel_count, accent

def text2template(text, stemmer: Callable[[str], str] = None):
    lines = []
    for line in get_cyrillic_lines(text):
        if stemmer is None:
            tokens = get_cyrillic_words(line)
        else:
            tokens = [(stemmer(w)) for w in get_cyrillic_words(line)]
        lines.append([get_phoneme(t) for t in tokens])
    return lines

def find_token(phoneme, corpus):
    word_end, vowel_count, accent = phoneme
    min_dist = 100000
    result_tokens = []

    key = "%s-%s" % (vowel_count, accent)
    if key not in corpus:
        key = choice([i for i in corpus])

    mini_corpus = corpus[key]

    for token, token_phoneme in mini_corpus.items():
        dist = levenshtein_distance(word_end, token_phoneme)

        if dist < min_dist:
            min_dist = dist
            result_tokens = [token]
        elif dist == min_dist:
            result_tokens += [token]

    return choice(result_tokens)


def generate_text_lines(template, corpus) -> List[str]:
    text = []
    for temp_line in template:
        row = []
        for t in temp_line:
            row.append(find_token(t, corpus))
        text.append(" ".join(row))

    return text

def build_corpus(words: Iterator[str]) -> Dict[str, List[str]]:
    corpus = dict()
    for token in words:
        word_end, vowel_count, accent = get_phoneme(token)

        key = "%s-%s" % (vowel_count, accent)
        w = corpus.get(key, dict())

        w[token] = word_end
        corpus[key] = w

    return corpus

# seed -> List[word]
# word -> List[word2] (collocations)
# word2 in poet lyric -> poet lines
# lines -> poem
class Generator1:
    poems_by_poet: Dict[Poet, List[Poem]]

    def __init__(self, log: Logger, reader: DataReader):
        self.poems_by_poet = defaultdict(lambda: [])
        self.log = log
        self.reader = reader

    def start(self):
        for poem in self.reader.read_classic_poems():
            self.poems_by_poet[poem.poet].append(poem)
        self.corpusw2v = CorpusW2v(WikiCorpus(self.reader, 'lemm'), self.reader)
        self.corpusw2v.load()

    def generate(self, poet_id: str, seed: str) -> PoemResult:
        request = PoemRequest(Poet.by_poet_id(poet_id), seed)
        poet = Poet.by_poet_id(poet_id)
        poems = self.poems_by_poet[poet]
        poem = choice(poems)
        words = self.corpusw2v.find_similar_words(request.get_cyrillic_words())
        lines = generate_text_lines(text2template(poem.content, lemma), build_corpus(words))
        return PoemResult(request, poem, lines)