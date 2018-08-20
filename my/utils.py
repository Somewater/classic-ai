from typing import List, Iterator, Optional
import re
from nltk.stem.snowball import RussianStemmer
from pymystem3 import Mystem
from my.profiler import profiler

def is_cyrillic(char: str) -> bool:
    o = ord(char)
    return (o >= 1040 and o <= 1103) or o == 1105 or o == 1025

NBSP = chr(160)
HYPHEN1 = chr(173)
HYPHEN2 = chr(8208)
HYPHEN3 = chr(8211)
HYPHEN4 = chr(8212)
STRESS = chr(769)
SPACE1 = '\u2003'
SPACE2 = '\u2004'
AllowedPunctuation = set([' ', '.', ',', ';', '!', '?', ':', '…', '«', '»', '"', "'", '(', ')', '[', ']', '{', '}',
                          chr(47),
                          '<', '>', '„', '“', '”', NBSP, '_', chr(769), '-', HYPHEN1, HYPHEN2, HYPHEN3, HYPHEN4,
                          SPACE1, SPACE2, '*', '‹', '›', '|'])
WebCharPattern = re.compile('&(#\d{1,4}|\w{1,4});')
ManySpaces = re.compile('\s+')
rus_stemmer = RussianStemmer()
rus_lemmatizer = Mystem()
#rus_lemmatizer.start()

def get_lines(content: str) -> Iterator[str]:
    for line in content.splitlines():
        line = WebCharPattern.sub('', ManySpaces.sub(' ', line))
        if line:
            yield line

def get_cyrillic_lines(content: str) -> Iterator[str]:
    for line in content.splitlines():
        line = WebCharPattern.sub('', ManySpaces.sub(' ', line))
        if line:
            cyrillic = True
            for c in line:
                if not (is_cyrillic(c) or c in AllowedPunctuation):
                    cyrillic = False
                    break
            if cyrillic:
                yield line

def is_cyrillic_word(word: str) -> bool:
    for c in word:
        if not is_cyrillic(c) and c != '-':
            return False
    return True

def unify_chars(line: str) -> str:
    # return line.replace('_', '').replace(STRESS, '').replace(NBSP, ' ') \
    #     .replace(HYPHEN1, '-').replace(HYPHEN2, '-').replace(HYPHEN3, '-').replace(HYPHEN4, '-') \
    #     .replace(SPACE1, ' ').replace(SPACE2, ' ').replace('*', '') \
    #     .replace('[', '').replace(']', '') \
    #     .replace('(', '').replace(')', '') \
    #     .replace('{', '').replace('}', '') \
    #     .replace('<', '').replace('>', '') \
    #     .replace('„', '').replace('“', '').replace('”', '') \
    #     .replace('‹', '').replace('›', '') \
    #     .replace('|', ' ')
    chars = []
    for c in line:
        skip_char = c == '_' or c == STRESS or c == '*' or c == '[' or c == ']' or c == '(' or c == '{' or c == '}'\
                    or c == '<' or c == '>' or c == '„' or c == '“' or c == '”' or c == '‹' or c == '›'
        if not skip_char:
            replace_to_space = c == NBSP or c == SPACE1 or c == SPACE2 or c == '|'
            if replace_to_space:
                chars.append(' ')
            else:
                replace_to_hyphen = c == HYPHEN1 or c == HYPHEN2 or c == HYPHEN3 or c == HYPHEN4
                if replace_to_hyphen:
                    chars.append('-')
                else:
                    chars.append(c)
    return ''.join(chars)



def get_cyrillic_words(line: str) -> List[str]:
    words = []
    word_chars = []
    wrong_word = False
    def append_word(words: List[str], word_chars: List[str], wrong_word: bool):
        word = "".join(word_chars)
        word_chars.clear()
        if word[0] == '-' or word[-1] == '-':
            wrong_word = True
        if wrong_word:
            pass
        else:
            words.append(word)

    for char in unify_chars(line):
        if is_cyrillic(char) or char == '-':
            word_chars.append(char)
        else:
            if word_chars:
                append_word(words, word_chars, wrong_word)
            wrong_word = False

    if word_chars:
        append_word(words, word_chars, wrong_word)

    return words

def get_cyrillic_words_and_punctuations(line: str) -> List[str]:
    words = []
    word_chars = []
    wrong_word = False
    def append_word(words: List[str], word_chars: List[str], wrong_word: bool):
        word = "".join(word_chars)
        word_chars.clear()
        if word[0] == '-' or word[-1] == '-':
            wrong_word = True
        if wrong_word:
            pass
        else:
            words.append(word)

    for char in unify_chars(line):
        if is_cyrillic(char) or char == '-':
            word_chars.append(char)
        else:
            if word_chars:
                append_word(words, word_chars, wrong_word)
            if char != ' ' and char in AllowedPunctuation:
                words.append(char)

    if word_chars:
        append_word(words, word_chars, wrong_word)

    return words

def stem(word: str) -> str:
    return rus_stemmer.stem(word)

def lemma(word: str) -> str:
    try:
        for w in rus_lemmatizer.lemmatize(word):
            return w
    except BrokenPipeError:
        rus_lemmatizer.start()
        for w in rus_lemmatizer.lemmatize(word):
            return w
    return word

def lemm_or_stem(word: str) -> str:
    for w in rus_lemmatizer.lemmatize(word):
        return w
    return rus_stemmer.stem(word)

class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)

def group_by_n(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def levenshtein_distance(s1: str, s2: str):
    if s1 == s2:
        return 0
    rows = len(s1)+1
    cols = len(s2)+1

    if not s1:
        return cols-1
    if not s2:
        return rows-1

    prev = None
    cur = range(cols)
    for r in range(1, rows):
        prev, cur = cur, [r] + [0]*(cols-1)
        for c in range(1, cols):
            deletion = prev[c] + 1
            insertion = cur[c-1] + 1
            edit = prev[c-1] + (0 if s1[r-1] == s2[c-1] else 1)
            cur[c] = min(edit, deletion, insertion)

    return cur[-1]