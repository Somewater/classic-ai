from typing import NamedTuple, Iterator, List, Dict, Optional, Tuple
from my.utils import get_cyrillic_lines, get_cyrillic_words, get_lines, MakeIter, is_cyrillic, lemma
from collections import namedtuple, defaultdict
import numpy as np
from math import *

class Topic(object):
    def get_title(self) -> str:
        return self.title

    def get_cyrillic_words(self) -> Iterator[str]:
        return self.words

class Corpus(object):
    def __init__(self, name, topic_iterator_factory, *args, **kwargs):
        self.topic_iterator_factory = topic_iterator_factory
        self.topic_iterator_factory_args = args
        self.topic_iterator_factory_kwargs = kwargs
        self._topics_cache = None
        self._name = name

    def name(self):
        return self._name

    def get_topics(self) -> Iterator[Topic]:
        if self._topics_cache is None:
            return self.topic_iterator_factory(*self.topic_iterator_factory_args, **self.topic_iterator_factory_kwargs)
        else:
            return self._topics_cache

    def load_to_memory(self):
        self._topics_cache = self.get_topics()

    def __repr__(self):
        return "Corpus(%s)" % self.name()

    def __str__(self):
        return self.__repr__()

class ContentBase(Topic):
    title: str
    content: str

    def get_cyrillic_lines(self) -> Iterator[str]:
        for l in get_cyrillic_lines(self.content):
            yield l

    def get_all_lines(self) -> Iterator[str]:
        for l in get_lines(self.content):
            yield l

    def get_cyrillic_words(self) -> Iterator[str]:
        for l in get_lines(self.content):
            for w in get_cyrillic_words(l):
                yield w

    def get_sentence_lemms(self) -> Iterator[List[str]]:
        for line in self.get_cyrillic_lines():
            line = line.strip().lower()
            if line:
                words = get_cyrillic_words(line)
                yield [lemma(w) for w in words]

# content -> str
class ContentBaseImpl(namedtuple('ContentBaseImpl', ['title', 'content']), ContentBase):
    pass

# sentences -> List[List[str]]
class ContentBaseImpl2(namedtuple('ContentBaseImpl', ['title', 'sentences']), ContentBase):
    title: str
    sentences: List[List[str]]

    def get_sentence_lemms(self) -> List[List[str]]:
        return self.sentences

    def get_cyrillic_lines(self) -> Iterator[str]:
        for words in self.sentences:
            yield " ".join(words)

    def get_all_lines(self) -> Iterator[str]:
        for words in self.sentences:
            yield " ".join(words)

    def get_cyrillic_words(self) -> Iterator[str]:
        for words in self.sentences:
            for word in words:
                yield word

# words -> List[str]
class ContentBaseImpl3(namedtuple('ContentBaseImpl', ['title', 'words']), ContentBase):
    title: str
    words: List[str]

    def get_sentence_lemms(self) -> Iterator[List[str]]:
        raise RuntimeError('ha only words')

    def get_cyrillic_lines(self) -> Iterator[str]:
        raise RuntimeError('ha only words')

    def get_all_lines(self) -> Iterator[str]:
        raise RuntimeError('ha only words')

    def get_cyrillic_words(self) -> Iterator[str]:
        return self.words

class Poet(NamedTuple):
    id: str
    _by_poet_id = dict()

    @classmethod
    def by_poet_id(cls, poet_id: str) -> 'Poet':
        if not poet_id in cls._by_poet_id:
            cls._by_poet_id[poet_id] = Poet(poet_id)
        return cls._by_poet_id[poet_id]

    @classmethod
    def all(cls):
        return list(cls._by_poet_id.values())

    @classmethod
    def recover(cls, poet_id):
        char = poet_id.lower()[0]
        if char == 'p': return 'pushkin'
        elif char == 'e': return 'esenin'
        elif char == 'm': return 'mayakovskij'
        elif char == 'b': return 'blok'
        elif char == 't': return 'tyutchev'
        else: return poet_id

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.id.__eq__(other.id)
        else:
            return self.id.__eq__(other)

class Poem(ContentBase, namedtuple('Poem', ['poet', 'title', 'content'])):
    poet: Poet
    title: str
    content: str

class PoemRequest(NamedTuple):
    poet: Poet
    seed: str
    def get_title(self) -> str:
        return self.seed

    def get_cyrillic_words(self) -> Iterator[str]:
        return get_cyrillic_words(self.seed.lower())

class PoemResult(NamedTuple):
    request: PoemRequest
    source: 'PoemTemplate'
    lines: List[str]
    time_seconds: float
    source_offset: int
    replaces: dict = None

    def content(self):
        return '\n'.join(self.lines)

    def __repr__(self):
        s = 'PoemResult(request=%s, time=%f, offset=%d\n' % (repr(self.request), self.time_seconds, self.source_offset)
        for i, line in enumerate(self.source.get_template()):
            line = " ".join(line)
            delta = min(abs(i - self.source_offset), abs(self.source_offset + len(self.lines) - i))
            if delta <= 4:
                source_line = i >= self.source_offset and i < self.source_offset + len(self.lines)
                if source_line:
                    s += '>     ' + line + '\n'
                else:
                    s += '      ' + line + '\n'
        s += '>>>\n'
        for line in self.lines:
            s += '      ' + line + '\n'
        s += ')\n'
        return s

    def __str__(self):
        return self.__repr__()

# read from wikipedia xml dump
class WikiPage(ContentBase, namedtuple('WikiPage', ['id', 'parentid', 'title', 'content'])):
    id: int
    parentid: int
    title: str
    content: str

    def __str__(self):
        return "WikiPage(title=%s, text=%s)" % (self.title, self.content[:100] + '…')

    def __repr__(self):
        return self.__str__()

# read from prepared CSV file
class WikiPage2(namedtuple('WikiPage2', ['id', 'parentid', 'title', 'words']), Topic):
    id: int
    parentid: int
    title: str
    words: List[str]

    def __str__(self):
        return "WikiPage2(title=%s, words=%s)" % (self.title, ' '.join(self.words[:5]))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.words)

class OpCorpText(namedtuple('OpCorpText', ['id', 'title', 'paragraphs']), Topic):
    id: int
    title: str
    paragraphs: List['OpCorpParagraph']

    def get_cyrillic_words(self) -> Iterator[str]:
        for paragraph in self.paragraphs:
            for sentence in paragraph.sentences:
                for token in sentence.tokens:
                    cyrillic = True
                    for c in token.text:
                        if not (is_cyrillic(c) or c == '-'):
                            cyrillic = False
                            break
                    if cyrillic:
                        yield token.text.lower()

class OpCorpParagraph(NamedTuple):
    sentences: List['OpCorpSentence']

class OpCorpSentence(NamedTuple):
    id: int
    content: str
    tokens: List['OpCorpToken']

class OpCorpToken(NamedTuple):
    id: int
    text: str
    gs: List[str]
    pos: str # part of speech

class SCTM(namedtuple('SCTM', ['id', 'title', 'categories', 'words']), Topic):
    id: int
    title: str
    categories: List[str]
    words: List[str]

    @staticmethod
    def corpus(sctms: Iterator['SCTM']) -> Dict[str, List[Topic]]:
        categories = {'Кино',
                      'Космонавтика',
                      'Медицина',
                      'Музыка',
                      'Транспорт',
                      'Финансы',
                      'Футбол',
                      'Шахматы'}
        posts = defaultdict(list)
        for p in sctms:
            pc = [c for c in p.categories if c in categories]
            if len(pc) == 1:
                category = pc[0]
                posts[category].append(p)
        return posts

class PoemTemplate(namedtuple('PoemTemplate', ['poem', 'lines', 'lines_count']), ContentBase):
    poem: Poem
    lines: List[List[str]]
    lines_count: int

    def get_template(self) -> Iterator[List[str]]:
        return self.lines

    def __repr__(self):
        s = 'PoemTemplate(poet=%s, title=%s\n' % (repr(self.poem.poet), self.poem.title)
        for line in self.lines:
            s += '      ' + " ".join(line) + '\n'
        return s + ')'

    def __str__(self):
        return self.__repr__()

class Seed:
    text: str
    words: List[str] # cyrilic words
    lemms: List[str]
    vector_indexies: List[Tuple[Optional[np.array], Optional[int]]]
    mean_vector: np.array
    frequencies: List[float]

    idfs: List[float]
    weighted_vectors: List[Tuple[np.array, int, float, float]] # vector, vector_index, idf, freq

    def __init__(self, text, words, lemms, vector_indexies, mean_vector, frequencies):
        assert len(words) > 0
        assert len(words) == len(lemms)
        assert len(words) == len(vector_indexies)
        assert len(words) == len(frequencies)

        self.text = text
        self.words = words
        self.lemms = lemms
        self.vector_indexies = vector_indexies
        self.mean_vector = mean_vector
        self.frequencies = frequencies
        self.max_idf = log(1000000 / (0.0003))
        self.max_seed_words_count = 10
        if len([vi for vi in self.vector_indexies if vi[0] is not None]) > self.max_seed_words_count:
            self._truncate_vector()

        self.weighted_vectors = []
        self.idfs = []
        for vi, ipm in zip(self.vector_indexies, self.frequencies):
            if vi[0] is not None and ipm > 0:
                idf = log(1000000 / ipm)
                self.idfs.append(idf)
                self.weighted_vectors.append((vi[0], vi[1], idf, ipm))

    def _truncate_vector(self):
        selected_indexies = []
        selected_index_to_ipm = dict()
        for i, (vi, ipm) in enumerate(zip(self.vector_indexies, self.frequencies)):
            if vi[0] is not None and ipm > 0:
                selected_index_to_ipm[i] = ipm
                if len(selected_indexies) < self.max_seed_words_count:
                    selected_indexies.append(i)
                else:
                    index_to_replace = None
                    index_to_replace_ipm = None
                    for si in selected_indexies:
                        si_ipm = selected_index_to_ipm[si]
                        if ipm < si_ipm and (index_to_replace_ipm is None or index_to_replace_ipm < si_ipm):
                            index_to_replace = si
                            index_to_replace_ipm = si_ipm
                    if index_to_replace is not None:
                        selected_indexies.remove(index_to_replace)
                        selected_indexies.append(i)
        vector_indexies = []
        frequencies = []
        words = []
        lemms = []
        for i in selected_indexies:
            vector_indexies.append(self.vector_indexies[i])
            frequencies.append(self.frequencies[i])
            words.append(self.words[i])
            lemms.append(self.lemms[i])
        self.vector_indexies = vector_indexies
        self.frequencies = frequencies
        self.words = words
        self.lemms = lemms
