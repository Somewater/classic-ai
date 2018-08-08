from typing import NamedTuple, Iterator, List
from my.utils import get_cyrillic_lines, get_cyrillic_words, get_lines, MakeIter, is_cyrillic
from collections import namedtuple

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

class Poem(ContentBase, namedtuple('WikiPage', ['poet', 'title', 'content'])):
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
    source: Poem
    lines: List[str]

    def content(self):
        return '\n'.join(self.lines)

    def __repr__(self):
        s = 'PoemResult(request=%s\n' % repr(self.request)
        for line in get_lines(self.source.content):
            s += '\t' + line + '\n'
        s += '>>>\n'
        for line in self.lines:
            s += '\t' + line + '\n'
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
