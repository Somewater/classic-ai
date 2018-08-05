from typing import NamedTuple, Iterator, List
from my.utils import get_cyrillic_lines, get_cyrillic_words, get_lines, MakeIter, is_cyrillic
from collections import namedtuple

class Topic(object):
    def get_title(self) -> str:
        return self.title

    def get_cyrillic_words(self) -> Iterator[str]:
        return self.words

class Corpus(object):
    def name(self):
        pass

    def get_topics(self) -> Iterator[Topic]:
        pass

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

class Poem(ContentBase, namedtuple('WikiPage', ['poet', 'title', 'content'])):
    poet: Poet
    title: str
    content: str

class PoemRequest(NamedTuple):
    poet: Poet
    seed: str

class PoemResult(NamedTuple):
    request: PoemRequest
    lines: List[str]

    def content(self):
        return '\n'.join(self.lines)

class WikiPage(ContentBase, namedtuple('WikiPage', ['id', 'parentid', 'title', 'content'])):
    id: int
    parentid: int
    title: str
    content: str

    def __str__(self):
        return "WikiPage(title=%s, text=%s)" % (self.title, self.content[:100] + '…')

    def __repr__(self):
        return self.__str__()

class WikiPage2(NamedTuple, Topic):
    id: int
    parentid: int
    title: str
    words: List[str]

    def __str__(self):
        return "WikiPage(title=%s, words=%s)" % (self.title, ' '.join(self.words[:5]))

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

class OpCorpus(Corpus):
    def __init__(self, reader):
        self.reader = reader
        self._topics = None

    def name(self):
        return 'opcorpora'

    def get_topics(self) -> Iterator[Topic]:
        if self._topics is None:
            return self.reader.read_opcorpora()
        else:
            return self._topics

    def load_to_memory(self):
        self._topics = list(self.reader.read_opcorpora())
