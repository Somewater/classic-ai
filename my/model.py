from typing import NamedTuple, Iterator, List
from my.utils import get_cyrillic_lines, get_cyrillic_words, get_lines
from collections import namedtuple

class ContentBase(object):
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