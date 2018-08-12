from typing import NamedTuple, Iterator, List, Dict
from my.utils import get_cyrillic_lines, get_cyrillic_words, get_lines, MakeIter, is_cyrillic, lemma
from collections import namedtuple, defaultdict

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

    def get_sentence_lemms(self) -> Iterator[List[str]]:
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
