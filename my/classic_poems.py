from my import DataReader
from my.model import *
import random
from collections import defaultdict
from my.utils import *

class ClassicPoems:
    def __init__(self, reader: DataReader, random: random.Random):
        self.reader = reader
        self.min_lines = 3
        self.random = random

    def load_from_files(self):
        self._poet_templates = defaultdict(list)
        self.random = random

        for poem in self.reader.read_classic_poems():
            template_lines = []
            words_count = 0
            for line in get_cyrillic_lines(poem.content):
                if len(line) < 80 and len(line) > 2:
                    words = get_cyrillic_words_and_punctuations(line.lower().strip())
                    words_count += len(words)
                    template_lines.append(words)
            if len(template_lines) >= self.min_lines:
                if len(template_lines) >= 4 and words_count > 30:
                    template = PoemTemplate(poem, template_lines, len(template_lines))
                    self._poet_templates[poem.poet].append(template)

    def save(self):
        pass # TODO

    def load(self):
        self.load_from_files() # TODO

    def get_random_template(self, poet: Poet) -> PoemTemplate:
        if not self._poet_templates[poet]:
            raise KeyError('Unknown poet "%s"' % poet)
        return self.random.choice(self._poet_templates[poet])

    def poems(self) -> Iterator[PoemTemplate]:
        for poet, poems in self._poet_templates.items():
            for poem in poems:
                yield poem
