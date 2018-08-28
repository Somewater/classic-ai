from my import DataReader
from my.model import *
import random
from collections import defaultdict, Counter
from my.utils import *

class ClassicPoems:
    def __init__(self, reader: DataReader, random: random.Random):
        self.reader = reader
        self.min_lines = 3
        self.random = random
        self._poet_lines = None
        self._poet_templates = None

    def load_from_files(self):
        self._poet_templates = defaultdict(list)
        self._poet_lines = Counter()
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
                    lines_count = len(template_lines)
                    template = PoemTemplate(poem, template_lines, lines_count)
                    self._poet_templates[poem.poet.id].append(template)
                    self._poet_lines[poem.poet.id] += lines_count

    def save(self):
        pass # TODO self._poet_templates, self._poet_lines

    def load(self):
        self.load_from_files() # TODO

    def get_random_template(self, poet: Poet) -> PoemTemplate:
        if not self._poet_templates[poet.id]:
            raise KeyError('Unknown poet "%s"' % poet)
        lines_offset_all = self.random.randint(0, self._poet_lines[poet.id] - 1 - 8)
        lines_offset = 0
        for template in self._poet_templates[poet.id]:
            lines_offset += template.lines_count
            if lines_offset >= lines_offset_all:
                return template
        print('Error, template not choosed from %d lines, offset=%d' % (lines_offset_all, lines_offset))
        return self.random.choice(self._poet_templates[poet])

    def poems(self) -> Iterator[PoemTemplate]:
        for poet, poems in self._poet_templates.items():
            for poem in poems:
                yield poem
