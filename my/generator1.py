from logging import Logger
from typing import Dict, List

from my import DataReader, Poem, Poet, PoemResult, PoemRequest
from collections import defaultdict

# seed -> List[word]
# word -> List[word2] (collocations)
# word2 in poet lyric -> poet lines
# lines -> poem
class Generator1:
    poems_by_poet: Dict[Poet, List[Poem]]

    def __init__(self, log: Logger):
        self.poems_by_poet = defaultdict(lambda: [])
        self.log = log

    def start(self):
        self.reader = DataReader()
        for poem in self.reader.read_classic_poems():
            self.poems_by_poet[poem.poet].append(poem)

    def generate(self, poet_id: str, seed: str) -> PoemResult:
        request = PoemRequest(Poet.by_poet_id(poet_id), seed)
        return PoemResult(request, [])