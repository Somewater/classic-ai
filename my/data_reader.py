from typing import List
from model import Poet, Poem
import os
import json

class DataReader:
    def read_classic_poems(self) -> List[Poem]:
        poems: List[Poem] = []
        with open(os.path.join('data', 'classic_poems.json')) as f:
            for entry in json.load(f):
                poem: Poem = Poem(entry['poet_id'], entry['title'], entry['content'])
                poems.append(poem)
        return poems