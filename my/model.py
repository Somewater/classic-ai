class Poet:
    id: str

class Poem:
    poet_id: str
    title: str
    content: str

    def __init__(self, poet_id: str, title: str, content: str):
        self._poet = None

    def poet(self):
        if self._poet is None:
            self._poet = self._fetch_poet()
        return self._poet

    def _fetch_poet(self) -> Poet:
        return Poet(self.poet_id)

class Model:
    def __init__(self):
        pass

    def start(self):
        pass

    def generate(self, poet_id: str, seed: str):
        pass