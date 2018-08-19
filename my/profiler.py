import time

class _ProfEntry:
    def __init__(self, name: str, parent: 'Profiler'):
        self.name = name
        self.parent = parent

    def __enter__(self):
        self.parent.enter(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.exit()

class Profiler:
    def __init__(self, disabled: bool = False):
        self.disabled = disabled
        self._data = dict()
        self._context = []
        self._start_time_by_context = dict()
        self.separator = '/'

    def __call__(self, *args, **kwargs):
        if args:
            return self.profile(args[0])

    def profile(self, name):
        return _ProfEntry(name, self)

    def enter(self, name: str):
        if not self.disabled:
            if self.separator in name:
                raise RuntimeError("Unsupported context name, don't use '%s': %s" % (self.separator, name))
            self.context.append(name)
            self._start_time[self.separator.join(self._context)] = time.time()

    def exit(self):
        if not self.disabled:
            prev_context_name = self.separator.join(self._context)
            self._context.pop()
            delta = time.time() - self._start_time_by_context[prev_context_name]
            self._data = (self._data.get(prev_context_name) or 0.0) + delta

    def clear(self):
        self._data.clear()
        self._start_time_by_context.clear()
        self._start_time_by_context.clear()

    def tree(self):
        root_tree = dict()
        for path, value in self._data.items():
            path = path.split(self.separator)
            tree = root_tree
            for p in path:
                if not p in tree:
                    tree[p] = dict()
                    tree = tree[p]
            tree['__value__'] = value
        return tree

    def print(self):
        def traverse(tree):
            pass
        traverse(self.tree())
