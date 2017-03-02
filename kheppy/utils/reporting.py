import pickle
import warnings
from collections import OrderedDict


class Reporter:

    def __init__(self, entries_names=None):
        self.entries = None
        if entries_names is not None:
            self.entries = OrderedDict([(entry, [[]]) for entry in entries_names])

    def move_to_new_series(self):
        for entry in self.entries.values():
            entry.append([])

    def put(self, entry, value):
        if isinstance(entry, list) and isinstance(value, list) and len(entry) == len(value):
            for e, v in zip(entry, value):
                self._put(e, v)
        else:
            self._put(entry, value)

    def _put(self, entry, value):
        if entry in self.entries:
            self.entries[entry][-1].append(value)
        else:
            warnings.warn('Omitting entry {}.'.format(entry))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.entries, f)

    @staticmethod
    def load(path):
        rep = Reporter()
        with open(path, 'rb') as f:
            rep.entries = pickle.load(f)
        return rep
