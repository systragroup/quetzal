import abc
from typing import List


# abstract class to create trackers
class Tracker(abc.ABC):
    @abc.abstractmethod
    def __init__(self, track_links_list: List):
        pass

    @abc.abstractmethod
    def init(self, links_sparse_index, links_to_sparse):
        pass

    @abc.abstractmethod
    def __call__(self) -> bool:
        pass

    @abc.abstractmethod
    def assign(self, ab_volumes, odv, pred, seg, it):
        pass

    @abc.abstractmethod
    def add_weights(self, phi, beta, relgap, it):
        pass

    @abc.abstractmethod
    def merge(self):
        pass
