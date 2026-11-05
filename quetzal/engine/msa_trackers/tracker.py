import abc
import pandas as pd


# abstract class to create trackers
class Tracker(abc.ABC):
    @abc.abstractmethod
    def __init__(self, track_links_list: list):
        pass

    @abc.abstractmethod
    def init(self, links_sparse_index, links_to_sparse, volumes: pd.DataFrame):
        pass

    @abc.abstractmethod
    def __call__(self) -> bool:
        pass

    @abc.abstractmethod
    def assign(self, odv, pred, origins, seg, it):
        pass

    @abc.abstractmethod
    def add_weights(self, phi, beta, it):
        pass

    @abc.abstractmethod
    def merge(self, *args, **kwargs):
        pass
