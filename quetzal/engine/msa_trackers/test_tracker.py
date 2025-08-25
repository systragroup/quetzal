from typing import List, Dict, Union
from quetzal.engine.msa_trackers.tracker import Tracker
from collections import namedtuple

# need to name the class the same as the namedTuple name for pickle.
TrackedAssign = namedtuple('TrackedAssign', 'ab_volumes odv pred seg it')
TrackedWeight = namedtuple('TrackedWeight', 'iteration phi beta relgap')


class TestTracker(Tracker):
    def __init__(self, track_links_list: List[str] = []):
        self.track_links_list = track_links_list
        self.weights = []
        self.tracked_mat: List[TrackedAssign] = []

    def init(
        self,
        links_sparse_index: Union[List[int], List[tuple[int, int]]],
        links_to_sparse: Union[Dict[str, int], Dict[str, tuple[int, int]]],
    ):
        self.links_sparse_index = links_sparse_index
        self.links_to_sparse = links_to_sparse
        self.sparse_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return True

    def assign(self, ab_volumes, odv, pred, seg, it):
        # just save everything.
        self.tracked_mat.append(TrackedAssign(ab_volumes, odv, pred, seg, it))

    def add_weights(self, phi, beta, relgap, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta, relgap=relgap))

    def merge(self):
        pass
