import pandas as pd
from typing import List, Dict, Union
import numba as nb
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csc_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
from collections import namedtuple

TrackedVolume = namedtuple('TrackedVolume', 'iteration seg volumes')
TrackedWeight = namedtuple('TrackedWeight', 'iteration phi beta relgap')


class TupleTracker(Tracker):
    def __init__(self, track_links_list: List[tuple] = []):
        # list of road_link index to track
        # use self.merge() at the end.
        self.track_links_list = track_links_list
        self.weights = []
        self.tracked_mat: List[TrackedVolume] = []

    def init(
        self,
        links_sparse_index: Union[List[int], List[tuple[int, int]]],
        links_to_sparse: Union[Dict[str, int], Dict[str, tuple[int, int]]],
    ):
        self.links_sparse_index = links_sparse_index
        self.sparse_links_list = [[*map(links_to_sparse.get, ls)] for ls in self.track_links_list]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def assign(self, ab_volumes, odv, pred, seg, it):
        # volumes = [ab_volumes.copy() for _ in self.sparse_links_list]
        ab_keys = [k for k in ab_volumes.keys()]
        mat = get_paths_matrix(odv, pred, ab_keys)
        od_indexes_list = get_od_indexes(mat, self.sparse_links_list)
        volumes = []
        for od_indexes in od_indexes_list:
            volumes.append(odv[od_indexes, 2].sum())

        self.tracked_mat.append(TrackedVolume(it, seg, np.array(volumes)))

    def add_weights(self, phi, beta, relgap, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta, relgap=relgap))

    def merge(self) -> Dict[str, pd.DataFrame]:
        # apply frank wolfe for each iteration on each segments
        merged = _merge(self.tracked_mat, self.weights)
        return pd.DataFrame(merged, index=self.track_links_list)


def _merge(mat_datas, weights):
    # read info [iteration,segment,file]
    iterations = [el.iteration for el in weights]
    segments = segments = set([el.seg for el in mat_datas])

    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}

    beta_dict = {w.iteration: w.beta for w in weights}

    def apply_biconjugated_frank_wolfe():
        flow = pd.DataFrame()
        sk_1 = pd.DataFrame()
        sk_2 = pd.DataFrame()
        for it in iterations:
            phi = phi_dict.get(it)
            beta = beta_dict.get(it)
            mat_data = [mat for mat in mat_datas if mat.iteration == it and mat.seg == seg][0]

            aux_flow = mat_data.volumes
            if it > 2:
                aux_flow = beta[0] * aux_flow + beta[1] * sk_1 + beta[2] * sk_2

            sk_2 = sk_1.copy()
            sk_1 = aux_flow.copy()

            if it == 0:
                flow = aux_flow.copy()
            else:
                flow = (1 - phi) * flow + phi * aux_flow
        return flow

    res = {}
    for seg in segments:
        res[seg] = apply_biconjugated_frank_wolfe()
    return res


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def get_paths(odv, predecessors):
    # volumes is a numba dict with all the key initialized
    rows, cols, vals = [], [], []
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        path = get_node_path(predecessors, origin, destination)
        for j in range(len(path)):
            rows.append(i)
            cols.append(path[j])
            vals.append(True)

    return rows, cols, vals


def get_paths_matrix(odv, pred, ab_keys):
    n_cols = len(ab_keys)
    n_rows = len(odv)
    rows, cols, vals = get_paths(odv, pred)  # List for Each od. the index of links.
    return csc_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


def get_od_indexes(mat, track_index_list):
    # return, for each track_index_list, indexes of odv
    resp = []
    for pair in track_index_list:
        res = mat[:, pair[0]]
        for col in pair[1:]:
            res = res.multiply(mat[:, col])
        idx = res.nonzero()[0]
        resp.append(idx)
    return resp
