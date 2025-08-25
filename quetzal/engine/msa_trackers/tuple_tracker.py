import pandas as pd
from typing import List, Dict, Union
import numba as nb
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csc_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
from collections import namedtuple

# need to name the class the same as the namedTuple name for pickle.
TrackedArrays = namedtuple('TrackedArrays', 'iteration seg volumes')
TrackedWeight = namedtuple('TrackedWeight', 'iteration phi beta relgap')


class TupleTracker(Tracker):
    def __init__(self, track_links_list: List[tuple] = []):
        # list of tuple [(rlink1, rlink2), (rlink1, rlink100),...]
        self.track_links_list = track_links_list
        self.weights = []
        self.tracked_mat: List[TrackedArrays] = []

    def init(
        self,
        links_sparse_index: Union[List[int], List[tuple[int, int]]],
        links_to_sparse: Union[Dict[str, int], Dict[str, tuple[int, int]]],
    ):
        self.links_sparse_index = links_sparse_index
        self.sparse_links_list = [[*map(links_to_sparse.get, ls)] for ls in self.track_links_list]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}
        self.tracked_links_set = set(np.concatenate(self.sparse_links_list))

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def assign(self, ab_volumes, odv, pred, seg, it):
        n_cols = len(ab_volumes)
        mat = get_paths_matrix(odv, pred, n_cols, self.tracked_links_set)
        od_indexes_list = get_od_indexes(mat, self.sparse_links_list)
        volumes = []
        for od_indexes in od_indexes_list:
            volumes.append(odv[od_indexes, 2].sum())

        self.tracked_mat.append(TrackedArrays(it, seg, np.array(volumes)))

    def add_weights(self, phi, beta, relgap, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta, relgap=relgap))

    def merge(self) -> pd.DataFrame:
        # apply frank wolfe for each iteration on each segments
        merged = _merge(self.tracked_mat, self.weights)
        return pd.DataFrame(merged, index=self.track_links_list)


def _merge(mat_datas, weights):
    # read info [iteration,segment,file]
    segments = segments = set([el.seg for el in mat_datas])
    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}
    beta_dict = {w.iteration: w.beta for w in weights}
    res = {}
    for seg in segments:
        filtered = [mat for mat in mat_datas if mat.seg == seg]
        res[seg] = apply_biconjugated_frank_wolfe(filtered, phi_dict, beta_dict)
    return res


def apply_biconjugated_frank_wolfe(mat_datas, phi_dict, beta_dict):
    flow = pd.DataFrame()
    sk_1 = pd.DataFrame()
    sk_2 = pd.DataFrame()
    for mat_data in mat_datas:
        it = mat_data.iteration
        phi = phi_dict.get(it)
        beta = beta_dict.get(it)
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


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})
def get_paths_matrix_data(odv, predecessors, path_set):
    # only append data if its in the set.
    # so we only keep the cols of links we will use (the one we track)
    rows, cols, vals = [], [], []
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        path = get_node_path(predecessors, origin, destination)
        for p in path:
            if p in path_set:
                rows.append(i)
                cols.append(p)
                vals.append(True)

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.bool)


def get_paths_matrix(odv, pred, n_cols, cols_set):
    # cols: links. rows: OD
    # having dtypes in index make this way faster.
    n_rows = len(odv)
    rows, cols, vals = get_paths_matrix_data(odv, pred, cols_set)  # List for Each od. the index of links.
    return csc_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


def get_od_indexes(mat, track_index_list) -> List[List[int]]:
    # this assume that every track_index_list is a tuple of 2 values.
    # we could extend to more (everyone must have the same), and multiply more matrix

    first_link_list = [el[0] for el in track_index_list]
    second_link_list = [el[1] for el in track_index_list]

    first_mat = mat[:, first_link_list]
    second_mat = mat[:, second_link_list]
    res = first_mat.multiply(second_mat)

    cols, rows = res.T.nonzero()

    od_list = [[] for _ in range(len(track_index_list))]
    for col, row in zip(cols, rows):
        od_list[col].append(row)
    return od_list
