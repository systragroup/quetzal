import pandas as pd
from typing import List, Dict, Union
import numba as nb
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csr_matrix
from collections import namedtuple

TrackedVolume = namedtuple('TrackedVolume', 'iteration seg mat row_labels col_labels')
TrackedWeight = namedtuple('TrackedWeight', 'iteration phi beta relgap')


class LinksTracker:
    def __init__(self, track_links_list: List[str] = []):
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
        self.sparse_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def assign_volume_on_links(self, ab_volumes, odv, pred, seg, it):
        volumes = [ab_volumes.copy() for _ in self.sparse_links_list]
        volumes = assign_tracked_volumes(odv, pred, volumes, self.sparse_links_list)
        ab_keys = [k for k in ab_volumes.keys()]
        self.add_volumes(volumes, seg, ab_keys, it)

    def add_volumes(self, volumes, seg, ab_keys, it):
        sparse_mat = dict_list_to_sparse_matrix(volumes, ab_keys)
        rows_labels = [*map(self.sparse_to_links.get, ab_keys)]
        cols_labels = self.track_links_list

        self.tracked_mat.append(TrackedVolume(it, seg, sparse_mat, rows_labels, cols_labels))

    def add_weights(self, phi, beta, relgap, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta, relgap=relgap))

    def merge(self) -> Dict[str, pd.DataFrame]:
        # apply frank wolfe for each iteration on each segments
        return _merge(self.tracked_mat, self.weights)


def _mat_to_df(mat_data):
    return pd.DataFrame(mat_data.mat.toarray(), index=mat_data.row_labels, columns=mat_data.col_labels)


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

            aux_flow = _mat_to_df(mat_data)
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


@nb.njit()
def flatten_dict_list(dict_list):
    rows, cols, vals = [], [], []
    i = 0
    for d in dict_list:
        for k, v in d.items():
            if v > 0:
                cols.append(i)
                rows.append(k)
                vals.append(v)
        i += 1
    return rows, cols, vals


def dict_list_to_sparse_matrix(dict_list: List[Dict], ab_keys: List[int | tuple]) -> csr_matrix:
    n_cols = len(dict_list)
    n_rows = len(ab_keys)
    rows, cols, vals = flatten_dict_list(dict_list)
    # need to reindex rows as keys can be tuple (or anything)
    # cols are already sparse as they are index of list.
    row_dict = dict(zip(ab_keys, range(n_rows)))
    rows = [*map(row_dict.get, rows)]

    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volumes(odv, predecessors, volumes, track_index_list):
    # volumes is a numba dict with all the key initialized
    nlen = len(track_index_list)
    is_tuple = isinstance(track_index_list[0], tuple)
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            if is_tuple:  # not expanded: we have tuple (a,b). else its a sparse index (link directly)
                path = list(zip(path[:-1], path[1:]))
            for j in range(nlen):
                track_index = track_index_list[j]
                if track_index in path:
                    for key in path:
                        volumes[j][key] += v
    return volumes
