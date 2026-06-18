import pandas as pd
import numpy as np
import numba as nb
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csr_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
from typing import NamedTuple


class TrackedVolume(NamedTuple):
    iteration: int
    seg: str
    volumes: csr_matrix


class TrackedWeight(NamedTuple):
    iteration: int
    phi: float
    beta: list[float]


class TurnTracker(Tracker):
    def __init__(self, track_links_list: list[str] = []):
        print('This tracker only work with turn_penalties')
        self.track_links_list = track_links_list  # row_labels
        self.weights: list[TrackedWeight] = []
        self.tracked_mat: list[TrackedVolume] = []

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def init(self, links_sparse_index: list[int], links_to_sparse: dict[str, int]) -> None:
        self.sparse_track_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}
        self.links_list = list(self.sparse_to_links.values())  # col_labels

    def assign(self, odv, pred, seg, it) -> None:
        n_rows, n_cols = len(self.sparse_track_links_list), len(self.links_list)
        volumes = np.zeros((n_rows, n_cols))
        # TODO: could parallelize: split sparse_track_links_list in N, and create N volumes mat: concat at the end
        volumes = assign_tracked_volumes(odv, pred, self.sparse_track_links_list, volumes)
        sparse_mat = array_to_csr_matrix(volumes)  # transform numpy to csr_matrix for easier lighter storage
        self.tracked_mat.append(TrackedVolume(it, seg, sparse_mat))

    def add_weights(self, phi, beta, it) -> None:
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta))

    def merge(self) -> pd.DataFrame:
        # apply frank wolfe for each iteration on each segments
        mats = _merge(self.tracked_mat, self.weights)
        row_labels = np.array(self.track_links_list)
        col_labels = np.array(self.links_list)
        output = pd.DataFrame(columns=['from', 'to'])
        for seg, mat in mats.items():
            rows, cols = mat.nonzero()
            df = pd.DataFrame({'from': row_labels[rows], 'to': col_labels[cols], seg: mat.data})
            output = pd.merge(output, df, on=['from', 'to'], how='outer')
        return output


def _merge(mat_datas: list[TrackedVolume], weights: list[TrackedWeight]) -> dict[str, csr_matrix]:
    # read info [iteration,segment,file]
    segments = segments = set([el.seg for el in mat_datas])
    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}
    beta_dict = {w.iteration: w.beta for w in weights}
    output = {}
    for seg in segments:
        filtered = [mat for mat in mat_datas if mat.seg == seg]
        output[seg] = apply_biconjugated_frank_wolfe(filtered, phi_dict, beta_dict)
    return output


def _mat_to_df(mat: csr_matrix, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(mat.toarray(), index=row_labels, columns=col_labels)


def apply_biconjugated_frank_wolfe(mat_datas: list[TrackedVolume], phi_dict: dict, beta_dict: dict) -> csr_matrix:
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


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volumes(odv, predecessors, track_index_list, volumes) -> np.ndarray:
    nrow = len(track_index_list)
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            for j in range(len(path) - 1):
                p = path[j]
                for row in range(nrow):
                    if track_index_list[row] == p:
                        # we get volume on next link (turn table)
                        volumes[row][path[j + 1]] += v
    return volumes


def array_to_csr_matrix(array: np.ndarray) -> csr_matrix:
    n_rows, n_cols = array.shape
    rows, cols, vals = _array_to_sparse(array)
    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


@nb.njit()
def _array_to_sparse(volumes: np.ndarray):
    # only append data if its in the set.
    # so we only keep the cols of links we will use (the one we track)
    rows, cols, vals = [], [], []
    nrows, ncols = volumes.shape
    for i in range(nrows):
        for j in range(ncols):
            v = volumes[i, j]
            if v != 0:
                rows.append(i)
                cols.append(j)
                vals.append(v)

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64)
