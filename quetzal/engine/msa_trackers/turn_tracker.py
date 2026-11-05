import pandas as pd
import numpy as np
import numba as nb
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csr_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
from quetzal.engine.msa_trackers.tracker_utils import TrackedVolume, TrackedWeight, apply_biconjugated_frank_wolfe


class TurnTracker(Tracker):
    def __init__(self, track_links_list: list[str] = [], num_threads: int = 1):
        print('This tracker only work with turn_penalties')
        # if we want to make this work with normal msa. we should track nodes (intersection)
        # so the check in the paths is easy. if p == node take p-1 and p+1
        self.track_links_list = track_links_list  # row_labels
        self.weights: list[TrackedWeight] = []
        self.tracked_mat: list[TrackedVolume] = []
        self.num_threads = num_threads

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def init(self, links_to_sparse: dict[str, int], *args, **kwargs) -> None:
        self.sparse_track_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}
        self.links_list = list(self.sparse_to_links.values())  # col_labels

    def assign(self, odv, pred, seg, it, *args, **kwargs) -> None:
        volumes = assign_tracked_volumes_parallel(odv, pred, self.sparse_track_links_list, self.num_threads)
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
    segments = set([el.seg for el in mat_datas])
    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}
    beta_dict = {w.iteration: w.beta for w in weights}
    output = {}
    for seg in segments:
        filtered = [mat for mat in mat_datas if mat.seg == seg]
        output[seg] = apply_biconjugated_frank_wolfe(filtered, phi_dict, beta_dict)
    return output


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volumes(odv, predecessors, track_index_list, volumes) -> np.ndarray:
    nrows = len(track_index_list)
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            for j in range(len(path) - 1):
                p = path[j]
                for row in range(nrows):
                    if track_index_list[row] == p:
                        # we get volume on next link (turn table)
                        volumes[row][path[j + 1]] += v
    return volumes


@nb.njit(locals={'predecessors': nb.int32[:, ::1]}, parallel=True)
def assign_tracked_volumes_parallel(odv, predecessors, track_index_list, num_threads=1) -> np.ndarray:
    nb.set_num_threads(num_threads)
    nrows = len(track_index_list)
    ncols = predecessors.shape[1]
    volumes = np.zeros((nrows, ncols))
    track_index_split = np.array_split(track_index_list, num_threads)
    volumes_split = np.array_split(volumes, num_threads, axis=0)
    for j in nb.prange(num_threads):
        volumes_split[j] = assign_tracked_volumes(odv, predecessors, track_index_split[j], volumes_split[j])
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
