import pandas as pd
import numba as nb
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csc_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
from collections import namedtuple
from quetzal.engine.msa_trackers.tracker_utils import (
    TrackedVolume,
    TrackedWeight,
    tracked_od_to_matrix,
    apply_biconjugated_frank_wolfe,
    sparse_mat_to_df,
)

# need to name the class the same as the namedTuple name for pickle.
TrackedArrays = namedtuple('TrackedArrays', 'iteration seg volumes')


class TupleTracker(Tracker):
    def __init__(self, track_links_list: list[tuple] = []):
        print('This tracker only work with turn_penalties')
        # list of tuple [(rlink1, rlink2), (rlink1, rlink100),...]
        self.track_links_list = track_links_list
        self.weights = []
        self.tracked_mat: list[TrackedArrays] = []
        self.tracked_od: list[TrackedVolume] = []

    def init(
        self,
        links_sparse_index: list[int] | list[tuple[int, int]],
        links_to_sparse: dict[str, int] | dict[str, tuple[int, int]],
        volumes: pd.DataFrame,
        *args,
        **kwargs,
    ):
        self.n_cols = len(links_sparse_index)
        self.sparse_links_list = [[*map(links_to_sparse.get, ls)] for ls in self.track_links_list]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}
        self.tracked_links_set = set(np.concatenate(self.sparse_links_list))

        volumes = volumes.reset_index(drop=True).reset_index()  # make sure index are 0 to len
        self.sparse_od_dict = volumes.set_index(['origin_sparse', 'destination_sparse'])['index'].to_dict()
        self.od_list = [*zip(volumes['origin'], volumes['destination'])]

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def assign(self, odv, pred, origins, seg, it, *args, **kwargs):
        mat = get_paths_matrix(odv, pred, self.n_cols, self.tracked_links_set)
        od_indexes_list = get_od_indexes(mat, self.sparse_links_list)
        volumes = []
        for od_indexes in od_indexes_list:
            volumes.append(odv[od_indexes, 2].sum())
        self.tracked_mat.append(TrackedArrays(it, seg, np.array(volumes)))
        self.add_od(od_indexes_list, odv, origins, seg, it)

    def add_weights(self, phi, beta, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta))

    def merge(self, od=False) -> pd.DataFrame | dict[str, pd.DataFrame]:
        # apply frank wolfe for each iteration on each segments
        if od:
            output = {}
            row_labels = self.od_list
            col_labels = self.track_links_list
            mats = apply_biconjugated_frank_wolfe(self.tracked_od, self.weights)
            for seg, mat in mats.items():
                output[seg] = sparse_mat_to_df(mat, row_labels, col_labels)
            return output
        else:
            merged = apply_biconjugated_frank_wolfe(self.tracked_mat, self.weights)
            return pd.DataFrame(merged, index=self.track_links_list)

    def add_od(self, od_is_used: list[list[int]], odv: np.ndarray, origins: np.ndarray, seg: str, it):
        sparse_mat = tracked_od_to_matrix(od_is_used, odv, origins, self.sparse_od_dict)
        self.tracked_od.append(TrackedVolume(it, seg, sparse_mat))


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


def get_od_indexes(mat, track_index_list) -> list[list[int]]:
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
