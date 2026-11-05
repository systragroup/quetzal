import pandas as pd
import numba as nb
from quetzal.engine.pathfinder_utils import get_node_path
from scipy.sparse import csr_matrix
from quetzal.engine.msa_trackers.tracker import Tracker
import numpy as np
from quetzal.engine.msa_trackers.tracker_utils import (
    TrackedVolume,
    TrackedWeight,
    apply_biconjugated_frank_wolfe,
    sparse_mat_to_df,
)


class LinksTracker(Tracker):
    def __init__(self, track_links_list: list[str] = []):
        self.track_links_list = track_links_list
        self.weights: list[TrackedWeight] = []
        self.tracked_mat: list[TrackedVolume] = []
        self.tracked_od: list[TrackedVolume] = []

    def init(
        self,
        links_sparse_index: list[int] | list[tuple[int, int]],
        links_to_sparse: dict[str, int] | dict[str, tuple[int, int]],
        volumes: pd.DataFrame,
        *args,
        **kwargs,
    ):
        # TODO: change for new LUT assigment method (faster and parallel), not numba dict assignment
        if isinstance(links_sparse_index[0], tuple):
            self.ab_volumes = init_ab_volumes(links_sparse_index)  # type: ignore
        else:
            self.ab_volumes = init_numba_volumes(links_sparse_index)  # type: ignore
        self.sparse_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}

        volumes = volumes.reset_index(drop=True).reset_index()  # make sure index are 0 to len
        self.sparse_od_dict = volumes.set_index(['origin_sparse', 'destination_sparse'])['index'].to_dict()
        self.od_list = [*zip(volumes['origin'], volumes['destination'])]

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def assign(self, odv, pred, origins, seg, it, *args, **kwargs):
        volumes = [self.ab_volumes.copy() for _ in self.sparse_links_list]
        volumes, od_is_used = assign_tracked_volumes(odv, pred, volumes, self.sparse_links_list)
        self.add_od(od_is_used, odv, origins, seg, it)
        self.add_volumes(volumes, seg, it)

    def add_volumes(self, volumes, seg, it):
        ab_keys = [k for k in self.ab_volumes.keys()]
        sparse_mat = dict_list_to_sparse_matrix(volumes, ab_keys)
        self.tracked_mat.append(TrackedVolume(it, seg, sparse_mat))

    def add_od(self, od_is_used: np.ndarray, odv: np.ndarray, origins: np.ndarray, seg: str, it):
        index_zone = {i: k for i, k in enumerate(origins)}
        rows, cols, data = [], [], []
        for i, is_used in enumerate(od_is_used):
            filtered = odv[is_used]
            origins_index = filtered[:, 0]
            sparse_destinations = filtered[:, 1].astype(np.int32)
            volumes = filtered[:, 2]
            sparse_origins = [*map(index_zone.get, origins_index)]
            od = [*zip(sparse_origins, sparse_destinations)]
            index = [*map(self.sparse_od_dict.get, od)]
            rows.extend(index)
            cols.extend([i] * len(index))
            data.extend(volumes)

        sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(self.sparse_od_dict), len(od_is_used)))
        self.tracked_od.append(TrackedVolume(it, seg, sparse_mat))

    def add_weights(self, phi, beta, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta))

    def merge(self, od=False) -> dict[str, pd.DataFrame]:
        # id OD: return volumes per od. [od, selectLinks]
        # else return volumes on links [links, selectLinks]
        # apply frank wolfe for each iteration on each segments
        output = {}
        if od:
            row_labels = self.od_list
            col_labels = self.track_links_list
            mats = _merge(self.tracked_od, self.weights)
            for seg, mat in mats.items():
                output[seg] = sparse_mat_to_df(mat, row_labels, col_labels)
        else:
            ab_keys = [k for k in self.ab_volumes.keys()]
            row_labels = [*map(self.sparse_to_links.get, ab_keys)]
            col_labels = self.track_links_list
            mats = _merge(self.tracked_mat, self.weights)
            for seg, mat in mats.items():
                output[seg] = sparse_mat_to_df(mat, row_labels, col_labels)

        return output


def _merge(mat_datas: list[TrackedVolume], weights):
    # read info [iteration,segment,file]
    segments = {el.seg for el in mat_datas}
    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}
    beta_dict = {w.iteration: w.beta for w in weights}
    res = {}
    for seg in segments:
        filtered = [mat for mat in mat_datas if mat.seg == seg]
        res[seg] = apply_biconjugated_frank_wolfe(filtered, phi_dict, beta_dict)
    return res


# for standard msa (not expanded)
def init_ab_volumes(indexes: list[tuple]) -> dict[tuple, float]:
    numba_volumes = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.types.int64, 2), value_type=nb.types.float64)
    for key in indexes:
        numba_volumes[key] = 0
    return numba_volumes


# # for extended
def init_numba_volumes(indexes: list[int]) -> dict[int, float]:
    numba_volumes = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)
    for key in indexes:
        numba_volumes[key] = 0
    return numba_volumes


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


def dict_list_to_sparse_matrix(dict_list: list[dict], ab_keys: list[int | tuple]) -> csr_matrix:
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
    tracked_od = np.full((nlen, odv.shape[0]), False)
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
                    tracked_od[j][i] = True
                    for key in path:
                        volumes[j][key] += v
    return volumes, tracked_od
