import numpy as np
from typing import List, Dict, Union, Any
from quetzal.engine.fast_utils import compute_offsets
import numba as nb
import pandas as pd


@nb.njit()
def get_path_3d(predecessors, i, j, s, reverse=False):
    path = []
    k = j
    while k != -9999:
        path.append(k)
        k = predecessors[s, i, k]
    if reverse:
        return path
    else:
        return path[::-1]


@nb.njit(parallel=True)
def get_flat_path(od_list, predecessors, offsets, total_len, reverse=False) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    flat_paths = np.zeros(total_len, dtype=np.int32)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s, reverse)
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
    return flat_paths


@nb.njit(parallel=True)
def compute_path_lengths(od_list, predecessors):
    # return the length of each path in predecessor.
    n = len(od_list)
    lengths = np.zeros(n, dtype=np.int32)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s)
        lengths[i] = len(path)
    return lengths


@nb.njit(parallel=True)
def get_paths_hash(od_list, predecessors) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    hash_list = np.empty(n, dtype=np.int64)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s)
        h = 0
        for x in path:
            h ^= hash(x) + 0x9E3779B9 + (h << 6) + (h >> 2)  # boost::hash_combine
        hash_list[i] = h

    return hash_list


@nb.njit()
def sum_jagged_array(flat_paths, offsets):
    row_sums = np.zeros(len(offsets) - 1, dtype=flat_paths.dtype)
    for i in nb.prange(len(offsets) - 1):
        row_sums[i] = flat_paths[offsets[i] : offsets[i + 1]].sum()
    return row_sums


def remap_arr(arr: np.ndarray[int], mapper: dict[int, Any]) -> np.ndarray[Any]:
    # take max of array for the Lookup table
    max_key = arr.max() + 1
    # determine array dtype with input dict and create an empty array
    first = next(iter(mapper.values()))
    lut = np.empty(max_key, dtype=object) if isinstance(first, str) else np.zeros(max_key, dtype=type(first))
    for k, v in mapper.items():
        if k < max_key:  # mapper could have more values than the array
            lut[k] = v
    return lut[arr]


class LosPaths:
    def __init__(self, predecessors, node_index: Dict[str, int], sources: List[str], reverse: bool = False) -> int:
        self.predecessors = np.array([predecessors])
        self.node_index = node_index  # {link_0: 0, link_1: 1, ...}
        self.sources = sources  # list of zones
        self.reverse = reverse

        self.index_node = {k: v for v, k in node_index.items()}  # {0: link_0, 1: link1, ...}
        self.source_index = dict(zip(sources, range(len(sources))))  # {zone_0:0, zone_1: 1, ...}

    def __repr__(self):
        return f'Lazy paths with {self.predecessors.shape[0]} sessions'

    def __hash__(self):
        return id(self)

    def append(self, predecessor, index_node) -> int:
        new_pred = remap_predecessor(self.predecessors[0], self.node_index, predecessor, index_node)
        self.predecessors = np.vstack([self.predecessors, [new_pred]])
        return self.predecessors.shape[0]

    def _get_int_path(self, o: int, d: int, s: int) -> List[int]:
        return get_path_3d(self.predecessors, o, d, s, self.reverse)

    def _remap_od_list(self, od_list: np.ndarray) -> np.ndarray[int, int]:
        o_list = [*map(self.source_index.get, od_list[:, 0])]
        d_list = [*map(self.node_index.get, od_list[:, 1])]
        s_list = np.array(od_list[:, 2], dtype=np.int32)
        od_index_list = np.stack([o_list, d_list, s_list], axis=1)
        return od_index_list

    def _get_jagged_int_path(self, od_index_list: np.ndarray):
        lengths = compute_path_lengths(od_index_list, self.predecessors)
        offsets, total_len = compute_offsets(lengths)
        flat_paths = get_flat_path(od_index_list, self.predecessors, offsets, total_len, self.reverse)
        return flat_paths, offsets

    #
    # exposed
    #
    def get_path(self, origin: str, destination: str, session: int, remap: bool = True) -> Union[List[str], List[int]]:
        path = self._get_int_path(self.source_index[origin], self.node_index[destination], session)
        if remap:
            path = [*map(self.index_node.get, path)]
        return path

    def get_all_paths(self, od_list: np.ndarray, remap: bool = True) -> list[list]:
        od_list = self._remap_od_list(od_list)
        flat_paths, offsets = self._get_jagged_int_path(od_list)
        if remap:
            flat_paths = remap_arr(flat_paths, self.index_node)

        return [flat_paths[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]

    def get_all_paths_hash(self, od_list: np.ndarray) -> list[list]:
        od_list = self._remap_od_list(od_list)
        return get_paths_hash(od_list, self.predecessors)

    def get_all_paths_len(self, od_list: np.ndarray) -> list[list]:
        od_list = self._remap_od_list(od_list)
        return compute_path_lengths(od_list, self.predecessors)

    def apply_dict_on_all_paths(self, od_list: np.ndarray, mapper: Dict[str, float]):
        od_list = self._remap_od_list(od_list)
        flat_paths, offsets = self._get_jagged_int_path(od_list)

        mapper = {self.node_index.get(k): v for k, v in mapper.items()}
        flat_paths = remap_arr(flat_paths, mapper)
        return sum_jagged_array(flat_paths, offsets)


def remap_predecessor(full_pred, full_dict, pruned_pred, pruned_dict):

    pruned_to_full = {k: full_dict[v] for k, v in pruned_dict.items()}

    # create a lookup table :  prune to full index
    pruned_to_full_map = np.empty(pruned_pred.shape[1], dtype=np.int32)
    full_indexes = np.array(list(pruned_to_full.values()))
    pruned_indexes = np.array(list(pruned_to_full.keys()))
    pruned_to_full_map[pruned_indexes] = full_indexes

    # remap value in pruned_predecessor to the one in full_pred
    mask = pruned_pred != -9999  # need to mask if we want to keep -9999,else it will look at index -9999 in lut
    pruned_pred[mask] = pruned_to_full_map[pruned_pred[mask]]

    # reorder rows to the full predecessor indexing. (and shape it as the full_pred)
    new_predecessor = np.full_like(full_pred, -9999, dtype=np.int32)
    new_predecessor[:, full_indexes] = pruned_pred[:, pruned_indexes]

    # assert full_paths.source_index == pruned_paths.source_index, 'TODO need to remap source too.'

    return new_predecessor


def concat(los_list):
    pt_los = pd.concat(los_list, ignore_index=True)
    lazy_path_list = pt_los['path'].unique()
    path_to_group = {p: i for i, p in enumerate(lazy_path_list)}
    pt_los['lazy_session'] = pt_los['path'].map(path_to_group)
    lazy_path = lazy_path_list[0]
    for obj in lazy_path_list[1:]:
        lazy_path.append(obj.predecessors[0], obj.index_node)
    pt_los['path'] = lazy_path
    return pt_los
