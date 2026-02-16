import numpy as np
import numba as nb
import pyarrow as pa
from typing import List, Tuple


@nb.njit(locals={'predecessors': nb.int32[:, ::1], 'i': nb.int32, 'j': nb.int32})
def get_path(predecessors, i, j, reverse=False):
    path = []
    k = j
    while k != -9999:
        path.append(k)
        k = predecessors[i, k]
    if reverse:
        return path
    else:
        return path[::-1]


@nb.njit(parallel=True)
def compute_path_lengths(od_list, predecessors):
    # return the length of each path in predecessor.
    n = len(od_list)
    lengths = np.zeros(n, dtype=np.int32)
    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path(predecessors, o, d)
        lengths[i] = len(path)
    return lengths


def compute_offsets(lengths):
    # compute offset with lengths
    n = len(lengths)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = offsets[-1]
    return offsets, total


@nb.njit(parallel=True)
def get_flat_path(od_list, predecessors, offsets, total_len, reverse=False) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    flat_paths = np.zeros(total_len, dtype=np.int32)
    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path(predecessors, o, d, reverse)
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
    return flat_paths


def get_jagged_path(sparse_od_list: List[Tuple[int, int]], predecessors, reverse=False) -> pa.ListArray:
    # deconstruct predecessor and return a list of list (each od path)
    # pyArrow listArray is super optimzed and fast
    #
    # it is way faster and thread safe to fist compute the lengths of each paths and then create a single flat array of each path
    # we then have a flat_path array and an offsets array. Pyarrow support this type of list with pa.ListArray
    lengths = compute_path_lengths(sparse_od_list, predecessors)
    offsets, total_len = compute_offsets(lengths)
    flat_paths = get_flat_path(sparse_od_list, predecessors, offsets, total_len, reverse)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_paths))
