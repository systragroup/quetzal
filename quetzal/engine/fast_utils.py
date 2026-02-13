import numpy as np
import numba as nb
import pyarrow as pa
from typing import List, Tuple


@nb.njit(locals={'predecessors': nb.int32[:, ::1], 'i': nb.int32, 'j': nb.int32})
def get_path(predecessors, i, j):
    path = []
    k = j
    while k != -9999:
        path.append(k)
        k = predecessors[i, k]
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
def get_flat_path(od_list, predecessors, offsets, total_len) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    flat_paths = np.zeros(total_len, dtype=np.int32)
    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path(predecessors, o, d)
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
    return flat_paths


def get_jagged_path(sparse_od_list: List[Tuple[int, int]], predecessors) -> pa.ListArray:
    # deconstruct predecessor and return a list of list (each od path)
    # pyArrow listArray is super optimzed and fast
    #
    # it is way faster and thread safe to fist compute the lengths of each paths and then create a single flat array of each path
    # we then have a flat_path array and an offsets array. Pyarrow support this type of list with pa.ListArray
    lengths = compute_path_lengths(sparse_od_list, predecessors)
    offsets, total_len = compute_offsets(lengths)
    flat_paths = get_flat_path(sparse_od_list, predecessors, offsets, total_len)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_paths))


# TODO: add support for remap
# def remap_int_arrow_array(arr: List[int], mapping: Dict[int, str]) -> pa.Array:
#     map_keys = pa.array(list(mapping.keys()), type=pa.int32())
#     map_values = pa.array(list(mapping.values()), type=pa.string())
#     # Find index of each value in the mapping keys
#     indices = pa.compute.index_in(arr, map_keys)
#     # Take strings using these indices
#     return pa.compute.take(map_values, indices)


# def get_jagged_path(sparse_od_list, predecessors, index_node=None):
#     lengths = compute_path_lengths(sparse_od_list, predecessors)
#     offsets, total_len = compute_offsets(lengths)
#     flat_paths = get_flat_path(sparse_od_list, predecessors, offsets, total_len)
#     if index_node is not None:
#         flat_paths = remap_int_arrow_array(flat_paths, index_node)
#     return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_paths))
