import numpy as np
import numba as nb
import pyarrow as pa
from typing import List, Tuple, Any, Optional
import numpy.typing as npt


def dict_to_lut(mapper: dict[int, Any], max_key: Optional[int] = None) -> npt.NDArray[Any]:
    # return a lookup table from a dictionary.
    # providing max key: array [0,1,2,3] with dict {1:'a',2:'b'} will return [None,a,b,None], else Error.
    if max_key is None:
        max_key = max(mapper.keys()) + 1
    # determine array dtype with input dict and create an empty array
    first = next(iter(mapper.values()))
    lut = np.empty(max_key, dtype=object) if isinstance(first, str) else np.zeros(max_key, dtype=type(first))
    for k, v in mapper.items():
        if k < max_key:  # dict could have more values than the array
            lut[k] = v
    return lut


def remap_int_array(arr: npt.NDArray[np.int_], mapper: dict[int, Any]) -> npt.NDArray[Any]:
    # apply a dict on an array
    # return an array
    # providing max key: array [0,1,2,3] with dict {1:'a',2:'b'} will return [None,a,b,None]
    max_key = arr.max() + 1  # take max of array for the Lookup table
    lut = dict_to_lut(mapper, max_key)
    return lut[arr]


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
    offsets = np.zeros(len(lengths), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    return offsets


@nb.njit(parallel=True)
def get_flat_path(od_list, predecessors, offsets, reverse=False) -> npt.NDArray:
    # get all paths as a single flat array.
    n = len(od_list)
    total_len = offsets[-1]
    flat_paths = np.zeros(total_len, dtype=np.int32)
    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path(predecessors, o, d, reverse)
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
    return flat_paths


# TODO: to remove / change
def get_jagged_path(sparse_od_list: List[Tuple[int, int]], predecessors, reverse=False) -> pa.ListArray:
    # deconstruct predecessor and return a list of list (each od path)
    # pyArrow listArray is super optimzed and fast
    #
    # it is way faster and thread safe to fist compute the lengths of each paths and then create a single flat array of each path
    # we then have a flat_path array and an offsets array. Pyarrow support this type of list with pa.ListArray
    lengths = compute_path_lengths(sparse_od_list, predecessors)
    offsets = compute_offsets(lengths)
    flat_paths = get_flat_path(sparse_od_list, predecessors, offsets, reverse)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat_paths))


def get_all_paths(
    od_list: npt.NDArray[np.int_], predecessors, index_node: dict[int, Any] = None, reverse=False
) -> List[List]:
    lengths = compute_path_lengths(od_list, predecessors)
    offsets = compute_offsets(lengths)
    flat_paths = get_flat_path(od_list, predecessors, offsets, reverse)
    if index_node:
        flat_paths = remap_int_array(flat_paths, index_node)

    return [flat_paths[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def reconstruct_flat_and_offsets(paths: npt.NDArray):
    # los['path'].values
    # this recreate a flat path from a dataframe
    flat = np.concatenate(paths)
    offsets = np.zeros(len(paths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(el) for el in paths])
    return flat, offsets
