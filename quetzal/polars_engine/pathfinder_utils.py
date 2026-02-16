from quetzal.engine.pathfinder_utils import fast_dijkstra, sparse_matrix_with_access_penalty, get_reversed_path
from quetzal.engine.fast_utils import get_jagged_path, get_path
import numpy as np
import polars as pl
import numba as nb


@nb.njit(parallel=True)
def compute_path_lengths(od_list, predecessors, reverse=False):
    n = len(od_list)
    lengths = np.zeros(n, dtype=np.int32)
    get_path_function = get_reversed_path if reverse else get_path
    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path_function(predecessors, o, d)  # returns Python list
        lengths[i] = len(path)

    return lengths


def compute_offsets(lengths):
    n = len(lengths)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = offsets[-1]
    return offsets, total  # total = size of flat array


@nb.njit(parallel=True)
def fill_paths_flat(od_list, predecessors, offsets, total_len):
    n = len(od_list)
    flat_paths = np.zeros(total_len, dtype=np.int32)
    flat_index = np.zeros(total_len, dtype=np.int32)

    for i in nb.prange(n):
        o, d = od_list[i]
        path = get_path(predecessors, o, d)  # returns Python list
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
            flat_index[start + j] = i
    return flat_paths, flat_index


def paths_from_edges(
    edges, sources=None, targets=None, od_set=None, cutoff=np.inf, penalty=1e9, log=True, num_threads=1
):
    o_set = {o for o, d in od_set}
    d_set = {d for o, d in od_set}
    sources = list(o_set)
    targets = list(d_set)
    reverse = len(sources) > len(targets)
    if reverse:
        if log:
            print(len(sources), 'sources', len(targets), 'targets', 'transposed search')
        sources, targets = targets, sources
    elif log:
        print(len(sources), 'sources', len(targets), 'targets', 'direct search')

    st = set(sources).union(targets)

    csgraph, node_index = sparse_matrix_with_access_penalty(edges, sources=st, penalty=penalty)

    if reverse:
        csgraph = csgraph.transpose()

    # INDEX
    source_indices = [node_index[s] for s in sources]
    target_indices = [node_index[t] for t in targets]
    source_index = dict(zip(sources, range(len(sources))))
    index_node = {v: k for k, v in node_index.items()}
    # DIKSTRA
    dist_matrix, predecessors = fast_dijkstra(
        csgraph=csgraph,
        indices=source_indices,
        return_predecessors=True,
        limit=cutoff + penalty,
        num_threads=num_threads,
    )
    dist_matrix = dist_matrix[:, target_indices]

    flat_matrix = dist_matrix.flatten()
    rows = [x for x in sources for _ in range(len(target_indices))]
    cols = targets * len(source_indices)
    df = pl.DataFrame({'origin': rows, 'destination': cols, 'time': flat_matrix})
    df = df.with_columns(pl.col('time') - penalty)
    df = df.filter(pl.col('time') < np.inf)
    df = df.with_row_index()
    sparse_od_list = df.select(
        pl.col('origin').replace_strict(source_index), pl.col('destination').replace_strict(node_index)
    ).to_numpy()

    paths = get_jagged_path(sparse_od_list, predecessors, reverse)
    df = df.with_columns(pl.from_arrow(paths).alias('path'))
    df = df.with_columns(pl.col('path').list.eval(pl.element().replace_strict(index_node, default=None)))

    if reverse:
        df[['origin', 'destination']] = df[['destination', 'origin']]
    return df
