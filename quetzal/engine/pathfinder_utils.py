import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import dijkstra
import numba as nb
from copy import deepcopy
from quetzal.os.parallel_call import parallel_executor
from typing import Optional

import fast_dijkstra as fd
import polars as pl
from quetzal.engine.lazy_path import LazyPaths
from quetzal.engine.fast_utils import get_all_paths


# Wrapper to split the indices (destination) into parallel batchs and compute the shortest path on each batchs.
def fast_dijkstra(
    csgraph: csr_matrix,
    indices: Optional[list[str]] = None,
    return_predecessors: bool = True,
    limit: float = np.inf,
    num_threads: int = -1,
):
    """
    C++ dijkstra Faster than scipy when parallelize.
    On windows with 16 threads. it is almost 6 times faster than scipy (parallilize scipy)
    """
    if isinstance(csgraph, csc_matrix):
        csgraph = csgraph.tocsr()
    if indices is None:
        indices = csgraph.indptr
    distances, predecessor = fd.dijkstra(csgraph.indptr, csgraph.indices, csgraph.data, indices, limit, num_threads)
    if return_predecessors:
        return distances, predecessor
    else:
        return distances


# Wrapper to split the indices (destination) into parallel batchs and compute the shortest path on each batchs.
def parallel_dijkstra(csgraph, indices=None, return_predecessors=True, num_core=1, **kwargs):
    """
    num_core = 1 : number of threads.
    """
    if num_core == 1:
        return dijkstra(csgraph=csgraph, indices=indices, return_predecessors=return_predecessors, **kwargs)

    indices_mat = np.array_split(indices, num_core)

    results = parallel_executor(
        dijkstra,
        num_workers=num_core,
        parallel_kwargs={'indices': indices_mat},
        csgraph=csgraph,
        return_predecessors=return_predecessors,
        **kwargs,
    )

    if return_predecessors:  # result is a tuple
        dist_matrix = np.concatenate([res[0] for res in results], axis=0)
        predecessors = np.concatenate([res[1] for res in results], axis=0).astype(np.int32)
        return dist_matrix, predecessors
    else:
        dist_matrix = np.concatenate(results, axis=0)
        return dist_matrix


def simple_edge_routing(edges, origins, destinations, return_predecessors=False, **kwargs):
    # simple routing
    #
    mat, node_index = sparse_matrix(edges)

    reverse = len(origins) > len(destinations)
    if reverse:
        origins, destinations = destinations, origins
        mat = mat.transpose()
    index_node = {v: k for k, v in node_index.items()}
    # liste des origines pour le dijkstra
    origin_sparse = [node_index[x] for x in origins]

    # dijktra on the road network from node = incices to every other nodes.
    # from b to a.
    response = fast_dijkstra(csgraph=mat, indices=origin_sparse, return_predecessors=return_predecessors, **kwargs)

    dist_matrix = response[0] if return_predecessors else response
    predecessors = response[1] if return_predecessors else None

    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.index = origins
    # filtrer. on garde seulement les destination d'intéret
    destination_sparse = [node_index[x] for x in destinations]
    dist_matrix = dist_matrix[destination_sparse]
    dist_matrix = dist_matrix.rename(columns=index_node)
    if reverse:
        dist_matrix = dist_matrix.T
    if return_predecessors:
        return dist_matrix, predecessors, node_index
    else:
        return dist_matrix


def simple_routing(origin, destination, links, weight_col='time', **kwargs):
    # simple routing with with df
    edges = links[['a', 'b', weight_col]].values
    return simple_edge_routing(edges, origin, destination, **kwargs)


@nb.njit(locals={'predecessors': nb.int32[:, ::1], 'i': nb.int32, 'j': nb.int32})
def get_path(predecessors, i, j):
    path = []
    k = j
    while k != -9999:
        path.append(k)
        k = predecessors[i, k]
    return path[::-1]


@nb.njit()
def get_reversed_path(predecessors, i, j):
    path = [j]
    k = j
    p = 0
    while p != -9999:
        k = p = predecessors[i, k]
        path.append(p)
    return path[:-1]


@nb.njit()
def get_node_path(predecessors, i, j):
    # remove zones nodes (first and last one)
    return get_path(predecessors, i, j)[1:-1]


def get_edge_path(p):
    return list(zip(p[:-1], p[1:]))


def get_first_and_last(path, link_dict):
    s = set()

    for i in path:
        try:
            s.add(link_dict[i])
            break
        except KeyError:
            pass

    for i in reversed(path):
        try:
            s.add(link_dict[i])
            break
        except KeyError:
            pass
    return s


def get_all(path, link_dict):
    return {link_dict.get(link) for link in path} - {None}


def path_and_duration_from_graph(
    nx_graph,
    pole_set,
    od_set=None,
    sources=None,
    reversed_nx_graph=None,
    reverse=False,
    ntlegs_penalty=1e9,
    cutoff=np.inf,
    **kwargs,
):
    sources = pole_set if sources is None else sources
    source_los = sparse_los_from_nx_graph(
        nx_graph, pole_set, sources=sources, cutoff=cutoff + ntlegs_penalty, od_set=od_set, **kwargs
    )
    source_los['reversed'] = False

    reverse = reverse or reversed_nx_graph is not None
    if reverse:
        if reversed_nx_graph is None:
            reversed_nx_graph = nx_graph.reverse()

        try:
            reversed_od_set = {(d, o) for o, d in od_set}
        except TypeError:
            reversed_od_set = None

        target_los = sparse_los_from_nx_graph(
            reversed_nx_graph,
            pole_set,
            sources=sources,
            cutoff=cutoff + ntlegs_penalty,
            od_set=reversed_od_set,
            **kwargs,
        )
        target_los['reversed'] = True
        target_los['path'] = target_los['path'].apply(lambda x: list(reversed(x)))
        target_los[['origin', 'destination']] = target_los[['destination', 'origin']]

    los = pd.concat([source_los, target_los]) if reverse else source_los
    los.loc[los['origin'] != los['destination'], 'gtime'] -= ntlegs_penalty
    tuples = [tuple(l) for l in los[['origin', 'destination']].values.tolist()]
    los = los.loc[[t in od_set for t in tuples]]
    return los


def split_od_set(od_set, maxiter=10000):
    df = pd.DataFrame(od_set)
    o = d = 'init'
    o_set = set()
    d_set = set()

    i = 0
    while len(df) and i < maxiter:
        i += 1

        vco = df[0].value_counts()
        vcd = df[1].value_counts()

        firsto = list(vco.loc[vco >= vcd.max()].index)
        firstd = list(vcd.loc[vcd > vco.max()].index)
        if len(firsto):
            df = df.loc[~df[0].isin(firsto)]
            o_set = o_set.union(firsto)
        if len(firstd):
            df = df.loc[~df[1].isin(firstd)]
            d_set = d_set.union(firstd)

    o_od_set = {(o, d) for o, d in od_set if o in o_set}
    d_od_set = {(o, d) for o, d in od_set if d in d_set}

    assert o_od_set.union(d_od_set) == od_set
    return o_od_set, d_od_set - o_od_set


def efficient_od_sets(od_set, factor=1, verbose=False):
    o_od_set, d_od_set = split_od_set(od_set)
    no = len({o for o, d in o_od_set})
    nd = len({d for o, d in d_od_set})
    minod = min(len({o for o, d in od_set}), len({d for o, d in od_set}))
    if no + nd < minod * factor:
        if verbose:
            print(no, '+', nd, '=', no + nd, '<', factor, '*', minod, 'splitting od_set')
        return o_od_set, d_od_set
    if verbose:
        print(no, '+', nd, '=', no + nd, '>=', factor, '*', minod, 'keeping od_set')

    return od_set, set()


def sparse_los_from_nx_graph(nx_graph, pole_set, sources=None, cutoff=np.inf, od_set=None):
    import networkx as nx

    sources = pole_set if sources is None else sources
    if od_set is not None:
        sources = {o for o, d in od_set if o in sources}
    # INDEX
    pole_list = sorted(list(pole_set))  # fix order
    source_list = sorted([zone for zone in pole_list if zone in sources])

    nodes = list(nx_graph.nodes)
    node_index = dict(zip(nodes, range(len(nodes))))

    zones = [node_index[zone] for zone in source_list]
    source_index = dict(zip(source_list, range(len(source_list))))
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    # SPARSE GRAPH
    sparse = nx.to_scipy_sparse_matrix(nx_graph)
    graph = csr_matrix(sparse)
    dist_matrix, predecessors = dijkstra(
        csgraph=graph, directed=True, indices=zones, return_predecessors=True, limit=cutoff
    )

    # LOS LAYOUT
    df = pd.DataFrame(dist_matrix)

    df.index = [zone for zone in pole_list if zone in sources]
    df.columns = list(nx_graph.nodes)
    df.columns.name = 'destination'
    df.index.name = 'origin'
    stack = df[pole_list].stack()
    stack.name = 'gtime'
    los = stack.reset_index()

    # QUETZAL FORMAT
    los = los.loc[los['gtime'] < np.inf]
    if od_set is not None:
        tuples = [tuple(l) for l in los[['origin', 'destination']].values.tolist()]
        los = los.loc[[t in od_set for t in tuples]]

    # BUILD PATH FROM PREDECESSORS
    od_list = los[['origin', 'destination']].values.tolist()
    paths = [[nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])] for o, d in od_list]

    los['path'] = paths
    return los


# buildindex
def build_index(edges):
    nodelist = sorted({e[0] for e in edges}.union({e[1] for e in edges}))
    nlen = len(nodelist)
    return dict(zip(nodelist, range(nlen)))


# build matrix
def sparse_matrix(edges, index=None):
    if index is None:
        index = build_index(edges)
    nlen = len(index)
    row = np.array([*map(index.get, [e[0] for e in edges])], dtype=np.int32)
    col = np.array([*map(index.get, [e[1] for e in edges])], dtype=np.int32)
    data = [e[2] for e in edges]
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index


def sparse_matrix_with_access_penalty(edges, sources=set(), penalty=1e9):
    penalty_edges = []
    for u, v, w in edges:
        if u in sources:
            w += penalty
        penalty_edges.append((u, v, w))
    return sparse_matrix(penalty_edges)


def _link_edges(links, boarding_time=None, alighting_time=None):
    """
    :param links: Description
    :param boarding_time: if None: column boarding_time
    :param alighting_time: if None: column alighting_time
    """
    assert not (boarding_time is not None and 'boarding_time' in links.columns)
    boarding_time = 0 if boarding_time is None else boarding_time

    assert not (alighting_time is not None and 'alighting_time' in links.columns)
    alighting_time = 0 if alighting_time is None else alighting_time

    l = links.copy()
    l['index'] = l.index
    l['next'] = l['link_sequence'] + 1

    l['_cost'] = l['time'] + l['headway'] / 2

    if 'boarding_time' not in l.columns:
        l['boarding_time'] = boarding_time

    if 'alighting_time' not in l.columns:
        l['alighting_time'] = alighting_time

    l['total_time'] = l['boarding_time'] + l['_cost']

    transit = pd.merge(
        l[['index', 'next', 'trip_id']],
        l[['index', 'link_sequence', 'trip_id', 'time']],
        left_on=['trip_id', 'next'],
        right_on=['trip_id', 'link_sequence'],
    )
    boarding_edges = l[['a', 'index', 'total_time']]
    alighting_edges = l[['index', 'b', 'alighting_time']]
    transit_edges = transit[['index_x', 'index_y', 'time']]
    return boarding_edges, alighting_edges, transit_edges


def link_edges(links, boarding_time=None, alighting_time=None):
    boarding_e, alighting_e, transit_e = _link_edges(
        links=links, boarding_time=boarding_time, alighting_time=alighting_time
    )
    boarding_edges = boarding_e.values.tolist()
    alighting_edges = alighting_e.values.tolist()
    transit_edges = transit_e.values.tolist()

    return boarding_edges + transit_edges + alighting_edges


def link_edge_array(links, boarding_time=None, alighting_time=None):
    boarding_e, alighting_e, transit_e = _link_edges(
        links=links, boarding_time=boarding_time, alighting_time=alighting_time
    )
    boarding_edges = boarding_e.values
    alighting_edges = alighting_e.values
    transit_edges = transit_e.values

    return np.concatenate([boarding_edges, transit_edges, alighting_edges])


def index_access_pruned_matrix(matrix, index, pruned):
    """
    copy a matrix and returns a matrix with infitine costs for a given row
    the index is a dict name:ix
    pruned is a list of names to remove
    """
    pmatrix = deepcopy(matrix)
    for vertex in pruned:
        i = index[vertex]
        for j in pmatrix[i].indices:
            pmatrix[i, j] = np.inf
    return pmatrix, index


def pruned_matrix(matrix, index, pruned):
    """
    copy a matrix and returns a matrix with infitine costs for a given row
    the index is a dict name:ix
    pruned is a list of names to remove
    """
    pruned_ix = {index[p] for p in pruned}
    kept_index = sorted(list(set(index.values()) - set(pruned_ix)))

    # REINDEX
    rank = 0  # {former_ix : new_ix}
    reindex = {}
    for i in kept_index:
        reindex[i] = rank
        rank += 1

    pindex = {}  # {name : new_ix}
    for k, v in index.items():
        try:
            pindex[k] = reindex[v]
        except KeyError:
            pass

    pmatrix = csr_matrix(csc_matrix(matrix[kept_index, :])[:, kept_index])
    return pmatrix, pindex


def paths_from_edges(
    edges,
    sources=None,
    targets=None,
    od_set=None,
    cutoff=np.inf,
    penalty=1e9,
    log=False,
    # edges can be transmitted as a CSR matrix
    csgraph=None,  # CSR matrix
    node_index=None,  # {name such as 'link_123': matrix index}
    lazy_paths=False,
    num_cores=1,
):
    if od_set:
        o_set = {o for o, d in od_set}
        d_set = {d for o, d in od_set}
        if sources is not None:
            sources = [s for s in sources if s in o_set]
        else:
            sources = list(o_set)
        if targets is not None:
            targets = [t for t in targets if t in d_set]
        else:
            targets = list(d_set)
    reverse = len(sources) > len(targets)
    if reverse:
        if log:
            print(len(sources), 'sources', len(targets), 'targets', 'transposed search')
        sources, targets = targets, sources
    elif log:
        print(len(sources), 'sources', len(targets), 'targets', 'direct search')

    st = set(sources).union(targets)
    if csgraph is None or node_index is None:
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
        csgraph=csgraph, indices=source_indices, return_predecessors=True, limit=cutoff + penalty, num_threads=num_cores
    )

    dist_matrix = dist_matrix[:, target_indices]
    df = pl.DataFrame(dist_matrix, schema=targets)
    df = df.insert_column(0, pl.Series('origin', sources))
    df = df.unpivot(index='origin', variable_name='destination', value_name='length')
    if od_set is not None:
        mask = pl.DataFrame({'origin': [od[0] for od in od_set], 'destination': [od[1] for od in od_set]})  # for late
        df = df.join(mask, on=['origin', 'destination'], how='semi')  # drop rows not in od_set

    los = df.to_pandas()
    los['length'] -= penalty
    los = los.loc[los['length'] < np.inf]

    if lazy_paths:
        los_paths = LazyPaths(predecessors, node_index, sources, reverse)
        los.path = los_paths
        los['path'] = los_paths

    else:
        odl = los[['origin', 'destination']].values
        od_sparse_list = np.stack([[*map(source_index.get, odl[:, 0])], [*map(node_index.get, odl[:, 1])]], axis=1)
        los['path'] = get_all_paths(od_sparse_list, predecessors, index_node, reverse)
        # los['hash'] = get_all_paths_hash

    if reverse:
        los[['origin', 'destination']] = los[['destination', 'origin']]

    return los


def adjacency_matrix(links, ntlegs, footpaths, ntlegs_penalty=1e9, boarding_time=None, alighting_time=None):
    ntlegs = ntlegs.copy()

    # ntlegs and footpaths
    ntlegs.loc[ntlegs['direction'] == 'access', 'time'] += ntlegs_penalty
    ntleg_edges = ntlegs[['a', 'b', 'time']].values.tolist()
    footpaths_edges = footpaths[['a', 'b', 'time']].values.tolist()

    edges = link_edges(links, boarding_time, alighting_time)
    edges += footpaths_edges + ntleg_edges
    return sparse_matrix(edges)


def los_from_graph(
    csgraph,  # graph is assumed to be a scipy csr_matrix
    node_index=None,
    pole_set=None,
    sources=None,
    cutoff=np.inf,
    od_set=None,
    ntlegs_penalty=1e9,
):
    sources = pole_set if sources is None else sources
    if od_set is not None:
        sources = {o for o, d in od_set if o in sources}
    # INDEX
    pole_list = sorted(list(pole_set))  # fix order
    source_list = sorted([zone for zone in pole_list if zone in sources])

    zones = [node_index[zone] for zone in source_list]
    source_index = dict(zip(source_list, range(len(source_list))))
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    # SPARSE GRAPH
    dist_matrix, predecessors = dijkstra(
        csgraph=csgraph, directed=True, indices=zones, return_predecessors=True, limit=cutoff + ntlegs_penalty
    )

    # LOS LAYOUT
    df = pd.DataFrame(dist_matrix)
    indexed_nodes = {v: k for k, v in node_index.items()}
    df.rename(columns=indexed_nodes, inplace=True)

    df.index = [zone for zone in pole_list if zone in sources]

    df.columns.name = 'destination'
    df.index.name = 'origin'

    od_weight = df[pole_list]
    if od_set is not None:
        mask = pd.Series(index=od_set, data=1).unstack()
        mask.index.name = 'origin'
        mask.columns.name = 'destination'
        od_weight = od_weight.multiply(mask)

    stack = od_weight.stack()
    stack.name = 'gtime'
    los = stack.reset_index()

    # QUETZAL FORMAT
    los = los.loc[los['gtime'] < np.inf]
    los.loc[los['origin'] != los['destination'], 'gtime'] -= ntlegs_penalty
    if od_set is not None:
        tuples = [tuple(l) for l in los[['origin', 'destination']].values.tolist()]
        los = los.loc[[t in od_set for t in tuples]]

    # BUILD PATH FROM PREDECESSORS
    od_list = los[['origin', 'destination']].values.tolist()
    paths = [[indexed_nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])] for o, d in od_list]

    los['path'] = paths
    return los


def paths_from_graph(csgraph, node_index, sources, targets, od_set=None, cutoff=np.inf):
    reverse = False
    if od_set:
        o_set = {o for o, d in od_set}
        d_set = {d for o, d in od_set}
        sources = [s for s in sources if s in o_set]
        targets = [t for t in targets if t in d_set]

    if len(sources) > len(targets):
        reverse = True
        sources, targets, csgraph = targets, sources, csgraph.T

    # INDEX
    source_indices = [node_index[s] for s in sources]
    target_indices = [node_index[t] for t in targets]
    source_index = dict(zip(sources, range(len(sources))))
    index_node = {v: k for k, v in node_index.items()}

    # DIKSTRA
    dist_matrix, predecessors = dijkstra(
        csgraph=csgraph, directed=True, indices=source_indices, return_predecessors=True, limit=cutoff
    )

    dist_matrix = dist_matrix.T[target_indices].T
    df = pd.DataFrame(dist_matrix, index=sources, columns=targets)

    df.columns.name = 'destination'
    df.index.name = 'origin'

    if od_set is not None:
        od_index = {(d, o) for o, d in od_set} if reverse else od_set
        mask = pd.Series(index=od_index, data=1).unstack()
        mask.index.name = 'origin'
        mask.columns.name = 'destination'
        df = df.multiply(mask)

    stack = df.stack()

    stack.name = 'length'
    odl = stack.reset_index()
    od_list = odl[['origin', 'destination']].values
    path = get_reversed_path if reverse else get_path
    paths = [[index_node[i] for i in path(predecessors, source_index[o], node_index[d])] for o, d in od_list]
    odl['path'] = paths

    if reverse:
        odl[['origin', 'destination']] = odl[['destination', 'origin']]
    return odl
