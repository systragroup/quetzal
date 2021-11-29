import itertools
import networkx as nx
import numpy as np
import pandas as pd
from quetzal.engine import engine
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def get_path(predecessors, i, j):
    path = [j]
    k = j
    p = 0
    while p != -9999:
        k = p = predecessors[i, k]
        path.append(p)
    return path[::-1][1:]

@jit(nopython=True)
def get_reversed_path(predecessors, i, j):
    path = [j]
    k = j
    p = 0
    while p != -9999:
        k = p = predecessors[i, k]
        path.append(p)
    return path[:-1]

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
    **kwargs
):
    sources = pole_set if sources is None else sources
    source_los = sparse_los_from_nx_graph(
        nx_graph, pole_set, sources=sources,
        cutoff=cutoff + ntlegs_penalty, od_set=od_set, **kwargs
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
            reversed_nx_graph, pole_set, sources=sources,
            cutoff=cutoff + ntlegs_penalty, od_set=reversed_od_set, **kwargs)
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

    assert o_od_set.union(d_od_set)==od_set
    return o_od_set, d_od_set-o_od_set

def efficient_od_sets(od_set, factor=1, verbose=False):
    o_od_set, d_od_set = split_od_set(od_set)
    no = len({o for o, d in o_od_set})
    nd = len({d for o, d in d_od_set})
    minod = min(len({o for o, d in od_set}), len({d for o, d in od_set}))
    if no + nd < minod*factor:
        if verbose:
            print(no, '+', nd, '=', no + nd, '<',factor, '*', minod, 'splitting od_set')
        return o_od_set, d_od_set
    if verbose:
        print(no, '+', nd, '=', no + nd, '>=',factor, '*', minod, 'keeping od_set')
    
    return od_set, set()

def sparse_los_from_nx_graph(
    nx_graph,
    pole_set,
    sources=None,
    cutoff=np.inf,
    od_set=None,
):
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
        csgraph=graph,
        directed=True,
        indices=zones,
        return_predecessors=True,
        limit=cutoff
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
    paths = [
        [nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])]
        for o, d in od_list
    ]

    los['path'] = paths
    return los


def sparse_matrix(edges):
    nodelist = {e[0] for e in edges}.union({e[1] for e in edges})
    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(*((index[u], index[v], w) for u, v, w in edges))
    row, col, data = coefficients
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index




def _link_edges(links, boarding_time=None, alighting_time=None):
    assert not (boarding_time is not None and 'boarding_time' in links.columns)
    boarding_time = 0 if boarding_time is None else boarding_time

    assert not (alighting_time is not None and 'alighting_time' in links.columns)
    alighting_time = 0 if alighting_time is None else alighting_time

    l = links.copy()
    l['index'] = l.index
    l['next'] = l['link_sequence'] + 1

    if 'cost' not in l.columns:
        l['cost'] = l['time'] + l['headway'] / 2

    if 'boarding_time' not in l.columns:
        l['boarding_time'] = boarding_time

    if 'alighting_time' not in l.columns:
        l['alighting_time'] = alighting_time

    l['total_time'] = l['boarding_time'] + l['cost']

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
        links=links, 
        boarding_time=boarding_time, 
        alighting_time=alighting_time
    )
    boarding_edges = boarding_e.values.tolist()
    alighting_edges =  alighting_e.values.tolist()
    transit_edges = transit_e.values.tolist()

    return boarding_edges + transit_edges + alighting_edges


def link_edge_array(links, boarding_time=None, alighting_time=None):
    boarding_e, alighting_e, transit_e = _link_edges(
        links=links, 
        boarding_time=boarding_time, 
        alighting_time=alighting_time
    )
    boarding_edges = boarding_e.values
    alighting_edges =  alighting_e.values
    transit_edges = transit_e.values

    return np.concatenate([boarding_edges,transit_edges,alighting_edges])

def sparse_matrix_with_access_penalty(edges, sources=set(), penalty=1e9):
    nodelist = {e[0] for e in edges}.union({e[1] for e in edges})
    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    penalty_edges = []
    for u, v, w in edges:
        if u in sources:
            w += penalty
        penalty_edges.append((u, v, w))
    coefficients = zip(*((index[u], index[v], w) for u, v, w in penalty_edges))
    row, col, data = coefficients
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index

def paths_from_edges(
    edges,
    sources=None,
    targets=None,
    od_set=None,
    cutoff=np.inf,
    penalty=1e9,
    log=False
):

    reverse = False
    if od_set:
        o_set = {o for o, d in od_set}
        d_set = {d for o, d in od_set}
        if sources is not None:
            sources = [s for s in sources if s in o_set]
        else :
            sources = list(o_set)
        if targets is not None:
            targets = [t for t in targets if t in d_set]
        else:
            targets = list(d_set)
        
    
    if len(sources) > len(targets):
        reverse = True
        if log:
            print(len(sources), 'sources', len(targets), 'targets', 'transposed search')
        sources, targets = targets, sources
        edges = [(b, a, w) for a, b, w in edges]
    elif log :
        print(len(sources), 'sources', len(targets), 'targets', 'direct search')
        
    st = set(sources).union(targets)
    csgraph, node_index = sparse_matrix_with_access_penalty(
        edges, sources=st, penalty=penalty
    )
    
    # INDEX
    source_indices = [node_index[s] for s in sources]
    target_indices = [node_index[t] for t in targets]
    source_index = dict(zip(sources, range(len(sources))))
    index_node = {v: k for k, v in node_index.items()}
    # DIKSTRA

    dist_matrix, predecessors = dijkstra(
        csgraph=csgraph,
        directed=True,
        indices=source_indices,
        return_predecessors=True,
        limit=cutoff+penalty
    )

    dist_matrix = dist_matrix.T[target_indices].T
    df = pd.DataFrame(dist_matrix, index=sources, columns=targets)

    df.columns.name = 'destination'
    df.index.name = 'origin'

    od_index = {(d, o) for o, d in od_set} if reverse else od_set
    if od_set is not None:
        mask = pd.Series(index=od_index, data=1).unstack()
        mask.index.name = 'origin'
        mask.columns.name = 'destination'
        df = df.multiply(mask)

    stack = df.stack()

    stack.name = 'length'
    stack -= penalty
    stack = stack.loc[stack < np.inf]
    odl = stack.reset_index()
    od_list = odl[['origin', 'destination']].values
    path = get_reversed_path if reverse else get_path
    
    
    paths = [
        [
            index_node[i] for i in
            path(predecessors, source_index[o], node_index[d])
        ]
        for o, d in od_list
    ]
    odl['path'] = paths

    if reverse:
        odl[['origin', 'destination']] = odl[['destination', 'origin']]
    return odl


def adjacency_matrix(
    links,
    ntlegs,
    footpaths,
    ntlegs_penalty=1e9,
    boarding_time=None,
    alighting_time=None,
    **kwargs
):
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
    ntlegs_penalty=1e9
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
        csgraph=csgraph,
        directed=True,
        indices=zones,
        return_predecessors=True,
        limit=cutoff + ntlegs_penalty
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
    paths = [
        [indexed_nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])]
        for o, d in od_list
    ]

    los['path'] = paths
    return los


def paths_from_graph(
    csgraph,
    node_index,
    sources,
    targets,
    od_set=None,
    cutoff=np.inf
):
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
        csgraph=csgraph,
        directed=True,
        indices=source_indices,
        return_predecessors=True,
        limit=cutoff
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
    paths = [
        [
            index_node[i] for i in
            path(predecessors, source_index[o], node_index[d])
        ]
        for o, d in od_list
    ]
    odl['path'] = paths

    if reverse:
        odl[['origin', 'destination']] = odl[['destination', 'origin']]
    return odl