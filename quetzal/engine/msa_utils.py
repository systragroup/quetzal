from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path, parallel_dijkstra
import numba as nb
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar


def get_zone_index(links: pd.DataFrame, v: pd.DataFrame, index: dict[str, int]) -> Tuple[pd.DataFrame, list[int]]:
    # return volumes with [origin_sparse, destination_sparse] anz zone_list (sparse zones)
    seta = set(links['a'])
    setb = set(links['b'])
    v = v.loc[v['origin'].isin(seta) & v['destination'].isin(setb)]

    sources = set(v['origin']).union(v['destination'])

    pole_list = sorted(list(sources))  # fix order
    source_list = [zone for zone in pole_list if zone in sources]

    zones_list = [index[zone] for zone in source_list]
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    v['origin_sparse'] = v['origin'].apply(zone_index.get)
    v['destination_sparse'] = v['destination'].apply(index.get)

    return v, zones_list


def get_sparse_matrix(edges, index):
    row, col, data = edges.T
    nlen = len(index)
    return csr_matrix((data, (row, col)), shape=(nlen, nlen))


def shortest_path(
    links: pd.DataFrame, weight_col: str, index: dict[str, int], origins: list[int], num_cores: int
) -> Tuple[np.ndarray, np.ndarray]:
    # from  df index(a,b) and weight col, return predecessor.
    edges = links[weight_col].reset_index().values  # build the edges again, useless
    sparse = get_sparse_matrix(edges, index=index)
    # shortest path
    weight_matrix, predecessors = parallel_dijkstra(
        sparse, directed=True, indices=origins, return_predecessors=True, num_core=num_cores
    )
    return weight_matrix, predecessors


def jam_time(links: pd.DataFrame, vdf, flow: str = 'flow', time_col: str = 'time') -> pd.Series:
    # vdf is a {function_name: function } to apply on links
    # 3 types of functions are accepted:
    # 1) str that will be evaluated with pandas.eval
    # 2) python function of the form func(df, flow_col, time_col).
    # 3) Jit function with single input matrix ['alpha','beta','limit',flow_col ,time_col,'penalty','capacity']]
    keys = set(links['vdf'])
    for key in keys:
        if isinstance(vdf[key], str):
            str_expression = vdf[key].replace('flow', flow).replace('time', time_col)
            links.loc[links['vdf'] == key, 'result'] = links.loc[links['vdf'] == key].eval(str_expression)
        elif type(vdf[key]).__name__ == 'function':  # normal python function.
            links.loc[links['vdf'] == key, 'result'] = vdf[key](links.loc[links['vdf'] == key], flow, time_col)
        else:  # numba function.
            links.loc[links['vdf'] == key, 'result'] = vdf[key](
                links.loc[links['vdf'] == key, ['alpha', 'beta', 'limit', flow, time_col, 'penalty', 'capacity']].values
            )

    return links['result']


def z_prime(links, vdf, phi, **kwargs):
    # min sum(on links) integral from 0 to flow + φΔ of Cost(f) df
    # using a single trapez (ok if phi small), which is the case after <10 iteration.
    # This give not perfect phi to begin with, but thats ok.
    # approx  ( Cost(flow) + Cost(flow + φΔ)  ) x φΔ /2
    # Δ = links['auxiliary_flow'] - links['flow']
    delta = (links['auxiliary_flow'] - links['flow']).values
    links['new_flow'] = delta * phi + links['flow']
    cost_del = jam_time(links, vdf=vdf, flow='new_flow', **kwargs).values
    cost_flow = links['jam_time'].values
    z = delta * phi * (cost_flow + cost_del) * 0.5
    return np.ma.masked_invalid(z).sum()


def find_phi(links, vdf, maxiter=10, tol=1e-4, bounds=(0, 1), **kwargs):
    return minimize_scalar(
        lambda x: z_prime(links, vdf, x, **kwargs),
        bounds=bounds,
        method='Bounded',
        tol=tol,
        options={'maxiter': maxiter},
    ).x


def get_relgap(links) -> float:
    # modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
    a = np.sum((links['flow']) * links['jam_time'])  # currentFlow x currentCost
    b = np.sum((links['auxiliary_flow']) * links['jam_time'])  # AON_flow x currentCost
    return 100 * (a - b) / a


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_volume(odv, predecessors, volumes):
    # volumes is a numba dict with all the key initialized
    for i in range(len(odv)):  # nb.prange(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            path = list(zip(path[:-1], path[1:]))
            for key in path:
                volumes[key] += v
    return volumes


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volume(odv, predecessors, volumes, track_index):
    # volumes is a numba dict with all the key initialized
    for i in range(len(odv)):  # nb.prange(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            path = list(zip(path[:-1], path[1:]))
            if track_index in path:
                for key in path:
                    volumes[key] += v
    return volumes


def init_ab_volumes(indexes: List[Tuple]) -> Dict[Tuple, float]:
    numba_volumes = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.types.int64, 2), value_type=nb.types.float64)
    for key in indexes:
        numba_volumes[key] = 0
    return numba_volumes


def init_extended_track_volumes(base_flow: Dict[int, float], track_links_list: List[int]) -> List[Dict[int, float]]:
    numba_volumes = base_flow.copy()
    for key in numba_volumes.keys():
        numba_volumes[key] = 0
    return [numba_volumes.copy() for _ in track_links_list]


def find_beta(links, phi_1, segments):
    # The Stiff is Moving - Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment from Mitradjieva maria
    b = [0, 0, 0]

    s_k_1 = links[[(seg, 's_k-1') for seg in segments]].sum(axis=1)
    s_k_2 = links[[(seg, 's_k-2') for seg in segments]].sum(axis=1)
    aux = links[[(seg, 'auxiliary_flow') for seg in segments]].sum(axis=1)
    flow = links['flow']
    derivative = links['derivative']

    dk_1 = s_k_1 - flow
    dk_2 = phi_1 * s_k_1 + (1 - phi_1) * s_k_2 - flow
    dk = aux - flow
    # put a try here except mu=0 if we have a division by 0...
    mu = -sum(dk_2 * derivative * dk) / sum(dk_2 * derivative * (s_k_2 - s_k_1))
    mu = max(0, mu)  # beta_k >=0
    # same try here.
    nu = -sum(dk_1 * derivative * dk) / sum(dk_1 * derivative * dk_1) + (mu * phi_1 / (1 - phi_1))
    nu = max(0, nu)
    b[0] = 1 / (1 + mu + nu)
    b[1] = nu * b[0]
    b[2] = mu * b[0]
    return b


def get_bfw_auxiliary_flow(links, i, prev_phi, segments) -> pd.DataFrame:
    if i > 2:
        b = find_beta(links, prev_phi, segments)  # this is the previous phi (phi_-1)
        for seg in segments:  # track per segments
            col = (seg, 'auxiliary_flow')
            links[col] = b[0] * links[col] + b[1] * links[(seg, 's_k-1')] + b[2] * links[(seg, 's_k-2')]

    for seg in segments:
        if i > 1:
            links[(seg, 's_k-2')] = links[(seg, 's_k-1')]
        links[(seg, 's_k-1')] = links[(seg, 'auxiliary_flow')]

    return links


def get_derivative(links, vdf, flow_col='flow', h=0.001, **kwargs):
    links['x1'] = links[flow_col] + h
    links['x2'] = links[flow_col] - h
    links['x1'] = jam_time(links, vdf, flow='x1', **kwargs)
    links['x2'] = jam_time(links, vdf, flow='x2', **kwargs)
    return (links['x1'] - links['x2']) / (2 * h)


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_volume_on_links(odv, predecessors, volumes):
    # volumes is a numba dict with all the key initialized
    for i in range(len(odv)):  # nb.prange(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            # our nodes are alreadt links in the original Graph
            path = get_node_path(predecessors, origin, destination)
            for key in path:
                volumes[key] += v
    return volumes


def init_numba_volumes(indexes: List[int]) -> Dict[int, float]:
    numba_volumes = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)
    for key in indexes:
        numba_volumes[key] = 0
    return numba_volumes


@nb.njit(locals={'predecessors': nb.int32[:, ::1]}, parallel=True)
def assign_tracked_volumes_on_links_parallel(odv, predecessors, volumes_list, key_list):
    for i in nb.prange(len(key_list)):
        assign_tracked_volume_on_links(odv, predecessors, volumes_list[i], key_list[i])
    return volumes_list


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volume_on_links(odv, predecessors, volumes, track_index):
    # volumes is a numba dict with all the key initialized
    for i in range(len(odv)):  # nb.prange(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            # our nodes are alreadt links in the original Graph
            path = get_node_path(predecessors, origin, destination)
            if track_index in path:
                for key in path:
                    volumes[key] += v
    return volumes
