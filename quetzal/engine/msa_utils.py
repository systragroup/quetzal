from typing import List, Tuple, Dict, Union
import pandas as pd
import polars as pl
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path, fast_dijkstra

import numba as nb
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar
import re


def get_sparse_volumes(volumes: pd.DataFrame, index: dict[str, int]):
    sources = set(volumes['origin'])
    sources = sorted(list(sources))  # fix order
    origins = [*map(index.get, sources)]
    zone_index = dict(zip(sources, range(len(sources))))

    volumes['origin_sparse'] = volumes['origin'].apply(zone_index.get)
    # could do this once before to save 100ms/it.
    volumes['destination_sparse'] = volumes['destination'].apply(index.get)
    return volumes, origins


def get_sparse_matrix(edges, index):
    row, col, data = edges.T
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    nlen = len(index)
    return csr_matrix((data, (row, col)), shape=(nlen, nlen))


def shortest_path(
    links: pd.DataFrame, weight_col: str, index: dict[str, int], origins: list[int], num_cores: int
) -> Tuple[np.ndarray, np.ndarray]:
    # from  df index(a,b) and weight col, return predecessor.
    edges = links[weight_col].reset_index().values  # build the edges again, useless
    csgraph = get_sparse_matrix(edges, index=index)
    # shortest path
    weight_matrix, predecessors = fast_dijkstra(
        csgraph, indices=origins, return_predecessors=True, num_threads=num_cores
    )
    return weight_matrix, predecessors


# convert string to Polars expression
def pl_expr_from_str(expr_str: str) -> pl.Expr:
    tokens = set(re.findall(r'[A-Za-z_]\w*', expr_str))
    for tok in tokens:
        expr_str = re.sub(rf'\b{tok}\b', f'pl.col("{tok}")', expr_str)
    return eval(expr_str, {'pl': pl})


def apply_segment_cost(links: Union[pd.DataFrame, pl.DataFrame], str_expression: str) -> np.ndarray:
    # simply evealute string expression on dataframe for a pandas or polars dataframe.
    if isinstance(links, pd.DataFrame):
        return links.eval(str_expression).values
    else:
        expr = pl_expr_from_str(str_expression)
        return links.select(expr).to_series().to_numpy()


def jam_time(links: pl.DataFrame, vdf, flow: str = 'flow', time_col: str = 'time') -> np.ndarray:
    # using polars is almost 30X faster than pandas here
    expr = pl.when(False).then(None)
    for key, str_expression in vdf.items():
        str_expression = str_expression.replace('flow', flow).replace('time', time_col)
        expr = expr.when(pl.col('vdf') == key).then(pl_expr_from_str(str_expression))
        # return a numpy array with filled with time_col
    return links.select(expr.fill_nan(pl.col(time_col))).to_series().to_numpy()


def z_prime(plinks: pl.DataFrame, segments, vdf, cost_functions, phi, **kwargs):
    # min sum(on links) integral from 0 to flow + φΔ of Cost(f) df
    # using a single trapez (ok if phi small), which is the case after <10 iteration.
    # This give not perfect phi to begin with, but thats ok.
    # approx  ( Cost(flow) + Cost(flow + φΔ)  ) x φΔ /2
    # Δ = links['auxiliary_flow'] - links['flow']

    # compute a new jam_time for a given phi
    plinks = plinks.with_columns((pl.col('flow') * (1 - phi) + pl.col('auxiliary_flow') * phi).alias('new_flow'))
    plinks = plinks.with_columns(pl.Series(jam_time(plinks, vdf=vdf, flow='new_flow', **kwargs)).alias('jam_time'))
    tot_z = 0
    for seg in segments:
        cost_delta = apply_segment_cost(plinks, cost_functions.get(seg))  # get cost for new jam_time
        delta = (plinks[str((seg, 'auxiliary_flow'))] - plinks[str((seg, 'flow'))]).to_numpy()  # str(tuple) for polars
        current_cost = plinks[str((seg, 'cost'))].to_numpy()  # current cost
        z = delta * phi * (current_cost + cost_delta) * 0.5
        # tot_z += np.ma.masked_invalid(z).sum()
        tot_z += z.sum()
    return tot_z


def find_phi(plinks: pl.DataFrame, segments, vdf, cost_functions, maxiter=10, tol=1e-4, bounds=(0, 1), **kwargs):
    return minimize_scalar(
        lambda x: z_prime(plinks=plinks, segments=segments, vdf=vdf, cost_functions=cost_functions, phi=x, **kwargs),
        bounds=bounds,
        method='Bounded',
        tol=tol,
        options={'maxiter': maxiter},
    ).x


def get_relgap(links: pd.DataFrame, segments: list[str]) -> float:
    # modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
    # NOTE: base_flow is ignored now while it was considered before. they dont move and they dont have cost, so I think its ok
    flow_cost = 0
    aux_cost = 0
    for seg in segments:
        flow_cost += (links[(seg, 'flow')] * links[(seg, 'cost')]).sum()  # currentFlow x currentCost
        aux_cost += (links[(seg, 'auxiliary_flow')] * links[(seg, 'cost')]).sum()  # AON_flow x currentCost
    return 100 * (flow_cost - aux_cost) / flow_cost


@nb.njit(locals={'predecessors': nb.int32[:, ::1]}, parallel=True)  # parallel=> not thread safe. do not!
def assign_volume_parallel(odv, predecessors, volumes, num_cores=1):
    # volumes is a numba dict with all the key initialized
    if num_cores == 1:
        return assign_volume(odv, predecessors, volumes)
    nb.set_num_threads(num_cores)
    odv_mat = np.array_split(odv, num_cores)
    volumes_mat = [volumes.copy() for _ in range(num_cores)]
    for j in nb.prange(num_cores):
        assign_volume(odv_mat[j], predecessors, volumes_mat[j])
    for d in volumes_mat:
        for k in d:
            volumes[k] += d[k]

    return volumes


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


def init_ab_volumes(indexes: List[Tuple]) -> Dict[Tuple, float]:
    numba_volumes = nb.typed.Dict.empty(key_type=nb.types.UniTuple(nb.types.int64, 2), value_type=nb.types.float64)
    for key in indexes:
        numba_volumes[key] = 0
    return numba_volumes


def init_expanded_track_volumes(base_flow: Dict[int, float], track_links_list: List[int]) -> List[Dict[int, float]]:
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


def get_bfw_auxiliary_flow(links, i, b, segments) -> pd.DataFrame:
    if i > 2:
        for seg in segments:  # track per segments
            col = (seg, 'auxiliary_flow')
            links[col] = b[0] * links[col] + b[1] * links[(seg, 's_k-1')] + b[2] * links[(seg, 's_k-2')]

    for seg in segments:
        if i > 1:
            links[(seg, 's_k-2')] = links[(seg, 's_k-1')]
        links[(seg, 's_k-1')] = links[(seg, 'auxiliary_flow')]

    return links


def get_derivative(links: pl.DataFrame, vdf, flow_col='flow', h=0.001, **kwargs):
    x1 = jam_time(links.with_columns((pl.col(flow_col) + h).alias('x1')), vdf=vdf, flow='x1', **kwargs)
    x2 = jam_time(links.with_columns((pl.col(flow_col) - h).alias('x2')), vdf=vdf, flow='x2', **kwargs)
    return (x1 - x2) / (2 * h)


@nb.njit(locals={'predecessors': nb.int32[:, ::1]}, parallel=True)  # parallel=> not thread safe. do not!
def assign_volume_on_links_parallel(odv, predecessors, volumes, num_cores=1):
    # volumes is a numba dict with all the key initialized
    if num_cores == 1:
        return assign_volume_on_links(odv, predecessors, volumes)
    nb.set_num_threads(num_cores)
    odv_mat = np.array_split(odv, num_cores)
    volumes_mat = [volumes.copy() for _ in range(num_cores)]
    for j in nb.prange(num_cores):
        assign_volume_on_links(odv_mat[j], predecessors, volumes_mat[j])
    for d in volumes_mat:
        for k in d:
            volumes[k] += d[k]

    return volumes


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
