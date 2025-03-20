import numpy as np
import pandas as pd
from quetzal.engine.pathfinder_utils import build_index
from quetzal.engine.msa_utils import (
    get_zone_index,
    assign_volume,
    jam_time,
    find_phi,
    get_car_los,
    find_beta,
    get_derivative,
    shortest_path,
)
from quetzal.engine.vdf import default_bpr, free_flow
import numba as nb


class RoadPathFinder:
    def __init__(self, model, method='aon', time_col='time', access_time='time', ntleg_penalty=10e9):
        # self.zones = model.zones.copy()
        self.method = method
        self.road_links = model.road_links.copy()
        self.zone_to_road = model.zone_to_road.copy()
        self.time_col = time_col
        assert not self.road_links.set_index(['a', 'b']).index.has_duplicates, (
            'there is duplicated road links (same a,b for a link)'
        )
        assert self.time_col in self.road_links.columns, f'time_column: {time_col} not found in road_links.'
        try:
            self.volumes = model.volumes.copy()
        except AttributeError:
            print('self.volumes does not exist. od generated with self.zones')
            od_set = []
            for o in model.zones.index.values:
                for d in model.zones.index.values:
                    od_set.append((o, d))
            self.volumes = pd.DataFrame(od_set, columns=['origin', 'destination'])
            self.volumes['volume'] = 0

        assert len(self.volumes) > 0

        self.zone_to_road = zone_to_road_preparation(self.zone_to_road, time_col, access_time, ntleg_penalty)
        # create DataFrame with road_links and zone to road
        self.network = add_connector_to_roads(self.road_links, self.zone_to_road, time_col, method == 'aon')


def zone_to_road_preparation(zone_to_road, time_col='time', access_time='time', ntleg_penalty=1e9):
    # prepare zone_to_road_links to the same format as road_links
    # and initialize it's parameters
    zone_to_road = zone_to_road.copy()
    zone_to_road[time_col] = zone_to_road[access_time]
    zone_to_road.loc[zone_to_road['direction'] == 'access', time_col] += ntleg_penalty
    if 'vdf' not in zone_to_road.columns:
        zone_to_road['vdf'] = 'free_flow'
    zone_to_road = zone_to_road
    return zone_to_road
    # keep track of it (need to substract it in car_los)


def add_connector_to_roads(road_links, zone_to_road, time_col='time', aon=False):
    if not aon:
        if 'vdf' not in road_links.columns:
            road_links['vdf'] = 'default_bpr'
            print("vdf not found in road_links columns. Values set to 'default_bpr'")

        df = pd.concat([road_links, zone_to_road])
        df['flow'] = 0
        df['auxiliary_flow'] = 0
        return df
    else:  # if aon
        columns = ['a', 'b', time_col]
        df = pd.concat([road_links[columns], zone_to_road[columns]])
        return df


def aon_roadpathfinder(df, volumes, od_set, time_col='time', ntleg_penalty=10e9, num_cores=1):
    index = build_index(df[['a', 'b']].values)
    df['sparse_a'] = df['a'].apply(lambda x: index.get(x))
    df['sparse_b'] = df['b'].apply(lambda x: index.get(x))
    df = df.set_index(['sparse_a', 'sparse_b'])

    # Handle volumes and sparsify keys
    if od_set is not None:
        volumes = volumes.set_index(['origin', 'destination']).reindex(od_set).reset_index()
    volumes, zones = get_zone_index(df, volumes, index)

    df['jam_time'] = df[time_col]

    reversed_index = {v: k for k, v in index.items()}
    return get_car_los(volumes, df, index, reversed_index, zones, ntleg_penalty, num_cores)


def msa_roadpathfinder(
    df,
    volumes,
    maxiters=10,
    tolerance=0.01,
    log=False,
    vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
    volume_column='volume_car',
    method='bfw',
    beta=None,
    num_cores=1,
    od_set=None,
    time_col='time',
    ntleg_penalty=10e9,
):
    """
    maxiters = 10 : number of iteration.
    tolerance = 0.01 : stop condition for RelGap. (in percent)
    log = False : log data on each iteration.
    vdf = {'default_bpr': default_bpr, 'free_flow': free_flow} : dict of function for the jam time.
    volume_column='volume_car' : column of self.volumes to use for volume
    method = bfw, fw, msa, aon
    od_set = None. od_set
    beta = None. give constant value foir BFW betas. ex: [0.7,0.2,0.1]
    num_cores = 1 : for parallelization.
    **kwargs: ntleg_penalty=1e9, access_time='time'  for zone to roads.
    """
    # preparation
    nb.set_num_threads(num_cores)

    # reindex links to sparse indexes (a,b)
    index = build_index(df[['a', 'b']].values)
    df['sparse_a'] = df['a'].apply(lambda x: index.get(x))
    df['sparse_b'] = df['b'].apply(lambda x: index.get(x))
    df = df.set_index(['sparse_a', 'sparse_b'])

    # Handle volumes and sparsify keys
    if od_set is not None:
        volumes = volumes.set_index(['origin', 'destination']).reindex(od_set).reset_index()
    volumes, zones = get_zone_index(df, volumes, index)
    odv = volumes[['origin_sparse', 'destination_sparse', volume_column]].values
    volumes_sparse_keys = df.index.values

    # first AON assignment
    _, predecessors = shortest_path(df, time_col, index, zones, num_cores=num_cores)
    ab_volumes = assign_volume(odv, predecessors, volumes_sparse_keys)

    df['auxiliary_flow'] = pd.Series(ab_volumes)
    df['auxiliary_flow'].fillna(0, inplace=True)
    df['flow'] += df['auxiliary_flow']  # do not do += in a cell where the variable is not created! bad
    if maxiters == 0:  # no iteration.
        df['jam_time'] = df[time_col]
    else:
        df['jam_time'] = jam_time(df, vdf, 'flow', time_col=time_col)
        df['jam_time'].fillna(df[time_col], inplace=True)

    rel_gap = []
    phi = 1
    if log:
        print('iteration | Phi |  Rel Gap (%)')

    for i in range(maxiters):
        # CREATE EDGES AND SPARSE MATRIX
        _, predecessors = shortest_path(df, 'jam_time', index, zones, num_cores=num_cores)
        ab_volumes = assign_volume(odv, predecessors, volumes_sparse_keys)
        df['auxiliary_flow'] = pd.Series(ab_volumes)
        df['auxiliary_flow'].fillna(0, inplace=True)
        if method == 'bfw':  # if biconjugate: takes the 2 last direction : direction is flow-auxflow.
            if i >= 2:
                if not beta:  # find beta
                    df['derivative'] = get_derivative(df, vdf, h=0.001, flow_col='flow', time_col=time_col)
                    b = find_beta(df, phi)  # this is the previous phi (phi_-1)
                else:  # beta was provided in function args (debugging)
                    assert sum(beta) == 1, 'beta must sum to 1.'
                    b = beta
                df['auxiliary_flow'] = b[0] * df['auxiliary_flow'] + b[1] * df['s_k-1'] + b[2] * df['s_k-2']

            if i > 0:
                df['s_k-2'] = df['s_k-1']
            df['s_k-1'] = df['auxiliary_flow']

        if method == 'msa':
            phi = 1 / (i + 2)
        else:  # fw or bfw
            phi = find_phi(df.reset_index(drop=True), vdf, 0, 0.8, 10, time_col=time_col)

        # print Relgap
        # modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
        a = np.sum((df['flow']) * df['jam_time'])
        b = np.sum((df['auxiliary_flow']) * df['jam_time'])
        rel_gap.append(100 * (a - b) / a)
        if log:
            print(i, round(phi, 4), round(rel_gap[-1], 3))

        # conventional frank-wolfe
        # df['flow'] + phi*(df['auxiliary_flow'] - df['flow'])  flow +step x direction
        df['flow'] = (1 - phi) * df['flow'] + phi * df['auxiliary_flow']
        df['flow'].fillna(0, inplace=True)

        df['jam_time'] = jam_time(df, vdf, 'flow', time_col=time_col)
        df['jam_time'].fillna(df[time_col], inplace=True)
        if rel_gap[-1] <= tolerance:
            break

    # finish.. format to quetzal object
    # road_links['flow'] = road_links.set_index(['a', 'b']).index.map(df['flow'].to_dict().get)
    # road_links['jam_time'] = road_links.set_index(['a', 'b']).index.map(df['jam_time'].to_dict().get)
    # remove penalty from jam_time
    # keep it.
    # self.road_links['jam_time'] -= self.road_links['penalty']
    reversed_index = {v: k for k, v in index.items()}
    car_los = get_car_los(volumes, df, index, reversed_index, zones, ntleg_penalty, num_cores)
    return df, car_los, rel_gap
