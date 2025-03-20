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

    # initialization
    df['jam_time'] = df[time_col]
    df['flow'] = 0
    relgap_list = []

    print('iteration | Phi |  Rel Gap (%)') if log else None
    # note: first iteration i == 0 is AON to initialized.
    for i in range(maxiters + 1):
        # Routing and assignment
        _, predecessors = shortest_path(df, 'jam_time', index, zones, num_cores=num_cores)
        ab_volumes = assign_volume(odv, predecessors, volumes_sparse_keys)
        df['auxiliary_flow'] = pd.Series(ab_volumes)
        df['auxiliary_flow'].fillna(0, inplace=True)
        #
        # ffnd Phi, and BFW auxiliary flow modification
        #
        if i == 0:
            phi = 1  # first iteration is AON
        elif method == 'bfw':  # if biconjugate: takes the 2 last direction
            df = get_bfw_auxiliary_flow(df, vdf, i, time_col, phi)
            phi = find_phi(df, vdf, 0, 0.8, 10, time_col=time_col)
        elif method == 'fw':
            phi = find_phi(df, vdf, 0, 0.8, 10, time_col=time_col)
        else:  # msa
            phi = 1 / (i + 2)
        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #
        if i > 0:
            relgap = get_relgap(df)
            relgap_list.append(relgap)
            print(i, round(phi, 4), round(relgap, 3)) if log else None
        #
        # Update flow and jam_time (frank-wolfe)
        #
        df['flow'] = (1 - phi) * df['flow'] + (phi * df['auxiliary_flow'])
        df['flow'].fillna(0, inplace=True)
        df['jam_time'] = jam_time(df, vdf, 'flow', time_col=time_col)
        df['jam_time'].fillna(df[time_col], inplace=True)
        # skip first iteration (AON asignment) as relgap in -inf
        if i > 0:
            if relgap <= tolerance:
                print('tolerance reached')
                break

    reversed_index = {v: k for k, v in index.items()}
    car_los = get_car_los(volumes, df, index, reversed_index, zones, ntleg_penalty, num_cores)
    return df, car_los, relgap_list


def get_relgap(df) -> float:
    # modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
    a = np.sum((df['flow']) * df['jam_time'])  # currentFlow x currentCost
    b = np.sum((df['auxiliary_flow']) * df['jam_time'])  # AON_flow x currentCost
    return 100 * (a - b) / a


def get_bfw_auxiliary_flow(df, vdf, i, time_col, prev_phi) -> pd.DataFrame:
    if i > 2:
        df['derivative'] = get_derivative(df, vdf, h=0.001, flow_col='flow', time_col=time_col)
        beta = find_beta(df, prev_phi)  # this is the previous phi (phi_-1)
        df['auxiliary_flow'] = beta[0] * df['auxiliary_flow'] + beta[1] * df['s_k-1'] + beta[2] * df['s_k-2']
    if i > 1:
        df['s_k-2'] = df['s_k-1']
    df['s_k-1'] = df['auxiliary_flow']
    return df
