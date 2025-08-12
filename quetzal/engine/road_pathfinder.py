import numpy as np
import geopandas as gpd
import pandas as pd

from quetzal.engine.pathfinder_utils import build_index, get_path
from quetzal.engine.msa_utils import (
    get_derivative,
    init_ab_volumes,
    assign_volume,
    jam_time,
    find_phi,
    find_beta,
    get_bfw_auxiliary_flow,
    assign_tracked_volumes_parallel,
    shortest_path,
    init_numba_volumes,
    assign_volume_on_links,
    get_relgap,
    get_sparse_volumes,
)
from typing import List, Dict, Tuple
from quetzal.engine.vdf import default_bpr, free_flow


def init_network(
    sm, method='aon', segments=['car'], time_col='time', access_time='time', ntleg_penalty=10e9, log=False
):
    """
    Initialize the road network for the pathfinder.
    return links and zone_to_road concat with the correct columns
    """
    road_links = sm.road_links.copy()
    zone_to_road = sm.zone_to_road.copy()
    assert not road_links.set_index(['a', 'b']).index.has_duplicates, (
        'there is duplicated road links (same a,b for a link) please remove duplicated'
    )
    assert time_col in road_links.columns, f'time_column: {time_col} not found in road_links.'
    aon = method == 'aon'

    test_zone_to_road_network(road_links, zone_to_road)
    zone_to_road = zone_to_road_preparation(zone_to_road, segments, time_col, access_time, ntleg_penalty, aon, log)
    # create DataFrame with road_links and zone to road
    network = concat_connectors_to_roads(road_links, zone_to_road, segments, time_col, aon, log)
    return network


def init_volumes(sm, od_set=None):
    # apply od_set to self.volumes
    try:
        volumes = sm.volumes.copy()
        if od_set is not None:
            volumes = volumes[volumes.set_index(['origin', 'destination']).index.isin(od_set)]
    except AttributeError:
        print('self.volumes does not exist. od generated with self.zones, od_set')
        if od_set is not None:
            print('od_set ignored')
        zones = sm.zones.index.values
        od_set = []
        for o in zones:
            for d in zones:
                od_set.append((o, d))
        volumes = pd.DataFrame(od_set, columns=['origin', 'destination'])
        volumes['volume'] = 0

    assert len(volumes) > 0
    test_zone_to_road_volumes(volumes, sm.zone_to_road)

    return volumes


def test_zone_to_road_network(road_links, zone_to_road):
    """
    test that all nodes in zone_to_road are in the road_links
    """
    assert 'direction' in zone_to_road, 'need direction with {access | eggress} in zone_to_road'

    access_nodes = set(zone_to_road[zone_to_road['direction'] == 'access']['b'])
    eggress_nodes = set(zone_to_road[zone_to_road['direction'] == 'eggress']['a'])

    nodes_set = set(road_links['a']).union(set(road_links['b']))

    if not access_nodes.issubset(nodes_set):
        missing_nodes = access_nodes - nodes_set
        raise ValueError('access: Some road_nodes in zone_to_road are missing in road_links', missing_nodes)
    if not eggress_nodes.issubset(nodes_set):
        missing_nodes = eggress_nodes - nodes_set
        raise ValueError('eggress: Some road_nodes in zone_to_road are missing in road_links', missing_nodes)


def test_zone_to_road_volumes(volumes, zone_to_road):
    zone_to_road_set = set(zone_to_road['a']).union(set(zone_to_road['b']))
    volumes_set = set(volumes['origin']).union(set(volumes['destination']))
    # check if all zones in volumes_set are in zone_to_road_set
    if not volumes_set.issubset(zone_to_road_set):
        # print elements that are not in zone_to_road_set
        missing_zones = volumes_set - zone_to_road_set
        raise ValueError(
            'Some zones in volumes are not in zone_to_road. add to zones_to_road or drop in volumes.', missing_zones
        )


def zone_to_road_preparation(
    zone_to_road, segments, time_col='time', access_time='time', ntleg_penalty=1e9, aon=False, log=False
):
    # prepare zone_to_road_links to the same format as road_links
    zone_to_road = zone_to_road.copy()
    zone_to_road[time_col] = zone_to_road[access_time] + ntleg_penalty
    if not aon:
        if 'vdf' not in zone_to_road.columns:
            print("vdf not found in zone_to_road columns. Values set to 'free_flow'") if log else None
            zone_to_road['vdf'] = 'free_flow'
        if 'segments' not in zone_to_road.columns:
            print("segments not found in zone_to_road columns. set all segments allow'") if log else None
            zone_to_road['segments'] = [set(segments) for _ in range(len(zone_to_road))]

    return zone_to_road


def concat_connectors_to_roads(road_links, zone_to_road, segments, time_col='time', aon=False, log=False):
    if aon:
        columns = ['a', 'b', time_col]
        links = pd.concat([road_links[columns], zone_to_road[columns]])
        links.index = links.index.astype(str)
        return links
    else:
        if 'vdf' not in road_links.columns:
            print("vdf not found in road_links columns. Values set to 'default_bpr'") if log else None
            road_links['vdf'] = 'default_bpr'
        if 'segments' not in road_links.columns:
            print("segments not found in road_links columns. set all segments allowed on each links'") if log else None
            road_links['segments'] = [set(segments) for _ in range(len(road_links))]

        links = pd.concat([road_links, zone_to_road])
        links.index = links.index.astype(str)

        links['flow'] = 0
        links['auxiliary_flow'] = 0
        for seg in segments:
            links[(seg, 'flow')] = 0
            links[(seg, 'auxiliary_flow')] = 0

        if 'base_flow' not in links.columns:
            links['base_flow'] = 0
        links['base_flow'] = links['base_flow'].fillna(0)
    return links


def assert_vdf_on_links(links, vdf):
    keys = set(links['vdf'])
    missing_vdf = keys - set(vdf.keys())
    assert len(missing_vdf) == 0, 'you should provide methods for the following vdf keys' + str(missing_vdf)


def get_car_los(volumes, links, index, zones, weight_col, num_cores=1):
    """get the car los paths for the given volumes and links"""
    reversed_index = {v: k for k, v in index.items()}
    car_los = volumes[['origin', 'destination', 'origin_sparse', 'destination_sparse']]
    _, predecessors = shortest_path(links, weight_col, index, zones, num_cores=num_cores)
    odlist = list(zip(car_los['origin_sparse'].values, car_los['destination_sparse'].values))

    path_dict = {}
    for origin, destination in odlist:
        path = get_path(predecessors, origin, destination)
        path = [*map(reversed_index.get, path)]
        path_dict[(origin, destination)] = path

    car_los['path'] = car_los.set_index(['origin_sparse', 'destination_sparse']).index.map(path_dict)
    car_los = car_los.drop(columns=['origin_sparse', 'destination_sparse'])

    return car_los


def get_car_los_time(car_los, links, zone_to_road, time_col='jam_time', access_time='time'):
    """
    Calculate the time for each path in car_los with on the road_links and zone_to_road.
    params:
        time_col: time column in road_links
        access_time: time column in zone_to_road
    return:
         car_los with [time, gtime, edge_path] columns
    """
    car_los['edge_path'] = car_los['path'].apply(lambda ls: list(zip(ls, ls[1:])))

    time_dict = {}
    time_dict.update(links.set_index(['a', 'b'])[time_col].to_dict())
    time_dict.update(zone_to_road.set_index(['a', 'b'])[access_time].to_dict())
    car_los['time'] = car_los['edge_path'].apply(lambda ls: sum([*map(time_dict.get, ls)]))

    car_los.loc[car_los['origin'] == car_los['destination'], 'time'] = 0.0
    car_los['gtime'] = car_los['time']
    return car_los


def links_to_expanded_links(links_with_zone_to_road: gpd.GeoDataFrame, u_turns=False):
    df = links_with_zone_to_road.reset_index()
    df1 = df.rename(columns={'index': 'from_link', 'a': 'a1', 'b': 'b1'})
    df2 = df[['index', 'a', 'b']].rename(columns={'index': 'to_link', 'a': 'a2', 'b': 'b2'})

    expanded_links = pd.merge(df1, df2, left_on='b1', right_on='a2')
    if not u_turns:  # remove U turn
        expanded_links = expanded_links[expanded_links['a1'] != expanded_links['b2']].reset_index(drop=True)

    # rename, reorder, clean
    expanded_links = expanded_links.rename(columns={'a1': 'from_node', 'b2': 'to_node', 'a2': 'mid_node'}).drop(
        columns=['b1']
    )
    expanded_links.index.name = 'index'
    expanded_links.index = 'ex_link_' + expanded_links.index.astype(str)

    def _reorder_columns(df, first=['from_link', 'to_link', 'from_node', 'mid_node', 'to_node']):
        all_cols = df.columns.tolist()
        remaining_cols = [col for col in all_cols if col not in first]
        return df[first + remaining_cols]

    expanded_links = _reorder_columns(expanded_links)
    return expanded_links


def expanded_path_to_nodes(path: List[str], links_to_nodes_dict: Dict[str, Tuple[str, str]]) -> List[str]:
    """
    Convert a path of links to a path of nodes, removing duplicate nodes from overlaps.
    First and last elements are zones. The rest are link IDs.
    Example: ['zone1', 'link1', 'link2', 'zone2'] => ['zone1', 'node1', 'node2', 'node3', 'zone2']
    """
    if len(path) <= 2:
        return path  # just zone-to-zone, no links

    nodes = []
    for i, link_id in enumerate(path[1:-1]):
        tup = links_to_nodes_dict.get(link_id)
        if i == 0:
            nodes.extend(tup)  # include both on first
        else:
            nodes.append(tup[1])  # only append the new one

    return [path[0]] + nodes + [path[-1]]


def fix_zone_to_road(expanded_links: gpd.GeoDataFrame, volumes: gpd.geodataframe) -> gpd.GeoDataFrame:
    """
    change zone_to_road expanded links to only the zone index.
    Usefull to do routing on zones when the graph is on links.
    Also, remove links that go through zones and zone-to-zone links.
    """
    links = expanded_links.copy()
    zones_set = set(volumes['origin']).union(set(volumes['destination']))
    # rename zone-to-link
    cond = links['from_node'].isin(zones_set)
    links.loc[cond, 'from_link'] = links.loc[cond, 'from_node']
    # rename link-to-zone
    cond = links['to_node'].isin(zones_set)
    links.loc[cond, 'to_link'] = links.loc[cond, 'to_node']
    # remove links that go through zones.
    links = links[~links['mid_node'].isin(zones_set)]
    # remove zone to zone links.
    links = links[~(links['from_node'].isin(zones_set) & links['to_node'].isin(zones_set))]

    return links


def aon_roadpathfinder(links, volumes, time_col='time', num_cores=1):
    index = build_index(links[['a', 'b']].values)
    links['sparse_a'] = links['a'].apply(lambda x: index.get(x))
    links['sparse_b'] = links['b'].apply(lambda x: index.get(x))
    links = links.set_index(['sparse_a', 'sparse_b'])
    # sparsify volumes keys
    volumes, origins = get_sparse_volumes(volumes, index)

    return get_car_los(volumes, links, index, origins, time_col, num_cores)


def msa_roadpathfinder(
    links,
    volumes,
    segments=['volume'],
    vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
    method='bfw',
    maxiters=10,
    tolerance=0.01,
    log=False,
    time_col='time',
    track_links_list=[],
    num_cores=1,
):
    """
    links: road_network with zone_to_road
    volumes: volumes to assign
    segments: list of segments (in volumes) to assign
    vdf = dict of function for the jam time.
    method = bfw, fw, msa
    maxiters = 10 : number of iteration.
    tolerance = 0.01 : stop condition for RelGap. (in percent)
    log = False : log data on each iteration.
    time_col='freeflow time column'. replace 'time' in vdf
    track_links_list=[] list of link index to track flow at each iteration
    num_cores = 1 : for parallelization.
    """
    # preparation

    assert_vdf_on_links(links, vdf)

    # reindex links to sparse indexes (a,b)
    index = build_index(links[['a', 'b']].values)
    links['sparse_a'] = links['a'].apply(lambda x: index.get(x))
    links['sparse_b'] = links['b'].apply(lambda x: index.get(x))
    links = links.reset_index().set_index(['sparse_a', 'sparse_b'])

    # init numba dict with (a, b, base_flow)
    # in the volumes assignment, start with base_flow (ex: network preloaded with buses on road)
    ab_volumes = init_ab_volumes(links.index)
    # initialization
    links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
    links['jam_time'].fillna(links[time_col], inplace=True)

    # init track links
    _tmp_dict = links[links['index'].isin(track_links_list)]['index'].to_dict()
    sparse_to_links = {v: k for k, v in _tmp_dict.items()}
    tracker = LinksTracker(
        track_links_list=track_links_list,
        links_index=links.index,
        index_dict=index,
        column_dict=sparse_to_links,
        path='road_pathfinder',
    )

    relgap = np.inf
    relgap_list = []
    print('it  |  Phi    |  Rel Gap (%)') if log else None
    # note: first iteration i == 0 is AON to initialized.
    for i in range(maxiters + 1):
        #
        # Routing and assignment
        #

        for seg in segments:
            segment_volumes, origins = get_sparse_volumes(volumes[volumes[seg] > 0], index)
            odv = segment_volumes[['origin_sparse', 'destination_sparse', seg]].values

            segment_links = links[links['segments'].apply(lambda x: seg in x)]  # filter links to allowed segment
            _, pred = shortest_path(segment_links, 'jam_time', index, origins, num_cores=num_cores)

            links[(seg, 'auxiliary_flow')] = assign_volume(odv, pred, ab_volumes.copy())

            if tracker():
                tracker.assign_volume_on_links(ab_volumes, odv, pred, seg)

        flow_cols = [(seg, 'auxiliary_flow') for seg in segments] + ['base_flow']
        links['auxiliary_flow'] = links[flow_cols].sum(axis=1)
        #
        # find Phi, and BFW auxiliary flow modification
        #
        if i == 0:
            phi = 1  # first iteration is AON
            beta = [1, 0, 0]
        elif method == 'bfw':  # if biconjugate: takes the 2 last direction
            links['derivative'] = get_derivative(links, vdf, h=0.001, flow_col='flow', time_col=time_col)
            if i > 2:
                beta = find_beta(links, phi, segments)  # this is the previous phi (phi_-1)
            links = get_bfw_auxiliary_flow(links, i, beta, segments)
            links['auxiliary_flow'] = links[flow_cols].sum(axis=1)
            max_phi = 1 / i**0.5  # limit search space
            phi = find_phi(links, vdf, maxiter=10, bounds=(0, max_phi), time_col=time_col)
        elif method == 'fw':
            max_phi = 1 / i**0.5  # limit search space
            phi = find_phi(links, vdf, maxiter=10, bounds=(0, max_phi), time_col=time_col)
        else:  # msa
            phi = 1 / (i + 2)
        #
        # Update flow and jam_time (frank-wolfe)
        #
        for seg in segments:  # track per segments
            links[(seg, 'flow')] = (1 - phi) * links[(seg, 'flow')] + (phi * links[(seg, 'auxiliary_flow')])
        flow_cols = [(seg, 'flow') for seg in segments] + ['base_flow']
        links['flow'] = links[flow_cols].sum(axis=1)

        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #
        if i > 0:
            relgap = get_relgap(links)
            relgap_list.append(relgap)
            print(f'{i:2}  |  {phi:.3f}  |  {relgap:.8f} ') if log else None

        if tracker():
            tracker.add_weights(phi, beta, relgap, i)
            tracker.save(it=i)

        #
        # Update Time on links
        #
        links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
        links['jam_time'].fillna(links[time_col], inplace=True)
        # skip first iteration (AON asignment) as relgap in -inf
        if relgap <= tolerance:
            print('tolerance reached') if log else None
            break
    #
    # Finish: reindex back and get car_los
    #
    car_los = pd.DataFrame()
    for seg in segments:
        # filter links to allowed segment
        segment_volumes, origins = get_sparse_volumes(volumes[volumes[seg] > 0], index)
        segment_links = links[links['segments'].apply(lambda x: seg in x)]  # filter links to allowed segment
        temp_los = get_car_los(segment_volumes, segment_links, index, origins, 'jam_time', num_cores)
        temp_los['segment'] = seg
        car_los = pd.concat([car_los, temp_los], ignore_index=True)

    links = links.set_index(['a', 'b'])  # go back to original indexes

    # drop columns
    to_drop = ['x1', 'x2', 'result', 'new_flow', 'derivative', 'auxiliary_flow']
    to_drop += [(seg, 's_k-1') for seg in segments]
    to_drop += [(seg, 's_k-2') for seg in segments]
    to_drop += [(seg, 'auxiliary_flow') for seg in segments]
    links = links.drop(columns=to_drop, errors='ignore')

    return links, car_los, relgap_list


from quetzal.io.hdf_io import to_zippedpickle
import os


class LinksTracker:
    def __init__(
        self,
        track_links_list: List[str],
        links_index: List,
        index_dict: Dict[str, int],
        column_dict: Dict[str, int],
        path: str = 'road_pathfinder',
    ):
        self.links_index = links_index
        self.sparse_links_list = [*map(column_dict.get, track_links_list)]
        self.reversed_index_dict = {v: k for k, v in index_dict.items()}
        self.reversed_column_dict = {v: k for k, v in column_dict.items()}
        # export data
        self.weights = pd.DataFrame()
        self.info = pd.DataFrame()
        self.tracked_df: Dict[str, pd.DataFrame] = {}
        self.path = path
        if not os.path.exists(path) & self():  # __call__()
            os.makedirs(path)

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.sparse_links_list) > 0

    def init_df(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.links_index, columns=self.sparse_links_list).fillna(0)

    def assign_volume_on_links(self, ab_volumes, odv, pred, seg):
        volumes = [ab_volumes.copy() for _ in self.sparse_links_list]
        volumes = assign_tracked_volumes_parallel(odv, pred, volumes, self.sparse_links_list)
        self.add_volumes(volumes, seg)

    def add_volumes(self, volumes, seg):
        df = self.init_df()
        for link_index, vols in zip(self.sparse_links_list, volumes):
            df[link_index] = vols
        self.tracked_df[seg] = df

    def add_weights(self, phi, beta, relgap, it):
        new_line = pd.DataFrame([{'iteration': it, 'phi': phi, 'beta': beta, 'relgap': relgap}])
        self.weights = pd.concat([self.weights, new_line], ignore_index=True)

    def add_file_info(self, seg, name, it):
        new_line = pd.DataFrame([{'iteration': it, 'segment': seg, 'file': name}])
        self.info = pd.concat([self.info, new_line], ignore_index=True)

    def save(self, it):
        for seg, df in self.tracked_df.items():
            df = df.rename(index=self.reversed_index_dict, columns=self.reversed_column_dict)
            df.index.name = 'index'
            name = f'tracked_volume_{seg}_{it}.zippedpickle'
            to_zippedpickle(df, os.path.join(self.path, name))
            self.add_file_info(seg, name, it)

        self.weights.to_csv(os.path.join(self.path, 'weights.csv'), index=False)
        self.info.to_csv(os.path.join(self.path, 'info.csv'), index=False)
        self.tracked_df = {}


def expanded_roadpathfinder(
    links,
    volumes,
    zones,
    segments=['volume'],
    vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
    method='bfw',
    maxiters=10,
    tolerance=0.01,
    log=False,
    time_col='time',
    zone_penalty=10e9,
    turn_penalty=10e3,
    track_links_list=[],
    turn_penalties={},
    num_cores=1,
):
    """
    links: road_network with zone_to_road
    volumes: volumes to assign
    zones: zones of the step model.
    segments: list of segments (in volumes) to assign
    vdf = dict of function for the jam time.
    method = bfw, fw, msa, aon
    maxiters = 10 : number of iteration.
    tolerance = 0.01 : stop condition for RelGap. (in percent)
    log = False : log data on each iteration.
    time_col='freeflow time column'. replace 'time' in vdf
    zone_penalty = 10e9 : penalty for zone to road links.
    turn_penalty = 10e3 : penalty for turn links.
    track_links_list=[] list of link index to track flow at each iteration
    turn_penalties: dict of turn penalties {from_link: [to_link]}
    num_cores = 1 : for parallelization.
    """
    assert_vdf_on_links(links, vdf)
    #
    # preparation
    #

    expanded_links = links_to_expanded_links(links, False)
    expanded_links = fix_zone_to_road(expanded_links, volumes)
    expanded_links = expanded_links.rename(columns={'from_link': 'a', 'to_link': 'b'})
    expanded_links = expanded_links[['a', 'b', 'from_node', 'mid_node', 'to_node', 'time', 'segments']]

    # reindex links to sparse indexes
    index = build_index(expanded_links[['a', 'b']].values)
    expanded_links['sparse_a'] = expanded_links['a'].apply(lambda x: index.get(x))
    expanded_links['sparse_b'] = expanded_links['b'].apply(lambda x: index.get(x))
    expanded_links['sparse_index'] = expanded_links['sparse_a']  # original link_a to add cost.
    expanded_links = expanded_links.reset_index().set_index(['sparse_a', 'sparse_b'])

    # turn penalties to sparse index tuple dict {(0,1): turn_penalty}
    turn_penalties = {(index[k], index[v]): turn_penalty for k, values in turn_penalties.items() for v in values}
    expanded_links['turn_penalty'] = expanded_links.index.map(turn_penalties).fillna(0)

    # add sparse index to links
    links['sparse_index'] = links.index.map(index)
    # remove zone to road here. dont need them anymore.
    links = links[~links['sparse_index'].isnull()]
    links = links.reset_index().set_index('sparse_index')  # keep index as column
    links.index = links.index.astype(int)

    # init numba dict with (links, 0)
    ab_volumes = init_numba_volumes(links.index)

    # initialization time
    links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
    links['jam_time'].fillna(links[time_col], inplace=True)
    # init cost on expanded Links
    jam_time_dict = links['jam_time'].to_dict()
    expanded_links['cost'] = expanded_links['sparse_index'].apply(lambda x: jam_time_dict.get(x, zone_penalty))
    expanded_links['cost'] += expanded_links['turn_penalty']

    # init track links
    tracker = LinksTracker(
        track_links_list=track_links_list,
        links_index=links.index,
        index_dict=index,
        column_dict=index,
        path='road_pathfinder',
    )

    relgap = np.inf
    relgap_list = []
    print('it  |  Phi    |  Rel Gap (%)') if log else None
    # note: first iteration i == 0 is AON to initialized.
    for i in range(maxiters + 1):
        #
        # Routing and assignment
        #
        for seg in segments:
            # filter links to allowed segment
            segment_volumes, origins = get_sparse_volumes(volumes[volumes[seg] > 0], index)
            odv = segment_volumes[['origin_sparse', 'destination_sparse', seg]].values

            segment_links = expanded_links[expanded_links['segments'].apply(lambda x: seg in x)]
            _, pred = shortest_path(segment_links, 'cost', index, origins, num_cores=num_cores)
            # assign volume
            links[(seg, 'auxiliary_flow')] = assign_volume_on_links(odv, pred, ab_volumes.copy())
            if tracker():
                tracker.assign_volume_on_links(ab_volumes, odv, pred, seg)

        auxiliary_flow_cols = [(seg, 'auxiliary_flow') for seg in segments] + ['base_flow']
        links['auxiliary_flow'] = links[auxiliary_flow_cols].sum(axis=1)  # for phi and relgap

        #
        # find Phi, and BFW auxiliary flow modification
        #

        if i == 0:
            phi = 1  # first iteration is AON
            beta = [1, 0, 0]
        elif method == 'bfw':  # if biconjugate: takes the 2 last direction
            links['derivative'] = get_derivative(links, vdf, h=0.001, flow_col='flow', time_col=time_col)
            if i > 2:
                beta = find_beta(links, phi, segments)  # this is the previous phi (phi_-1)
            links = get_bfw_auxiliary_flow(links, i, beta, segments)

            links['auxiliary_flow'] = links[auxiliary_flow_cols].sum(axis=1)
            max_phi = 1 / i**0.5  # limit search space
            phi = find_phi(links, vdf, maxiter=10, bounds=(0, max_phi), time_col=time_col)
        elif method == 'fw':
            max_phi = 1 / i**0.5  # limit search space
            phi = find_phi(links, vdf, maxiter=10, bounds=(0, max_phi), time_col=time_col)
        else:  # msa
            phi = 1 / (i + 2)

        #
        # Update flow on links
        #

        for seg in segments:  # track per segments
            links[(seg, 'flow')] = (1 - phi) * links[(seg, 'flow')] + (phi * links[(seg, 'auxiliary_flow')])

        flow_cols = [(seg, 'flow') for seg in segments] + ['base_flow']
        links['flow'] = links[flow_cols].sum(axis=1)

        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #

        if i > 0:
            relgap = get_relgap(links)
            relgap_list.append(relgap)
            print(f'{i:2}  |  {phi:.3f}  |  {relgap:.8f} ') if log else None

        if tracker():
            tracker.add_weights(phi, beta, relgap, i)
            tracker.save(it=i)

        #
        # Update Time on links
        #

        links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
        links['jam_time'].fillna(links[time_col], inplace=True)
        jam_time_dict = links['jam_time'].to_dict()
        expanded_links['cost'] = expanded_links['sparse_index'].apply(lambda x: jam_time_dict.get(x, zone_penalty))
        expanded_links['cost'] += expanded_links['turn_penalty']

        # skip first iteration (AON asignment) as relgap in -inf
        if relgap <= tolerance:
            print('tolerance reached') if log else None
            break
    #
    # Finish: reindex back and get car_los
    #
    car_los = pd.DataFrame()
    for seg in segments:
        # filter links to allowed segment
        segment_volumes, origins = get_sparse_volumes(volumes[volumes[seg] > 0], index)
        segment_links = expanded_links[expanded_links['segments'].apply(lambda x: seg in x)]
        temp_los = get_car_los(segment_volumes, segment_links, index, origins, 'cost', num_cores)
        temp_los['segment'] = seg
        car_los = pd.concat([car_los, temp_los], ignore_index=True)

    # go back to original indexes
    links = links.set_index(['a', 'b'])
    # change path of links to path of nodes
    links_to_nodes_dict = {v: k for k, v in links['index'].to_dict().items()}
    car_los['path'] = car_los['path'].apply(lambda ls: expanded_path_to_nodes(ls, links_to_nodes_dict))

    # drop columns
    to_drop = ['x1', 'x2', 'result', 'new_flow', 'derivative', 'auxiliary_flow']
    to_drop += [(seg, 's_k-1') for seg in segments]
    to_drop += [(seg, 's_k-2') for seg in segments]
    to_drop += [(seg, 'auxiliary_flow') for seg in segments]
    links = links.drop(columns=to_drop, errors='ignore')

    return links, car_los, relgap_list
