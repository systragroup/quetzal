import numpy as np
import geopandas as gpd
import pandas as pd

from quetzal.engine.pathfinder_utils import build_index, get_path
from quetzal.engine.msa_utils import (
    get_zone_index,
    init_ab_volumes,
    assign_volume,
    jam_time,
    find_phi,
    get_bfw_auxiliary_flow,
    init_track_volumes,
    assign_tracked_volume,
    assign_tracked_volumes_on_links_parallel,
    shortest_path,
    init_numba_volumes,
    assign_volume_on_links,
    get_relgap,
)
from typing import List, Dict, Tuple
from quetzal.engine.vdf import default_bpr, free_flow


def init_network(sm, method='aon', segments=['car'], time_col='time', access_time='time', ntleg_penalty=10e9):
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
    zone_to_road = zone_to_road_preparation(zone_to_road, segments, time_col, access_time, ntleg_penalty, aon)
    # create DataFrame with road_links and zone to road
    network = concat_connectors_to_roads(road_links, zone_to_road, segments, time_col, aon)
    return network


def init_volumes(sm, od_set=None):
    # apply od_set to self.volumes
    try:
        volumes = sm.volumes.copy()
        if od_set is not None:
            volumes = volumes.set_index(['origin', 'destination']).reindex(od_set).reset_index()
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
    return volumes


def zone_to_road_preparation(zone_to_road, segments, time_col='time', access_time='time', ntleg_penalty=1e9, aon=False):
    # prepare zone_to_road_links to the same format as road_links
    zone_to_road = zone_to_road.copy()
    zone_to_road[time_col] = zone_to_road[access_time] + ntleg_penalty
    if not aon:
        if 'vdf' not in zone_to_road.columns:
            print("vdf not found in zone_to_road columns. Values set to 'free_flow'")
            zone_to_road['vdf'] = 'free_flow'
        if 'segments' not in zone_to_road.columns:
            print("segments not found in zone_to_road columns. set all segments allowed on each links'")
            zone_to_road['segments'] = [set(segments) for _ in range(len(zone_to_road))]

    return zone_to_road


def concat_connectors_to_roads(road_links, zone_to_road, segments, time_col='time', aon=False):
    if aon:
        columns = ['a', 'b', time_col]
        links = pd.concat([road_links[columns], zone_to_road[columns]])
        return links
    else:
        if 'vdf' not in road_links.columns:
            print("vdf not found in road_links columns. Values set to 'default_bpr'")
            road_links['vdf'] = 'default_bpr'
        if 'segments' not in road_links.columns:
            print("segments not found in road_links columns. set all segments allowed on each links'")
            road_links['segments'] = [set(segments) for _ in range(len(road_links))]

        links = pd.concat([road_links, zone_to_road])
        links['flow'] = 0
        links['auxiliary_flow'] = 0
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


def links_to_extended_links(links_with_zone_to_road: gpd.GeoDataFrame, u_turns=False):
    df = links_with_zone_to_road.reset_index()
    df1 = df.rename(columns={'index': 'from_link', 'a': 'a1', 'b': 'b1'})
    df2 = df[['index', 'a', 'b']].rename(columns={'index': 'to_link', 'a': 'a2', 'b': 'b2'})

    extended_links = pd.merge(df1, df2, left_on='b1', right_on='a2')
    if not u_turns:  # remove U turn
        extended_links = extended_links[extended_links['a1'] != extended_links['b2']].reset_index(drop=True)

    # rename, reorder, clean
    extended_links = extended_links.rename(columns={'a1': 'from_node', 'b2': 'to_node', 'a2': 'mid_node'}).drop(
        columns=['b1']
    )
    extended_links.index.name = 'index'
    extended_links.index = 'ex_link_' + extended_links.index.astype(str)

    def _reorder_columns(df, first=['from_link', 'to_link', 'from_node', 'mid_node', 'to_node']):
        all_cols = df.columns.tolist()
        remaining_cols = [col for col in all_cols if col not in first]
        return df[first + remaining_cols]

    extended_links = _reorder_columns(extended_links)
    return extended_links


def extended_path_to_nodes(path: List[str], links_to_nodes_dict: Dict[str, Tuple[str, str]]) -> List[str]:
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


def fix_zone_to_road(extended_links: gpd.GeoDataFrame, zones: gpd.geodataframe) -> gpd.GeoDataFrame:
    """
    change zone_to_road extended links to only the zone index.
    Usefull to do routing on zones when the graph is on links.
    Also, remove links that go through zones and zone-to-zone links.
    """
    links = extended_links.copy()
    zones_list = zones.index
    # rename zone-to-link
    cond = links['from_node'].isin(zones_list)
    links.loc[cond, 'from_link'] = links.loc[cond, 'from_node']
    # rename link-to-zone
    cond = links['to_node'].isin(zones_list)
    links.loc[cond, 'to_link'] = links.loc[cond, 'to_node']
    # remove links that go through zones.
    links = links[~links['mid_node'].isin(zones_list)]
    # remove zone to zone links.
    links = links[~(links['from_node'].isin(zones_list) & links['to_node'].isin(zones_list))]

    return links


def aon_roadpathfinder(links, volumes, time_col='time', num_cores=1):
    index = build_index(links[['a', 'b']].values)
    links['sparse_a'] = links['a'].apply(lambda x: index.get(x))
    links['sparse_b'] = links['b'].apply(lambda x: index.get(x))
    links = links.set_index(['sparse_a', 'sparse_b'])

    # sparsify volumes keys
    volumes, zones = get_zone_index(links, volumes, index)

    return get_car_los(volumes, links, index, zones, time_col, num_cores)


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
    method = bfw, fw, msa, aon
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

    #  sparsify volumes keys
    volumes, zones = get_zone_index(links, volumes, index)
    # init numba dict with (a, b, base_flow)
    # in the volumes assignment, start with base_flow (ex: network preloaded with buses on road)
    base_flow = init_ab_volumes(links['base_flow'].to_dict())
    # initialization
    links['flow'] = 0
    links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
    links['jam_time'].fillna(links[time_col], inplace=True)
    # init track links aux_flow dict and flow in links
    track_links = len(track_links_list) > 0
    if track_links:
        _tmp_dict = links[links['index'].isin(track_links_list)]['index'].to_dict()
        track_links_index_dict = {v: k for k, v in _tmp_dict.items()}
        track_links_list = [*map(track_links_index_dict.get, track_links_list)]
        tracked_df = pd.DataFrame(index=links.index)
        tracked_df[[str(idx) for idx in track_links_list]] = 0

    relgap_list = []
    print('it  |  Phi    |  Rel Gap (%)') if log else None
    # note: first iteration i == 0 is AON to initialized.
    for i in range(maxiters + 1):
        #
        # Routing and assignment
        #
        ab_volumes = base_flow.copy()  # init numba dict with (a,b,0)
        if track_links:
            tracked_volumes = init_track_volumes(base_flow, track_links_list)

        for seg in segments:
            filtered = links[links['segments'].apply(lambda x: seg in x)]  # filter links to allowed segment
            _, predecessors = shortest_path(filtered, 'jam_time', index, zones, num_cores=num_cores)
            odv = volumes[['origin_sparse', 'destination_sparse', seg]].values

            ab_volumes = assign_volume(odv, predecessors, ab_volumes)
            if track_links:
                for idx in track_links_list:
                    tracked_volumes[idx] = assign_tracked_volume(odv, predecessors, tracked_volumes[idx], idx)

        links['auxiliary_flow'] = ab_volumes
        #
        # find Phi, and BFW auxiliary flow modification
        #
        if i == 0:
            phi = 1  # first iteration is AON
        elif method == 'bfw':  # if biconjugate: takes the 2 last direction
            links = get_bfw_auxiliary_flow(links, vdf, i, time_col, phi)
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
        links['flow'] = (1 - phi) * links['flow'] + (phi * links['auxiliary_flow'])
        links['flow'].fillna(0, inplace=True)
        if track_links:
            for idx in track_links_list:
                tracked_df[str(idx)] = (1 - phi) * tracked_df[str(idx)] + (phi * pd.Series(tracked_volumes[idx]))
        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #
        if i > 0:
            relgap = get_relgap(links)
            relgap_list.append(relgap)
            print(f'{i:2}  |  {phi:.3f}  |  {relgap:.8f} ') if log else None

        #
        # Update Time on links
        #
        links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
        links['jam_time'].fillna(links[time_col], inplace=True)
        # skip first iteration (AON asignment) as relgap in -inf
        if i > 0:
            if relgap <= tolerance:
                print('tolerance reached')
                break
    #
    # Finish: reindex back and get car_los
    #
    car_los = get_car_los(volumes, links, index, zones, 'jam_time', num_cores)

    if track_links:  # rename with original indexes
        rename_dict = {str(v): k for k, v in track_links_index_dict.items()}
        tracked_df = tracked_df.rename(columns=rename_dict)
        links = pd.merge(links, tracked_df, left_index=True, right_index=True, how='left')
    links = links.set_index(['a', 'b'])  # go back to original indexes

    return links, car_los, relgap_list


def extended_roadpathfinder(
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
    ntleg_penalty=10e9,
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
    track_links_list=[] list of link index to track flow at each iteration
    num_cores = 1 : for parallelization.
    """
    assert_vdf_on_links(links, vdf)

    # preparation
    extended_links = links_to_extended_links(links, False)
    extended_links = fix_zone_to_road(extended_links, zones)
    extended_links = extended_links.rename(columns={'from_link': 'a', 'to_link': 'b'})
    extended_links = extended_links[['a', 'b', 'from_node', 'mid_node', 'to_node', 'time', 'segments']]

    # reindex links to sparse indexes
    index = build_index(extended_links[['a', 'b']].values)
    extended_links['sparse_a'] = extended_links['a'].apply(lambda x: index.get(x))
    extended_links['sparse_b'] = extended_links['b'].apply(lambda x: index.get(x))
    extended_links['sparse_index'] = extended_links['sparse_a']  # original link_a to add cost.
    extended_links = extended_links.reset_index().set_index(['sparse_a', 'sparse_b'])

    # turn penalties to sparse index tuple dict {(0,1): turn_penalty}
    turn_penalties = {
        (index.get(k), index.get(v)): turn_penalty for k, values in turn_penalties.items() for v in values
    }
    extended_links['turn_penalty'] = extended_links.index.map(turn_penalties).fillna(0)

    # add sparse index to links
    links['sparse_index'] = links.index.map(index)
    # remove zone to road here. dont need them anymore.
    links = links[~links['sparse_index'].isnull()]
    links = links.reset_index().set_index('sparse_index')  # keep index as column
    links.index = links.index.astype(int)

    #  sparsify volumes keys
    volumes, zones = get_zone_index(extended_links, volumes, index)

    # init numba dict with (links, base_flow)
    # in the volumes assignment, start with base_flow (ex: network preloaded with buses on road)
    base_flow = init_numba_volumes(links['base_flow'].to_dict())

    # initialization of flow and time
    links['flow'] = 0
    links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
    links['jam_time'].fillna(links[time_col], inplace=True)
    # init cost on extended Links
    jam_time_dict = links['jam_time'].to_dict()
    extended_links['cost'] = extended_links['sparse_index'].apply(lambda x: jam_time_dict.get(x, ntleg_penalty))
    extended_links['cost'] += extended_links['turn_penalty']

    # init track links aux_flow dict and flow in links
    track_links = len(track_links_list) > 0
    if track_links:
        track_links_list = [*map(index.get, track_links_list)]
        tracked_df = pd.DataFrame(index=links.index, columns=track_links_list).fillna(0)
    relgap = np.inf
    relgap_list = []
    print('it  |  Phi    |  Rel Gap (%)') if log else None
    # note: first iteration i == 0 is AON to initialized.
    for i in range(maxiters + 1):
        #
        # Routing and assignment
        #
        # init numba dict with (a,b,0)
        links_volumes = base_flow.copy()
        if track_links:
            tracked_volumes = [base_flow.copy() for _ in track_links_list]
        # loop ons segments
        for seg in segments:
            # filter links to allowed segment
            filtered = extended_links[extended_links['segments'].apply(lambda x: seg in x)]
            _, pred = shortest_path(filtered, 'cost', index, zones, num_cores=num_cores)
            odv = volumes[['origin_sparse', 'destination_sparse', seg]].values
            # assign volume
            links_volumes = assign_volume_on_links(odv, pred, links_volumes)
            if track_links:
                tracked_volumes = assign_tracked_volumes_on_links_parallel(odv, pred, tracked_volumes, track_links_list)

        links['auxiliary_flow'] = links_volumes
        #
        # find Phi, and BFW auxiliary flow modification
        #
        if i == 0:
            phi = 1  # first iteration is AON
        elif method == 'bfw':  # if biconjugate: takes the 2 last direction
            links = get_bfw_auxiliary_flow(links, vdf, i, time_col, phi)
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
        links['flow'] = (1 - phi) * links['flow'] + (phi * links['auxiliary_flow'])
        links['flow'].fillna(0, inplace=True)
        if track_links:
            for idx, aux_flow in zip(track_links_list, tracked_volumes):
                tracked_df[idx] = (1 - phi) * tracked_df[idx] + (phi * pd.Series(aux_flow))
        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #
        if i > 0:
            relgap = get_relgap(links)
            relgap_list.append(relgap)
            print(f'{i:2}  |  {phi:.3f}  |  {relgap:.8f} ') if log else None

        #
        # Update Time on links
        #
        links['jam_time'] = jam_time(links, vdf, 'flow', time_col=time_col)
        links['jam_time'].fillna(links[time_col], inplace=True)
        jam_time_dict = links['jam_time'].to_dict()
        extended_links['cost'] = extended_links['sparse_index'].apply(lambda x: jam_time_dict.get(x, ntleg_penalty))
        extended_links['cost'] += extended_links['turn_penalty']
        # skip first iteration (AON asignment) as relgap in -inf
        if relgap <= tolerance:
            print('tolerance reached')
            break
    #
    # Finish: reindex back and get car_los
    #

    car_los = get_car_los(volumes, extended_links, index, zones, 'cost', num_cores)

    if track_links:  # put tracked flow as columns in links
        reversed_index = {v: k for k, v in index.items()}
        tracked_df = tracked_df.rename(columns=reversed_index)
        links = pd.merge(links, tracked_df, left_index=True, right_index=True, how='left')

    links = links.set_index(['a', 'b'])  # go back to original indexes
    # change path of links to path of nodes
    links_to_nodes_dict = {v: k for k, v in links['index'].to_dict().items()}
    car_los['path'] = car_los['path'].apply(lambda ls: extended_path_to_nodes(ls, links_to_nodes_dict))

    return links, car_los, relgap_list
