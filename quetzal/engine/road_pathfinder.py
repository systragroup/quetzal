import numpy as np
import pandas as pd
from quetzal.engine.pathfinder_utils import build_index
from quetzal.engine.msa_utils import (
    get_zone_index,
    init_ab_volumes,
    assign_volume,
    jam_time,
    find_phi,
    get_car_los,
    find_beta,
    get_derivative,
    init_track_volumes,
    assign_tracked_volume,
    shortest_path,
)
from quetzal.engine.vdf import default_bpr, free_flow
import numba as nb


def init_network(sm, method='aon', segments=['car'], time_col='time', access_time='time', ntleg_penalty=10e9):
    #
    #
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
    zone_to_road[time_col] = zone_to_road[access_time]
    zone_to_road.loc[zone_to_road['direction'] == 'access', time_col] += ntleg_penalty
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


def aon_roadpathfinder(links, volumes, time_col='time', ntleg_penalty=10e9, num_cores=1):
    index = build_index(links[['a', 'b']].values)
    links['sparse_a'] = links['a'].apply(lambda x: index.get(x))
    links['sparse_b'] = links['b'].apply(lambda x: index.get(x))
    links = links.set_index(['sparse_a', 'sparse_b'])

    # sparsify volumes keys
    volumes, zones = get_zone_index(links, volumes, index)

    links['jam_time'] = links[time_col]

    reversed_index = {v: k for k, v in index.items()}
    return get_car_los(volumes, links, index, reversed_index, zones, ntleg_penalty, num_cores)


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
    ntleg_penalty=10e9,
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
    nb.set_num_threads(num_cores)
    track_links = len(track_links_list) > 0
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

    if track_links:
        tmp_dict = links[links['index'].isin(track_links_list)]['index'].to_dict()
        track_links_index_dict = {v: k for k, v in tmp_dict.items()}
        track_links_list = [*map(track_links_index_dict.get, track_links_list)]
        for idx in track_links_list:
            links[f'flow_{idx}'] = 0

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
                links[f'flow_{idx}'] = (1 - phi) * links[f'flow_{idx}'] + (phi * pd.Series(tracked_volumes[idx]))
        #
        # Get relGap. Skip first iteration (AON and relgap = -inf)
        #
        if i > 0:
            relgap = get_relgap(links)
            relgap_list.append(relgap)
            print(f'{i:2}  |  {phi:.3f}  |  {relgap:.4f} ') if log else None

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

    reversed_index = {v: k for k, v in index.items()}
    car_los = get_car_los(volumes, links, index, reversed_index, zones, ntleg_penalty, num_cores)
    links = links.set_index(['a', 'b'])  # go back to original indexes

    if track_links:  # rename with original indexes
        rename_dict = {f'flow_{v}': f'flow_{k}' for k, v in track_links_index_dict.items()}
        links = links.rename(columns=rename_dict)

    return links, car_los, relgap_list


def get_relgap(links) -> float:
    # modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
    a = np.sum((links['flow']) * links['jam_time'])  # currentFlow x currentCost
    b = np.sum((links['auxiliary_flow']) * links['jam_time'])  # AON_flow x currentCost
    return 100 * (a - b) / a


def get_bfw_auxiliary_flow(links, vdf, i, time_col, prev_phi) -> pd.DataFrame:
    if i > 2:
        links['derivative'] = get_derivative(links, vdf, h=0.001, flow_col='flow', time_col=time_col)
        b = find_beta(links, prev_phi)  # this is the previous phi (phi_-1)
        links['auxiliary_flow'] = b[0] * links['auxiliary_flow'] + b[1] * links['s_k-1'] + b[2] * links['s_k-2']
    if i > 1:
        links['s_k-2'] = links['s_k-1']
    links['s_k-1'] = links['auxiliary_flow']
    return links
