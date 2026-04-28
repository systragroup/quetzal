import bisect
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
import numba as nb


def time_footpaths(links, footpaths):
    footpaths['model_index'] = footpaths.index

    left = pd.merge(
        links[['b', 'arrival_time']],
        footpaths[['a', 'b', 'duration', 'model_index']],
        left_on='b',
        right_on='a',
        suffixes=['_link', ''],
    )

    left['ready_time'] = left['arrival_time'] + left['duration']
    right = links[['a', 'departure_time']]

    # EFFICIENT MERGE GROUPBY FIRST
    left['ka'] = left['b']
    left['kb'] = left['ready_time']

    right['ka'] = right['a']
    right['kb'] = right['departure_time']

    a_intindex = {}
    i = 0
    for k in set(left['ka']).union(set(right['ka'])):
        i += 1
        a_intindex[k] = i * 1e9

    b_intindex = {}
    i = 0
    for k in sorted(set(left['kb']).union(set(right['kb']))):
        i += 1
        b_intindex[k] = i

    left['a_intindex'] = [a_intindex[v] for v in left['ka']]
    right['a_intindex'] = [a_intindex[v] for v in right['ka']]

    left['b_intindex'] = [b_intindex[v] for v in left['kb']]
    right['b_intindex'] = [b_intindex[v] for v in right['kb']]

    left['key'] = left['a_intindex'] + left['b_intindex']
    right['key'] = right['a_intindex'] + right['b_intindex']

    left.sort_values('key', inplace=True)
    right.sort_values('key', inplace=True)

    transfers = pd.merge_asof(
        left[['a', 'b', 'key', 'arrival_time', 'duration', 'model_index']],
        right[['departure_time', 'key']],
        on='key',
        direction='forward',
        tolerance=1e6,
    )
    # END MERGE GROUPBY

    # enables the earliest departure from b, for each arrival at a
    transfers.sort_values('departure_time', ascending=True, inplace=True)
    transfers = transfers.groupby(['a', 'b', 'arrival_time'], as_index=False).first()

    # enables the latest arrival at a, for each departure from b
    transfers.sort_values('arrival_time', ascending=False, inplace=True)
    transfers = transfers.groupby(['a', 'b', 'departure_time'], as_index=False).first()

    transfers[['departure_time', 'arrival_time']] = transfers[['arrival_time', 'departure_time']]
    transfers['str'] = transfers['model_index'].astype(str)
    transfers['str'] += '_' + transfers['departure_time'].astype(str)
    transfers['str'] += '_' + transfers['arrival_time'].astype(str)
    transfers['csa_index'] = 'footpath_' + transfers['str']
    transfers['trip_id'] = 'footpath_trip_' + transfers['str']
    columns = ['a', 'b', 'departure_time', 'arrival_time', 'trip_id', 'csa_index', 'model_index']
    return transfers[columns]


def time_zone_to_transit(links, zone_to_transit, reindex=False, step=None):
    ztt = zone_to_transit
    ztt['model_index'] = ztt.index
    # access
    left = ztt.loc[ztt['direction'] == 'access']
    df = pd.merge(
        left[['a', 'b', 'time', 'model_index']],
        links[['a', 'b', 'departure_time']],
        left_on='b',
        right_on='a',
        suffixes=['_ztt', '_link'],
    )
    df['arrival_time'] = df['departure_time']
    df['departure_time'] = df['arrival_time'] - df['time']
    df['a'] = df['a_ztt']
    df['b'] = df['b_ztt']
    df['direction'] = 'access'
    access = df.copy()

    # egress
    left = ztt.loc[ztt['direction'] != 'access']
    df = pd.merge(
        ztt[['a', 'b', 'time', 'model_index']],
        links[['a', 'b', 'arrival_time']],
        left_on='a',
        right_on='b',
        suffixes=['_ztt', '_link'],
    )
    df['departure_time'] = df['arrival_time']
    df['arrival_time'] = df['departure_time'] + df['time']
    df['a'] = df['a_ztt']
    df['b'] = df['b_ztt']
    df['direction'] = 'egress'
    egress = df.copy()
    df = pd.concat([access, egress])

    df['str'] = range(len(df))
    if reindex:
        df['csa_index'] = 'ztt_' + df['str'].astype(str)
    else:
        df['csa_index'] = 'ztt_' + df['model_index'].astype(str) + '_' + df['str'].astype(str)

    df['trip_id'] = 'ztt_trip_' + df['str'].astype(str)
    if step is not None:
        df['departure_time_round'] = df['departure_time'] // (step)
        df = df.sort_values(by='departure_time')
        df = df.drop_duplicates(['a', 'b', 'departure_time_round', 'direction'])
    return df[['a', 'b', 'departure_time', 'arrival_time', 'trip_id', 'csa_index', 'model_index', 'direction']]


def csa_profile(connections, target, stop_set=None, Ttrip=None):
    if stop_set is None:
        stop_set = {c['a'] for c in connections}.union({c['b'] for c in connections})
    if Ttrip is None:
        Ttrip = {c['trip_id']: float('inf') for c in connections}

    profile = {stop: [[0, float('inf'), 'root']] for stop in stop_set}
    profile[target] = [[0, 0]]
    predecessor = {target: 'root'}

    for c in connections:
        # SCAN
        a, b, index, trip_id = c['a'], c['b'], c['csa_index'], c['trip_id']

        # EVALUATE
        t_min_stop = float('inf')
        if b == target:
            p = 'root'
            t_min_stop = c['arrival_time']
        else:
            for td, ta, p in reversed(profile[b]):
                if td >= c['arrival_time']:
                    t_min_stop = ta
                    break

        t_min_trip = Ttrip[trip_id]

        if t_min_stop < t_min_trip:
            Ttrip[trip_id] = t_min_stop
            t_min = t_min_stop

            predecessor[trip_id] = index
            predecessor[index] = p

        else:
            t_min = t_min_trip
            predecessor[index] = predecessor.get(trip_id, trip_id)
        # END EVALUATE

        # UPDATE
        if t_min < profile[a][-1][1]:
            profile[a].append([c['departure_time'], t_min, index])
            predecessor[a] = index
        # END UPDATE
    # END SCAN
    profile = {k: v[1:] for k, v in profile.items() if len(v) > 1}
    return profile, predecessor


def get_path(predecessor, source, maxiter=1000):
    path = []
    current = source
    for i in range(maxiter):
        try:
            path.append(current)
            current = predecessor[current]
        except KeyError:  # root
            return path[:-1]


def trip_bit(c_in: int, c_out: int, trip_connections: list[int]):
    left = bisect.bisect_left(trip_connections, c_in)
    right = bisect.bisect_left(trip_connections, c_out)
    return trip_connections[left:right]


def get_full_csa_path(path, trip_dict, trip_connections):
    full_path = []
    for j in range(len(path) - 1):
        c_in = path[j]
        c_out = path[j + 1]
        trip_in = trip_dict[c_in]
        trip_out = trip_dict[c_out]
        full_path.append(c_in)
        if trip_in == trip_out:
            # first value already append
            bit = trip_bit(c_in, c_out, trip_connections.get(trip_in))[1:]
            full_path += bit
    full_path.append(path[-1])
    return full_path


def pathfinder(pseudo_connections, zone_set, time_interval=None, cutoff=np.inf, targets=None, workers=1, od_set=None):
    targets = list(set(targets))
    if workers > 1:
        results = {}
        chunksize = len(targets) // workers
        if len(targets) % workers > 0:
            chunksize += 1
        slices = [[i * chunksize, (i + 1) * chunksize] for i in range(workers)]
        target_slices = [targets[s[0] : min(s[1], len(targets))] for s in slices]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(workers):
                results[i] = executor.submit(
                    pathfinder,
                    targets=target_slices[i],
                    pseudo_connections=pseudo_connections,
                    zone_set=zone_set,
                    time_interval=time_interval,
                    cutoff=cutoff,
                    workers=1,
                    od_set=od_set,
                )
        to_return = pd.concat([r.result() for r in results.values()])
        return to_return.reset_index(drop=True)

    # DROP EGRESS
    egress = pseudo_connections.set_index('direction').loc['egress']
    egress_index = set(egress['csa_index'])
    egress_by_zone = egress.groupby(['b'])['csa_index'].agg(set).to_dict()
    all_egress = set(egress['csa_index'])

    # OD SET ACCESS FILTER
    targets = zone_set if targets is None else zone_set.intersection(targets)
    if od_set is not None:
        if targets is None:
            targets = {d for o, d in od_set}
        else:
            od_set = {(o, d) for o, d in od_set if d in targets}
            targets = {d for o, d in od_set}.intersection(targets)

        target_sources = {t: [] for t in targets}
        for o, d in od_set:
            target_sources[d].append(o)

        access = pseudo_connections.set_index('direction').loc['access']
        access_by_zone = access.groupby(['a'])['csa_index'].agg(set).to_dict()
        all_access = set(access['csa_index'])
        access_by_target = {t: set() for t in targets}
        for t in targets:
            for source in target_sources[t]:
                try:
                    access_by_target[t].update(access_by_zone[source])
                except KeyError:
                    pass

    # TIME FILTER
    # all links reaching b
    egress_time_dict = egress.groupby('b')['departure_time'].agg(list).to_dict()
    departure_times = list(pseudo_connections['departure_time'])[::-1]

    stop_set = set(pseudo_connections['a']).union(set(pseudo_connections['b']))
    Ttrip_inf = {t: float('inf') for t in set(pseudo_connections['trip_id'])}
    columns = ['a', 'b', 'departure_time', 'arrival_time', 'csa_index', 'trip_id']
    decreasing_departure_connections = pseudo_connections[columns].to_dict(orient='records')

    pareto = []

    for target in tqdm(targets):
        # BUILD CONNECTIONS
        start, end = time_interval[0], time_interval[1] + cutoff
        end = max([t for t in egress_time_dict[target] if t <= end] + [0])
        slice_end = bisect.bisect_left(departure_times, start)
        slice_start = bisect.bisect_right(departure_times, end)
        if slice_end == 0:
            ti_connections = decreasing_departure_connections[-slice_start:]
        else:
            ti_connections = decreasing_departure_connections[-slice_start:-slice_end]

        forbidden = all_egress - egress_by_zone[target]
        if od_set is not None:
            forbidden.update(all_access - access_by_target[target])
        connections = [c for c in ti_connections if c['csa_index'] not in forbidden]

        profile, predecessor = csa_profile(connections, target=target, stop_set=stop_set, Ttrip=Ttrip_inf.copy())
        predecessor.update({i: 'root' for i in egress_index})

        for source, source_profile in profile.items():
            if source not in zone_set:
                continue
            for departure, arrival, c in source_profile:
                path = get_path(predecessor, c)
                path = [source] + path + [target]
                pareto.append((source, target, departure, arrival, c, path))

    pt_los = pd.DataFrame(
        pareto, columns=['origin', 'destination', 'departure_time', 'arrival_time', 'last_connection', 'csa_path']
    )
    return pt_los


def pathfinder_on_stops(pseudo_connections: pd.DataFrame):
    targets = list(pseudo_connections['b'].unique())
    stop_set = set(pseudo_connections['a']).union(set(pseudo_connections['b']))
    Ttrip_inf = {t: float('inf') for t in set(pseudo_connections['trip_id'])}
    columns = ['a', 'b', 'departure_time', 'arrival_time', 'csa_index', 'trip_id']
    decreasing_departure_connections = pseudo_connections[columns].to_dict(orient='records')

    pareto = []
    for target in targets:
        ti_connections = decreasing_departure_connections
        connections = [c for c in ti_connections]
        profile, predecessor = csa_profile(connections, target=target, stop_set=stop_set, Ttrip=Ttrip_inf.copy())
        for source, source_profile in profile.items():
            for departure, arrival, c in source_profile:
                path = get_path(predecessor, c)
                pareto.append((source, target, departure, arrival, path))

    pt_los = pd.DataFrame(pareto, columns=['origin', 'destination', 'departure_time', 'arrival_time', 'csa_path'])
    return pt_los


def get_footpaths_time(pt_los: pd.DataFrame, pseudo_connections: pd.DataFrame, footpaths: pd.DataFrame) -> pd.Series:
    footpaths_set = set(footpaths['model_index'])
    pseudo_footpaths = pseudo_connections[pseudo_connections['model_index'].isin(footpaths_set)]
    pseudo_footpaths['time'] = pseudo_footpaths['arrival_time'] - pseudo_footpaths['departure_time']
    footpaths_time_dict = pseudo_footpaths.set_index('csa_index')['time'].to_dict()
    return pt_los['csa_path'].apply(lambda ls: sum([footpaths_time_dict.get(x, 0) for x in ls]))


def merge_on_connector(
    pt_los: pd.DataFrame,
    zone_to_transit: pd.DataFrame,
    od_set: list[tuple],
    groupby=['origin', 'destination'],
    walk_penalty=1,
):
    """
    takes the pt_los station to station, and merge on all the zone_to_road and apply pareto to each group in groupby
    pt_los : csa pt_los station station.
    groupby = ['origin', 'destination']: for pareto groups. can add route_type_access, route_type_egress
    stop_egress
    walk_penalty : float >= 1, optional, default 1 (1 = no penaly)
            penalty to be applied on access/egress and footpath time for the pareto filter

    """

    # TODO  have a way to select columns to merge. we could want to groupby mode_type_egress for example
    ztt_cols = ['a', 'b', 'time', 'route_type', 'model_index']
    for col in ['route_type', 'model_index']:  # convert to category before merge: save memory.
        zone_to_transit[col] = zone_to_transit[col].astype('category')
    access = zone_to_transit[zone_to_transit['direction'] == 'access'][ztt_cols]
    egress = zone_to_transit[zone_to_transit['direction'] == 'eggress'][ztt_cols]
    # rename before merging. this is faster than renamming and dropping after when df is big.
    access = access.rename(columns={'a': 'origin', 'b': 'stop_access'})
    egress = egress.rename(columns={'b': 'destination', 'a': 'stop_egress'})

    # init the pt_los df (each od)
    df = pd.DataFrame(od_set, columns=['origin', 'destination'])
    # merge access and egress connector
    df = df.merge(access, on='origin')
    df = df.merge(egress, on='destination', suffixes=['_access', '_egress'])

    # convert to category before merge. this save a lot of memory
    for col in ['origin', 'destination', 'stop_access', 'stop_egress']:
        df[col] = df[col].astype('category')
    for col in ['origin', 'destination']:
        pt_los[col] = pt_los[col].astype('category')

    # merge station to station csa on OD
    pt_los = pt_los.rename(columns={'origin': 'stop_access', 'destination': 'stop_egress'})
    df = df.merge(pt_los, on=['stop_access', 'stop_egress'])

    # compute actal time with access and egress
    df['departure_time'] = df['departure_time'] - df['time_access']
    df['arrival_time'] = df['arrival_time'] + df['time_egress']

    # compute a pseudo time with access and egress time penalty
    df['pseudo_departure_time'] = df['departure_time'] - df['time_access'] * (walk_penalty - 1)
    df['pseudo_arrival_time'] = df['arrival_time'] + (df['time_egress'] + df['footpath_time']) * (walk_penalty - 1)

    # group data for the pareto by group [['origin', 'destination']]
    df['pareto_group'] = df.groupby(groupby, group_keys=False).ngroup()
    df = df.sort_values('pareto_group')

    # filter big los with pareto
    mask = pareto_per_groups(
        df['pseudo_departure_time'].values, df['pseudo_arrival_time'].values, df['pareto_group'].values
    )
    df = df.iloc[mask]

    # add access and egress ntlegs. usefull to create a complete path (and drop dup later)
    df['ntlegs'] = [*zip(df['model_index_access'], df['model_index_egress'])]
    df = df.drop(columns=['model_index_access', 'model_index_egress', 'pseudo_departure_time', 'pseudo_arrival_time'])

    return df  # this is pt_los


def pareto(departures: np.ndarray, arrivals: np.ndarray) -> np.ndarray[bool]:
    """
    sort departures and arrivals then apply pareto.
    """
    order = np.lexsort((arrivals, -departures))
    return _pareto_sweep(arrivals, order)


@nb.njit()
def _pareto_sweep(arrivals: np.ndarray, order: np.ndarray) -> np.ndarray[bool]:
    """
    order = np.lexsort((arrivals, -departures))
    """
    keep = np.full(len(arrivals), False)
    best_arr = np.inf

    for i in order:
        if arrivals[i] < best_arr:
            keep[i] = True
            best_arr = arrivals[i]

    return keep


def compute_offset(groups: np.ndarray) -> np.ndarray:
    changes = np.where(groups[1:] != groups[:-1])[0] + 1
    offsets = np.concatenate(([0], changes, [len(groups)]))
    return offsets


def pareto_per_groups(all_departures: np.ndarray, all_arrivals: np.ndarray, groups: np.ndarray) -> np.ndarray[bool]:
    """
    Data must be sorted by groups (df = df.sort_values("group"), or continious by groups
    """
    # we have long array labels with groups. get offset (start:end) opf each groups
    pareto_mask = np.full(len(all_departures), False)
    offsets = compute_offset(groups)
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        arrival = all_arrivals[start:end]
        departure = all_departures[start:end]
        pareto_mask[start:end] = pareto(departure, arrival)
    return pareto_mask
