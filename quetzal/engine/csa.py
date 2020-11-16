import pandas as pd
import bisect

def time_footpaths(links, footpaths):
    footpaths['model_index'] = footpaths.index

    left = pd.merge(
        links[[ 'b', 'arrival_time']], 
        footpaths[['a', 'b', 'duration', 'model_index']], 
        left_on='b', right_on='a', suffixes=['_link', '']
    )

    left['ready_time'] = left['arrival_time'] + left['duration']
    right = links[['a', 'departure_time']]

    # EFFICIENT MERGE GROUPBY FIRST
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
        left[['a','b', 'key', 'arrival_time', 'duration', 'model_index']], 
        right[['departure_time', 'key']], 
        on='key', direction='forward', tolerance=1e6
    )
    # END MERGE GROUPBY

    # enables the earliest departure from b, for each arrival at a
    transfers.sort_values('departure_time', ascending=True, inplace=True) 
    transfers = transfers.groupby(['a', 'b', 'arrival_time'], as_index=False).first()

    # enables the latest arrival at a, for each departure from b
    transfers.sort_values('arrival_time', ascending=False, inplace=True)
    transfers = transfers.groupby(['a', 'b', 'departure_time'], as_index=False).first()

    transfers[['departure_time', 'arrival_time']]  = transfers[['arrival_time', 'departure_time']]
    transfers['str'] = [str(i) for i in transfers.index]
    transfers['csa_index'] = 'footpath_' + transfers['str']
    transfers['trip_id'] = 'footpath_trip_' + transfers['str']
    columns = ['a', 'b', 'departure_time', 'arrival_time', 'trip_id', 'csa_index', 'model_index']
    return transfers[columns]

def time_zone_to_transit(links, zone_to_transit):
    ztt = zone_to_transit
    ztt['model_index'] = ztt.index
    # access
    left = ztt.loc[ztt['direction'] == 'access']
    df = pd.merge(
        left[['a','b', 'time', 'model_index']], 
        links[['a','b', 'departure_time']],
        left_on='b', right_on='a',suffixes=['_ztt', '_link'] 
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
        ztt[['a','b', 'time', 'model_index']], 
        links[['a','b', 'arrival_time']],
        left_on='a', right_on='b',suffixes=['_ztt', '_link'] 
    )
    df['departure_time'] = df['arrival_time'] 
    df['arrival_time'] = df['departure_time'] + df['time']
    df['a'] = df['a_ztt']
    df['b'] = df['b_ztt']
    df['direction'] = 'egress'
    egress = df.copy()
    df = pd.concat([access, egress])
    df['str'] = range(len(df))
    df['csa_index'] = 'ztt_' + df['str'].astype(str)
    df['trip_id'] = 'ztt_trip_' + df['str'].astype(str)
    return df[['a', 'b', 'departure_time', 'arrival_time', 'trip_id',
     'csa_index','model_index' ,'direction']]


def csa_profile(
    connections, 
    target,
    stop_set=None,
    Ttrip=None,
    ):
    
    if stop_set is None:
        stop_set = {c['a'] for c in connections}.union({c['b'] for c in connections})
    if Ttrip is None:
        Ttrip = {c['trip_id']: float('inf') for c in connections}
        
    profile = {stop: [[0, float('inf'), 'root']] for stop in stop_set}
    profile[target] =[[0, 0]]
    predecessor = {target: 'root'}
    
    for c in connections:
    ############## SCAN
        a, b, index, trip_id = c['a'], c['b'], c['csa_index'], c['trip_id']
        
        ########## EVALUATE
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
            predecessor[index] = p#predecessor[b]
            
        else :
            t_min = t_min_trip
            predecessor[index] = predecessor.get(trip_id, trip_id)
        ########## END EVALUATE
            
        ########## UPDATE
        if  t_min < profile[a][-1][1]:
            profile[a].append([c['departure_time'], t_min, index])
            predecessor[a] = index
        ########## END UPDATE
        
    ############## END SCAN
    
    profile = {k: v[1:] for k, v in profile.items() if len(v) > 1}
    return profile, predecessor

def get_path(predecessor, source, maxiter=1000):
    path = []
    current = source
    for i in range(maxiter):
        try:
            path.append(current)
            current = predecessor[current] 
        except KeyError: # root
            return path[:-1]



def trip_bit(t, c_in, c_out, trip_connections):
    trip = trip_connections[t]
    left = bisect.bisect_left(trip, c_in)
    right = bisect.bisect_left(trip, c_out)
    return trip[left:right]

def path_to_boarding_links_and_boarding_path(csa_path, connection_trip, trip_connections):
    link_path = []
    trips = set()
    boarding_links = []
    
    for i in range(len(csa_path) -2):
        

        c_in, c_out = csa_path[i: i+2]
        
        try:
            trip_in =  connection_trip[c_in]
            link_path.append(c_in)
            if trip_in not in trips:
                trips.add(trip_in)
                boarding_links.append(c_in)
            assert trip_in == connection_trip[c_out]
            link_path += trip_bit(connection_trip[c_in], c_in, c_out, trip_connections)
        except (KeyError, AssertionError):
            pass
        link_path.append(c_out)
        
    link_path = list(dict.fromkeys(link_path))
    return link_path, boarding_links

def pathfinder(
    time_expended_zone_to_transit, 
    pseudo_connections,
    zone_set,
    min_transfer_time=0, time_interval=None, cutoff=np.inf, 
    targets=None, workers=1
):

    targets = list(set(targets))
    if workers > 1:
        results = {}
        chunksize = len(targets) // workers
        if len(targets) % workers > 0:
            chunksize += 1
        slices = [[i*chunksize, (i+1)*chunksize] for i in range(workers)]
        target_slices = [
            targets[s[0]: min(s[1], len(targets))]
            for s in slices
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(workers):
                results[i] = executor.submit(
                    pathfinder,
                    targets=target_slices[i],
                    time_expended_zone_to_transit=time_expended_zone_to_transit,
                    pseudo_connections=pseudo_connections,
                    zone_set=zone_set,
                    min_transfer_time=min_transfer_time,
                    time_interval=time_interval,
                    cutoff=cutoff,
                    workers=1
                )
        to_return = pd.concat([r.result() for r in results.values()])
        return to_return.reset_index(drop=True)

    # DROP EGRESS
    egress = pseudo_connections.set_index('direction').loc['egress']
    egress_index = set(egress['csa_index'])
    egress_by_zone = egress.groupby(['b'])['csa_index'].agg(set).to_dict()
    all_egress = set(egress['csa_index'])

    # TIME FILTER
    # all links reaching b
    egress_time_dict = egress.groupby('b')['departure_time'].agg(list).to_dict()
    departure_times = list(pseudo_connections['departure_time'])[::-1]

    stop_set = set(pseudo_connections['a']).union(set(pseudo_connections['b']))
    Ttrip_inf = {t: float('inf') for t in set(pseudo_connections['trip_id'])}
    columns = ['a', 'b', 'departure_time', 'arrival_time', 'csa_index', 'trip_id']
    decreasing_departure_connections = pseudo_connections[columns].to_dict(orient='record')

    pareto = []
    targets = zone_set if targets is None else zone_set.intersection(targets)
    for target in tqdm(targets):

        # BUILD CONNECTIONS
        start, end = time_interval[0], time_interval[1] + cutoff
        slice_end = bisect.bisect_left(departure_times, start)
        end = max([t for t in egress_time_dict[target] if t <= end] + [0]) 
        slice_start = bisect.bisect_right(departure_times, end)
        if slice_end == 0:
            ti_connections = decreasing_departure_connections[-slice_start:]
        else:
            ti_connections = decreasing_departure_connections[-slice_start: -slice_end]

        forbidden = all_egress - egress_by_zone[target]
        connections = [c for c in ti_connections if c['csa_index'] not in forbidden]

        profile, predecessor = csa_profile(
            connections, target=target,
            stop_set=stop_set, Ttrip=Ttrip_inf.copy()
        )
        predecessor.update({i: 'root' for i in egress_index})

        for source, source_profile in profile.items():
            if source not in zone_set:
                continue
            for departure, arrival, c in source_profile:
                path = get_path(predecessor, c)
                path = [source] + path + [target]
                pareto.append((source, target, departure, arrival, c, path))

    pt_los = pd.DataFrame(
        pareto,
        columns=[
            'origin', 'destination',
            'departure_time', 'arrival_time', 'last_connection', 'csa_path'
        ]
    )
    return pt_los
