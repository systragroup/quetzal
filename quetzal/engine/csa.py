import pandas as pd

def time_footpaths(links, footpaths):
    left = pd.merge(links[[ 'b', 'arrival_time']], footpaths[['a', 'b', 'duration']], on='b')
    transfers = pd.merge(links[['a', 'departure_time']], left, on='a')
    transfers = transfers.loc[transfers['a'] != transfers['b']]
    transfers = transfers.loc[transfers['departure_time'] >= transfers['arrival_time']]
    
    # enables the earliest departure from b, for each arrival at a
    transfers.sort_values('departure_time', ascending=True, inplace=True) 
    transfers = transfers.groupby(['a', 'b', 'arrival_time'], as_index=False).first()

    # enables the latest arrival at a, for each departure from b
    transfers.sort_values('arrival_time', ascending=False, inplace=True)
    transfers = transfers.groupby(['a', 'b', 'departure_time'], as_index=False).first()

    transfers['negative_time'] =  transfers['arrival_time'] - transfers['departure_time']
    
    transfers = transfers.sort_values(['a','b','arrival_time', 'negative_time'], ascending=False)

    # discard dominated transfers
    # il se semble pas y avoir des transferts dominés, mais bon...

    _ab = 'start'
    kept = []
    for a, b, departure in transfers[['a', 'b', 'departure_time']].values:
        keep = False
        if (a, b) != _ab :
            _departure = float('inf')
            _ab = (a, b)
        if departure < _departure:
            keep = True
            _departure = departure
        kept.append(keep)

    transfers['keep'] = kept
    transfers = transfers.loc[transfers['keep'] == True]

    transfers[['a', 'b', 'departure_time', 'arrival_time']] = transfers[['b', 'a', 'arrival_time', 'departure_time']]
    transfers['str'] = [str(i) for i in transfers.index]
    transfers['ix'] = 'footpath_' + transfers['str']
    transfers['trip_id'] = 'footpath_trip_' + transfers['str']

    return transfers

def time_zone_to_transit(links, zone_to_transit):
    ztt = zone_to_transit
    # access
    left = ztt.loc[ztt['direction'] == 'access']
    df = pd.merge(
        left[['a','b', 'time']], 
        links[['a','b', 'departure_time']],
        left_on='b', right_on='a',suffixes=['_ztt', '_link'] 
    )
    df['arrival_time'] = df['departure_time']
    df['departure_time'] = df['arrival_time'] - df['time']
    df['a'] = df['a_ztt']
    df['b'] = df['b_ztt']
    df['direction'] = 'acess'
    access = df.copy()

    # egress

    left = ztt.loc[ztt['direction'] != 'access']
    df = pd.merge(
        ztt[['a','b', 'time']], 
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
    df['ix'] = 'ztt_' + df['str'].astype(str)
    df['trip_id'] = 'ztt_trip_' + df['str'].astype(str)
    return df[['a', 'b', 'departure_time', 'arrival_time', 'trip_id', 'ix', 'direction']]


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
        a, b, index, trip_id = c['a'], c['b'], c['ix'], c['trip_id']
        
        ########## EVALUATE
        t_min_stop = float('inf')
        if b == target:
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
            predecessor[index] = predecessor[b]
            
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
    right = bisect.bisect_right(trip, c_out)
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