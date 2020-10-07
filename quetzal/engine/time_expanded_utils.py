import pandas as pd
import geopandas as gpd
import numpy as np
from syspy.skims import skims
from shapely import geometry
import pyproj


def create_connection_boarding_edges(links, boarding_time=0):
    """
    connection_edges: from (a, departure, departure_time) to 
        (b, arrival, arrival_time), at cost 'time'
    boarding_edges: from (a, transfer, departure_time) to 
        (a, departure, departure_time), at cost 0 
    """
    connection_edges = []
    boarding_edges = []
    
    for (a, b, time, dep_time, arr_time, t_id), i in zip(links[
        ['a', 'b', 'time', 'departure_time', 'arrival_time', 'trip_id']
    ].values, links.index):
        node_dep = tuple([a, 'departure', dep_time])
        connection_edges.append(
            [
                node_dep,
                tuple([b, 'arrival', arr_time]),
                time,
                {'id': i, 'trip_id': t_id},
                dep_time
            ]
        )
        boarding_edges.append(
            [
                tuple([a, 'transfer', dep_time]),
                node_dep,
                boarding_time,
                {'trip_id': t_id, 'id': i},
                dep_time
            ]
        )
        
    connection_edges = pd.DataFrame(
        data=connection_edges,
        columns=['a', 'b', 'weight', 'data', 'start_time']
    )
    connection_edges['type'] = 'connection'
    
    boarding_edges = pd.DataFrame(
        data=boarding_edges,
        columns=['a', 'b', 'weight', 'data', 'start_time']
    )
    boarding_edges['type'] = 'boarding'
    
    return connection_edges, boarding_edges


def create_transit_edges(links):
    """
    transit_edges: from (b, arrival, arrival_time) to 
    (b, departure, departure_time) if staying in the same vehicle
    """
    def transits_from_trip_connections(g, trip_id):
        trip_transit_edges = []
        for (tuple_b, arr_time),(tuple_a,dep_time) in zip(
            g[['tuple_b', 'arrival_time']][:-1].values.tolist(),
            g[['tuple_a', 'departure_time']][1:].values.tolist()
        ):
            trip_transit_edges.append(
                [tuple_b, tuple_a, dep_time - arr_time,  {'trip_id': trip_id}, arr_time]
            )

        return trip_transit_edges
    
    links = links.copy()
    tuple_b = []
    tuple_a = []
    for b, arr_time, a, dep_time in links[['b', 'arrival_time', 'a', 'departure_time']].values.tolist():
        tuple_b.append(tuple([b, 'arrival', arr_time]))
        tuple_a.append(tuple([a, 'departure', dep_time]))
    links['tuple_b'] = tuple_b
    links['tuple_a'] = tuple_a
        
    links = links.sort_values('link_sequence').reset_index(drop=True)
    transit_edges = []
    for name, group in links.groupby('trip_id'):
        transit_edges += transits_from_trip_connections(group, name)

    transit_edges = pd.DataFrame(
        data = transit_edges,
        columns=['a', 'b', 'weight', 'data', 'start_time']
    )
    transit_edges['type'] = 'transit'
    
    return transit_edges

def create_waiting_edges(nodes):
    """
    waiting_edges: between subsequent transfer nodes of the same stop
    transfer edges: to join a "waiting node"
    """
    transfer_nodes = nodes.sort_values(['stop', 'time'])
    a = []
    for s, tp, tm in transfer_nodes[['stop', 'type','time']].values.tolist():
        a.append(tuple([s,tp,tm]))
    transfer_nodes['a'] = a

    def waiting_edges_from_transfers(g): 
        stop_waiting_edges = []
        gl = g[['a', 'time']].values.tolist()
        for (ax, tx), (ay, ty) in zip(gl[:-1], gl[1:]):
            stop_waiting_edges.append([ax, ay, ty - tx, tx])

        return stop_waiting_edges
    
    waiting_edges = []
    for name, group in transfer_nodes.groupby('stop'):
        if len(group)>1:
            waiting_edges += waiting_edges_from_transfers(group)
    
    waiting_edges = pd.DataFrame(
        data = waiting_edges,
        columns = ['a', 'b', 'weight', 'start_time']
    )
    
    waiting_edges['type'] = 'waiting'
    waiting_edges['data'] = [{}]*len(waiting_edges)

    return waiting_edges

def create_transfers_edges(arrival_nodes, transfer_nodes, min_transfer_time=0):
    """
    Transfers_edges: within one stop, from arrival node to 
    earliest transfer node above transfer threshold
    """
    try:
        transfer_nodes['min_transfer_time'] = np.maximum(transfer_nodes['transfer_duration'].values, min_transfer_time) # l'un sinon l'autre
    except KeyError:
        print('cluster transfer duration not defined')
        transfer_nodes['min_transfer_time'] = min_transfer_time
    transfers = arrival_nodes.merge(transfer_nodes, on='stop', suffixes=('_arrival', '_transfer'))
    transfers['time'] = transfers['time_transfer'] - transfers['time_arrival']
    transfers = transfers.loc[transfers['time'] >= transfers['min_transfer_time']].sort_values('time_transfer')
    transfers = transfers.groupby(['stop', 'time_arrival'], as_index=False).first()

    transfer_edges = []
    
    for stop, time_arrival, time_transfer, time, in transfers[
        ['stop', 'time_arrival', 'time_transfer', 'time']].values.tolist():
        transfer_edges.append(
            [
                tuple([stop, 'arrival', time_arrival]),
                tuple([stop, 'transfer', time_transfer]),
                time,
                time_arrival
            ]
        )
    
    transfer_edges = pd.DataFrame(
        data = transfer_edges,
        columns=['a', 'b', 'weight', 'start_time']
    )
    transfer_edges['type'] = 'transfer'
    transfer_edges['data'] = [{}]*len(transfer_edges)
    
    return transfer_edges

def create_footpath_edges(nodes, footpaths):
    """
    Create footpath_edges between nodes such as:
    - a footpath exists between the two stops
    - the timing is consistent, ie arrival_time + footpath_duration <= departure_time
    """
    arrival_nodes = nodes.loc[nodes['type']=='arrival']
    transfer_nodes = nodes.loc[nodes['type']=='transfer']
    
    temp = arrival_nodes.merge(footpaths.reset_index(), left_on='stop', right_on='a')
    temp = temp.merge(transfer_nodes, left_on='b', right_on='stop', suffixes=('_arrival', '_tr'))
    # filter on walk duration
    temp = temp.loc[temp['time_arrival'] + temp['duration'] < temp['time_tr']]
    temp = temp.groupby(['stop_arrival', 'time_arrival', 'stop_tr']).first().reset_index()

    temp['w'] = temp['time_tr'] - temp['time_arrival']

    footpath_edges = []
    for sa, ta, st, tt, duration, index, w in temp[
        ['stop_arrival', 'time_arrival', 'stop_tr', 'time_tr', 'duration', 'index', 'w']
    ].values.tolist():
        footpath_edges.append(
            [
                tuple([sa, 'arrival', ta]),
                tuple([st, 'transfer', tt]),
                w,
                {'walking_duration': duration, 'id': index},
                ta
            ]
        )
        
    footpath_edges = pd.DataFrame(
        data=footpath_edges,
        columns=['a', 'b', 'weight', 'data', 'start_time']
    )
    footpath_edges['type'] = 'footpath'

    return footpath_edges


def create_egress_edges(nodes, zone_to_transit, ntlegs_penalty=1e9, time_interval=None):
    # eggress: every arrival node arriving after departure time is connected to a zone
    arrival_nodes = nodes.loc[nodes['type']=='arrival']
    
    egress = zone_to_transit.loc[zone_to_transit['direction']=='eggress'][['a', 'b', 'time']].rename(
        columns={'time':'duration', 'b': 'zone'}
    )
    temp = arrival_nodes.merge(egress.reset_index(), left_on='stop', right_on='a')
    
    if time_interval is not None:
        temp = temp.loc[temp['time'] > time_interval[0]]
    
    temp['arrival_time'] = temp['time'] + temp['duration']
    
    egress_edges = []
    zone_to_zone = []
    for stop, time, zone, duration, index, arr_time in temp[
        ['stop', 'time', 'zone', 'duration', 'index', 'arrival_time']
    ].values.tolist():
        egress_edges.append(
            [
                tuple([stop, 'arrival', time]),
                tuple([zone, 'zone', arr_time]),
                duration,
                {'arrival': arr_time, 'id': index},
                time
            ]
        )
        zone_to_zone.append(
            [
                tuple([zone, 'zone', arr_time]),
                tuple([zone, 'zone']),
                arr_time
            ]
        )
    
    egress_edges = pd.DataFrame(
        data = egress_edges,
        columns = ['a', 'b', 'weight', 'data', 'start_time']
    )
    egress_edges['type'] = 'egress'

    zone_to_zone = pd.DataFrame(
        data = zone_to_zone,
        columns = ['a', 'b', 'start_time']
    )
    zone_to_zone['data'] = [{}] * len(zone_to_zone)
    zone_to_zone['weight'] = ntlegs_penalty
    zone_to_zone['type'] = 'zone_to_zone'
        
    return pd.concat([egress_edges, zone_to_zone])

def create_access_edges(nodes, zone_to_transit, ntlegs_penalty=1e9, time_interval=None):

    transfer_nodes = nodes.loc[nodes['type']=='transfer']

    access = zone_to_transit.loc[zone_to_transit['direction']=='access'][['a', 'b', 'time']].rename(
        columns={'time':'duration', 'a': 'zone'}
    )
    temp = transfer_nodes.merge(access.reset_index(), left_on='stop', right_on='b')

    temp['start_time'] = temp['time'] - temp['duration'] # TODO: round duration to minute
    
    if time_interval is not None:
        temp = temp.loc[
            (temp['start_time'] >= time_interval[0])&
            (temp['start_time'] < time_interval[1])
        ]

    access_edges = []
    for zone, stop, start_time, time, duration, index, in temp[
        ['zone', 'stop', 'start_time', 'time', 'duration', 'index']].values.tolist():
        access_edges.append(
            [
                tuple([zone, 'zone', start_time]),
                tuple([stop, 'transfer', time]),
                duration,
                {'id': index},
                start_time
            ]
        )
    access_edges = pd.DataFrame(data=access_edges, columns=['a', 'b', 'weight', 'data', 'start_time'])
    access_edges['weight'] += ntlegs_penalty
    access_edges['type'] = 'access'

    return access_edges

def pt_edges_and_nodes_from_links(links, nodes, time_interval=None, boarding_time=0, min_transfer_time=0):
    if time_interval is not None:
        links = links.loc[
            (links['departure_time'] >= time_interval[0])
        ] 
    
    connection_edges, boarding_edges = create_connection_boarding_edges(links, boarding_time=boarding_time)
    transit_edges = create_transit_edges(links)
    
    edges = pd.concat([connection_edges, boarding_edges, transit_edges])
    
    temp = pd.concat([edges['a'], edges['b']]).drop_duplicates()
    
    graph_nodes = pd.DataFrame(
        data=temp.map(list).values.tolist(),
        columns=['stop', 'type', 'time']
    )
    # add data such as transfer duration for clusters
    graph_nodes = graph_nodes.merge(nodes[['transfer_duration']], left_on='stop', right_index=True)
    
    arrival_nodes = graph_nodes.loc[graph_nodes['type']=='arrival']
    transfer_nodes = graph_nodes.loc[graph_nodes['type']=='transfer']
    
    waiting_edges = create_waiting_edges(transfer_nodes)
    transfer_edges = create_transfers_edges(arrival_nodes, transfer_nodes, min_transfer_time=0)
    
    edges = pd.concat([edges, waiting_edges, transfer_edges])

    return edges, graph_nodes.reset_index(drop=True)

# Scipy pathfinder

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx

def get_path(predecessors, i, j):
    pred = predecessors[i]
    path = [j]
    k = j
    p = 0
    while p != -9999:
        k = p = pred[k]
        path.append(p)
    return path[::-1][1:]

def sparse_los_from_nx_graph(nx_graph, pole_set, sources=None, cutoff=np.inf):

    sources = pole_set if sources is None else sources
    # INDEX
    pole_list = sorted(list(pole_set)) # fix order
    source_list = [zone for zone in pole_list if zone in sources]

    nodes = list(nx_graph.nodes)
    node_index = dict(zip(nodes, range(len(nodes))))
    
    zones = [node_index[zone] for zone in source_list]
    source_index = dict(zip(source_list, range(len(source_list))))
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    # SPARSE GRAPH
    sparse = nx.to_scipy_sparse_matrix(nx_graph)
    graph = csr_matrix(sparse)
    dist_matrix, predecessors = dijkstra(
        csgraph=graph, 
        directed=True, 
        indices=zones, 
        return_predecessors=True,
        limit=cutoff
    )
    # LOS LAYOUT
    df = pd.DataFrame(dist_matrix)

    df.index = source_list
    df.columns = list(nx_graph.nodes)
    df.columns.name = 'destination'
    df.index.name = 'origin'
    stack = df[pole_list].stack()
    stack.name = 'gtime'
    los = stack.reset_index()

    # QUETZAL FORMAT
    los = los.loc[los['gtime'] < np.inf]


    # BUILD PATH FROM PREDECESSORS
    od_list = los[['origin', 'destination']].values.tolist()
    paths = [
        [nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])]
        for o, d in od_list
    ]
    los['path'] = paths
    return los

import operator 
from functools import reduce

def get_edge_path(los):
    paths = []
    for p in los['node_path'].values:
        paths.append(list(zip(p[:-1], p[1:])))
    los['edge_path'] = paths
    return los

def get_model_link_path(los, all_edges):
    if not 'edge_path' in los.columns:
        los = get_edge_path(los)

    all_edges['index'] = [x.get('id') for x in all_edges['data'].values] # remove get?
    ab_indexed_dict = all_edges.set_index(['a', 'b']).sort_index()['index'].to_dict()
    ml_paths = []
    for lp in los['edge_path'].values:
#         ml_paths.append([l for l in list(map(ab_indexed_dict.get, lp)) if l is not None])
        ml_paths.append([l for l in [ab_indexed_dict[x] for x in lp] if l is not None])
    # los['link_path'] = ml_paths

    los['link_path'] = [
        list(dict.fromkeys(p)) for p in ml_paths # remove duplicated link ids (link_i, link_i, etc.)
    ]
    return los

def get_edges_per_type_set(los, all_edges):
    # edges per type
    link_path_sets = los['edge_path'].map(set).values
    for edge_type in all_edges.type.unique():
        type_edges = set(
            [tuple([a,b]) for a, b in all_edges.loc[all_edges['type']==edge_type][['a','b']].values]
        )
        edge_links = []
        for lp in link_path_sets:
            edge_links.append(lp.intersection(type_edges))

        los[edge_type + '_edges'] = edge_links
        
    return los

def get_edges_per_type_dict(los, all_edges):
    ab_indexed_dict = all_edges.set_index(['a', 'b']).sort_index()['type'].to_dict()
    edges_per_type = []
    for lp in los['edge_path'].values:
        temp = {x: ab_indexed_dict[x] for x in lp}
        inverted_dict = dict()
        for key, value in temp.items():
            inverted_dict.setdefault(value, list()).append(key)
        edges_per_type.append(inverted_dict)
    
    los['edges_per_type'] = edges_per_type
    
        
    return los
    
def get_model_node_path(los):
    los['model_node_path'] = [
        list(dict.fromkeys([n[0] for n in p])) for p in los['node_path'].values
    ]
    return los

def merge_node_link_paths(node_path, link_path): 
        return list(reduce(operator.add, zip(node_path, link_path))) + [node_path[-1]]
    
def get_model_path(los):
    link_path_loc = (los['link_path'].apply(len) > 0)
    m_paths = []
    for mlp, mnp in los.loc[link_path_loc, ['link_path', 'model_node_path']].values:
        m_paths.append(merge_node_link_paths(mnp, mlp))
    los.loc[link_path_loc, 'path'] = m_paths
    return los

def analysis_paths(los, all_edges, typed_edges=False, force=False):
    # edge_path
    if not 'edge_path' in los.columns or force:
        los = get_edge_path(los)
   
    # model_link_path
    if not 'link_path' in los.columns or force:
        los = get_model_link_path(los, all_edges)
    

    if (typed_edges and not 'edges_per_type' in los.columns) or force:
        los = get_edges_per_type_dict(los, all_edges)

    # model_node_path
    if not 'model_node_path' in los.columns or force:
        los = get_model_node_path(los)

    # model_path (node, link, node, link, …)
    if not 'model_path' in los.columns or force:
        los = get_model_path(los)
    
    return los

def analysis_lengths(los, links, footpaths, zone_to_transit):
    # LOS Lengths
    model_edges_length =  gpd.GeoSeries(pd.concat([links.geometry, footpaths.geometry, zone_to_transit.geometry])).length
    length_dict = model_edges_length.to_dict()
    los['length'] = los['link_path'].apply(lambda x: sum(map(length_dict.get, x)))

    links_length_dict = links.geometry.length.to_dict()
    los['in_vehicle_length'] = los['link_path'].apply(
        lambda x: sum(map(lambda y: links_length_dict.get(y,0), x))
    )

    footpath_length_dict = footpaths.geometry.apply(lambda x: x.length).to_dict()
    los['footpath_length'] = los['link_path'].apply(
        lambda x: sum(map(lambda y: footpath_length_dict.get(y,0), x))
    )

    ntlegs_length_dict = zone_to_transit.geometry.apply(lambda x: x.length).to_dict()
    los['ntlegs_length'] = los['link_path'].apply(
        lambda x: sum(map(lambda y: ntlegs_length_dict.get(y,0), x))
    )
    
    return los

def analysis_transfers(los):
    # LOS transfers
    los['transfers'] = los['edges_per_type'].apply(lambda x: max(0, len(x.get('boarding', [])) - 1))
    return los

def analysis_durations(los, all_edges, ntlegs_penalty):
    typed_edges_weight = {}
    for edge_type in all_edges.type.unique():
        typed_edges = all_edges.loc[all_edges['type']==edge_type]
        typed_edges_weight[edge_type]  = typed_edges.set_index(['a', 'b']).sort_index()['weight'].to_dict()
        
    # LOS durations
    ## in-vehicle_duration: connection['weight'], transit['weight']
    los['in_vehicle_duration'] = los['edges_per_type'].apply(
    lambda x: sum(map(typed_edges_weight['connection'].get,x['connection'])) +\
              sum(map(typed_edges_weight['transit'].get, x.get('transit', [])))
    )
    ## footpaths_duration: footpaths['data']['duration']
    los['footpaths_duration'] = los['edges_per_type'].apply(
        lambda x: sum(map(typed_edges_weight['footpath'].get, x.get('footpath', [])))
    )
    ## ntlegs_duration access - ntlegs_penalty, egress - ntlegs_penalty
    los['ntlegs_duration'] = los['edges_per_type'].apply(
        lambda x: sum(map(typed_edges_weight['access'].get, x.get('access', []))) - ntlegs_penalty + 
                  sum(map(typed_edges_weight['egress'].get, x.get('egress', []))) 

    )
    
    los['ntlegs_duration'] = los['ntlegs_duration'].clip(0)

    ## waiting_duration: transfer['weight'], footpaths['weight'] - footpaths['data']['duration'] TODO
    los['waiting_duration'] = los['edges_per_type'].apply(
        lambda x: sum(map(typed_edges_weight['transfer'].get, x.get('transfer',[])))
    )
    
    return los


def expand_volumes_with_time(volumes, time_interval, bins, volume_columns):
    # departure times
    duration = time_interval[1] - time_interval[0]
    departures = [time_interval[0] + duration / bins * (x + 0.5) for x in np.arange(bins)]
    
    time_volumes = pd.DataFrame()
    for departure in departures:
        volumes['wished_departure_time'] = departure
        time_volumes = pd.concat([time_volumes, volumes])
    for col in volume_columns:
        time_volumes[col] *= 1 / len(departures)
    
    time_volumes.reset_index(inplace=True, drop=True)
    return time_volumes

def build_dense_footpaths(nodes, max_length=1000, walking_speed=3):
    footpaths = skims.euclidean(nodes, coordinates_unit='meter')
    footpaths = footpaths.loc[footpaths['euclidean_distance'] < max_length]
    footpaths['duration'] = footpaths['euclidean_distance'] / (walking_speed / 3.6) # km/h to m/s
    footpaths = footpaths.loc[footpaths['origin']!=footpaths['destination']]
    footpaths['geometry'] = footpaths.apply(
        lambda x: geometry.LineString(
            [
                [x['x_origin'], x['y_origin']],
                [x['x_destination'], x['y_destination']],
            ]
        ),
        1
    )
    footpaths = footpaths[['origin', 'destination', 'duration', 'geometry']].rename(
        columns={'origin': 'a', 'destination': 'b'}
    )
    return footpaths