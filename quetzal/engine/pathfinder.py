# -*- coding: utf-8 -*-

from quetzal.engine import engine

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import pandas as pd
import numpy as np
import networkx as nx

import itertools
from syspy.skims import skims
from syspy.spatial import spatial
from syspy.routing.frequency import graph as frequency_graph
import syspy.assignment.raw as assignment_raw
from tqdm import tqdm
from copy import deepcopy
import json
from quetzal.engine.subprocesses import filepaths
from quetzal.os import parallel_call

def get_path(predecessors, i, j):
    path = [j]
    k = j
    p = 0
    while p != -9999:
        k = p = predecessors[i, k]
        path.append(p)
    return path[::-1][1:] 

def path_and_duration_from_graph(
    nx_graph, 
    pole_set,
    od_set=None,
    sources=None,
    reversed_nx_graph=None,
    reverse=False,
    ntlegs_penalty=1e9,
    cutoff=np.inf,
    **kwargs
):
    sources = pole_set if sources is None else sources
    source_los=sparse_los_from_nx_graph(
        nx_graph, pole_set, sources=sources, 
        cutoff=cutoff+ntlegs_penalty, od_set=od_set, **kwargs)
    source_los['reversed'] = False
    
    reverse = reverse or reversed_nx_graph is not None
    if reverse:
        if reversed_nx_graph is None:
            reversed_nx_graph = nx_graph.reverse()
        
        try:
            reversed_od_set = {(d, o) for o, d in od_set}
        except TypeError:
            reversed_od_set = None

        target_los=sparse_los_from_nx_graph(
            reversed_nx_graph, pole_set, sources=sources, 
            cutoff=cutoff+ntlegs_penalty, od_set=reversed_od_set, **kwargs)
        target_los['reversed'] = True
        target_los['path'] = target_los['path'].apply(lambda x: list(reversed(x)))
        target_los[['origin', 'destination']] = target_los[['destination', 'origin']]
        
    los = pd.concat([source_los, target_los]) if reverse else source_los
    los.loc[los['origin'] != los['destination'],'gtime'] -= ntlegs_penalty
    tuples = [tuple(l) for l in  los[['origin', 'destination']].values.tolist()]
    los = los.loc[[t in od_set for t in tuples]]
    return los

def sparse_los_from_nx_graph(
    nx_graph, 
    pole_set, 
    sources=None, 
    cutoff=np.inf,
    od_set=None,
):

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

    df.index = [zone for zone in pole_list if zone in sources]
    df.columns = list(nx_graph.nodes)
    df.columns.name = 'destination'
    df.index.name = 'origin'
    stack = df[pole_list].stack()
    stack.name = 'gtime'
    los = stack.reset_index()

    # QUETZAL FORMAT
    los = los.loc[los['gtime'] < np.inf]
    if od_set is not None:
        tuples = [tuple(l) for l in  los[['origin', 'destination']].values.tolist()]
        los = los.loc[[t in od_set for t in tuples]]


    # BUILD PATH FROM PREDECESSORS
    od_list = los[['origin', 'destination']].values.tolist()
    paths = [
        [nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])]
        for o, d in od_list
    ]

    los['path'] = paths
    return los

def sparse_matrix(edges):
    nodelist = {e[0] for e in edges}.union({e[1] for e in edges})
    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(*((index[u], index[v], w ) for u, v, w in edges))
    row, col, data = coefficients
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index

def adjacency_matrix(
    links, 
    ntlegs, 
    footpaths,
    ntlegs_penalty=1e9, 
    boarding_time=0, 
    alighting_time=0,
    **kwargs
    ):
    l = links.copy()
    ntlegs = ntlegs.copy()
    l['index']= l.index
    l['next'] = l['link_sequence'] + 1

    if 'cost' not in l.columns:
        l['cost'] = l['time'] + l['headway'] / 2

    if 'boarding_time' not in l.columns:
        l['boarding_time'] = boarding_time

    if 'alighting_time' not in l.columns:
        l['alighting_time'] = alighting_time

    l['total_time'] = l['boarding_time'] + l['cost']

    boarding_edges = l[['a', 'index', 'total_time']].values.tolist()
    alighting_edges = l[['index', 'b', 'alighting_time']].values.tolist()    

    transit = pd.merge(
        l[['index', 'next', 'trip_id']], 
        l[['index', 'link_sequence', 'trip_id', 'time']],
        left_on=['trip_id', 'next'],
        right_on=['trip_id', 'link_sequence'],
    )
    transit_edges = transit[['index_x', 'index_y', 'time']].values.tolist()

    # ntlegs and footpaths
    ntlegs.loc[ntlegs['direction']=='access', 'time'] += ntlegs_penalty
    ntleg_edges =  ntlegs[['a', 'b', 'time']].values.tolist()
    footpaths_edges = footpaths[['a', 'b', 'time']].values.tolist()

    edges = boarding_edges + transit_edges + alighting_edges
    edges += footpaths_edges + ntleg_edges
    return sparse_matrix(edges)

def los_from_graph(
    csgraph, # graph is assumed to be a scipy csr_matrix
    node_index=None,
    pole_set=None, 
    sources=None, 
    cutoff=np.inf,
    od_set=None,
    ntlegs_penalty=1e9
):
    sources = pole_set if sources is None else sources
    # INDEX
    pole_list = sorted(list(pole_set)) # fix order
    source_list = [zone for zone in pole_list if zone in sources]

    zones = [node_index[zone] for zone in source_list]
    source_index = dict(zip(source_list, range(len(source_list))))
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    # SPARSE GRAPH
    dist_matrix, predecessors = dijkstra(
        csgraph=csgraph, 
        directed=True, 
        indices=zones, 
        return_predecessors=True,
        limit=cutoff+ntlegs_penalty
    )

    # LOS LAYOUT
    df = pd.DataFrame(dist_matrix)
    indexed_nodes = {v: k for k, v in node_index.items()}
    df.rename(columns=indexed_nodes, inplace=True)

    df.index = [zone for zone in pole_list if zone in sources]

    df.columns.name = 'destination'
    df.index.name = 'origin'

    stack = df[pole_list].stack()

    stack = df[pole_list].stack()
    stack.name = 'gtime'
    los = stack.reset_index()

    # QUETZAL FORMAT
    los = los.loc[los['gtime'] < np.inf]
    los.loc[los['origin'] != los['destination'],'gtime'] -= ntlegs_penalty
    if od_set is not None:
        tuples = [tuple(l) for l in  los[['origin', 'destination']].values.tolist()]
        los = los.loc[[t in od_set for t in tuples]]
        
    # BUILD PATH FROM PREDECESSORS
    od_list = los[['origin', 'destination']].values.tolist()
    paths = [
        [indexed_nodes[i] for i in get_path(predecessors, source_index[o], node_index[d])]
        for o, d in od_list
    ]

    los['path'] = paths
    return los

class PublicPathFinder:
    def __init__(self, model, walk_on_road=False):
        self.zones = model.zones.copy()
        self.links = engine.graph_links(model.links.copy())

        if walk_on_road:
            road_links = model.road_links.copy()
            road_links['time'] = road_links['walk_time']
            self.footpaths = pd.concat([model.footpaths, road_links, model.road_to_transit])
            self.ntlegs = pd.concat(
                [model.zone_to_road,  model.zone_to_transit]
            )
        else:
            self.footpaths = model.footpaths.copy()
            self.ntlegs = model.zone_to_transit.copy()

        try :
            self.centroids = model.centroids.copy()
        except AttributeError:
            self.centroids = self.zones.copy()
            self.centroids['geometry'] = self.centroids['geometry'].apply(
                lambda g: g.centroid
            )
            
        

    def first_link(self, path):
        for n in path:
            if n in self.links.index:
                return n
            
    def last_link(self, path):
        for n in reversed(path):
            if n in self.links.index:
                return n

    def build_route_zones(self, route_column):
        """
        find origin zones that are likely to be affected by the removal 
        each one of the routes
        """

        los = self.best_paths.copy()
        los['first_link'] = los['path'].apply(self.first_link)
        los['last_link'] = los['path'].apply(self.last_link)
        
        right = self.links[[route_column]]
        
        merged = pd.merge(los, right, left_on='first_link', right_index=True)
        merged = pd.merge(merged, right, left_on='last_link', right_index=True, suffixes=['_first',  '_last'])
        
        first = merged[['origin', route_column + '_first']]
        first.columns = ['zone', 'route']
        last = merged[['destination', route_column + '_last']]
        last.columns = ['zone', 'route']
        zone_route = pd.concat([first, last]).drop_duplicates(subset=['zone', 'route'])

        route_zone_sets = zone_route.groupby('route')['zone'].apply(set)
        self.route_zones = route_zone_sets.to_dict()

    def build_route_breaker(self, route_column='route_id'):
        self.build_route_zones(route_column=route_column)
    
    def build_mode_breaker(self, mode_column='route_type'):
        self.build_mode_combinations(mode_column='route_type')

    def build_mode_combinations(self, mode_column='route_type'):
    
        mode_list = sorted(list(set(self.links[mode_column])))
        boolean_array = list(itertools.product((True, False), repeat=len(mode_list)))
        mode_combinations = []
        for booleans in boolean_array:
            combination = {
                mode_list[i] 
                for i in range(len(mode_list) )
                if booleans[i]
            }
            mode_combinations.append(combination)
        self.mode_combinations = mode_combinations

        links = self.links.copy()
        links.drop('index', axis=1, inplace=True, errors='ignore')
        links.index.name = 'index'

        self.mode_links = links.reset_index().groupby(
            [mode_column]
        )['index'].apply(lambda s: set(s)).to_dict()

    def find_best_path(
        self, 
        od_set=None, 
        cutoff=np.inf, 
        ntlegs_penalty=1e9, 
        boarding_time=0,
        **kwargs
    ):
        pole_set=set(self.zones.index)
        matrix, node_index = adjacency_matrix(
            links=self.links, 
            ntlegs=self.ntlegs, 
            footpaths=self.footpaths,
            ntlegs_penalty=ntlegs_penalty, 
            boarding_time=boarding_time,
            **kwargs
        )

        los = los_from_graph(
            csgraph=matrix,
            node_index=node_index,
            pole_set=pole_set,
            od_set=od_set, 
            cutoff=cutoff,
            ntlegs_penalty=ntlegs_penalty
        )
        
        los['pathfinder_session'] = 'best_path'
        los['reversed'] = False
        self.best_paths = los

    def find_broken_route_paths(
        self, 
        od_set=None, 
        cutoff=np.inf, 
        route_column='route_id',
        ntlegs_penalty=1e9, 
        boarding_time=0,
        speedup=True,
        **kwargs
        
    ):
        pole_set=set(self.zones.index)
        do_set = {(d, o) for o, d in od_set} if od_set is not None else None
        to_concat = []
        iterator = tqdm(self.route_zones.items())
        for route_id, zones in iterator:
            if not speedup:
                zones = set(self.zones.index).intersection(set(self.ntlegs['a']))
            iterator.desc = 'breaking route: ' + str(route_id) + ' '
            matrix, node_index = adjacency_matrix(
                links=self.links.loc[self.links[route_column]!=route_id],
                ntlegs=self.ntlegs, 
                footpaths=self.footpaths,
                ntlegs_penalty=ntlegs_penalty, 
                boarding_time=boarding_time,
                **kwargs
            )

            los = los_from_graph(
                csgraph=matrix,
                node_index=node_index,
                pole_set=pole_set,
                od_set=od_set, 
                sources=zones,
                cutoff=cutoff,
                ntlegs_penalty=ntlegs_penalty
            )
            los['reversed'] = False
            los['broken_route'] = route_id
            los['pathfinder_session'] = 'route_breaker'

            to_concat.append(los)
            los = los_from_graph(
                csgraph=matrix.transpose(),
                node_index=node_index,
                pole_set=pole_set,
                od_set=do_set, 
                sources=zones,
                cutoff=cutoff,
                ntlegs_penalty=ntlegs_penalty
            )
            los[['origin', 'destination']] = los[['destination', 'origin']]
            los['path'] = los['path'].apply(lambda p: list(reversed(p)))
            los['reversed'] = True
            los['broken_route'] = route_id
            los['pathfinder_session'] = 'route_breaker'
            to_concat.append(los)
        self.broken_route_paths = pd.concat(to_concat)
    
    def find_broken_mode_paths(
        self, 
        od_set=None, 
        cutoff=np.inf, 
        mode_column='mode_type',
        ntlegs_penalty=1e9, 
        boarding_time=0,
        **kwargs
    ):
        pole_set=set(self.zones.index)
        to_concat = []
        iterator =  tqdm(self.mode_combinations)
        for combination in iterator:
            iterator.desc = 'breaking modes: ' + str(combination) + ' '
            matrix, node_index = adjacency_matrix(
                links=self.links.loc[~self.links[mode_column].isin(combination)],
                ntlegs=self.ntlegs, 
                footpaths=self.footpaths,
                ntlegs_penalty=ntlegs_penalty, 
                boarding_time=boarding_time,
                **kwargs
            )

            los = los_from_graph(
                csgraph=matrix,
                node_index=node_index,
                pole_set=pole_set,
                od_set=od_set, 
                cutoff=cutoff,
                ntlegs_penalty=ntlegs_penalty
            )
            los['reversed'] = False
            los['pathfinder_session'] = 'mode_breaker'
            los['broken_modes'] = [combination for i in range(len(los))]
            to_concat.append(los)
            
        self.broken_mode_paths = pd.concat(to_concat)

    def find_best_paths(
        self, 
        route_column='route_id',
        mode_column='route_type',
        broken_routes=False,
        broken_modes=False,
        drop_duplicates=True,
        speedup=True,
        cutoff=np.inf,
        od_set=None,
        boarding_time=0,
        **kwargs
    ):
        
        to_concat = []
        if broken_routes:
            self.find_best_path(
                boarding_time=boarding_time, 
                cutoff=cutoff, 
                od_set=od_set, 
                **kwargs
            ) # builds the graph
            to_concat.append(self.best_paths)
            self.build_route_breaker(route_column=route_column)
            self.find_broken_route_paths(
                speedup=speedup, 
                od_set=od_set, 
                boarding_time=boarding_time, 
                cutoff=cutoff,
                route_column=route_column
            )
            to_concat.append(self.broken_route_paths)

        if broken_modes:
            self.build_graph(**kwargs)
            self.build_mode_combinations(mode_column=mode_column)
            self.find_broken_mode_paths(
                od_set=od_set, 
                cutoff=cutoff,
                boarding_time=boarding_time,
                mode_column=mode_column
            )
            to_concat.append(self.broken_mode_paths)

        if (broken_modes or broken_routes) == False:
            self.find_best_path(
                cutoff=cutoff, 
                od_set=od_set, 
                boarding_time=boarding_time,
                **kwargs
            )
            to_concat.append(self.best_paths)

        self.paths = pd.concat(to_concat)
        self.paths['path'] = self.paths['path'].apply(tuple)

        if drop_duplicates:
            self.paths.drop_duplicates(subset=['path'], inplace=True)


