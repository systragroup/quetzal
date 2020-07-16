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

# def graph_from_links_and_ntlegs(
#     links,
#     ntlegs,
#     footpaths=None,
#     boarding_cost=300
# ):
    
#     links = links.copy()

#     links['index'] = links.index # to be consistent with frequency_graph

#     nx_graph, _ = frequency_graph.graphs_from_links(
#         links,
#         include_edges=[],
#         include_igraph=False,
#         boarding_cost=boarding_cost
#     )

#     nx_graph.add_weighted_edges_from(ntlegs[['a', 'b', 'time']].values.tolist())

#     if footpaths is not None:
#         nx_graph.add_weighted_edges_from(
#             footpaths[['a', 'b', 'time']].values.tolist()
#         )
#     return nx_graph

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
        cutoff=cutoff+ntlegs_penalty, **kwargs)
    source_los['reversed'] = False
    
    reverse = reverse or reversed_nx_graph is not None
    if reverse:
        if reversed_nx_graph is None:
            reversed_nx_graph = nx_graph.reverse()
        
        target_los=sparse_los_from_nx_graph(
            reversed_nx_graph, pole_set, sources=sources, 
            cutoff=cutoff+ntlegs_penalty, od_set=od_set, **kwargs)
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

    # SPARSE GRAPH
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

def dumps_kwargs(kwargs):
    data = kwargs
    for key in ['nx_graph', 'reversed_nx_graph']:
        try:
            data[key] = nx.node_link_data(data[key])
        except KeyError:
            pass
    try:
        data['pole_set'] = list(data['pole_set'])
    except KeyError: 
        pass
    
    try:
        data['sources'] = list(data['sources'])
    except KeyError: 
        pass
    
    s = json.dumps(data)
    return s

def dump_kwargs(kwargs, filepath):
    s = dumps_kwargs(kwargs)
    with open(filepath, 'w') as file:
        file.write(s)

def loads_kwargs(s):
    data = json.loads(s)
    for key in ['nx_graph', 'reversed_nx_graph']:
        try:
            data[key] = nx.node_link_graph(data[key])
        except KeyError:
            pass
    
    return data

def load_kwargs(filepath):
    with open(filepath, 'r') as file:
        s = file.read()
    return loads_kwargs(s)

def dumps_result(result):
    # reset_index to be sure index is unique
    return result.reset_index(drop=True).to_json()

def dump_result(result, filepath):
    s = dumps_result(result)
    with open(filepath, 'w') as file:
        file.write(s)

def load_result(filepath):
    return pd.read_json(filepath)

def path_and_duration_from_graph_json(input_json, output_json):
    kwargs = load_kwargs(input_json)
    result = path_and_duration_from_graph(**kwargs)
    dump_result(result, output_json)

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
            
    def build_graph(self, **kwargs):
        self.nx_graph = engine.multimodal_graph(
            self.links,
            ntlegs=self.ntlegs,
            pole_set=set(self.zones.index),
            footpaths=self.footpaths,
            **kwargs
        )
        
    def find_best_path(self, cutoff=np.inf, od_set=None, **kwargs):
        self.nx_graph = engine.multimodal_graph(
            self.links,
            ntlegs=self.ntlegs,
            pole_set=set(self.zones.index),
            footpaths=self.footpaths,
            **kwargs
        )
        pt_los = path_and_duration_from_graph(
            self.nx_graph, 
            pole_set=set(self.zones.index),
            cutoff=cutoff,
            od_set=od_set
        )

        pt_los['pathfinder_session'] = 'best_path'
        pt_los['reversed'] = False
        self.best_paths = pt_los

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

        self.reversed_nx_graph = self.nx_graph.reverse()

        links = self.links.copy()
        links.drop('index', axis=1, inplace=True, errors='ignore')
        links.index.name = 'index'

        self.route_links = links.reset_index().groupby(
            [route_column]
        )['index'].apply(lambda s: set(s)).to_dict()

    def broken_route_graphs(
        self,
        route,
        nx_graph=None, 
        reversed_nx_graph=None,
    ):
        if nx_graph is None:
            nx_graph = self.nx_graph

        if reversed_nx_graph is None:
            reversed_nx_graph = self.reversed_nx_graph

        to_drop = self.route_links[route]
        
        ebunch = [
            e for e in list(nx_graph.edges) 
            if to_drop.intersection(e) #  un des noeuds de l'arrête appartient aussi à la route
        ]

        reversed_ebunch = [
            e for e in list(reversed_nx_graph.edges) 
            if to_drop.intersection(e)
        ]
        
        broken = nx_graph.copy()
        reversed_broken = reversed_nx_graph.copy()
        
        broken.remove_edges_from(ebunch)
        reversed_broken.remove_edges_from(reversed_ebunch)
        return broken, reversed_broken

    def find_broken_route_paths(
        self, 
        speedup=False, 
        workers=1, 
        cutoff=np.inf, 
        *args, 
        **kwargs
    ):

        if workers > 1:
            self.find_broken_route_paths_parallel(
                speedup=speedup, 
                workers=workers, 
                *args, 
                **kwargs
            )
            return 
        los_tuples = []
        iterator =  tqdm(self.route_zones.items())
        for route, zones in iterator:
            if not speedup:
                zones = set(self.zones.index).intersection(set(self.ntlegs['a']))

            iterator.desc = 'breaking route: ' + str(route) + ' '
            
            broken, reversed_broken = self.broken_route_graphs(route)

            los = path_and_duration_from_graph(
                nx_graph=broken,
                reversed_nx_graph=reversed_broken,
                pole_set=set(self.zones.index).intersection(set(self.ntlegs['a'])),
                sources=set(zones),
                cutoff=cutoff,
                **kwargs
            )
            
            los_tuples.append([route, los]) 
            
        to_concat = [] 
        for route, los in los_tuples:
            los['pathfinder_session'] = 'route_breaker' 
            los['broken_route'] = route
            to_concat.append(los)
            
        self.broken_route_paths =  pd.concat(to_concat)

    def find_broken_route_paths_parallel(self, speedup=False, workers=2, sleep=0, *args, **kwargs):

        # fichier ou est stocké le sous process «path_and_duration...»
        root = filepaths.__file__.split('filepaths.py')[0]
        subprocess_file = 'subprocess_path_and_duration_from_graph_json.py'
        subprocess_filepath = root + subprocess_file

        iterator =  tqdm(self.route_zones.items())
        
        routes = []
        kwarg_list = []

        for route, zones in iterator:
            if not speedup:
                zones = set(self.zones.index).intersection(set(self.ntlegs['a']))

            iterator.desc = 'breaking route: ' + str(route) + ' '
            
            broken, reversed_broken = self.broken_route_graphs(route)

            kwarg_dict = {
                'nx_graph': broken,
                'reversed_nx_graph': reversed_broken,
                'sources': set(zones),
                'pole_set': set(self.zones.index).intersection(set(self.ntlegs['a'])),
            }
            routes.append(route)
            kwarg_list.append(kwarg_dict)
            
        results = parallel_call.parallel_call_subprocess(
            subprocess_filepath=subprocess_filepath, 
            kwarg_list=kwarg_list, 
            dump_kwargs=dump_kwargs, 
            load_result=load_result,
            workers=workers,
            sleep=sleep
        )
            
        los_tuples = list(zip(routes, results))
        to_concat = [] 
        for route, los in los_tuples:
            los['pathfinder_session'] = 'route_breaker' 
            los['broken_route'] = route
            to_concat.append(los)
            
        self.broken_route_paths =  pd.concat(to_concat)


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

    def broken_mode_graph(
        self,
        drop_modes,
        nx_graph=None, 
    ):

        if nx_graph is None:
            nx_graph = self.nx_graph

        if len(drop_modes) == 0:
            return nx_graph

        to_drop = set.union(*[self.mode_links[mode] for mode in drop_modes])
        
        ebunch = [
            e for e in list(nx_graph.edges) 
            if to_drop.intersection(e)  #  un des noeuds de l'arrête appartient aussi à la route
        ]
        
        broken = nx_graph.copy()
        broken.remove_edges_from(ebunch)

        return broken

    def find_broken_mode_paths(self, workers=1, cutoff=np.inf, *args, **kwargs):

        if workers > 1:
            self.find_broken_mode_paths_parallel(
                workers=workers,
                *args,
                **kwargs
            )
            return 
        los_tuples = []
        iterator =  tqdm(self.mode_combinations)
        for combination in iterator:
            if len(combination) == 0:
                pass
                #continue # their is no broken mode
            iterator.desc = 'breaking modes: ' + str(combination) + ' '
            
            broken = self.broken_mode_graph(combination)
            
            los = path_and_duration_from_graph(
                nx_graph=broken,
                reversed_nx_graph=None,
                pole_set=set(self.zones.index).intersection(set(self.ntlegs['a'])),
                cutoff=cutoff,
                **kwargs
            )

            los_tuples.append([combination, los])
            
        to_concat = [] 
        for combination, los in los_tuples:
            los['pathfinder_session'] = 'mode_breaker' 
            los['broken_modes'] = [combination for i in range(len(los))]
            to_concat.append(los)
            
        self.broken_mode_paths =  pd.concat(to_concat)




    def find_broken_mode_paths_parallel(self, workers=2, sleep=0, leave=False,  *args, **kwargs):

        # fichier ou est stocké le sous process «path_and_duration...»
        root = filepaths.__file__.split('filepaths.py')[0]
        subprocess_file = 'subprocess_path_and_duration_from_graph_json.py'
        subprocess_filepath = root + subprocess_file

        iterator =  tqdm(self.mode_combinations)
        
        combinations = []
        kwarg_list = []
        
        for combination in iterator:
            if len(combination) == 0:
                continue # their is no broken mode
                
            iterator.desc = 'breaking modes: ' + str(combination) + ' '
            broken = self.broken_mode_graph(combination)
            
            kwarg_dict = {
                'nx_graph': broken,
                'pole_set': set(self.zones.index).intersection(set(self.ntlegs['a']))
            }
            combinations.append(combination)
            
            kwarg_list.append(kwarg_dict)
            
        results = parallel_call.parallel_call_subprocess(
            subprocess_filepath=subprocess_filepath, 
            kwarg_list=kwarg_list, 
            dump_kwargs=dump_kwargs, 
            load_result=load_result,
            workers=workers,
            sleep=sleep,
            leave=leave,
        )
            
        los_tuples = list(zip(combinations, results))

        to_concat = [] 
        for combination, los in los_tuples:
            los['pathfinder_session'] = 'mode_breaker' 
            los['broken_modes'] = [combination for i in range(len(los))]
            to_concat.append(los)

        self.broken_mode_paths =  pd.concat(to_concat)

    def find_best_paths(
        self, 
        route_column='route_id',
        mode_column='route_type',
        broken_routes=False,
        broken_modes=False,
        drop_duplicates=True,
        speedup=False,
        route_workers=1,
        mode_workers=1,
        cutoff=np.inf,
        od_set=None,
        **kwargs
    ):
        to_concat = []
        if broken_routes:
            self.find_best_path(cutoff=cutoff,**kwargs) # builds the graph
            to_concat.append(self.best_paths)
            self.build_route_breaker(
                route_column=route_column
            )
            self.find_broken_route_paths(
                speedup=speedup, od_set=od_set, 
                workers=route_workers, cutoff=cutoff)
            to_concat.append(self.broken_route_paths)

        if broken_modes:
            self.build_graph(**kwargs)
            self.build_mode_combinations(mode_column=mode_column)
            self.find_broken_mode_paths(
                workers=mode_workers, 
                od_set=od_set, cutoff=cutoff)
            to_concat.append(self.broken_mode_paths)


        if (broken_modes or broken_routes) == False:
            self.find_best_path(cutoff=cutoff, od_set=od_set)
            to_concat.append(self.best_paths)

        self.paths = pd.concat(to_concat)
        self.paths['path'] = self.paths['path'].apply(tuple)

        if drop_duplicates:
            self.paths.drop_duplicates(subset=['path'], inplace=True)


