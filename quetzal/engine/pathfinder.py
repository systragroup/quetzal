# -*- coding: utf-8 -*-

from quetzal.engine import engine


import pandas as pd
import numpy as np
import networkx as nx

import itertools
from syspy.skims import skims
from syspy.spatial import spatial
from syspy.routing.frequency import graph as frequency_graph
import syspy.assignment.raw as assignment_raw
from tqdm import tqdm

def graph_from_links_and_ntlegs(
    links,
    ntlegs,
    footpaths=None,
    boarding_cost=300
):
    
    links = links.copy()

    links['index'] = links.index # to be consistent with frequency_graph

    nx_graph, _ = frequency_graph.graphs_from_links(
        links,
        include_edges=[],
        include_igraph=False,
        boarding_cost=boarding_cost
    )

    nx_graph.add_weighted_edges_from(ntlegs[['a', 'b', 'time']].values.tolist())

    if footpaths is not None:
        nx_graph.add_weighted_edges_from(
            footpaths[['a', 'b', 'time']].values.tolist()
        )
    return nx_graph

def path_and_duration_from_graph(
    nx_graph,
    pole_set,
    sources=None,
    reversed_nx_graph=None,
    reverse=False
):

    
        
    allpaths = {}
    alllengths = {}
    if sources is None:
        sources = pole_set
        
    
    # sources

    
    iterator = set(sources).intersection(list(pole_set))
    for pole in iterator:
        alllengths[pole], allpaths[pole] = nx.single_source_dijkstra(
            nx_graph, source=pole)

    duration_stack = assignment_raw.nested_dict_to_stack_matrix(
        alllengths, pole_set, name='gtime')
    path_stack = assignment_raw.nested_dict_to_stack_matrix(
        allpaths, pole_set, name='path')
    source_los = pd.merge(duration_stack, path_stack, on=['origin', 'destination'])
    source_los['reversed'] = False
    
    # targets
    if reverse or reversed_nx_graph is not None:
        if reversed_nx_graph is None:
            reversed_nx_graph = nx_graph.reverse()

        iterator = set(sources).intersection(list(pole_set))
        for pole in iterator:
            alllengths[pole], allpaths[pole] = nx.single_source_dijkstra(
                reversed_nx_graph, source=pole)

        duration_stack = assignment_raw.nested_dict_to_stack_matrix(
            alllengths, pole_set, name='gtime')
        path_stack = assignment_raw.nested_dict_to_stack_matrix(
            allpaths, pole_set, name='path')
        target_los = pd.merge(duration_stack, path_stack, on=['origin', 'destination'])
        target_los['reversed'] = True
        target_los['path'] = target_los['path'].apply(lambda x: list(reversed(x)))
        target_los[['origin', 'destination']] = target_los[['destination', 'origin']]

        los = pd.concat([source_los, target_los])
    else:

        return source_los

    return los


class PublicPathFinder:
    def __init__(self, model):
        self.zones = model.zones.copy()
        self.links = engine.graph_links(model.links.copy())
        self.footpaths = model.footpaths.copy()
        self.ntlegs = model.zone_to_transit.copy()

        try :
            self.centroids = model.centroids.copy()
        except AttributeError:
            self.centroids = self.zones.copy()
            self.centroids['geometry'] = self.centroids['geometry'].apply(
                lambda g: g.centroid
            )
        
    def find_best_path(self, **kwargs):
        pt_los, self.nx_graph = engine.path_and_duration_from_links_and_ntlegs(
            self.links,
            ntlegs=self.ntlegs,
            pole_set=set(self.zones.index),
            footpaths=self.footpaths,
            **kwargs
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

    def build_route_braker(self, route_column='route_id'):

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

    def find_broken_route_paths(self, speedup=False):
        los_dict = {}
        
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
                sources=set(zones)
            )
            
            los_dict[route] = los
            
        to_concat = [] 
        for route, los in los_dict.items():
            los['pathfinder_session'] = 'route_braker' 
            los['broken_route'] = route
            to_concat.append(los)
            
        self.broken_route_paths =  pd.concat(to_concat)


    def build_mode_braker(self, mode_column='route_type'):
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

    def find_broken_mode_paths(self):
        los_tuples = []
        
        iterator =  tqdm(self.mode_combinations)
        for combination in iterator:
            iterator.desc = 'breaking modes: ' + str(combination) + ' '
            
            broken = self.broken_mode_graph(combination)
            
            los = path_and_duration_from_graph(
                nx_graph=broken,
                reversed_nx_graph=None,
                pole_set=set(self.zones.index).intersection(set(self.ntlegs['a'])),
            )
            
            los_tuples.append([combination, los])
            
        to_concat = [] 
        for combination, los in los_tuples:
            los['pathfinder_session'] = 'mode_braker' 
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
        **kwargs
    ):

        self.find_best_path(**kwargs)
        to_concat = [self.best_paths]

        if broken_routes:
            self.build_route_braker(
                route_column=route_column
            )
            self.find_broken_route_paths(speedup=speedup)
            to_concat.append(self.broken_route_paths)

        if broken_modes:
            self.build_mode_combinations(mode_column=mode_column)
            self.find_broken_mode_paths()
            to_concat.append(self.broken_mode_paths)
        
        self.paths = pd.concat(to_concat)
        self.paths['path'] = self.paths['path'].apply(tuple)

        if drop_duplicates:
            self.paths.drop_duplicates(subset=['path'], inplace=True)


