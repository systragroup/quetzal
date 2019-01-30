# -*- coding: utf-8 -*-

import pandas as pd
from syspy.assignment import raw as raw_assignment
from tqdm import tqdm
import networkx as nx
import numpy as np


def jam_time(links, ref_time='time', flow='load', alpha=0.15, beta=4, capacity=1500):
    
    alpha = links['alpha'] if 'alpha' in links.columns else alpha
    beta = links['beta'] if 'beta' in links.columns else beta
    capacity = links['capacity'] if 'capacity' in links.columns else capacity
    
    return links[ref_time] * (1 + alpha * np.power((links[flow] / capacity), beta))

def z_prime(row, phi):
    delta = row['aux_flow'] - row['former_flow']
    return (delta * row['time'] * (row['former_flow'] + phi * delta)).sum()

def find_phi(links, inf=0, sup=1, tolerance=1e-6):
    
    if z_prime(links, inf) > 0:
        print('fin: ', inf)
        return inf
    if z_prime(links, sup) < 0:
        return sup 
    m = (inf + sup) / 2

    if (sup - inf) < tolerance:
        return m

    z_prime_m = z_prime(links, m)
    if z_prime_m == 0:
        return m
    elif z_prime_m < 0:
        inf = m
    elif z_prime_m > 0:
        sup = m
    return find_phi(links, inf, sup, tolerance)

class RoadPathFinder:
    def  __init__(self, model):
        self.zones = model.zones.copy()
        self.road_links = model.road_links.copy()
        self.zone_to_road = model.zone_to_road.copy()
        try :
            self.volumes = model.volumes.copy()
        except AttributeError:
            pass
        

    def aon_road_pathfinder(self, time='time', **kwargs):
        """
        * requires: zones, road_links, zone_to_road
        * builds: road_paths
        """
        road_links = self.road_links
        road_links['index'] = road_links.index
        indexed = road_links.set_index(['a', 'b']).sort_index()
        ab_indexed_dict = indexed['index'].to_dict()

        def node_path_to_link_path(road_node_list, ab_indexed_dict):
            tuples = [
                (road_node_list[i], road_node_list[i+1]) 
                for i in range(len(road_node_list)-1)
            ]
            road_link_list = [ab_indexed_dict[t] for t in tuples]
            return road_link_list

        road_graph = nx.DiGraph()
        road_graph.add_weighted_edges_from(
            self.road_links[['a', 'b', time]].values.tolist()
        )
        road_graph.add_weighted_edges_from(
            self.zone_to_road[['a', 'b', 'time']].values.tolist()
        )

        l = []
        for origin in tqdm(list(self.zones.index)):
            lengths, paths = nx.single_source_dijkstra(road_graph, origin)
            for destination in list(self.zones.index):
                try:
                    length = lengths[destination]
                    path = paths[destination]
                    node_path = path[1:-1]
                    link_path = node_path_to_link_path(node_path, ab_indexed_dict)
                    try:
                        ntlegs = [(path[0], path[1]), (path[-2], path[-1])]
                    except IndexError:
                        ntlegs = []
                    l.append( [origin, destination, path, node_path, link_path, ntlegs, length])
                except KeyError:
                    l.append( [origin, destination, path, node_path, link_path, ntlegs, length])
                    
        self.car_los = pd.DataFrame(
            l, 
            columns=['origin', 'destination', 'path','node_path', 'link_path', 'ntlegs', 'time']
        )

    def frank_wolfe_step(
        self, 
        iteration=0, 
        log=False, 
        speedup=True, 
        volume_column='volume_car'
    ):
        links = self.road_links # not a copy
        
        # a 
        links['eq_jam_time'] = links['jam_time'].copy()
        links['jam_time'] = jam_time(links, ref_time='time', flow='flow')
        
        # b
        self.aon_road_pathfinder(time='jam_time')
        merged = pd.merge(
            self.volumes, 
            self.car_los, 
            on=['origin', 'destination']
        )
        auxiliary_flows = raw_assignment.assign(
            merged[volume_column], 
            merged['link_path']
        )
        
        auxiliary_flows.columns = ['flow']
        links['aux_flow'] = auxiliary_flows['flow']
        links['aux_flow'].fillna(0, inplace=True)
        links['former_flow'] = links['flow'].copy()
        # c
        phi = 2 / (iteration + 2)
        if iteration > 0 and speedup:
            phi = find_phi(links)
        if phi == 0:
            return True
        if log: 
            print('step: %i ' % iteration,  'moved = %.1f %%' % (phi * 100))
        
        self.car_los['iteration'] = iteration
        self.car_los['phi'] = phi
        
        links['flow'] = (1 - phi) * links['flow'] + phi * links['aux_flow']
        links['flow'].fillna(0, inplace=True)

        return False # fin de l'algorithme

    def process_car_los(self, car_los_list):
        df = pd.concat(car_los_list).sort_values('iteration')
        phi_series = df.groupby('iteration')['phi'].first()
        phi_series = phi_series.loc[phi_series > 0]
        
        # will not work if road_links.index have mixed types
        groupby = df.groupby(df['link_path'].apply(lambda l: tuple(l)))
        iterations = groupby['iteration'].apply(lambda s: tuple(s))
        los = groupby.first()
        los['iterations'] = iterations
        
        def path_weight(iterations):
            w = 0
            for i in phi_series.index:
                phi = phi_series[i]
                if i in iterations:
                    w = w + (1 - w) * phi
                else: 
                    w = w * (1 - phi)
            return w
        
        combinations = {
            i: path_weight(i) 
            for i in set(los['iterations'].apply(lambda l: tuple(l)))
        }

        # weight
        los['weight'] = los['iterations'].apply(lambda l: combinations[l])
        
        # ntleg_time
        time_dict = self.zone_to_road.set_index(['a', 'b'])['time'].to_dict()
        los['ntleg_time'] = los['ntlegs'].apply(lambda p: sum([time_dict[l] for l in p]))

        # equilibrium_jam_time
        time_dict = self.road_links['eq_jam_time'].to_dict()
        los['link_eq_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['eq_time'] =  los['link_eq_time'] + los['ntleg_time']

        # actual_time
        time_dict = self.road_links['jam_time'].to_dict()
        los['link_actual_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['actual_time'] =  los['link_actual_time'] + los['ntleg_time']
        
        # free_time
        time_dict = self.road_links['time'].to_dict()
        los['link_free_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p])) 
        los['free_time'] = los['link_free_time'] + los['ntleg_time']
        return los.reset_index(drop=True)

    def get_relgap(self, car_los):
        los = car_los.copy()
        los = pd.merge(los, self.volumes, on=['origin', 'destination'])
        min_time = los.groupby(['origin', 'destination' ], as_index=False)['actual_time'].min()
        los = pd.merge(los, min_time , on=['origin', 'destination'], suffixes=['',  '_minimum'])
        los['delta'] = los['actual_time'] - los['actual_time_minimum']
        gap = (los['delta']* los['weight'] * los['volume_car']).sum()
        total_time = (los['actual_time_minimum'] * los['weight'] * los['volume_car']).sum()
        return gap / total_time

    def frank_wolfe(
        self, 
        all_or_nothing=False,
        reset_jam_time=True, 
        maxiters=20, 
        tolerance=0.01,
        log=False,
        *args, 
        **kwargs
    ):
        if all_or_nothing:
            self.aon_road_pathfinder(*args, **kwargs)
            return 
            
        if reset_jam_time:
            self.road_links['flow'] = 0
            self.road_links['jam_time'] = self.road_links['time']
            
        car_los_list = []
        for i in range(maxiters):
            done = self.frank_wolfe_step(iteration=i, log=log,*args, **kwargs)
            c = self.car_los
            car_los_list.append(c)

            los = self.process_car_los(car_los_list)
            relgap = self.get_relgap(los)
            if log:
                print('relgap = %.1f %%' % (relgap * 100))
            if i > 0:
                if done or relgap < tolerance:
                    break

            
        
        self.car_los = los.reset_index(drop=True)
