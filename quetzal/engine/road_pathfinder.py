import networkx as nx
import numpy as np
import pandas as pd
from quetzal.engine.pathfinder_utils import sparse_los_from_nx_graph, sparse_matrix, build_index
from syspy.assignment import raw as raw_assignment
from quetzal.engine.msa_utils import get_zone_index, assign_volume,default_bpr, free_flow, jam_time, find_phi, get_car_los, find_beta
from quetzal.model.integritymodel import deprecated_method

from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm



class RoadPathFinder:
    def __init__(self, model):
        self.zones = model.zones.copy()
        self.road_links = model.road_links.copy()
        self.zone_to_road = model.zone_to_road.copy()
        try:
            self.volumes = model.volumes.copy()
        except AttributeError:
            pass

    def aon_road_pathfinder(
        self,
        time='time',
        ntleg_penalty=1e9,
        cutoff=np.inf,
        access_time='time',
        **kwargs,
    ):
        road_links = self.road_links
        road_links['index'] = road_links.index
        indexed = road_links.set_index(['a', 'b']).sort_index()
        ab_indexed_dict = indexed['index'].to_dict()

        road_graph = nx.DiGraph()
        road_graph.add_weighted_edges_from(
            self.road_links[['a', 'b', time]].values.tolist()
        )
        zone_to_road = self.zone_to_road.copy()
        zone_to_road['time'] = zone_to_road[access_time]
        zone_to_road = zone_to_road[['a', 'b', 'direction', 'time']]
        zone_to_road.loc[zone_to_road['direction'] == 'access', 'time'] += ntleg_penalty
        road_graph.add_weighted_edges_from(
            zone_to_road[['a', 'b', 'time']].values.tolist()
        )

        def node_path_to_link_path(road_node_list, ab_indexed_dict):
            tuples = list(zip(road_node_list[:-1], road_node_list[1:]))
            road_link_list = [ab_indexed_dict[t] for t in tuples]
            return road_link_list

        def path_to_ntlegs(path):
            try:
                return [(path[0], path[1]), (path[-2], path[-1])]
            except IndexError:
                return []

        los = sparse_los_from_nx_graph(
            road_graph,
            pole_set=set(self.zones.index),
            cutoff=cutoff + ntleg_penalty,
            **kwargs
        )
        los['node_path'] = los['path'].apply(lambda p: p[1:-1])
        los['link_path'] = los['node_path'].apply(
            lambda p: node_path_to_link_path(p, ab_indexed_dict)
        )
        los['ntlegs'] = los['path'].apply(path_to_ntlegs)
        los.loc[los['origin'] != los['destination'], 'gtime'] -= ntleg_penalty
        self.car_los = los.rename(columns={'gtime': 'time'})

    def frank_wolfe_step(
        self,
        iteration=0,
        log=False,
        speedup=True,
        volume_column='volume_car',
        vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
        **kwargs
    ):
        links = self.road_links  # not a copy

        # a
        links['eq_jam_time'] = links['jam_time']
        links['jam_time'] = jam_time(links, vdf=vdf,flow='flow')

        # b
        self.aon_road_pathfinder(time='jam_time',**kwargs)
        merged = pd.merge(
            self.volumes,
            self.car_los,
            on=['origin', 'destination']
        )
        auxiliary_flows = raw_assignment.fast_assign(
            merged[volume_column],
            merged['link_path']
        )

        links['aux_flow'] = auxiliary_flows
        links['aux_flow'].fillna(0, inplace=True)
        links['former_flow'] = links['flow'].copy()
        # c
        phi = 2 / (iteration + 2)
        if iteration > 0 and speedup:
            phi = find_phi(links,vdf)
        if phi == 0:
            return True
        if log:
            print('step: %i ' % iteration, 'moved = %.1f %%' % (phi * 100))

        self.car_los['iteration'] = iteration
        self.car_los['phi'] = phi

        links['flow'] = (1 - phi) * links['flow'] + phi * links['aux_flow']
        links['flow'].fillna(0, inplace=True)
        return False  # fin de l'algorithme

    def process_car_los(self, car_los_list):
        df = pd.concat(car_los_list).sort_values('iteration')
        phi_series = df.groupby('iteration')['phi'].first()
        phi_series = phi_series.loc[phi_series > 0]

        # will not work if road_links.index have mixed types
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
        los['eq_time'] = los['link_eq_time'] + los['ntleg_time']

        # actual_time
        time_dict = self.road_links['jam_time'].to_dict()
        los['link_actual_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['actual_time'] = los['link_actual_time'] + los['ntleg_time']

        # free_time
        time_dict = self.road_links['time'].to_dict()
        los['link_free_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['free_time'] = los['link_free_time'] + los['ntleg_time']
        return los.reset_index(drop=True)

    def get_relgap(self, car_los):
        los = car_los.copy()
        los = pd.merge(los, self.volumes, on=['origin', 'destination'])
        min_time = los.groupby(['origin', 'destination'], as_index=False)['actual_time'].min()
        los = pd.merge(los, min_time, on=['origin', 'destination'], suffixes=['', '_minimum'])
        los['delta'] = los['actual_time'] - los['actual_time_minimum']
        gap = (los['delta'] * los['weight'] * los['volume_car']).sum()
        total_time = (los['actual_time_minimum'] * los['weight'] * los['volume_car']).sum()
        return gap / total_time

    @deprecated_method
    def frank_wolfe(
        self,
        all_or_nothing=False,
        reset_jam_time=True,
        maxiters=20,
        tolerance=0.01,
        log=False,
        vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
        *args,
        **kwargs
    ):
        if all_or_nothing:
            self.aon_road_pathfinder(*args, **kwargs)
            return

        assert 'vdf' in self.road_links.columns
        assert 'capacity' in self.road_links.columns

        if reset_jam_time:
            self.road_links['flow'] = 0
            self.road_links['jam_time'] = self.road_links['time']

        car_los_list = []
        for i in range(maxiters):
            done = self.frank_wolfe_step(iteration=i, log=log, vdf=vdf, *args, **kwargs)
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


    def msa(self, 
        maxiters=10,
        tolerance=0.01,
        log=False,
        vdf={'default_bpr': default_bpr,'free_flow':free_flow},
        volume_column='volume_car',
        BFW = False,
        beta=None):
        '''
        maxiters = 10 : number of iteration.
        tolerance = 0.01 : stop condition for RelGap.
        log = False : log data on each iteration.
        vdf = {'default_bpr': default_bpr,'free_flow':free_flow} : dict of function for the jam time.
        volume_column='volume_car' : column of self.volumes to use for volume
        '''

        v = self.volumes

        self.zone_to_road_preparation()

        #create DataFrame with road_links and zone top road
        df = self.init_df()

        # CREATE EDGES FOR SPARSE MATRIX
        edges = df['time'].reset_index().values # to build the index once and for all
        index = build_index(edges)
        reversed_index = {v:k for k, v in index.items()}
        # index V with sparse indexes, zones in sparse indexes
        v,zones = get_zone_index(df,v,index)
        
        sparse, _ = sparse_matrix(edges, index=index)
        
        _, predecessors = dijkstra(sparse, directed=True, indices=zones, return_predecessors=True)
        
        odv = list(zip(v['o'].values, v['d'].values, v[volume_column].values))
        
        # BUILD PATHS FROM PREDECESSORS AND ASSIGN VOLUMES
        # then convert index back to each links indexes
        ab_volumes = assign_volume(odv,predecessors,reversed_index)
        
        
        df['auxiliary_flow'] = pd.Series(ab_volumes)
        df['auxiliary_flow'].fillna(0, inplace=True)
        df['flow'] += df['auxiliary_flow'] # do not do += in a cell where the variable is not created! bad
        df['jam_time'] = jam_time(df,vdf,'flow')
        df['jam_time'].fillna(df['time'], inplace=True)

        rel_gap = []
        if log:
            print('iteration | Phi |  Rel Gap (%)')

        for i in range(maxiters):
            # CREATE EDGES AND SPARSE MATRIX
            edges = df['jam_time'].reset_index().values # build the edges again, useless
            sparse, _ = sparse_matrix(edges, index=index)
            #shortest path
            dist_matrix, predecessors = dijkstra(sparse, directed=True, indices=zones, return_predecessors=True)

            odv = list(zip(v['o'].values, v['d'].values, v[volume_column].values)) # volume for each od

            # BUILD PATHS FROM PREDECESSORS AND ASSIGN VOLUMES
            # then convert index back to each links indexes
            ab_volumes = assign_volume(odv,predecessors,reversed_index)
            df['auxiliary_flow'] = pd.Series(ab_volumes)
            df['auxiliary_flow'].fillna(0, inplace=True)
            if BFW: # if biconjugate: takes the 2 last direction : direction is flow-auxflow.
                if i>=2:
                    if not beta:
                        df['derivative'] = jam_time(df,vdf,'flow',der=True)
                        b = find_beta(df,phi) #this is the previous phi (phi_-1)
                    else :
                        assert sum(beta)==1 , 'beta must sum to 1.'
                        b = beta
                    df['auxiliary_flow'] = b[0]*df['auxiliary_flow'] + b[1]*df['s_k-1'] + b[2]*df['s_k-2']

                if i>0 :
                    df['s_k-2'] = df['s_k-1']
                df['s_k-1'] =  df['auxiliary_flow'] 
            
        
            phi = find_phi(df,vdf,0,0.8,10)
           # phi Olga
           # df['direction'] = df['auxiliary_flow']-df['flow']
           # df['derivative'] = jam_time(df,vdf,'flow',der=True)
           # phi = -sum(df['jam_time']*df['direction']) / sum(df['derivative']*df['direction']**2)


            #  modelling transport eq 11.11. SUM currentFlow x currentCost - SUM AONFlow x currentCost / SUM currentFlow x currentCost
            rel_gap.append(100*(np.sum(df['flow']*df['jam_time']) - np.sum(df['auxiliary_flow']*df['jam_time']))/np.sum(df['flow']*df['jam_time']))
            if log:
                print(i, round(phi,4), round(rel_gap[-1],3))
            #Aux-flow direction a l'etape

            # conventional frank-wolfe
            # df['flow'] + phi*(df['auxiliary_flow'] - df['flow'])  flow +step x direction
            df['flow'] = (1 - phi) * df['flow'] + phi * df['auxiliary_flow']     
            df['flow'].fillna(0, inplace=True)


            df['jam_time'] = jam_time(df,vdf,'flow')
            df['jam_time'].fillna(df['time'], inplace=True)
            if rel_gap[-1] <= tolerance*100:
                break

            
        self.road_links['flow'] = self.road_links.set_index(['a','b']).index.map(df['flow'].to_dict().get)
        self.road_links['jam_time'] = self.road_links.set_index(['a','b']).index.map(df['jam_time'].to_dict().get)
        #remove penalty from jam_time
        self.road_links['jam_time'] -= self.road_links['penalty']

        self.car_los = get_car_los(v,df,index,reversed_index,zones,self.ntleg_penalty)
        self.relgap  = rel_gap




    def zone_to_road_preparation(self,time='time', ntleg_penalty=1e9, access_time='time'):
        # prepare zone_to_road_links to the same format as road_links
        # and initialize it's parameters
        zone_to_road = self.zone_to_road.copy()
        zone_to_road['time'] = zone_to_road[access_time]
        zone_to_road['length'] = np.nan
        zone_to_road = zone_to_road[['a', 'b','length', 'direction', 'time']]
        zone_to_road.loc[zone_to_road['direction'] == 'access', 'time'] += ntleg_penalty 
        
        zone_to_road['capacity'] = np.nan
        zone_to_road['vdf'] = 'free_flow'
        zone_to_road['alpha'] = 0
        zone_to_road['limit'] = 0
        zone_to_road['beta'] = 0
        zone_to_road['penalty'] = 0
        self.zone_to_road = zone_to_road
        # keep track of it (need to substract it in car_los)
        self.ntleg_penalty = ntleg_penalty




    def init_df(self):
        #check if columns exist
        assert 'vdf' in self.road_links.columns, 'vdf not found in road_links columns.'
        assert 'capacity' in self.road_links.columns, 'capacity not found in road_links columns.'
        for col in ['limit','penalty','alpha','beta']:
            if col not in self.road_links.columns:
                self.road_links[col] = 0
                print(col, " not found in road_links columns. Values set to 0")
        
        columns = ['a', 'b','length', 'time', 'capacity', 'vdf', 'alpha', 'beta', 'limit','penalty']
        df = pd.concat([self.road_links[columns], self.zone_to_road[columns]]).set_index(['a', 'b'], drop=False)
        df['flow'] = 0
        df['auxiliary_flow'] = 0
        return df
