import numpy as np
import pandas as pd
from quetzal.engine.pathfinder_utils import sparse_matrix, build_index, parallel_dijkstra
from quetzal.engine.msa_utils import get_zone_index, assign_volume, default_bpr, limited_bpr, free_flow, jam_time, find_phi, get_car_los, find_beta
import numba as nb
import ray



class RoadPathFinder:
    def __init__(self, model):
        #self.zones = model.zones.copy()
        self.road_links = model.road_links.copy()
        self.zone_to_road = model.zone_to_road.copy()
        try:
            self.volumes = model.volumes.copy()
        except AttributeError:
            print('self.volumes does not exist. od generated with self.zones')
            od_set = []
            for o in model.zones.index.values:
                for d in model.zones.index.values:
                    if o!=d:
                        od_set.append((o,d))
            self.volumes = pd.DataFrame(od_set,columns=['origin','destination'])
            self.volumes['volume'] = 0   

        assert len(self.volumes)>0
    
    def msa(self, 
        maxiters=10,
        tolerance=0.01,
        log=False,
        vdf={'default_bpr': default_bpr,'limited_bpr':limited_bpr, 'free_flow': free_flow},
        volume_column='volume_car',
        method = 'bfw', 
        beta=None,
        num_cores=1,
        od_set=None,
        **kwargs):
        '''
        maxiters = 10 : number of iteration.
        tolerance = 0.01 : stop condition for RelGap. (in percent)
        log = False : log data on each iteration.
        vdf = {'default_bpr': default_bpr,'limited_bpr':limited_bpr, 'free_flow': free_flow} : dict of function for the jam time.
        volume_column='volume_car' : column of self.volumes to use for volume
        method = bfw, fw, msa, aon
        od_set = None. od_set
        beta = None. give constant value foir BFW betas. ex: [0.7,0.2,0.1]
        num_cores = 1 : for parallelization. 
        '''
        #preparation
        aon = True if method=='aon' else False
        v = self.volumes  
        if od_set is not None:
            v = v.set_index(['origin','destination']).loc[od_set].reset_index()

        self.zone_to_road_preparation(**kwargs)

        #create DataFrame with road_links and zone to road
        df = self.init_df(aon=aon)

        # CREATE EDGES FOR SPARSE MATRIX
        edges = df['time'].reset_index().values # to build the index once and for all
        index = build_index(edges)
        reversed_index = {v:k for k, v in index.items()}
        # index V with sparse indexes, zones in sparse indexes
        v,zones = get_zone_index(df,v,index)

        if aon==True:
            df['jam_time'] = df['time']
            self.car_los = get_car_los(v,df,index,reversed_index,zones,self.ntleg_penalty,num_cores)
            return

        sparse, _ = sparse_matrix(edges, index=index)
        
        _ , predecessors = parallel_dijkstra(sparse, directed=True, indices=zones, return_predecessors=True, num_core=num_cores, keep_running=True)

        
        df['sparse_a'] = df['a'].apply(lambda x:index.get(x))
        df['sparse_b'] = df['b'].apply(lambda x:index.get(x))
        volumes_sparse_keys = list(zip(df['sparse_a'],df['sparse_b']))
   

        odv = v[['o','d',volume_column]].values
        # BUILD PATHS FROM PREDECESSORS AND ASSIGN VOLUMES
        # then convert index back to each links indexes
        nb.set_num_threads(num_cores)
        ab_volumes = assign_volume(odv,predecessors,volumes_sparse_keys,reversed_index)
        
     
        
        df['auxiliary_flow'] = pd.Series(ab_volumes)
        df['auxiliary_flow'].fillna(0, inplace=True)
        df['flow'] += df['auxiliary_flow'] # do not do += in a cell where the variable is not created! bad
        if maxiters==0: # no iteration.
            df['jam_time'] = df['time']
        else:
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
            _, predecessors = parallel_dijkstra(sparse, directed=True, indices=zones, return_predecessors=True, num_core=num_cores, keep_running=True)

            # BUILD PATHS FROM PREDECESSORS AND ASSIGN VOLUMES
            ab_volumes = assign_volume(odv,predecessors,volumes_sparse_keys,reversed_index)
        
            df['auxiliary_flow'] = pd.Series(ab_volumes)
            df['auxiliary_flow'].fillna(0, inplace=True)
            if method == 'bfw': # if biconjugate: takes the 2 last direction : direction is flow-auxflow.
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
            
            if method == 'msa':
                phi = 1 / (i + 2)
            else:
                phi = find_phi(df,vdf,0,0.8,10)
            #
           
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
            if rel_gap[-1] <= tolerance:
                break

        # finish.. format to quetzal object
        self.road_links['flow'] = self.road_links.set_index(['a','b']).index.map(df['flow'].to_dict().get)
        self.road_links['jam_time'] = self.road_links.set_index(['a','b']).index.map(df['jam_time'].to_dict().get)
        #remove penalty from jam_time
        #keep it.
        #self.road_links['jam_time'] -= self.road_links['penalty']

        self.car_los = get_car_los(v,df,index,reversed_index,zones,self.ntleg_penalty,num_cores)
        self.relgap  = rel_gap
        if ray.is_initialized():
            print('shutdown ray')
            ray.shutdown()




    def zone_to_road_preparation(self, ntleg_penalty=1e9, access_time='time'):
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




    def init_df(self,aon=False):
        #check if columns exist
        if not aon:

            assert 'capacity' in self.road_links.columns, 'capacity not found in road_links columns. add a capacity for each link'
            
            if 'vdf' not in self.road_links.columns:
                self.road_links['vdf'] = 'default_bpr'
                print("vdf not found in road_links columns. Values set to 'default_bpr'")

            if 'alpha' not in self.road_links.columns:
                self.road_links['alpha'] = 0.15
                print("alpha not found in road_links columns. Values set to 0.15")
            if 'beta' not in self.road_links.columns:
                self.road_links['beta'] = 4
                print("beta not found in road_links columns. Values set to 4")

            for col in ['limit','penalty']:
                if col not in self.road_links.columns:
                    self.road_links[col] = 0
                    print(col, " not found in road_links columns. Values set to 0")
            
            columns = ['a', 'b','length', 'time', 'capacity', 'vdf', 'alpha', 'beta', 'limit','penalty']
            df = pd.concat([self.road_links[columns], self.zone_to_road[columns]]).set_index(['a', 'b'], drop=False)
            df['flow'] = 0
            df['auxiliary_flow'] = 0
        else:
            columns = ['a', 'b', 'time']
            df = pd.concat([self.road_links[columns], self.zone_to_road[columns]]).set_index(['a', 'b'], drop=False)
        
        return df
