# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.road_pathfinder import RoadPathFinder
from quetzal.engine import nested_logit, optimal_strategy
from quetzal.model import model, preparationmodel

from syspy.assignment import raw as raw_assignment
from syspy.skims import skims
from tqdm import tqdm
import networkx as nx

class OptimalModel(preparationmodel.PreparationModel):

    def get_optimal_strategy_edges(
        self,
        boarding_cost = 300,
        alighting_cost = 0,
        alpha=0.5,
        target=None,
        inf=1e9
        ):
        zero = 1 / inf
        links = self.links.copy()
        
        # transit edges
        links['j'] = [tuple(l) for l in links[['b', 'trip_id']].values]
        links['i'] = [tuple(l) for l in links[['a', 'trip_id']].values]
        links['f'] = inf
        links['c'] = links['time']
        transit_edges = links[['i','j', 'f' ,'c']].reset_index().values.tolist()

        # boarding edges
        links.index = 'boarding_' + links['index'].astype(str)
        links['f'] = 1 / links['headway'] / alpha
        links['c'] = boarding_cost
        boarding_edges = links[['a', 'i', 'f' ,'c' ]].reset_index().values.tolist()

        # alighting edges
        links.index = 'alighting_' + links['index'].astype(str)
        links['f'] = inf
        links['c'] = alighting_cost 
        alighting_edges = links[['j', 'b', 'f' ,'c']].reset_index().values.tolist()

        # access edges 
        access = self.zone_to_transit.copy()
        if target is not None:
            # we do not want to egress to a destination that is not the target
            access = access.loc[(access['direction'] == 'access') | (access['b'] == target)]
        access['f'] = inf
        access['c'] = access['time'] 
        access_edges = access[['a', 'b', 'f' ,'c']].reset_index().values.tolist()

        # footpaths 
        footpaths = self.footpaths.copy()
        footpaths['f'] = inf
        footpaths['c'] = footpaths['time']
        footpaths_edges = footpaths[['a', 'b', 'f' ,'c']].reset_index().values.tolist()

        edges = boarding_edges + transit_edges + alighting_edges + access_edges + footpaths_edges
        edges = [tuple(e) for e in edges]
        
        return edges

    def step_strategy_finder(self, *args, **kwargs):
    
        s_dict = {}
        node_df_list = []

        for destination in tqdm(self.zones.index):
            edges = self.get_optimal_strategy_edges(target=destination, *args, **kwargs)
            strategy, u, f = optimal_strategy.find_optimal_strategy(edges, destination)
            s_dict[destination] = strategy
            node_df = pd.DataFrame({'f': pd.Series(f), 'u':pd.Series(u)})
            node_df['destination'] = destination
            node_df_list.append(node_df)

        optimal_strategy_nodes = pd.concat(node_df_list)
        edges = self.get_optimal_strategy_edges(*args, **kwargs)
        optimal_strategy_sets = pd.Series(s_dict).apply(list)
        optimal_strategy_edges = pd.DataFrame(
            edges, columns=['ix', 'i', 'j', 'f', 'c']).set_index('ix')
        assert optimal_strategy_edges.index.is_unique
        
        self.optimal_strategy_edges = optimal_strategy_edges
        self.optimal_strategy_sets = optimal_strategy_sets
        self.optimal_strategy_nodes = optimal_strategy_nodes

    def step_strategy_assignment(self, volume_column):

        destination_indexed_volumes = self.volumes.set_index(['destination', 'origin'])[volume_column]
        destination_indexed_nodes =  self.optimal_strategy_nodes.set_index(
            'destination', append=True).swaplevel()
        destination_indexed_strategies = self.optimal_strategy_sets
        indexed_edges = self.optimal_strategy_edges[['i', 'j', 'f', 'c']]

        node_volume = {}
        edge_volume = {}
        
        for destination in tqdm(self.zones.index):

            try:
                sources = destination_indexed_volumes.loc[destination]
                subset = destination_indexed_strategies.loc[destination]
                edges = indexed_edges.loc[subset].reset_index().values.tolist()
                f = destination_indexed_nodes.loc[destination]['f'].to_dict()
                u = destination_indexed_nodes.loc[destination]['u'].to_dict()
            except KeyError:
                continue

            node_v, edge_v = optimal_strategy.assign_optimal_strategy(sources, edges, u, f)

            for k, v in node_v.items():
                node_volume[k] = node_volume.get(k, 0) + v
            for k, v in edge_v.items():
                edge_volume[k] = edge_volume.get(k, 0) + v
                
        loaded_edges = self.optimal_strategy_edges
        loaded_edges[volume_column] = pd.Series(edge_volume)
        df = loaded_edges[['i', 'j', volume_column]].dropna(subset=[volume_column])
        links = self.links.copy()
        links['index'] = links.index
        # transit edges
        links['j'] = [tuple(l) for l in links[['b', 'trip_id']].values]
        links['i'] = [tuple(l) for l in links[['a', 'trip_id']].values]

        transit = pd.merge(links, df, on=['i', 'j'])
        boardings = pd.merge(links,  df, left_on=['a', 'i'], right_on=['i', 'j'])
        alightings = pd.merge(links,  df, left_on=['j', 'b'], right_on=['i', 'j'])

        loaded_links = self.links.copy()
        loaded_links[volume_column] = transit.set_index('index')[volume_column]
        loaded_links['boardings'] = boardings.set_index('index')[volume_column]
        loaded_links['alightings'] = alightings.set_index('index')[volume_column]

        loaded_nodes = self.nodes.copy()
        loaded_nodes['boardings'] = boardings.groupby('a')[volume_column].sum()
        loaded_nodes['alightings'] = alightings.groupby('b')[volume_column].sum()
        
        self.loaded_edges = loaded_edges
        self.loaded_nodes = loaded_nodes
        self.loaded_links = loaded_links
