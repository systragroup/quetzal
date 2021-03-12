# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.road_pathfinder import RoadPathFinder
from quetzal.engine import nested_logit, optimal_strategy
from quetzal.model import preparationmodel

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
        if 'boarding_stochastic_utility' in links.columns:
            links['f'] *= np.exp(links['boarding_stochastic_utility'])
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

        edges = access_edges + boarding_edges + transit_edges + alighting_edges +  footpaths_edges
        edges = [tuple(e) for e in edges]
        
        return edges

    def step_strategy_finder(self, *args, **kwargs):
    
        s_dict = {}
        node_df_list = []

        all_edges = self.get_optimal_strategy_edges(*args, **kwargs)
        for destination in tqdm(self.zones.index):
            forbidden = set(self.zones.index) - {destination}
            edges = [e for e in all_edges if e[2] not in forbidden]
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

        nodes = optimal_strategy_nodes.copy()
        nodes.index.name = 'origin'
        nodes.set_index('destination', append=True, inplace=True)
        pt_los = nodes.loc[self.zones.index]['u'].reset_index().rename(columns={'u': 'gtime'})
        pt_los['pathfinder_session'] = 'optimal_strategy'
        self.pt_los = pt_los

    def step_strategy_assignment(self, volume_column, road=False):

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

        if road:
            self.road_links[volume_column] = raw_assignment.assign(
                volume_array=list(self.loaded_links[volume_column]), 
                paths=list(self.loaded_links['road_link_list'])
            )
            # todo remove 'load' from analysis module: 
            self.road_links['load'] = self.road_links[volume_column]

    def get_aggregated_edges(self, origin, destination):
        
        # restrection to the destination edges
        edges = self.optimal_strategy_edges[['i', 'j', 'f', 'c']].copy()
        edges = edges.loc[self.optimal_strategy_sets.loc[destination]]
        edges['ix'] = edges.index

        # removing the edges that are non relevant (p<1e-6)
        f_total = edges.groupby('i')[['f']].sum()
        edges = pd.merge(edges, f_total, left_on='i', right_index=True, suffixes=['', '_total'])
        edges['p'] = np.round(edges['f'] / edges['f_total'], 6)
        edges = edges.loc[edges['p'] > 0]

        #restriction to the origin
        g = nx.DiGraph()
        for e in edges.to_dict(orient='records'):
            g.add_edge(e['i'], e['j'])

        paths = list(nx.all_simple_paths(g, source=origin, target=destination))
        nodes = set.union(*[set(p) for p in paths])
        ode = edges.loc[edges['i'].isin(nodes) & edges['j'].isin(nodes)]

        # transform node -> (node, trip_id) to node -> trip_id
        links = self.links.copy()
        links['j'] = [tuple(l) for l in links[['b', 'trip_id']].values]
        links['i'] = [tuple(l) for l in links[['a', 'trip_id']].values]
        transit = pd.merge(links, ode[['i', 'j', 'ix']], on=['i', 'j'])
        boardings = pd.merge(links[['a', 'i', 'trip_id']],  ode[['i', 'j', 'ix']], left_on=['a', 'i'], right_on=['i', 'j'])
        alightings = pd.merge(links[['j', 'b', 'trip_id']],  ode[['i', 'j', 'ix']], left_on=['j', 'b'], right_on=['i', 'j'])

        inlegs = set(transit['ix']).union(boardings['ix']).union(alightings['ix'])
        remaining = ode.drop(list(inlegs))

        boardings = boardings.set_index('ix')
        boardings['i'] = boardings['a']
        boardings['j'] = boardings['trip_id']
        boardings['f'] = ode['f']
        boardings['p'] = ode['p']

        alightings = alightings.set_index('ix')
        alightings['j'] = alightings['b']
        alightings['i'] = alightings['trip_id']
        alightings['f'] = ode['f']
        alightings['p'] = 1

        c = ['i', 'j', 'f', 'p']
        a = pd.concat([boardings[c], alightings[c], remaining[c]])
        a = a.dropna(subset=['i', 'j'])

        return a
