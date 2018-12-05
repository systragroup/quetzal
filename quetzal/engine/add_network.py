# -*- coding: utf-8 -*-

import shapely

from syspy.graph import nearest_path
import pandas as pd
from tqdm import tqdm
import networkx as nx
from syspy.spatial.geometries import line_list_to_polyline
from syspy.spatial.spatial import nearest_geometry
from syspy.skims import skims
from syspy.spatial.graph import network

from shapely import geometry


class NetworkCaster:
    
    def __init__(self, nodes, links, road_nodes, road_links=None, weight='length'):
        
        self.nodes = nodes
        
        self.road_nodes = road_nodes
        self.links = links
        
        if road_links is not None:
            self.road_links = road_links
            road_graph = nx.DiGraph()
            road_graph.add_weighted_edges_from(
                road_links[['a', 'b', weight]].values.tolist()
            )
            
            self.road_graph = road_graph




    def dumb_cast(self, replace_geometry=True):
        links = self.links.copy()

        links['road_node_list'] = [[]]*len(links)
        links['road_link_list'] = [[]]*len(links)
        
        nearest = nearest_geometry(self.nodes, self.road_nodes)
        nearest_node = nearest.set_index('ix_one')['ix_many'].to_dict()
        reindex_node = lambda n: nearest_node[n]
        links['road_a'] = links['a'].apply(reindex_node)
        links['road_b'] = links['b'].apply(reindex_node)
        self.links = links

        def ab_geometry(link, nodes, a='a', b='b'):
            node_a = nodes.loc[link[a]]['geometry']
            node_b = nodes.loc[link[b]]['geometry']
            return shapely.geometry.LineString([node_a, node_b])

        links['road_geometry'] = links.apply(
            ab_geometry, 
            nodes=self.road_nodes, 
            a='road_a', 
            b='road_b', 
            axis=1
        )

        if replace_geometry:
            links.drop('geometry', axis=1, errors='ignore', inplace=True)
            links.rename(columns={'road_geometry': 'geometry'}, inplace=True)
            links = links.dropna(subset=['geometry'])
            node_geo = self.road_nodes['geometry'].to_dict()
            def roada_to_geometry(key):
                return geometry.LineString([node_geo[key], node_geo[key]])
            loc = links[(links['road_a'] == links['road_b'])].index
            null_geometry = links['road_a'].apply(roada_to_geometry)
            links.loc[loc, 'geometry'] = null_geometry.loc[loc]


    def build_nearest_neighbors(
        self, 
        nearest_method='nodes', 
        coordinates_unit='degree',
        **nearest_geometry_kwargs
    ):
        
        if nearest_method == 'nodes':
            nearest_neighbors = nearest_geometry(
                self.nodes, 
                self.road_nodes,
                **nearest_geometry_kwargs
            )
            neighbors=nearest_neighbors

        elif nearest_method == 'links':
            nearest_links = nearest_geometry(
                self.nodes, 
                self.road_links, 
                **nearest_geometry_kwargs
            )
            # focus on b

            right = self.road_links[['b', 'a']]
            nearest_links = pd.merge(
                nearest_links, 
                right , 
                left_on='ix_many', 
                right_index=True
            )
            nearest_links['rank'] = nearest_links['actual_rank']
            nearest_links['distance'] = nearest_links['actual_distance']

            # we want to offer a and b as a possible node
            b_neighbors = nearest_links.copy()
            a_neighbors = nearest_links.copy()

            b_neighbors['ix_many'] = nearest_links['b']
            a_neighbors['ix_many'] = nearest_links['a']

            nearest_links = pd.concat([a_neighbors, b_neighbors])
            neighbors = nearest_links.drop_duplicates(subset=['ix_one', 'ix_many'])

        if coordinates_unit == 'degree':
            neighbors['length'] = skims.distance_from_geometry(
                neighbors['geometry'],
                method='numpy'
            ) 
        elif coordinates_unit == 'meter':
            neighbors['length'] = neighbors['distance']

        self.neighbors = neighbors
        self.nearest_neighbors = neighbors[neighbors['rank'] == 0]

    def build(self, penalty_factor, **kwargs):
        self.build_nearest_neighbors(**kwargs)
        
        self.neighbors['penalty'] = self.neighbors['length'] * penalty_factor

        self.p_series = self.neighbors.set_index(
            ['ix_one', 'ix_many'])['penalty']
        self.penalties = self.p_series.to_dict() 
        self.add_road_nodes(penalties=self.penalties)
        self.build_road_dataframe()
        self.build_road_access()

    def add_road_nodes(self, penalties=None):
        self.links[['road_a', 'road_b']] = link_road_nodes(
            self.links,
            road_graph=self.road_graph,
            nearest_neighbors=self.neighbors,
            penalties=penalties,
            group='trip_id'
        )

    def build_road_dataframe(
        self, 
        merge_on_links=True, 
        replace_geometry=True
    ):

        self.road_dataframe = road_dataframe(
            links=self.links, 
            road_links=self.road_links,
            road_graph=self.road_graph,
        )

        if merge_on_links:
            # we want to be sure we are not going to create duplicated columns
            self.links.drop(
                self.road_dataframe.columns, 
                axis=1, 
                inplace=True, 
                errors='ignore'
            ) 
            links = pd.concat([self.links, self.road_dataframe], axis=1)
            if replace_geometry:
                links.drop('geometry', axis=1, errors='ignore', inplace=True)
                links.rename(columns={'road_geometry': 'geometry'}, inplace=True)
                links = links.dropna(subset=['geometry'])
                node_geo = self.road_nodes['geometry'].to_dict()
                def roada_to_geometry(key):
                    return geometry.LineString([node_geo[key], node_geo[key]])
                loc = links[(links['road_a'] == links['road_b'])].index
                null_geometry = links['road_a'].apply(roada_to_geometry)
                links.loc[loc, 'geometry'] = null_geometry.loc[loc]
            self.links = links.copy()

    
    def build_road_access(self):
        a = [tuple(l) for l in self.links[['a', 'road_a']].values.tolist()]
        b = [tuple(l) for l in self.links[['b', 'road_b']].values.tolist()]
        road_access = self.neighbors.set_index(
            ['ix_one', 'ix_many']).loc[list(set(a + b))]
        road_access.index.names = ['node', 'road_node']
        self.road_access = road_access
 

def find_node_path(links, line):
    line_links = links[links['trip_id'] == line].copy()
    line_links.sort_values('link_sequence')
    path = list(line_links['a'])
    path.append(list(line_links['b'])[-1])
    return path

def node_join_dataframe(
    node_path, 
    road_graph, 
    nearest_neighbors, 
    penalties=None
):
    options = nearest_path.find_road_node_options(
        node_path, 
        nearest_neighbors=nearest_neighbors
    )
    shortcut_graph = nearest_path.build_shortcut_ghaph(
        node_path,
        options,
        road_graph=road_graph,
        penalties=penalties
    )
    road_node_path = nearest_path.find_road_node_path(shortcut_graph)

    df = pd.DataFrame(
        {
            'node': node_path, 
            'road_node': road_node_path
        }
    )
    return df

def build_node_dict(
    links,
    road_graph,
    nearest_neighbors,
    penalties,
    group='trip_id'
):
    
    links = links.copy()

    to_concat = []
    lines = set(links[group])

    iterator = tqdm(lines)
    for line in iterator:
        iterator.desc = str(line)

        node_path = find_node_path(links, line)

        df = node_join_dataframe(
            node_path=node_path, 
            road_graph=road_graph, 
            nearest_neighbors=nearest_neighbors, 
            penalties=penalties
        )

        df[group] = line
        to_concat.append(df)

        concatenated =  pd.concat(to_concat)
        
    node_dict = concatenated.set_index(
        [group, 'node']
    )['road_node'].to_dict()

    return node_dict, concatenated, df, iterator
    

def link_road_nodes(
    links,
    road_graph,
    nearest_neighbors,
    penalties,
    group='trip_id'
):
    
    links = links.copy()

    to_concat = []
    lines = set(links[group])

    iterator = tqdm(lines)
    for line in iterator:
        iterator.desc = str(line)

        node_path = find_node_path(links, line)

        df = node_join_dataframe(
            node_path=node_path, 
            road_graph=road_graph, 
            nearest_neighbors=nearest_neighbors, 
            penalties=penalties
        )

        df[group] = line
        to_concat.append(df)

        concatenated =  pd.concat(to_concat)
        
    node_dict = concatenated.set_index(
        [group, 'node']
    )['road_node'].to_dict()
    

    def road_a_road_b(row):
        a, b = None, None
        try:
            a = node_dict[(row[group], row['a'] )]
            b = node_dict[(row[group], row['b'] )]
        except KeyError: # (('D10-42_bis', 'LSEAD'), 'occurred at index link_293')
            pass
        

        return pd.Series([a, b])
    
    return links.apply(road_a_road_b, axis=1)

def road_dataframe(links, road_links, road_graph):

    # index link data in order to improve performances of road_series
    road_links = road_links.copy()
    road_links['index'] = road_links.index
    indexed = road_links.set_index(['a', 'b']).sort_index()
    ab_indexed_dict = indexed['index'].to_dict()
    indexed = road_links.set_index(['index']).sort_index()
    indexed_geometries = indexed['geometry']

    def road_series(link, road_graph):

        try: 

            a = link['road_a']
            b = link['road_b']

            road_distance, road_node_list = nx.bidirectional_dijkstra(
                road_graph, a, b
            )

            tuples = [
                (road_node_list[i], road_node_list[i+1]) 
                for i in range(len(road_node_list)-1)
            ]

            road_link_list = [ab_indexed_dict[t] for t in tuples]
            line_list = list(indexed_geometries.loc[road_link_list])
            road_geometry = line_list_to_polyline(line_list)

            values = [
                road_node_list,
                road_link_list,
                road_distance,
                road_geometry
            ]
        except nx.NodeNotFound:
            values = [None, None, None, None]


        return pd.Series(values)

    # apply road_series to build a dataframe containing all the 
    # columns related to the road network
    tqdm.pandas(desc='road_paths ')
    road_df = links.progress_apply(
        lambda l: road_series(l, road_graph),
        axis=1
    )

    road_df.columns = [
        'road_node_list',
        'road_link_list',
        'road_length',
        'road_geometry'
    ]
    
    return road_df

