# -*- coding: utf-8 -*-

from syspy.spatial.graph import network
from syspy.spatial.geometries import reversed_polyline, simplify
import pandas as pd
from tqdm import tqdm


class GraphBuilder():
    """
    Use graphbuilders to build a graph from a collection of lines.
    INIT then LINE METHODS then LINE TO LINK then LINK METHODS:
      process the lines first by adding intersections and splitting them;
      build the nodes and the links from the lines;
      process the links (merge them, drop the dead ends etc...)

    example:
    ::
        gb = graphbuilder.GraphBuilder(main_roads)
        gb.build_nodes()
        gb.build_directed_links(
            direction_column='sens',
            direct='Direct',
            indirect='Inverse',
            both='Double',
            inplace=True
        )
        links, nodes = gb.links, gb.nodes
    """

    def __init__(self, lines=None):
        """
        :param lines: line GeoDataFrame with at least a 'geometry' column
        """
        self.lines = lines

    def split_polylines(self, inplace=True, **kwargs):
        """
        LINE METHOD
        Splits polylines in lines.
        The polylines are simplified then split at their checkpoints.
        """
        self.split_lines = simplify(self.lines, **kwargs)
        if inplace:
            self.lines = self.split_lines

    def build_planar_lines(
            self,
            buffer=1e-6,
            n_neighbors=100,
            inplace=True,
            seek_intersections=True,
            line_split_coord_dict=dict()
        ):
        """
        LINE METHOD
        Test every line of self.lines against its neigbors in order to find intersections.
        The links are split at their intersections with the other links.
        Performs network.split_geometries_at_nodes on self.lines
        """
        self.planar_lines = network.split_geometries_at_nodes(
            self.lines,
            buffer=buffer,
            n_neighbors=n_neighbors,
            seek_intersections=seek_intersections,
            line_split_coord_dict=line_split_coord_dict
        )

        if inplace:
            self.lines = self.planar_lines.reset_index(drop=True)

    def build_nodes(self):
        """
        LINE TO LINKS
        Builds self.links and self.nodes
        self.links have 'a' and 'b' columns which reference
        nodes from self.nodes (column 'n')
        """
        links, nodes = network.graph_from_links(self.lines)

        # les noeuds qui ne sont pas des carrefours
        # tester avec l'union pour ajouter les impasses
        join = network.constrained_nodes(links)
        nodes['join'] = nodes['n'].isin(join)
        self.links, self.nodes, self.join = links, nodes, join

    def merge_links(self, inplace=True,  **kwargs):
        """
        LINK METHOD
        Removes the nodes of degree 2 in the undirected graph (more or less)
        """
        self.merged_links = network.polyline_graph(self.links, **kwargs)
        if inplace:
            self.links = self.merged_links

    def build_directed_links(
            self,
            inplace=True,
            direction_column='oneway',
            direct=2,
            indirect=3,
            both=1
        ):
        """
        LINK METHOD
        Builds directed links where:
          one can go from a to b
          the geometry goes from a to b
        The links that enable both directions are duplicated
        The links tha enable only the indirect direction are reversed  
        example:
        ::
            gb.build_directed_links(
                direction_column='sens', 
                direct='Direct', 
                indirect='Inverse', 
                both='Double',
                inplace=True
            )     
        """
        merge_oneway = pd.DataFrame(
            {
                direction_column: [direct, indirect, both],
                'oneway_temp': [2, 3, 1]
            }
        )

        # we create a new column : oneway_temp which contains the direction:
        # 1 = both, 2 = direct, 3 = indirect
        self.links = pd.merge(
            self.links,
            merge_oneway,
            on=direction_column,
            suffixes=['_replaced', '']  # if the new oneway replace the former
        )

        frame = self.links.copy()

        # duplicates rows that allow both ways
        keep = frame[frame['oneway_temp'] > 1].copy()
        direct = frame[frame['oneway_temp'] == 1].copy()
        indirect = direct.copy()
        direct['oneway_temp'] = 2
        indirect['oneway_temp'] = 3
        frame = pd.concat([direct, indirect, keep])

        # reverse geometry, a and b for indirect links
        direct = frame[frame['oneway_temp'] == 2].copy()
        indirect = frame[frame['oneway_temp'] == 3].copy()

        indirect['geometry'] = indirect['geometry'].apply(reversed_polyline)
        indirect[['a', 'b']] = indirect[['b', 'a']]

        # in the end, all the links are direct
        indirect['oneway_temp'] = 2

        self.split_links = pd.concat([direct, indirect])
        if inplace:
            self.links = self.split_links

    def drop_secondary_components(self):
        """
        LINK METHOD
        keeps only the main component among the connected components of the graph
        built from links,
        then returns the links that form the main component.
        Wraps network.drop_secondary_components
        """
        self.links = network.drop_secondary_components(self.links)
        nodeset = set(self.links['a']).union(set(self.links['b']))
        self.nodes = self.nodes.loc[self.nodes['n'].isin(nodeset)]

    def drop_deadends(self, **kwargs):
        """
        LINK METHOD
        drops the dead ends.
        Wraps network.drop_deadends
        """
        self.links = network.drop_deadends(self.links, **kwargs)
        nodeset = set(self.links['a']).union(set(self.links['b']))
        self.nodes = self.nodes.loc[self.nodes['n'].isin(nodeset)]








def clean_geometries(links, nodes):
    links = links.copy()
    ng = nodes['geometry'].to_dict()

    def directed(row):
        return list(ng[row['a']].coords)[0] == list(row['geometry'].coords)[0]
    def return_directed(row):
        return list(ng[row['a']].coords)[0] == list(row['geometry'].coords)[-1]

    #tqdm.pandas('directed')
    
    reverse = links.apply(
        return_directed,
        axis=1
    )
    
    direct = links.apply(
        directed,
        axis=1
    )
    
    loc = (reverse == True, 'geometry')

    links.loc[loc] = links.loc[loc].apply(
        reversed_polyline
    )
    assert (reverse | direct).mean() == 1
    return links

def add_reversed_links(
    links, 
    direction_column='oneway', 
    reverse_value=False,
    reset_link_index=True,
):
    reversed_link = links.copy()
    reversed_link[ 'a'], reversed_link['b']  = links[ 'b'], links['a']
    reversed_link['geometry'] = links['geometry'].apply(reversed_polyline)

    direct_links = links
    indirect_links = reversed_link.loc[
        reversed_link[direction_column] == reverse_value
    ]

    bothway = pd.concat([direct_links, indirect_links])

    if reset_link_index:
        bothway['uid'] = range(1, len(bothway)+1)
        bothway.set_index('uid', inplace=True)
    return bothway

class OsmnxCleaner():
    def __init__(
        self,     
        links=None, 
        nodes=None, 
        a='from', 
        b='to', 
        node_id=None
    ):
        self.links=links.rename(columns ={a: 'a', b: 'b'})
        self.nodes = nodes
        if node_id is not None:
            self.nodes = self.nodes.set_index(node_id)
            
    def add_reversed_links(self, *args, **kwargs):
        self.links = add_reversed_links(self.links, *args, **kwargs)
        
    def clean_geometries(self, *args, **kwargs):
        self.links = clean_geometries(self.links, self.nodes, *args, **kwargs)

    def fix_nodeset_consistency(self):
        self.links = self.links.loc[self.links['a'].isin(set(self.nodes.index))]
        self.links = self.links.loc[self.links['b'].isin(set(self.nodes.index))]
        self.nodes = self.nodes.loc[set(self.links['a']).union(self.links['b'])]

