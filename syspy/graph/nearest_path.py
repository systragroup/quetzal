import itertools
import networkx as nx
from networkx import NetworkXNoPath


def find_road_node_options(node_path, nearest_neighbors):

    df = nearest_neighbors.loc[nearest_neighbors['ix_one'].isin(node_path)].copy()
    df.sort_values('rank', inplace=True)
    
    options = {}
    for node_index in range(len(node_path)):
        node = node_path[node_index]
        road_nodes = set(df[df['ix_one'] == node]['ix_many'])
        tuples = [(node, road_node) for road_node in road_nodes]
        options[node] = tuples
    
    return options


def build_shortcut_ghaph(
    node_path, 
    road_node_options, 
    road_graph, 
    penalties=None
):

    if True:

        indexed_road_node_options = {}
        indexed_node_path = []
        for node_index in range(len(node_path)):
            node = node_path[node_index]
            key = (node_index, node)
            value = [(node_index,) + option for option in road_node_options[node]]
            indexed_road_node_options[key] = value
            
            indexed_node_path.append(key)
            
        road_node_options = indexed_road_node_options
        node_path = indexed_node_path
    

    node_tuples = [
        (node_path[i], node_path[i+1]) 
        for i in range(len(node_path)-1)
    ]
    
    # Build the shortcut graph
    edges = []

    for tuple_index in range(len(node_tuples)):
        origin, destination = node_tuples[tuple_index]
        product = list(
            itertools.product(
                road_node_options[origin], 
                road_node_options[destination]
            )
        )
        
        edges += [p + (tuple_index,) for p in product]

    def penalty(node_road_node_tuple):
        if True:
            node_road_node_tuple = node_road_node_tuple[1:]

        if penalties:
            return penalties[node_road_node_tuple]
        else:
            return 0
    
    weighted_edges = []
    for edge_index in range(len(edges)):
        origin, destination, tuple_index = edges[edge_index]
        road_origin = origin[-1]
        road_destination = destination[-1]
        try:
            length, path = nx.bidirectional_dijkstra(
                road_graph,
                road_origin,
                road_destination
            )
        except NetworkXNoPath: 
            length, path = float('inf'), ['wrong']

        length += penalty(origin) + penalty(destination)

        weighted_edge = (
            origin, 
            destination, 
            length
        )
        weighted_edges.append(weighted_edge)
            

    origin_options = road_node_options[node_path[0]]
    destination_options = road_node_options[node_path[-1]]

    assert origin_options and destination_options
    o_edges= [['origin', o, penalty(o)] for o in origin_options]
    d_edges = [[o, 'destination', penalty(o)]for o in destination_options]
    
    shortcut_graph = nx.DiGraph()
    shortcut_graph.add_weighted_edges_from(
        weighted_edges + o_edges + d_edges
    )

    return shortcut_graph

def find_road_node_path(shortcut_graph):
    
    tuple_path = nx.dijkstra_path(shortcut_graph, 'origin', 'destination')[1:-1]
    
    return [t[-1] for t in tuple_path]