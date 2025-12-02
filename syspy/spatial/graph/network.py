"""
This module provides tools for network processing.
It processes geometries more than abstract graphs.
"""

__author__ = 'qchasserieau'

import networkx as nx
import pandas as pd
import shapely
from syspy import spatial
from tqdm import tqdm


def non_terminal_intersections(graph, n_neighbors=100):
    """
    graph is a link dataframe.
    return {link_index: intersections for link_index in graph}
    The intersections are the points where the link intersects anothe link.
    """
    geometry_series = graph['geometry']
    geometry_dict = geometry_series.to_dict()
    geometry_list = list(geometry_dict.items())
    intersections = {key: set() for key in geometry_dict.keys()}

    nearest_lines = spatial.spatial.nearest(
        graph,
        graph,
        n_neighbors=n_neighbors
    )

    grouped = nearest_lines.groupby('ix_one')['ix_many']
    potential_intersections = grouped.agg(lambda s: list(s)).to_dict()

    for i in tqdm(range(len(geometry_list))):
        index_a, geometry_a = geometry_list[i]
        for index_b in potential_intersections[index_a]:
            geometry_b = geometry_dict[index_b]
            if spatial.geometries.intersects_in_between(geometry_a, geometry_b):
                intersection = geometry_a.intersection(geometry_b)
                try:
                    intersections[index_a].add(list(intersection.coords)[0])
                    intersections[index_b].add(list(intersection.coords)[0])
                except Exception:
                    # the geometries intersect in various points
                    for intersection_point in intersection:
                        intersections[index_a].add(list(intersection_point.coords)[0])
                        intersections[index_b].add(list(intersection_point.coords)[0])
    return intersections


def split_geometries_at_nodes(
        links,
        buffer=1e-9,
        n_neighbors=100,
        seek_intersections=True,
        line_split_coord_dict={}
):
    """
    Test every link of links against its neigbors in order to find intersections.
    The links are split at their intersections with the other links.
    All the links are concatenated then returned. The returned dataframe contains
    more links than the one passed as an argument.
    """
    cut = line_split_coord_dict
    geometry_dict = links['geometry'].to_dict()

    if seek_intersections:
        intersections = non_terminal_intersections(
            links,
            n_neighbors=n_neighbors
        )
    else:
        intersections = {key: set() for key in links.index}

    for link, coord_set in cut.items():
        try:
            intersections[link] = intersections[link].union(cut[link])
        except KeyError:
            intersections[link] = cut[link]

    polylines = {}
    for key in intersections.keys():
        polylines[key] = spatial.geometries.add_centroids_to_polyline(
            geometry_dict[key],
            intersections[key],
            buffer
        )

    tuple_list = []
    for key, value in polylines.items():
        for geometry in value:
            tuple_list.append([key, geometry])
    left = pd.DataFrame(tuple_list, columns=['key', 'geometry']).set_index('key')

    to_return = pd.merge(
        left,
        links.drop('geometry', axis=1),
        left_index=True,
        right_index=True
    )
    return to_return


def graph_from_links(links, first_node=0, geometry=True):
    links['coordinates_a'] = links['geometry'].apply(lambda c: c.coords[0])
    links['coordinates_b'] = links['geometry'].apply(lambda c: c.coords[-1])

    coordinate_list = list(
        set(list(links['coordinates_a'])).union(list(links['coordinates_b'])))
    coordinate_dict = {
        first_node + i: coordinate_list[i]
        for i in range(len(coordinate_list))
    }

    nodes = pd.DataFrame(pd.Series(coordinate_dict)).reset_index()
    nodes.columns = ['n', 'coordinates']
    nodes.index = nodes['n']
    # both 'n' and the index carry the index of the nodes

    links = pd.merge(
        links,
        nodes.rename(columns={'coordinates': 'coordinates_a'}),
        on='coordinates_a',
        how='left'
    )
    links = pd.merge(
        links,
        nodes.rename(columns={'coordinates': 'coordinates_b'}),
        on='coordinates_b',
        how='left',
        suffixes=['_a', '_b']
    )

    # links.drop(['a', 'b', 'A', 'B', 'coordinates_a', 'coordinates_b'], axis=1, errors='ignore', inplace=True)

    links.drop(['a', 'b', 'A', 'B'], axis=1, errors='ignore', inplace=True)
    links.rename(columns={'n_a': 'a', 'n_b': 'b'}, inplace=True)
    links = links.groupby(['a', 'b'], as_index=False).first()

    coordinates = tqdm(list(nodes['coordinates']))
    nodes['geometry'] = [shapely.geometry.point.Point(c) for c in coordinates]
    return links, nodes


def chained_component(source, g):
    """ from a given source, return the longest path in g"""
    paths = nx.shortest_path(g, source)
    to_return = [source]
    for path in paths.values():
        if len(path) > len(to_return):
            to_return = path
    return to_return


def constrained_nodes(links):
    count_a = links['a'].value_counts()
    single_a = set(count_a[count_a == 1].index)

    count_b = links['b'].value_counts()
    single_b = set(count_b[count_b == 1].index)
    # tester avec l'union pour ajouter les impasses
    return single_a.intersection(single_b)


def non_trivial_chain_graph(links):
    """return a graph that chains all the edges that link constrained nodes"""
    join = constrained_nodes(links)
    borders = links[links['a'].isin(join) | links['b'].isin(join)]
    link_edges = borders[['a', 'b']].reset_index()
    node_edges = pd.merge(
        link_edges[['b', 'index']],
        link_edges[['a', 'index']],
        left_on='b',
        right_on='a',
        suffixes=['_a', '_b']
    )
    node_edges = node_edges[node_edges['b'].isin(join)]
    g = nx.DiGraph()
    g.add_edges_from(node_edges[['index_a', 'index_b']].values.tolist())
    return g


def polyline_geometry(link_list, geometries):
    geolist = geometries.loc[list(link_list)]
    return spatial.geometries.line_list_to_polyline(geolist)


def polyline_graph(
    links,
    intensive_columns=list(),
    extensive_columns=list(),
    shared_columns=list(),
    geometry=False,
    drop_circular=False
):
    """
    Merge links of a graph to leave only the nodes that belong to more than
    two links.
    """
    # todo : semble ne pas marcher dans un graphe directionnel
    # il y a un problème, on se retrouve avec 'b' en premier nœud
    links = links.copy()
    links.reset_index(inplace=True, drop=True)
    chain_graph = non_trivial_chain_graph(links)

    # sources gather all the vertices with no ancestor
    edf = pd.DataFrame(chain_graph.edges())
    sources = set(edf[0]) - set(edf[1])
    chained = [chained_component(s, chain_graph) for s in sources]

    link_dict = links.to_dict(orient='index')
    polylines = pd.DataFrame(pd.Series(chained), columns=['links'])
    polylines['links'] = chained
    polylines['a'] = polylines['links'].apply(lambda l: link_dict[l[0]]['a'])
    polylines['b'] = polylines['links'].apply(lambda l: link_dict[l[-1]]['b'])

    # group links and aggregate values to fill polylines
    aggregated_columns = intensive_columns + extensive_columns + shared_columns

    if len(aggregated_columns) or True:

        polyline_index = []
        link_lists = list(polylines['links'].to_dict().items())
        for index, link_list in link_lists:
            for link in link_list:
                polyline_index.append([index, link])

        links['polyline'] = pd.DataFrame(polyline_index).set_index(1)
        grouped = links.groupby(['polyline'])
        for column in intensive_columns:
            polylines[column] = grouped[column].mean()
        for column in extensive_columns:
            polylines[column] = grouped[column].sum()
        for column in shared_columns:
            polylines[column] = grouped[column].first()

    columns = ['a', 'b'] + aggregated_columns

    # build geometries by chaining links
    if geometry:
        chained_geometries = [
            polyline_geometry(chain, links['geometry'])
            for chain in tqdm(chained, desc='geometry')
        ]
        polylines['geometry'] = chained_geometries
        columns.append('geometry')

    if drop_circular:
        polylines = polylines[polylines['a'] != polylines['b']].copy()

    single_links = links[links['polyline'].isnull()]
    model_graph = pd.concat([single_links[columns], polylines[columns]])
    return model_graph


def drop_secondary_components(links):
    """
    keep only the main component among the connected components of the graph
    built from links,
    then returns the links that form the main component
    """
    g = nx.Graph()
    g.add_edges_from(links[['a', 'b']].values.tolist())
    # l = list(nx.connected_component_subgraphs(g))  --> Deprecated with nx 2.4
    l = [g.subgraph(c) for c in nx.connected_components(g)]
    subgraph_length_list = [len(sg.nodes) for sg in l]
    max_length = max(subgraph_length_list)

    main_graph = [sg for sg in l if len(sg.nodes) == max_length][0]
    main_nodes = set(main_graph.nodes.keys())
    main_links = links[links['a'].isin(main_nodes)]
    return main_links


def drop_deadends(links, cutoff=10, valid_deadends=[]):
    """
    drop the dead ends
    """
    graph = nx.DiGraph()
    graph.add_edges_from(links[['a', 'b']].values.tolist())

    all_pairs = nx.all_pairs_shortest_path_length(
        graph,
        cutoff=cutoff
    )

    # for example, if in 10 moves, we can reach less
    # than 10 nodes, we are in a dead end
    try:
        items = list(all_pairs.items())
    except AttributeError:  # 'generator' object has no attribute 'items'
        items = list(tqdm(all_pairs, desc='direct'))  # networkx 2.0 syntax

    direct_deadends = {
        key for key, value in items
        if len(value) < cutoff
    }

    reversed_graph = graph.reverse()

    reversed_all_pairs = nx.all_pairs_shortest_path_length(
        reversed_graph,
        cutoff=cutoff
    )

    # for example, if in 10 moves, we can reach less
    # than 10 nodes, we are in a dead end
    try:
        items = list(reversed_all_pairs.items())
    except AttributeError:  # 'generator' object has no attribute 'items'
        items = list(tqdm(reversed_all_pairs, desc='reversed'))  # networkx 2.0 syntax

    reversed_deadends = {
        key for key, value in items
        if len(value) < cutoff
    }

    deadends = direct_deadends.union(reversed_deadends)
    deadends = deadends - set(valid_deadends)

    # nodes
    nodes = set(links['a']).union(set(links['b'])) - deadends

    keep = links[links['a'].isin(nodes)]
    keep = keep[keep['b'].isin(nodes)]
    return keep


def reindex_nodes(links, nodes, start_from=0, reindex_node=None):
    if reindex_node is None:
        nodes = nodes.copy()
        links = links.copy()

        index = nodes.index
        rename_dict = {}
        current = start_from
        for n in index:
            rename_dict[n] = str(current)
            current += 1

        def reindex_node(node):
            return rename_dict[node]

    nodes.index = [reindex_node(n) for n in nodes.index]
    links['a'] = links['a'].apply(reindex_node)
    links['b'] = links['b'].apply(reindex_node)
    return links, nodes
