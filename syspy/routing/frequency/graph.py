# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'

from syspy.transitfeed import feed_links
import networkx as nx


def distinct(l, lists):
    """
    return 1 if every list of 'lists' contains an item that does not belong to l.
    The return value should then be converted to bool. (mostly useful when comparing the route set in a graph search)
    :param l: a list
    :param lists: list of lists or sets
    :return: n>0 if every list of 'lists' contains at list an item that does not belong to l.
    """
    differences = [len(set(r)-set(l)) for r in lists]
    try:
        return min(differences)
    except:
        return 1


def transit_edges(links, weight='time'):

    """
    Return a list of edges between transitlinks , given the following links of the same trip:
        * link1 = (a, b, 10min), link2 = (b, c, 3min) and link3 = (c, d, 4min)
        * we should return [(link1, link2, 3min), (link2, link3, 4min)]

    :param links: a transitlink DataFrame, the column named 'index' is used to identify them
    :return: a list of edges between links
    """
    assert len(links)
    next_links = feed_links.link_from_stop_times(
        links, 
        max_shortcut=1, 
        stop_id='index',
        in_sequence='link_sequence',
        out_sequence='connection_sequence',
        keep_origin_columns=[],
        keep_destination_columns=[weight]
    )
    return next_links[['index_origin', 'index_destination', weight]].values.tolist()


def combined_edges(links, index='index', weight='cost'):

    """
    Builds direct edges between the links in "links":
        * link a goes from station 1 to station 2
        * link b from 2 to 3
        * link c from 3 to 4
        * link d from 2 to 5
    The following edges should be returned : a-b, a-d, b-c

    :param links: transitlinks DataFrame
    :param index: name of the column that indexes the links
    :param weight: name of the column that holds the cost of the link (waiting + in vehicle time)
    :return: list of lists [[index_linka, index_linkb, weight], [index_linkb, index_linkc, weight], ...]
    """

    edges = []
    for i in set(links['origin']):
        arrivals = list(links[links['destination'] == i][index])
        t = links[links['origin'] == i][[index, weight]]
        for a in arrivals:
            for item in t.values.tolist():
                edges.append([a] + item)
    return edges


def graphs_from_links(
    links,
    include_igraph=False,
    boarding_cost=0,
    alighting_cost=0,
    shortcuts=False,
    include_edges=[]
):

    """
    Builds a graph from a table of transit links. The transit links and the stations are used as nodes.
        * boarding edges and alighting edges connect the transit links to the stations ;
        * one transit edge connects a transit link to the next transit link of the trip with no connection cost ;
        * one combined edge connects a transit link to all the transit links leaving from its arrival station ;
        * include edges may be used to handle footpaths that connect stations together ;

    :param links: transitlinks DataFrame
    :param include_igraph: if True: an igraph Graph object is also returned
    :param boarding_cost: added to the weight of the boarding edges
    :param alighting_cost: added to the weight of the alighting edges
    :param shortcuts: in True: combined edges are added to the graph
    :param include_edges: a list of edges to add to the graph [[a, b, weight(a-b)], [b, c , weight(b-c)], ...]
    :return: nx_graph is a networkx DiGraph, ig_graph is a directed igraph Graph
    """

    # drop the edges that connect nodes that are not linked by transitlinks
    stop_set = set(links['destination']).union(set(links['origin']))
    include_edges = [
        edge for edge in include_edges
        if edge[0] in stop_set and edge[1] in stop_set
    ]

    edges = combined_edges(links) + transit_edges(links) if shortcuts else transit_edges(links)
    edges += include_edges

    boarding_edges = []
    alighting_edges = []

    for name, link in links.iterrows():
        boarding_edges.append((link['origin'], link['index'], link['cost'] + boarding_cost))
        alighting_edges.append((link['index'], link['destination'], alighting_cost))

    edges += boarding_edges + alighting_edges

    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(edges)

    ig_graph = None
    if include_igraph:
        import igraph
        ig_graph = igraph.Graph.TupleList(
            edges=edges,
            directed=True,
            edge_attrs=['weight']
    )

    return nx_graph, ig_graph


def indexed_data(
    links,
    index='index',
    origin='origin',
    destination='destination',
    route='route_id',
    stop_route=0
):

    """
    Returns a dict that contains the data of the edges of a graph built with "links":
        * transit data contains the data of transit links (indexed by integers)
        * stop data contains the data of the stops (stations). Stop index is converted to strings
        to avoid collisions with transit links index

    :param links: transitlinks DataFrame
    :param index: name of the column that indexes the links
    :param origin: name of the origin column
    :param destination: name of the destination column
    :param route: name of a link attribute that groups links (trip_id, pattern, route_id etc...)
    :return: {link_id: {destination: d, route: g}... str(stop_id): {destination : stop_id, route: stop_route}}
    """

    transit_data = links.set_index(index)[
        [destination, route]].to_dict(orient='index')
    stop_set = set(links[origin]).union(set(links[destination]))
    stop_data = {str(d): {destination: d, route: stop_route} for d in stop_set}
    data = transit_data
    data.update(stop_data)
    return data


def multiple_sources_dijkstra(
    graph,
    sources,
    data,
    route='route_id',
    stop='destination'
):
    """
    Call single_source_dijkstra on a collection of sources and return the concatenated results.
    See single_source_dijkstra for args description.
    """
    paths = []
    for source in sources:
        paths += single_source_dijkstra(graph, source, data, route, stop)
    return paths


def single_source_dijkstra(
    graph,
    source,
    data,
    route='route_id',
    stop='destination',
    return_type='list'
):

    """
    :param graph: networkx DiGraph to perform the search in
    :param source: source of the dijkstra search
    :param data: dict that contains the data of the edges of the graph
        (besides the weight). route and destination station for example
    :param route: name of the group identifier (route_id, trip_id, pattern...). Most probably : 'route_id'
    :param stop: key to the stop_id in data:
    :return:
    """
    lengths, paths = nx.single_source_dijkstra(graph, source)

    to_return = {
        'stops': [
            {
                'source': source,
                'target': key,
                'path': [data[p][stop] for p in value],
                'routes': set([data[p][route] for p in value]),
                'transfers': len(set([data[p][route] for p in value])) - 2,
                'length':lengths[key]
            } for key, value in paths.items() if type(key) == str
        ],
        'links': [
            {
                'source': source,
                'target': key,
                'path': [data[p][stop] for p in value],
                'routes': set([data[p][route] for p in value]),
                'transfers': len(set([data[p][route] for p in value])) - 2,
                'length':lengths[key]
            } for key, value in paths.items() if type(key) == int
        ],
    }

    return to_return if return_type == 'list' else {
        kind: {item['target']: item for item in to_return[kind]}
        for kind in to_return.keys()
    }  # never tested, may be bugged


def single_source_labels(
    source,
    graph,
    data,
    start_from=0,
    infinity=999999,
    spread=1,
    absolute=0,
    unique_route_sets=True,
    max_transfer=3,
    stop_iteration=100000,
    route='route_id'
):

    """
    From a given source, search a graph for the best paths to all the stops.
    Takes parameters to not only look for the best path to every destination but a 'relevant' set of paths.

    :param source: source of the branch and bound search
    :param graph: searched graph (networkx DiGraph)
    :param data: edge data dictionary {edge_index: {'destination': edge_destination, 'route_id': edge_route}}
    :param stop_set: set of the stops of the actual network (stations)
    :param start_from: first label_id to use
    :param infinity: number to use as infinity in order to initiate the distance of the nodes to the source
    :param spread: if the cost to a node is bigger than spread*best_cost to this node : the search stops
    :param absolute: actually, the search only stops if cost > spread*best_cost AND cost-best_cost > absolute
    :param unique_route_sets: if True, when a path does not beat the best_cost to a node : it is only kept if it uses
        a route set that is not used by another path to the node.
    :param max_transfer: the search stops when the number of routes (footpaths and connections altogether count
        for a route) is over max_transfer + 2
    :param route
    :return: a list of labels that track the search for the best paths to all the stops from the source
    """

    stop_set = {int(n) for n in graph.nodes() if type(n) == str}

    root = {
        'stop': source,
        'node': str(source),
        'parent': 0,
        'cumulative': 0,
        'visited': [source],
        'route': frozenset([0]),
        'cost': 0,
        'first': False
    }

    pile = [root]
    label_id = iter(range(start_from, stop_iteration))
    store = []
    alighting = {stop: infinity for stop in stop_set}
    best = {node: infinity for node in graph.nodes()}
    routes = {stop: frozenset({}) for stop in stop_set}

    nodes = graph.edge.keys()

    def next_labels(label, label_id, route=route):

        stop = label['stop']
        node = label['node']
        current_route = label['route']
        cumulative = label['cumulative']
        cost = label['cost']
        label['label_id'] = label_id
        visited = label['visited']

        store.append(label)

        if len(current_route) - 2 > max_transfer:
            return []

        not_alighting = cost  # the eggress links have the save stop as the transit link that preceed them

        if node not in nodes:
            return []  # the node has no neighbors - no next labels

        neighbors = graph.edge[node]

        if cumulative < best[node]:
            best[node] = cumulative

        if stop not in alighting.keys() or cumulative < alighting[stop]:
            alighting[stop] = cumulative

        dominated = cumulative > alighting[stop] and cumulative > best[node]

        if dominated:

            if cumulative > alighting[stop]*spread and cumulative - alighting[stop] > absolute:
                return []

            if not distinct(current_route, routes[stop]) and unique_route_sets and not_alighting:
                return []

        routes[stop] = routes[stop].union({current_route})

        proto_labels = [
            {
                'node': key,
                'stop': data[key]['destination'],
                'parent': label_id,
                'cost': value['weight'],
                'cumulative':  cumulative + value['weight'],
                'visited': visited + [data[key]['destination']],
                'route': frozenset(current_route.union({data[key][route]})),
                'first': label['first'] if label['first'] else data[key][route]
            }
            for key, value in neighbors.items()
            if data[key]['destination'] not in visited[:-2]
        ]
        #  an egress has the same destination as the link it follows [:-2]
        return proto_labels

    while len(pile):
        # on remplace le dernier élément de la pile par tous ses enfants
        pile = next_labels(pile.pop(), next(label_id)) + pile

    return store




