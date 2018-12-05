# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx

from IPython.html.widgets import FloatProgress
from IPython.display import display
from syspy.syspy_utils import syscolors


def print_if_debug(to_print, debug):
    if debug:
        print(to_print)


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


def transfer_time(arrival_time, departure_time, period=24 * 3600):
    delta = departure_time - arrival_time
    return delta if delta >= 0 else delta + period


def transfer_times(
    stop,
    origin_indexed_links,
    origin_indexed_costs,
    destination_indexed_links,
    include_transfer_time=True
):
    departure_series = origin_indexed_links.loc[stop]
    departure_times = list(departure_series)

    arrival_series = destination_indexed_links.loc[stop]
    arrival_times = list(arrival_series)

    if not include_transfer_time:
        def pseudo_transfer_time(a, d):
            return 0
    else:
        def pseudo_transfer_time(a, d):
            return transfer_time(a, d)

    transfer_time_array = [
        [
            pseudo_transfer_time(arrival_time, departure_time)
            for departure_time in departure_times
        ] for arrival_time in arrival_times
    ]

    departure_cost = origin_indexed_costs.loc[stop]

    matrix = pd.DataFrame(
        transfer_time_array,
        index=arrival_series.index,
        columns=departure_series.index
    ) + departure_cost

    return matrix.stack()


def graph_from_links(
    links,
    boarding_cost=0,
    alighting_cost=0,
    transfer_stops=None,
    include_transfer_time=True
):

# transfer edges
    origin_indexed_links = links.set_index(
        ['origin', 'index'])['departure_time']
    origin_indexed_costs = links.set_index(
        ['origin', 'index'])['cost']
    destination_indexed_links = links.set_index(
        ['destination', 'index'])['arrival_time']

    od_intersection = set(links['origin']).intersection(links['destination'])
    if transfer_stops:
        transfer_stops = od_intersection.intersection(transfer_stops)
    else:
        transfer_stops = od_intersection

    concatenated = pd.concat(
        [
            transfer_times(
                stop,
                origin_indexed_links,
                origin_indexed_costs,
                destination_indexed_links,
                include_transfer_time
            ) for stop in transfer_stops
        ]
    )
    concatenated.index.names = ['from', 'to']
    transfer_edges = concatenated.reset_index().values.tolist()

# boarding and alighting edges
    boarding_edges = []
    alighting_edges = []

    for name, link in links.iterrows():
        boarding_edges.append(
            [
                'boarding_' + str(link['origin']),
                link['index'],
                link['cost'] + boarding_cost
            ]
        )
        alighting_edges.append(
            [
                link['index'],
                'alighting_' + str(link['destination']),
                alighting_cost
            ]
        )

    edges = transfer_edges + boarding_edges + alighting_edges

# networkx digraph and node dict update
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(edges)

    in_stop_node_dict = {}

    boarding_nodes = {
        'boarding_' + str(stop): {
            'destination': stop,
            'route_id': 0
        }
        for stop in set(links['origin'])
    }

    alighting_nodes = {
        'alighting_' + str(stop): {
            'destination': stop,
            'route_id': 0
        }
        for stop in set(links['destination'])
    }
    in_stop_node_dict.update(alighting_nodes)
    in_stop_node_dict.update(boarding_nodes)

    for node, data in in_stop_node_dict.items():
        nx_graph.node[node].update(data)

    link_dict = links.set_index('index')[
        ['destination', 'route_id']].to_dict(orient='index')

    for link, data in link_dict.items():
        nx_graph.node[link].update(data)

    return nx_graph


def dijkstra_powered_single_source_labels(
    source,
    graph,
    start_from=0,
    infinity=999999,
    spread=1,
    absolute=0,
    unique_route_sets=True,
    max_transfer=3,
    stop_iteration=100000,
    debug=False,
    cutoff=float('inf'),
    max_stack=100000
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
    :return: a list of labels that track the search for the best paths to all the stops from the source
    """

    stop_set = {node['destination'] for node in graph.node.values()}

    root = {
        'stop': graph.node[source]['destination'],
        'node': source,
        'parent': 0,
        'cumulative': 0,
        'visited': [source],
        'route': frozenset([0]),
        'cost': 0
    }

    pile = [root]
    label_id = iter(range(start_from, stop_iteration))
    store = []

    dijkstra = nx.single_source_dijkstra_path_length(graph, source)

    tolerated = {
        key: best * spread + absolute
        for key, best in dijkstra.items()
    }

    node_set = graph.edge.keys()
    data = graph.node

    stack_progress = FloatProgress(
        min=0,
        max=max_stack,
        width=975,
        height=10,
        color=syscolors.rainbow_shades[1],
        margin=5
    )

    stack_progress.value = 0
    display(stack_progress)

    iteration_progress = FloatProgress(
        min=0,
        max=stop_iteration,
        width=975,
        height=10,
        color=syscolors.rainbow_shades[0],
        margin=5
    )

    iteration_progress.value = 0
    display(iteration_progress)

    def next_labels(label, label_id):

        stop = label['stop']
        node = label['node']
        route = label['route']
        cumulative = label['cumulative']
        cost = label['cost']
        label['label_id'] = iteration_progress.value = label_id
        visited = label['visited']

        store.append(label)

        if len(route) - 2 > max_transfer:
            return []

        # the eggress links have the save stop as
        # the transit link that preceed them, they are free

        try:
            neighbors = graph.edge[node]
        except KeyError:
            print_if_debug('not in node_set', debug)
            return []  # the node has no neighbors - no next labels


        if cumulative > cutoff:
            print_if_debug('cutoff', debug)
            return []

        if cumulative > tolerated[node]:
            print_if_debug('dijkstra', debug)
            return[]

        proto_labels = [
            {
                'node': key,
                'stop': data[key]['destination'],
                'parent': label_id,
                'cost': value['weight'],
                'cumulative':  cumulative + value['weight'],
                'visited': visited + [data[key]['destination']],
                'route': frozenset(route.union({data[key]['route_id']}))
            }
            for key, value in neighbors.items()
            if data[key]['destination'] not in visited[:-1]
        ]
        #  an egress has the same destination as the link it follows [:-2]

        print_if_debug(
            ('proto_labels_length', len(proto_labels)),
            debug)
        return proto_labels

    while len(pile) and len(pile) < max_stack:
        # on remplace le dernier élément de la pile par tous ses enfants
        pile = next_labels(pile.pop(), next(label_id)) + pile
        stack_progress.value = len(pile)

    return store


def single_source_labels(
    source,
    graph,
    start_from=0,
    infinity=999999,
    spread=1,
    absolute=0,
    unique_route_sets=True,
    max_transfer=3,
    stop_iteration=100000,
    debug=False,
    cutoff=float('inf'),
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
    :return: a list of labels that track the search for the best paths to all the stops from the source
    """

    stop_set = {node['destination'] for node in graph.node.values()}

    root = {
        'stop': graph.node[source]['destination'],
        'node': source,
        'parent': 0,
        'cumulative': 0,
        'visited': [source],
        'route': frozenset([0]),
        'cost': 0
    }

    pile = [root]
    label_id = iter(range(start_from, stop_iteration))
    store = []
    alighting = {stop: infinity for stop in stop_set}
    best = {node: infinity for node in graph.nodes()}
    routes = {stop: frozenset({}) for stop in stop_set}

    node_set = graph.edge.keys()
    data = graph.node

    def next_labels(label, label_id):

        stop = label['stop']
        node = label['node']
        route = label['route']
        cumulative = label['cumulative']
        cost = label['cost']
        label['label_id'] = label_id
        visited = label['visited']

        store.append(label)

        if len(route) - 2 > max_transfer:
            return []

        # the eggress links have the save stop as
        # the transit link that preceed them, they are free
        not_alighting = cost

        if node not in node_set:
            print_if_debug('not in node_set', debug)
            return []  # the node has no neighbors - no next labels


        neighbors = graph.edge[node]

        if cumulative > cutoff:
            print_if_debug('cutoff', debug)
            return []

        if cumulative < best[node]:
            best[node] = cumulative

        if stop not in alighting.keys() or cumulative < alighting[stop]:
            alighting[stop] = cumulative

        dominated = cumulative > alighting[stop] and cumulative > best[node]

        if dominated:

            if (
                    cumulative > alighting[stop] * spread
                    and cumulative - alighting[stop] > absolute
            ):
                print_if_debug('out of tolerance', debug)
                return []

            if (
                    not distinct(route, routes[stop])
                    and unique_route_sets
                    and not_alighting
            ):
                print_if_debug('same route set', debug)
                return []

        routes[stop] = routes[stop].union({route})

        proto_labels = [
            {
                'node': key,
                'stop': data[key]['destination'],
                'parent': label_id,
                'cost': value['weight'],
                'cumulative':  cumulative + value['weight'],
                'visited': visited + [data[key]['destination']],
                'route': frozenset(route.union({data[key]['route_id']}))
            }
            for key, value in neighbors.items()
            if data[key]['destination'] not in visited[:-1]
        ]
        #  an egress has the same destination as the link it follows [:-2]

        print_if_debug(
            ('proto_labels_length', len(proto_labels)),
            debug)
        return proto_labels

    while len(pile):
        # on remplace le dernier élément de la pile par tous ses enfants
        pile = next_labels(pile.pop(), next(label_id)) + pile

    return store
