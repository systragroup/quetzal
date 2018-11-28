__author__ = 'qchasserieau'

import operator
import pandas as pd
from syspy.routing.frequency import graph


def couple_optimal_strategy(skim, challenger):
    ta, ha, tb, hb = skim[0], skim[1], challenger[0], challenger[1]

    if tb < (ha / 2 + ta):
        headway = ha * hb / (ha + hb)
        pa = hb / (ha + hb)
        pb = ha / (ha + hb)
        optimal = pa * ta + pb * tb, headway
    else:
        optimal = ta, ha
    return optimal


def group_optimal_strategy(name, group,):
    skim = tuple(group.iloc[0][['actual_cost', 'key_headway']])
    for i in range(len(group) - 1):
        challenger = tuple(group.iloc[i][['actual_cost', 'key_headway']])
        skim = couple_optimal_strategy(skim, challenger)
    return {'stop': name, 'best': skim[0], 'headway': skim[1], 'expected': skim[0] + skim[1]*0.5 }


def single_source_optimal_strategy(label_dataframe, headways):

    """
    * Works on the raw result of a single source multi_path graph search.
      Such result is provided as a label_dataframe all the labels of the graph
      search are gathered.
    * Performs the single source optimal strategy algorithm in order to keep
      only the relevant paths
    * Compute aggregated levels of service for every origin-destination couple

    :param label_dataframe: a dataframe built on the labels of a graph search
    :type label_dataframe:
        {'dtypes': {'cost': numpy.float64,
        'cumulative': numpy.float64,
        'first': numpy.int64,
        'label_id': numpy.int64,
        'node': int,
        'parent': numpy.int64,
        'route': frozenset,
        'stop': numpy.int64,
        'visited': [int]},
        'type': pandas.core.frame.DataFrame}

    :param headways: {route: headway} dict
    :type headways: dict

    :return: path
    :rtype path:
        {'dtypes': {'actual_cost': numpy.float64,
        'best': numpy.float64,
        'cost': numpy.float64,
        'cumulative': numpy.float64,
        'cumulative_best': numpy.float64,
        'expected': numpy.float64,
        'first': numpy.int64,
        'headway': numpy.float64,
        'key_headway': numpy.int64,
        'key_route': numpy.int64,
        'label_id': numpy.int64,
        'node': numpy.int64,
        'optimization': numpy.float64,
        'parent': numpy.int64,
        'route': frozenset,
        'stop': numpy.int64,
        'transfers': numpy.int64,
        'visited': [int]},
        'type': pandas.core.frame.DataFrame}
    """
    paths = label_dataframe.copy()
    paths['transfers'] = paths['route'].apply(lambda route: len(route) - 2)

    def key_tuple(route):
        headways_slice = {k: v for k, v in headways.items() if k in route}
        max_item = max(headways_slice.items(), key=operator.itemgetter(1))
        return {'key_route': max_item[0], 'key_headway':  max_item[1]}

    # drop_dominated: only a few labels are likely to lower the best expected cost
    paths.reset_index(drop=True, inplace=True)
    paths[['key_headway','key_route']] = pd.DataFrame(list(paths['route'].apply(key_tuple)))
    paths['actual_cost'] = paths['cumulative'] - paths['key_headway']/2
    to_merge = paths.sort_values('cumulative').groupby('stop', as_index=False).first()[['stop', 'cumulative']]
    paths = pd.merge(paths, to_merge, on='stop', suffixes=['', '_best'])
    paths = paths[paths['cumulative_best'] >= paths['actual_cost']]

    #  optimal_strategy: for a given destination, we iterate optimal strategy on all relevant paths.
    #  for a given destination, if two path have the same key_route (route with the worse headway), the fastest is the
    #  only relevant path for the optimal strategy

    paths = paths.sort_values('cumulative').groupby(['stop', 'key_route'], as_index=False).first()
    groups = paths.sort_values('cumulative').groupby('stop')['actual_cost','cumulative']
    optimal = pd.DataFrame([group_optimal_strategy(name, group) for name, group in groups])
    paths = pd.merge(paths, optimal, on='stop', suffixes=['', '_optimal'])
    paths['optimization'] = paths['cumulative_best'] - paths['expected']

    return paths


def key_tuple(route, headways):
    headways_slice = {k: v for k, v in headways.items() if k in route}
    max_item = max(headways_slice.items(), key=operator.itemgetter(1))
    return {'key_route': max_item[0], 'key_headway':  max_item[1]}


def path(label_id, parents, nodes):
    if label_id == 0:
        return [], []
    else:
        label_path, node_path = path(parents[label_id], parents, nodes)
        return label_path + [label_id], node_path + [nodes[label_id]]


def split_skims(all_labels, headways):
    paths = all_labels.copy()
    paths.reset_index(drop=True, inplace=True)
    paths[['key_headway','key_route']] = pd.DataFrame(list(paths['route'].apply(key_tuple, args=[headways])))
    paths['actual_cost'] = paths['cumulative'] - paths['key_headway']/2
    paths['transfers'] = paths['route'].apply(lambda route: len(route) - 2)
    return paths


def links_in_path(indexed_links, nodes):
    path_links = [item for item in nodes if type(item) == int]
    return indexed_links.loc[path_links]


def trip_leg_geometry(name, stops):
    # should be implemented to provide trip leg geometry to group_data
    return 'not implemented yet'


def group_data(name, group, geometry=False):
    to_return = {
        'trip_id': name,
        'stops': [group['stop_id_origin'].iloc[0]] + list(group['stop_id_destination']),
        'headway': group['headway'].mean(),
        'duration': group['duration'].sum()
    }

    if geometry:
        to_return['geometry'] = trip_leg_geometry(name, to_return['stops'])

    return to_return


def footpath_from_path_data(path):

    origins = path['stop_id_origin']
    destination = path['stop_id_destination'].iloc[:-1]
    destination.index = list(origins.index[1:])
    transfer = pd.DataFrame([origins.iloc[1:], destination]).T
    transfer['is_footpath'] = (transfer['stop_id_origin'] - transfer['stop_id_destination'] > 0)

    return transfer[transfer['is_footpath']][['stop_id_origin', 'stop_id_destination']].values.tolist()


def best_paths_from_labels(
    destination,
    labels,
    headways,
    indexed_links,
    first_only=False,
    include_data=False
):

    all_labels = pd.DataFrame(labels)
    parents = all_labels.set_index('label_id')['parent'].to_dict()
    nodes = all_labels.set_index('label_id')['node'].to_dict()

    destination_labels = all_labels[all_labels['stop'] == destination]

    if first_only:
        destination_labels = destination_labels.sort_values('cumulative').iloc[:1]
    optimal = single_source_optimal_strategy(destination_labels, headways)
    summary = optimal[['best', 'expected', 'transfers', 'headway', 'optimization']].mean().to_dict()
    summary['number'] = len(optimal)
    tags = optimal[['label_id', 'actual_cost', 'key_headway', 'transfers']].to_dict(orient='records')

    paths = []

    for tag in tags:
        path_data = links_in_path(indexed_links, path(tag['label_id'], parents, nodes)[1])
        grouped = path_data.groupby('trip_id')
        trip_data = [group_data(name, group) for name, group in grouped]
        footpaths = footpath_from_path_data(path_data)
        path_dict = {'summary': tag,
                     'path_data': path_data if include_data else 'not_included',
                     'trips': trip_data,
                     'footpaths': footpaths}

        paths.append(path_dict)

    return {'summary': summary, 'paths': paths}


def best_paths_from_frequency_links(
    frequency_links,
    origins=None, destinations=None, include_edges=[],
    single_source_labels_kwargs={},
    best_paths_from_labels_kwargs={}
):

    nx_graph, ig_graph = graph.graphs_from_links(frequency_links, include_edges=include_edges, include_igraph=False)

    stop_set = list({int(n) for n in nx_graph.nodes() if type(n) == str})

    origins = origins if origins is not None else stop_set
    destinations = destinations if destinations is not None else stop_set

    headways = frequency_links.groupby('route_id').first()['headway'].to_dict()
    headways[0] = 0

    indexed_links = frequency_links.set_index('index')

    to_return = {}
    for origin in origins:
        to_return[origin] = {}

        labels = graph.single_source_labels(
            origin,
            graph=nx_graph,
            data=graph.indexed_data(frequency_links),
            **single_source_labels_kwargs
        )

        for destination in destinations:

            try:
                to_return[origin][destination] = best_paths_from_labels(
                    destination, labels, headways, indexed_links, **best_paths_from_labels_kwargs)
            except ValueError:  # the label is not reachable
                pass

    return to_return



