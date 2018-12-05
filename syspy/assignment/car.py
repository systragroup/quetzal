# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'

import pandas as pd
import numpy as np
import networkx as nx
from syspy.assignment.raw import *
from tqdm import tqdm
from syspy.routing import networkx_wrapper as nxw

def path_matrix(edges, centroid_set):

    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(edges.values.tolist())
    allpaths = nx.all_pairs_dijkstra_path(digraph)

    matrix = pd.DataFrame(
        {
            key: {
                _key: _value for _key, _value in value.items()
                if _key in centroid_set
            }
            for key, value in allpaths.items() if key in centroid_set
         }
    )

    return matrix


def load_edges(edges, volumes):

    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(edges.values.tolist())

    centroid_set = set(volumes['origin']).union(set(volumes['destination']))

    allpaths = nxw.pairwise_function(
        single_source_function=nx.single_source_dijkstra_path,
        sources=centroid_set,
        targets=centroid_set,
        G=digraph
    )

    # todo : factorize ?
    matrix = pd.DataFrame(
        {key: {
            _key: _value
            for _key, _value in value.items() if _key in centroid_set
            }
         for key, value in allpaths.items() if key in centroid_set}
    )

    ab_index = edges.reset_index().set_index(['a', 'b'])['index'].to_dict()

    def edge_list_from_path(path):
        return [
            ab_index[ab]
            for ab in link_list_from_path(path)
            if ab in ab_index.keys()
        ]

    stack = matrix.T.stack().reset_index()
    stack.columns = ['origin', 'destination', 'path']

    stack['path'] = stack['path'].apply(edge_list_from_path)
    merged = pd.merge(volumes, stack, on=['origin', 'destination'])

    volume_array = merged['volume'].values
    paths = merged['path'].values

    assigned = assign(volume_array, paths)
    return pd.merge(edges, assigned, left_index=True, right_index=True, how='left').fillna(0)


def load_links(links, ntlegs, volumes):

    edges = build_edges(links, ntlegs)
    loaded_edges = load_edges(edges, label_volumes(volumes))

    links = label_links(links)
    loaded_links = pd.merge(
        links.drop('volume', axis=1, errors='ignore'),
        loaded_edges[['a', 'b', 'volume']])

    if 'load' in links.columns:
        loaded_links['load'] = loaded_links['load'] + loaded_links['volume']
    else:
        loaded_links['load'] = loaded_links['volume']
    return remove_ab_prefix(loaded_links)


def jam_time(links, ref_time='time', alpha=0.4, beta=4, capacity=1500):

    alpha = links['alpha'] if 'alpha' in links.columns else alpha
    beta = links['beta'] if 'beta' in links.columns else beta
    capacity = links['capacity'] if 'capacity' in links.columns else capacity
    return links[ref_time] * (1 + alpha * np.power((links['load'] / capacity), beta))


def assign_wardrop(
    volumes,
    links,
    ntlegs,
    n_slices=10,
    alpha=0.4,
    beta=4,
    capacity=1500,
    penalty=1e9,
    force=True
):

    penalty_ntlegs = ntlegs.copy()
    penalty_ntlegs['time'] += penalty

    slices = [
        (volumes.set_index(['origin', 'destination']) / n_slices).reset_index()
        for i in range(n_slices)
    ]

    links = links.copy()

    if force:
        links['capacity'] = capacity if capacity else links['capacity']
        links['alpha'] = alpha if alpha else links['alpha']
        links['beta'] = beta if beta else links['beta']

    links.drop('load', axis=1, errors='ignore', inplace=True)

    links['wardrop_reference_time'] = links['time']

    for volume_slice in slices:

        loaded_links = load_links(links, penalty_ntlegs, volume_slice)
        links['time'] = jam_time(
            loaded_links,
            ref_time='wardrop_reference_time',
            alpha=alpha,
            beta=beta,
            capacity=capacity
        )
        links = loaded_links

    return links


def time_matrix(links, ntlegs, penalty=1e9):

    penalty_ntlegs = ntlegs.copy()
    penalty_ntlegs['time'] += penalty

    centroid_set = set(label_ntlegs(penalty_ntlegs)['centroid'])

    edges = build_edges(links, penalty_ntlegs)
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(edges.values.tolist())

    alllengths = nxw.pairwise_function(
        single_source_function=nx.single_source_dijkstra_path_length,
        sources=centroid_set,
        targets=centroid_set,
        G=digraph
    )
    # alllengths = nx.all_pairs_dijkstra_path_length(digraph)

    matrix = pd.DataFrame(
        {
            key: {
                _key: _value for _key, _value in value.items()
                if _key in centroid_set
            }
            for key, value in alllengths.items() if key in centroid_set
        }
    )

    stack = matrix.stack().reset_index()
    stack.columns = ['origin', 'destination', 'length']

    # we remove the penalty added to the ntlegs
    stack.loc[stack['origin'] != stack['destination'], 'length'] -= 2 * penalty

    return remove_od_prefix(stack)
