# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'
import pandas as pd
import itertools
import collections


def ma_fonction_a_tester(a, b):
    return a + b


def nested_list(volume_array, paths):
    return [[volume_array[i]]*len(paths[i]) for i in range(len(volume_array))]


def assign(volume_array, paths, checkpoints=None, checkpoints_how='all'):

    if checkpoints is not None:
        checkpoints = set(checkpoints)
        if checkpoints_how == 'all':
            keep = [set(path) >= checkpoints for path in paths]
            volume_array = keep * volume_array
        elif checkpoints_how == 'any':
            keep = [bool(checkpoints.intersection(path)) for path in paths]
            volume_array = keep * volume_array
        else:
            print('checkpoints failed')

    nested_row_indices = [[i]*len(paths[i]) for i in range(len(volume_array))]
    row_indices = list(itertools.chain.from_iterable(nested_row_indices))
    column_indices = list(itertools.chain.from_iterable(paths))
    nested_volumes = nested_list(volume_array, paths)
    volumes = list(itertools.chain.from_iterable(nested_volumes))
    try:
        # volumes_array is an ndarray
        assert isinstance(volumes[0], collections.Iterable)
        sparse = pd.concat(
            (pd.DataFrame({'od': row_indices, 'link': column_indices}),
             pd.DataFrame(volumes)),
            axis=1
        )
    except AssertionError:  # volume_array is actually a 1d vector
        sparse = pd.DataFrame(
            {
                'od': row_indices,
                'link': column_indices,
                'volume': volumes
            }
        )

    return sparse.drop('od', axis=1).groupby('link').sum()


def remove_ab_prefix(df):
    df = df.copy()

    df['a'] = df['a'].apply(lambda s: s.split('_')[1])
    df['b'] = df['b'].apply(lambda s: s.split('_')[1])
    try:
        df[['a', 'b']] = df[['a', 'b']].astype(int)

    # nodes have never been ints...
    except ValueError:  # invalid literal for int() with base 10:
        pass

    return df


def remove_od_prefix(df):
    df = df.copy()
    df['origin'] = df['origin'].apply(lambda s: s.split('_')[1]).astype(int)
    df['destination'] = df['destination'].apply(lambda s: s.split('_')[1]).astype(int)
    return df


def label_links(links):
    links = links.copy()
    links['a'] = 'node_' + links['a'].astype(str)
    links['b'] = 'node_' + links['b'].astype(str)
    return links


def label_ntlegs(ntlegs):
    ntlegs = ntlegs.copy()
    ntlegs['centroid'] = 'centroid_' + ntlegs['centroid'].astype(str)
    ntlegs['node'] = 'node_' + ntlegs['node'].astype(str)
    return ntlegs


def label_volumes(volumes):
    volumes = volumes.copy()
    volumes['origin'] = 'centroid_' + volumes['origin'].astype(str)
    volumes['destination'] = 'centroid_' + volumes['destination'].astype(str)
    return volumes


def build_ntlinks(ntlegs):
    ntlinks = pd.concat(
        [
            ntlegs.rename(columns={'centroid': 'b', 'node': 'a'}),
            ntlegs.rename(columns={'centroid': 'a', 'node': 'b'})
        ]
    ).drop_duplicates(subset=('a', 'b'))
    return ntlinks


def build_edges(links, ntlegs):
    links = label_links(links)
    #ntlegs = label_ntlegs(ntlegs)
    ntlinks = build_ntlinks(ntlegs)
    return pd.concat([links, ntlinks])[['a', 'b', 'time']].reset_index(drop=True)


def link_list_from_path(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]


def nested_dict_to_stack_matrix(nested_dict, centroids, name='value'):
    tuples = []

    for origin, single_source_nested_dict in nested_dict.items():
        if origin in centroids:
            for destination, path in single_source_nested_dict.items():
                if destination in centroids:
                    tuples.append((origin, destination, path))

    stack = pd.DataFrame(tuples, columns=['origin', 'destination', name])

    return stack

