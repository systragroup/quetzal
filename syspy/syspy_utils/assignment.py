__author__ = 'qchasserieau'

import itertools

import pandas as pd


def nested_list(volume_array, paths):
    return [[volume_array[i]] * len(paths[i]) for i in range(len(volume_array))]


def assign(volume_array, paths):
    nested_row_indices = [[i] * len(paths[i]) for i in range(len(volume_array))]
    row_indices = list(itertools.chain.from_iterable(nested_row_indices))
    column_indices = list(itertools.chain.from_iterable(paths))
    nested_volumes = nested_list(volume_array, paths)
    volumes = list(itertools.chain.from_iterable(nested_volumes))
    try:
        test = volumes[0][0]  # volumes_array is an ndarray
        sparse = pd.concat(
            (
                pd.DataFrame(
                    {
                        'od': row_indices,
                        'link': column_indices
                    }
                ),
                pd.DataFrame(volumes)
            ),
            axis=1
        )
    except IndexError:  # volume_array is actually a 1d vector
        sparse = (pd.DataFrame({'od': row_indices, 'link': column_indices, 'volume': volumes}))
    return sparse.drop('od', axis=1).groupby('link').sum()
