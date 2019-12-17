# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'

import pandas as pd


def link_from_stop_times(
    stop_times, 
    max_shortcut=False,
    stop_id='stop_id',
    trip_id='trip_id',
    in_sequence='stop_sequence',
    out_sequence='link_sequence',
    stop_id_origin=False,
    stop_id_destination=False,
    keep_origin_columns=[],
    keep_destination_columns=[]
):
    """
    From a set of trips, represented as a table of events (stop time for
    example), returns a table of links between these events, given two trips:
    a-b-c-d and f-g-h. we should return : ab, bc, cd, fg and gh.


    :param stop_times: DataFrame
    :param max_shortcut:
    :param stop_id:
    :param in_sequence:
    :param out_sequence:
    :param stop_id_origin:
    :param stop_id_destination:
    :param keep_origin_columns:
    :param keep_destination_columns:
    :return:
    """
    origins = stop_times[[stop_id, trip_id,
                          in_sequence] + keep_origin_columns].copy()
    destinations = stop_times[[stop_id, trip_id,
                               in_sequence] + keep_destination_columns].copy()

    links = []
    
    max_sequence = stop_times[in_sequence].max()
    assert max_sequence
    max_shortcut = max_shortcut if max_shortcut and max_shortcut < max_sequence else max_sequence
    
    for i in range(int(max_shortcut)):
        origins['next'] = origins[in_sequence] + 1 + i
        links.append(pd.merge(origins, destinations,
                              left_on=[trip_id, 'next'],
                              right_on=[trip_id, in_sequence],
                              suffixes=['_origin', '_destination']
                              ))
    stop_id_origin = stop_id_origin if stop_id_origin else stop_id + '_origin'
    stop_id_destination = stop_id_destination if stop_id_destination else stop_id + '_destination'

    assert len(links), 'no link to concatenate'
    concat = pd.concat(links).rename(
        columns={
            in_sequence + '_origin': out_sequence,
            stop_id + '_origin': stop_id_origin,
            stop_id + '_destination': stop_id_destination
        }
    ).drop([in_sequence + '_destination', 'next'], axis=1)

    return concat


def clean_sequences(df, sequence='stop_sequence', group_id='trip_id'):
    """ Clean the sequence column, drop the index"""
    df = df.sort_values([group_id, sequence]).reset_index(drop=True)
    sequence_series = df.groupby(group_id).count()[
        sequence].sort_index()

    all_sequences = []
    for group_sequences in list(sequence_series):
        for seq in range(group_sequences):
            all_sequences.append(seq + 1)

    df[sequence] = pd.Series(all_sequences)

    return df