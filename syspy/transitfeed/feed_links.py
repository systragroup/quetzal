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


def get_trip_links_list(trip_id, links):
    """
    Given a trip_id and a links dataframe, return the list of links included in the trip.
    """
    arrays_result =  list(links[
        (links['trip_id'] == trip_id)&(links['a']!=links['b'])
    ][['a', 'b']].values)
    return [list(x) for x in arrays_result]


def same_direction_ratio(parent_links_list, links_list):
    """
    Given two lists of links, returns the shared directio ratio [-1,1]:
    if 1: all of the oriented links of links_list are included in parent_links_list
          and with the same orientation
    if -1: all of the oriented links of links_list are included in parent_links_list
           but in reversed orientation
    if 0: None of the links in links_list (or their reverse) are in parent_links_list
    """
    same_direction_score = 0
    for link in links_list:
        for link_0 in parent_links_list:
            if link == link_0:
                same_direction_score +=1
            elif link[::-1] == link_0:
                same_direction_score -= 1
    return same_direction_score / len(links_list)


def get_trips_direction(
    links, 
    stop_to_parent_station=None, 
    reference_trip_ids=None, 
    group='route_id', 
    direction_ratio_threshold=0.1,
    count=0,
    ):
    """
    Read a links dataframe and compute the direction for each trip within each group.
    Reference trip ids can be given to define trips that will be given the 0 direction.
    If not, the longest trip of each group (in terms of number of links) will be the reference.
    """
    # Create trip-direction df
    df = links.copy()[[group, 'trip_id']].drop_duplicates().reset_index(drop=True)
    df['direction_id'] = None
    # Aggregate links in list
    df['links_list'] = df['trip_id'].apply(lambda x: get_trip_links_list(x, links))
    if stop_to_parent_station is not None:
        df['links_list'] = df['links_list'].apply(lambda x: [[stop_to_parent_station[a] for a in l] for l in x])
        df['links_list'] = df['links_list'].apply(lambda a: [[x[0], x[1]] for x in a if x[0]!=x[1]])

    df['links_number'] = df['links_list'].apply(len)
    # Sort by descending length
    df = df.sort_values(
        [group, 'links_number'],
        ascending=False
    ).reset_index(drop=True)
    
    if reference_trip_ids is None:
        reference_trip_ids = df.groupby(group).first()['trip_id'].to_dict()
    # Set direction of reference trip to 0
    df['reference_trip'] = df[group].apply(lambda x: reference_trip_ids[x])
    # Compute shared direction ratio
    df['direction_0_ratio'] = df.apply(
        lambda x: same_direction_ratio(
            x['links_list'],
            df.loc[df['trip_id']==x['reference_trip'], 'links_list'].values[0]
        ),
        1
    )
    # Give 0 or 1 direction id if shared direction is not zero, 100 (=not found) otherwise
    df['direction_id'] = df['direction_0_ratio'].apply(
        lambda x: 0*(x>direction_ratio_threshold) + 1*(x<-direction_ratio_threshold) +\
                 100*(x>=-direction_ratio_threshold)*(x<=direction_ratio_threshold)
        )
    
    # While some direction id are 100 (not found): recursively call function
    
    while len(df[df['direction_id']==100])>0 and (count < 10):
        
        df_ = get_trips_direction(
            links.loc[links['trip_id'].isin(df[df['direction_id']==100]['trip_id'].values)],
            stop_to_parent_station=stop_to_parent_station,
            group=group,
            direction_ratio_threshold=direction_ratio_threshold,
            count=count+1
        )
        d=df[
            (df['direction_0_ratio']!=0)&(df['direction_id']<100)
        ].groupby(group)['direction_id'].max().to_dict()
        df_['direction_id'] = df_.apply(
            lambda x: 1 + x['direction_id'] + d[x[group]] if x['direction_id']!=100 else x,
            1
        )
        t = df_.set_index('trip_id')['direction_id'].to_dict()
        df['direction_id'] = df.apply(lambda x: t.get(x['trip_id'], x['direction_id']), 1)
    
    return df[[group, 'trip_id', 'direction_id', 'direction_0_ratio', 'links_list']]

