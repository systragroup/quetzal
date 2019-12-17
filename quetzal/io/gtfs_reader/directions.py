import gtfs_kit as gk
import pandas as pd
import numpy as np
from . import patterns

def build_directions(
    feed, group=['route_id'], on='stop_id', all_trips=False
):
    """
    Build the column direction_id of the table trips.
    This is done by comparing the ordered stop sequence of trip within
    a given group (default route_id).
    Args:
        - group (list, ['route_id']): column names to build the trip groups
        - on (str, 'stop_id'): stop entity whose sequences are compared. 
                can be 'cluster_id', 'parent_station'
        - all_trips (bool - False): compute missing directions only or all
    Returns:
        None
    """
    trip_stops = patterns.get_trip_stop_list(feed.stop_times, feed.stops, on=on)
    if all_trips:
        feed.trips['direction_id'] = np.nan
    trip_directions = feed.trips.groupby(group).apply(
        lambda x: get_trip_directions(x, trip_stops)
    ).reset_index()
    feed.trips.drop('direction_id', 1, inplace=True)
    feed.trips = feed.trips.merge(
        trip_directions[group + ['trip_id', 'direction_id']],
        on=group + ['trip_id']
    )

def stop_list_to_link_list(stop_list):
    """
    [a, b, c, d] -> [(a,b), (b,c), (c,d)]
    """
    return list(zip(stop_list[:-1], stop_list[1:]))

def same_direction_ratio(parent_link_list, link_list):
    """
    Given two lists of links, returns the shared direction ratio [-1,1]:
    if 1: all of the oriented links of link_list are included in parent_link_list
          and with the same orientation
    if -1: all of the oriented links of link_list are included in parent_link_list
           but in reversed orientation
    if 0: None of the links in link_list (or their reverse) are in parent_link_list
    Args:
        - parent_link_list: [(a,b), (b,c)]: the reference link list
        - link_list: [(a,b), (b,c)]
    Return:
        - direction_ratio (float): bewteen -1 and 1
    """
    same_direction_score = 0
    parent_link_list = [[x[0], x[1]] for x in parent_link_list if x[0]!=x[1]]
    link_list = [[x[0], x[1]] for x in link_list if x[0]!=x[1]]
    for link in link_list:
        for link_0 in parent_link_list:
            if link == link_0:
                same_direction_score +=1
            elif link[::-1] == link_0:
                same_direction_score -= 1
    return same_direction_score / len(link_list) if len(link_list) > 0 else 0

def get_trip_directions(
    trip_group, trip_stops, direction_ratio_threshold=0.1,
    count=0, max_iter=20
):
    """
    Given a group of trips, return their directions
    """
    trip_group = trip_group.copy()
    trip_group['stop_list'] = trip_group['trip_id'].apply(lambda x: trip_stops.loc[x])
    trip_group['link_list'] = trip_group['stop_list'].apply(stop_list_to_link_list)
    trip_group['n_stops'] = trip_group['stop_list'].apply(len)
    # Sort by descending length
    df = trip_group.sort_values('n_stops', ascending=False).reset_index(drop=True)
    # Get one trip for which direction is define or set longest one to 0
    if len(df[(df['direction_id'] == 0)|(df['direction_id'] == 1)]) >= 1:
        ref_id = df.loc[
            (df['direction_id'] == 0)|(df['direction_id']==1),
            'trip_id'
        ].values[0]
    else:
        ref_id = df.loc[0, 'trip_id']
        df.loc[0, 'direction_id'] = 0
    ref_direction = df.loc[df['trip_id']==ref_id, 'direction_id'].values[0]
    ref_link_list = df.loc[df['trip_id']==ref_id, 'link_list'].values[0]
    df['reference_trip'] = ref_id

    def get_direction_id(x):
        if not np.isnan(x['direction_id']) and x['direction_id'] is not None:
            return x['direction_id']
        else:
            dir_ratio = same_direction_ratio(x['link_list'], ref_link_list)
            to_return = ref_direction * (dir_ratio > direction_ratio_threshold) +\
                (1-ref_direction) * (dir_ratio < -direction_ratio_threshold) +\
                100 * (dir_ratio >= -direction_ratio_threshold) *\
                (dir_ratio <= direction_ratio_threshold)
            return to_return
    
    df['direction_id'] = df.apply(get_direction_id, 1)
    
    # While some direction id are 100 (not found): recursively call function
    count = 0
    while len(df[df['direction_id']==100])>0 and (count < max_iter):
        count += 1
        df_ = get_trip_directions(
            trip_group.loc[trip_group['trip_id'].isin(df[df['direction_id']==100]['trip_id'].values)],
            trip_stops, 
            direction_ratio_threshold=direction_ratio_threshold,
            count=count
        )
        max_d = df[(df['direction_id']<100)]['direction_id'].max()
        df_['direction_id'] = df_.apply(
            lambda x: 1 + x['direction_id'] + max_d if x['direction_id']!=100 else x,
            1
        )
        t = df_.set_index('trip_id')['direction_id'].to_dict()
        df['direction_id'] = df.apply(lambda x: t.get(x['trip_id'], x['direction_id']), 1)
    
    df['direction_id'] = df['direction_id'].map(int)

    return df[['trip_id', 'direction_id']]
