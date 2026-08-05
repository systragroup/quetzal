import pandas as pd
import polars as pl
import numpy as np
from syspy.spatial.geometries import line_list_to_polyline
from itertools import chain
from typing import Optional


def find_common_sets(links: pd.DataFrame) -> list[frozenset]:
    counter = links.groupby(['a', 'b'])[['trip_id']].agg(frozenset)
    counter = counter.rename(columns={'trip_id': 'trip_id_set'})
    counter['len'] = counter['trip_id_set'].apply(len)
    counter = counter[counter['len'] > 1]
    counter = counter.sort_values(by='len', ascending=False)
    trip_id_sets = counter['trip_id_set'].unique()
    return trip_id_sets


def _get_shared_stops(pl_stops_list: pl.DataFrame, trip_set: set[str]) -> np.ndarray:
    filtered_stops = pl_stops_list.filter(pl.col('trip_id').is_in(trip_set))
    _expr = [pl.col('trip_id').alias('trip_id_list'), pl.col('link_sequence'), pl.col('index')]
    shared_stops = filtered_stops.group_by('stop', maintain_order=True).agg(_expr)
    shared_stops = shared_stops.filter(pl.col('trip_id_list').list.n_unique() == len(trip_set))
    # if a link uses 2 trip. we dont want 3 links because its a loop and reuse multiple time the same link. we drop and it will be filled if necessary.
    shared_stops = shared_stops.filter(pl.col('trip_id_list').list.n_unique() == pl.col('index').list.n_unique())
    # remove stops not in order.
    if len(shared_stops) < 2:
        return []
    arr = shared_stops['link_sequence'].to_numpy()
    arr = np.stack(arr, axis=1)

    order = np.all(np.diff(arr) > 0, axis=0)  # i to i+1. so last is missing
    # check the last diff for the last stop (to append) if was decreasing (False) must be False also.
    order = np.append(order, order[-1])
    shared_stops = shared_stops.filter(order)

    shared_stops = shared_stops['stop'].to_numpy()
    return shared_stops


def bisect(arr: np.ndarray[str], start: str, end: str):
    a = np.where(arr == start)[0][0]
    b = np.where(arr == end)[0][0]
    return arr[a : b + 1]


def _agg_on_shared_stops(shared_stops: np.array, trip_stops: np.array) -> list[tuple]:
    """
    shared_stops: list of all shared stops (ordered)
    trip_stops: list of all the stop for a trip (ordered)

    return
        a list of edges from the first to the last shared trip.
    """
    first_stop = shared_stops[0]
    last_stop = shared_stops[-1]
    trip_stops = bisect(trip_stops, first_stop, last_stop)
    if len(trip_stops) < 2:
        # some trips are detedted as shared, but they are "loop". their end is at the start of the others.
        # in this case bisect return [], and we skip.
        # TODO we could crop to keep only the beginning, or create 2 new common links for example
        return []

    i = 1
    origin = first_stop
    edges = []
    for stop in shared_stops[1:]:
        target = stop
        stops_list = [origin]
        next_stop = trip_stops[i]
        if target not in trip_stops[i:]:
            continue
        while next_stop != target:
            stops_list.append(next_stop)
            i += 1
            next_stop = trip_stops[i]

        # create a list of tuple
        stops_list.append(target)
        edges.append(list(zip(stops_list[:-1], stops_list[1:])))
        origin = target
        i += 1
    return edges


def find_common_trips(links: pd.DataFrame, trip_id_sets: list[frozenset], log=False) -> pd.DataFrame:
    links = links[['a', 'b', 'trip_id', 'link_sequence']].reset_index()
    last = links.groupby(['trip_id']).agg('last').reset_index()
    last['a'] = last['b']
    last['link_sequence'] += 1
    last['index'] += '_last'
    stops_list = pd.concat([links, last], ignore_index=True).sort_values(['trip_id', 'link_sequence'])
    stops_list = stops_list.drop(columns=['b']).reset_index(drop=True).rename(columns={'a': 'stop'})
    stop_list_dict = stops_list.groupby('trip_id')['stop'].agg(np.array).to_dict()
    links_dict_per_trip = (
        links.reset_index().groupby('trip_id').apply(lambda g: g.set_index(['a', 'b'])['index'].to_dict()).to_dict()
    )
    pl_stops_list = pl.DataFrame(stops_list)
    common_list = {'edges': [], 'links': [], 'trip_id': [], 'common_trip_id': [], 'link_sequence': []}
    for i, trip_set in enumerate(trip_id_sets):
        shared_stops = _get_shared_stops(pl_stops_list, trip_set)
        if len(shared_stops) <= 2:
            print('skip: less than 2 shared stops: ', i, trip_set) if log else None
            continue

        result = {'edges': [], 'links': [], 'trip_id': [], 'link_sequence': []}
        for trip in trip_set:
            trip_stops = stop_list_dict.get(trip)
            edges = _agg_on_shared_stops(shared_stops, trip_stops)
            if len(edges) == 0:
                # if at least one trip was not well ordered (when bisect. last node is before first). we skip
                # we could keep the other trips as common, but they will exist somewhere in another trip_set
                result = None
                print('skip: cannot bissect: ', i, trip_set) if log else None
                break
            result['edges'].extend(edges)
            _dict = links_dict_per_trip.get(trip)
            result['links'].extend([[*map(_dict.get, ls)] for ls in edges])
            result['trip_id'].extend([trip] * len(edges))
            result['link_sequence'].extend([i for i in range(1, len(edges) + 1)])
        if result is None:
            continue
        common_list['edges'].extend(result['edges'])
        common_list['trip_id'].extend(result['trip_id'])
        common_list['common_trip_id'].extend([f'common_{i}'] * len(result['edges']))
        common_list['link_sequence'].extend(result['link_sequence'])
        common_list['links'].extend(result['links'])

    df = pd.DataFrame(common_list)
    df['a'] = df['edges'].apply(lambda ls: ls[0][0])
    df['b'] = df['edges'].apply(lambda ls: ls[-1][-1])

    df = (
        df.groupby(['common_trip_id', 'a', 'b'])
        .agg({'trip_id': list, 'link_sequence': 'first', 'links': list})
        .reset_index()
    )

    # only keeps common links using all the common trips.
    # TODO: I think its like the len(edges) == 0 skip. but the bisect does return some common links
    df['len'] = df['trip_id'].apply(len)
    df = df[df['len'] == df.groupby('common_trip_id')['len'].transform('max')]
    df = df[df['len'] > 1]  # same-ish reason. we can have a common with a single trip: remove as its not common

    df = df.rename(columns={'trip_id': 'trip_id_list', 'common_trip_id': 'trip_id', 'links': 'index_list'})
    df.index = 'link_' + df['trip_id'] + '_' + df['link_sequence'].astype(str)
    df.index.name = 'index'

    df = df[['a', 'b', 'trip_id', 'trip_id_list', 'index_list', 'link_sequence']]
    df = df.sort_values(['trip_id', 'link_sequence'])
    print(len(df['trip_id'].unique()), ' common_trips founded')
    return df


# delete new common trips if total time is smaller than min_time
def restrict_common_trips(common_trips: pd.DataFrame, links: pd.DataFrame, min_time=5 * 60) -> pd.DataFrame:
    # return common_trips
    time_dict = links['time'].to_dict()
    common_trips['time'] = common_trips['index_list'].apply(
        lambda lsls: np.mean([sum([*map(time_dict.get, ls)]) for ls in lsls])
    )

    to_keep = common_trips.groupby('trip_id')['time'].agg(sum) >= min_time
    to_keep = to_keep[to_keep].index
    common_trips = common_trips[common_trips['trip_id'].isin(to_keep)]
    print(len(common_trips['trip_id'].unique()), ' common trips after filtering')

    return common_trips.drop(columns=['time'])


def create_common_links(
    common_trips: pd.DataFrame, links: pd.DataFrame, merge_overwrite={}, agg_overwrite={}
) -> pd.DataFrame:
    """
    merge_overwrite: to change a column agg (in the merge)
        we take first of all cols, but merge geometry, sum time, etc
    agg_overwrite: to change a column agg (in the merge.)
        we take first of all cols, but average time, 1 / sum(1 / x) headway, etc

    """

    # create agg dict
    # to merge links (fillgap) when there are multiple links for 1 link
    merge_dict = {col: 'first' for col in links.columns}
    merge_dict['index'] = 'first'
    merge_dict['boarding_time'] = 'first'  # we board on first link. not a sum of all
    merge_dict['headway'] = 'first'
    merge_dict['geometry'] = line_list_to_polyline
    merge_dict['road_link_list'] = lambda x: list(chain.from_iterable(x))  # concat list
    merge_dict['time'] = sum
    merge_dict['length'] = sum
    merge_dict.update(merge_overwrite)  # change predefined agg func
    # agg dict
    agg_dict = {col: 'first' for col in links.columns}
    agg_dict['road_link_list'] = lambda x: list(chain.from_iterable(x))  # concat list
    agg_dict['headway'] = lambda x: 1 / sum(1 / x)
    agg_dict['time'] = np.mean
    agg_dict['length'] = np.mean
    agg_dict['boarding_time'] = np.mean
    agg_dict.update(agg_overwrite)  # change predefined agg func

    #
    # create links
    #

    common_links = common_trips[['a', 'b', 'trip_id', 'link_sequence', 'index_list']]
    common_links = common_links.explode('index_list')

    common_links['to_sum'] = range(0, len(common_links))
    common_links = common_links.explode('index_list')

    # merge all links to agg
    common_links = common_links.merge(
        links.drop(columns=common_links.columns, errors='ignore'), left_on='index_list', right_index=True
    ).reset_index()

    common_links = common_links.groupby('to_sum').agg(merge_dict)

    common_links = common_links.groupby('index').agg(agg_dict)

    common_links['speed'] = 3.6 * common_links['length'] / common_links['time']

    common_links = common_links.sort_values(by=['trip_id', 'link_sequence'])
    common_links.index.name = 'index'
    common_links['is_common'] = True
    common_links = common_links[links.columns]
    return common_links


def _get_links_inbetween_trips(a: str, b: str, trip_list: list[str], links: pl.DataFrame) -> list[list[str]]:
    # for each trips, return a list of links (index) starting at a and finishing at b
    # links need to be sorted here.
    links_list = []
    for trip in trip_list:
        tmp_links = links.filter(pl.col('trip_id') == trip)
        tmp_list = _get_links_inbetween(a, b, tmp_links)
        links_list.append(tmp_list)

    return links_list


def _get_links_inbetween(a: str, b: str, trip_links: pl.DataFrame) -> list[str]:
    # return a list of links (index) starting at a and finishing at b
    # trip_links need to be sorted here.
    arr = trip_links.select(['a', 'b', 'index']).to_numpy()
    filter_from = np.where(arr[:, 0] == a)[0][0]
    filter_to = np.where(arr[:, 1] == b)[0][-1]
    index_list = arr[filter_from : filter_to + 1, 2]
    return index_list


def distribute_commons_on_links(
    links: pd.DataFrame,
    common_trips: pd.DataFrame,
    all_cols=['volume'],
    first_cols=['boardings'],
    last_cols=['alightings'],
    keep_common=False,
    secondary_weight: Optional[str] = None,
):
    """
    distribute volumes, boardings, alightings from common to normal links.
    distribution is weighted with Headway and secondary weight if provided.
    when multiple links makes 1 common_link, boardings apply on first, alightings on last.
    ex: common_link1 = [link1, link2, link3] from trip1 and [link100] from trip2
        trip1 and trip2 have the same headway, so the common_volume (100) is distributed 50/50.
        we add 50 to link1, link2, link3 and link100. however, if there are boardings (let say 100)
        on the common links. we only want to add  50 boardings to link1 and link100. not link2 and link3.
    Parameters
    ----------
    links: sm.links
    common_trips : sm.common_trips
    all_cols: columns to distribute on all links
    first_cols: column to distribute on the first link only
    last_cols: column to distribute on the last link only
    keep_common: delete common_links from links if False, their volume is distributed. should drop.
    secondary_weight: add another weight (not just the headway). should be greater than 0.
        W_i = Wh_i * Ws_i / sum(Wh_i * Ws_i). where Wh is the headway weight and Ws the secondary weight
    """

    # we have common_link : [[links], [links]] # first list is each trip. second is the merge of links (fill gap)
    common_links = common_trips[['index_list', 'trip_id_list', 'trip_id']]
    # add each common_link volume to the df
    # we also need the common_link headway (to weight each trip in the common trip later)
    cols = [*all_cols, *first_cols, *last_cols]
    common_links = common_links.merge(links[[*cols, 'headway']], left_index=True, right_index=True).reset_index()
    # explode to have common_link: [links]. so each row is a trip that get a portion of the volume
    common_links = common_links.explode(['trip_id_list', 'index_list'])
    # add each trip headway to compute the weight.
    # we can take the fist link to have the trip headway
    # common_links['first_link'] = common_links['index_list'].apply(lambda x: x if type(x) is str else x[0])
    headway_dict = links.set_index('trip_id')['headway'].to_dict()
    common_links['trip_headway'] = common_links['trip_id_list'].apply(headway_dict.get)
    # Weight = combined_headway*(1/headway)
    common_links['weight'] = common_links['headway'] / common_links['trip_headway']
    # can add a secondary weight: like bpr of (volume/capacity).
    # this hep to move people on lines with less peoples that have similar headways.
    if secondary_weight is not None:
        _dict = links[secondary_weight].to_dict()
        common_links['secondary_weight'] = common_links['index_list'].apply(lambda ls: _dict.get(ls[0], 1))
        common_links['weight'] *= common_links['secondary_weight']

        tot_weight = common_links.groupby('index')['weight'].sum().to_dict()
        common_links['weight'] = common_links['weight'] / common_links['index'].apply(tot_weight.get)

    # now we have each Trip weight for each common links. we can explode.

    # The volume for Gap links is still weight X volume for each link
    common_links = common_links.drop(columns=['index'])
    all_links = common_links.explode('index_list').rename(columns={'index_list': 'index'})
    # for boardings: only apply on first.
    first_links = common_links.copy()
    first_links['index'] = first_links['index_list'].apply(lambda ls: ls[0])
    # for alightings: only aply on last
    last_links = common_links.copy()
    last_links['index'] = last_links['index_list'].apply(lambda ls: ls[-1])

    for col in all_cols:
        all_links[col] = all_links[col] * all_links['weight']

    for col in first_cols:
        first_links[col] = first_links[col] * first_links['weight']

    for col in last_cols:
        last_links[col] = last_links[col] * last_links['weight']

    # then sum on the links. (we can ahve multiple common links using the same link)
    all_links = all_links.groupby('index')[all_cols].agg(sum)
    first_links = first_links.groupby('index')[first_cols].agg(sum)
    last_links = last_links.groupby('index')[last_cols].agg(sum)
    if not keep_common:
        links = links[~links['is_common']]

    links[all_cols] = links[all_cols].add(all_links[all_cols], fill_value=0)
    links[first_cols] = links[first_cols].add(first_links[first_cols], fill_value=0)
    links[last_cols] = links[last_cols].add(last_links[last_cols], fill_value=0)
    return links
