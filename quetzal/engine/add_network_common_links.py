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


def find_common_trips(links: pd.DataFrame, trip_id_sets: list[frozenset]) -> pd.DataFrame:
    #
    # return common_trips (index  a	b	trip_id_list	index_list	link_sequence	trip_id)
    #
    ab_links = pl.DataFrame(links.reset_index()[['index', 'a', 'b', 'trip_id', 'link_sequence']])
    ab_links = ab_links.sort(['trip_id', 'link_sequence'])
    # we want index_list as we want a list of list when agg. at the end we have a,v:[[link1],[link2],...]. but there could
    # be multiple links (when we find missing links) for a given a,b,trip. so we need a list of list.
    ab_links = ab_links.with_columns(pl.col('index').map_elements(lambda x: [x]).alias('index_list'))
    new_links = []
    for i, trip_set in enumerate(trip_id_sets):
        # filter links for the one in the trip_set.
        filtered_links = ab_links.filter(pl.col('trip_id').is_in(trip_set))

        # first. we find the common links. links shared between all trips
        _expr = [
            pl.col('trip_id').alias('trip_id_list'),
            pl.col('index_list'),
            pl.col('link_sequence').first(),
            pl.col('link_sequence').alias('seq'),
        ]
        common_list = filtered_links.group_by(['a', 'b']).agg(_expr)
        # some links doesnt contain all our trip. check if list == set with a n_unique.(this method is valid because of the prior filter)
        common_list = common_list.filter(pl.col('trip_id_list').list.n_unique() == len(trip_set))
        # if a link uses 2 trip. we dont want 3 links because its a loop and reuse multiple time the same link. we drop and it will be filled if necessary.
        common_list = common_list.filter(pl.col('trip_id_list').list.n_unique() == pl.col('index_list').list.n_unique())
        common_list = common_list.sort(by='link_sequence')

        # check if link_sequences is consecutives for each trip. if not. its not to agg.
        # ex: we have trips_1 and_2 going from link_a to link_b. and  trip (trip_3) from link_b to link_a.
        # if its the case, we do not agg, we exit. there should be a trip_set in trip_id_sets that is only {trip_1, trip_2}
        arr = common_list['seq'].to_numpy()
        if len(arr) == 0:
            continue
        arr = np.stack(arr, axis=1)
        if not np.all(np.diff(arr) > 0):
            continue

        common_list = common_list.drop('seq')
        # find missing links. if a,b are not consecutive (b != next_a). we add a new link
        missing_list = common_list.with_columns(pl.col('a').shift(-1).alias('next_a'))
        missing_list = missing_list.filter(pl.col('next_a').is_not_null())
        missing_list = missing_list.filter(pl.col('b') != pl.col('next_a'))

        missing_list[['a', 'b']] = missing_list[['b', 'next_a']]
        missing_list = missing_list.with_columns(pl.col('link_sequence') + 1)
        missing_list = missing_list.drop(['next_a', 'index_list'])
        if len(missing_list) > 0:
            lsls = []
            for a, b, trips in missing_list[['a', 'b', 'trip_id_list']].to_numpy():
                lsls.append(_get_links_inbetween_trips(a, b, trips, filtered_links))
            missing_list = missing_list.with_columns(pl.Series('index_list', lsls))
            missing_list = missing_list.select(common_list.columns)
            common_list = pl.concat([common_list, missing_list])

        # finish up
        common_list = common_list.sort('link_sequence')

        common_list = common_list.with_columns(pl.arange(1, common_list.height + 1).alias('link_sequence'))
        common_list = common_list.with_columns(pl.lit(f'common_{i}').alias('trip_id'))
        common_list = common_list.with_columns(
            ('link_' + pl.col('trip_id') + '_' + pl.col('link_sequence').cast(str)).alias('index')
        )

        new_links.append(common_list)

    new_links = pl.concat(new_links)
    new_links.to_pandas().set_index('index')
    print(len(new_links['trip_id'].unique()), ' common_trips founded')
    return new_links.to_pandas().set_index('index')


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

    return common_trips


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

    common_links = common_trips.drop(columns=['trip_id_list'])
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
