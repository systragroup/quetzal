__author__ = 'qchasserieau'

import itertools

import pandas as pd
from shapely import geometry as shapely_geometry
from syspy.skims import skims
from syspy.transitfeed import feed_links


def linestring_geometry(row):
    return shapely_geometry.linestring.LineString([[row['x_origin'], row['y_origin']],
                                                   [row['x_destination'], row['y_destination']]])


def transfers_from_stops(
    stops,
    stop_times,
    transfer_type=2,
    trips=False,
    links_from_stop_times_kwargs={'max_shortcut': False, 'stop_id': 'stop_id'},
    euclidean_kwargs={'latitude': 'stop_lat', 'longitude': 'stop_lon'},
    seek_traffic_redundant_paths=True,
    seek_transfer_redundant_paths=True,
    max_distance=800,
    euclidean_speed=5 * 1000 / 3600 / 1.4,
    geometry=False,
    gtfs_only=False
):
    """
    Builds a relevant footpath table from the stop_times and stops tables of a transitfeed.
    The trips table may be used to spot the "dominated" footpaths that offer no new connection

    (compared to the pool of stops), for example:
    * line A stops at station i and station k;
    * line B stops at station j and station k;
    * no other line stops at a or b;
    * the footpath F goes from i to j;
    * In our understanding : F is dominated by the station k

    :param stops: DataFrame consistent with the GTFS table "trips"
    :param stop_times: DataFrame consistent with the GTFS table "trips"
    :param transfer_type: how to fill the 'transfer_type' column of the feed
    :param trips: DataFrame consistent with the GTFS table "trips"
    :param links_from_stop_times_kwargs: kwargs to pass to transitlinks.links_from_stop_times, called on stop_times
    :param euclidean_kwargs: kwargs to pass to skims.euclidean (the name of the latitude and longitude column)
    :param seek_traffic_redundant_paths: if True, only the footpaths that do not belong to the transit links are kept.
        the transit links are built from the stop times using transitlinks.links_from_stop_times. The maximum number of
        transit links to concatenate in order to look for redundancies may be passed in the kwargs ('max_shortcut').
        For example, if max_shortcut = 5: the footpath that can be avoided be taking a five stations ride will be tagged
        as "dominated".
    :param seek_transfer_redundant_paths: if True, the "trips" table is used to look for the dominated footpaths
    :param max_distance: maximum distance of the footpaths (meters as the crows fly)
    :param euclidean_speed: speed as the crows fly on the footpaths.
    :param geometry: If True, a geometry column (shapely.geometry.linestring.Linestring object) is added to the table
    :return: footpaths data with optional "dominated" tag
    """
    stop_id = links_from_stop_times_kwargs['stop_id']
    origin = stop_id + '_origin'
    destination = stop_id + '_destination'

    euclidean = skims.euclidean(stops.set_index(stop_id), **euclidean_kwargs)
    euclidean.reset_index(drop=True, inplace=True)
    euclidean['tuple'] = pd.Series(list(zip(list(euclidean['origin']), list(euclidean['destination']))))

    short_enough = euclidean[euclidean['euclidean_distance'] < max_distance]
    short_enough = short_enough[short_enough['origin'] != short_enough['destination']]

    footpath_tuples = {tuple(path) for path in short_enough[['origin', 'destination']].values.tolist()}
    paths = euclidean[euclidean['tuple'].isin(footpath_tuples)]

    paths['dominated'] = False

    _stop_times = stop_times

    if stop_id in stops.columns and stop_id not in stop_times.columns:
        _stop_times = pd.merge(
            stop_times, stops[['id', stop_id]], left_on='stop_id', right_on='id', suffixes=['', '_merged'])

    if seek_traffic_redundant_paths:

        links = feed_links.link_from_stop_times(_stop_times, **links_from_stop_times_kwargs).reset_index()
        in_links_tuples = {tuple(path) for path in links[[origin, destination]].values.tolist()}
        paths['trafic_dominated'] = paths['tuple'].isin(in_links_tuples)
        paths['dominated'] = paths['dominated'] | paths['trafic_dominated']

    stop_routes = {}
    stop_set = set(_stop_times[stop_id])

    # if two routes are connected by several footpaths we only keep the shortest one
    # if routes a and b are connected to route c, d and e by several footpaths :
    # we keep only the shortest one that does the job.
    if trips is not False:
        grouped = pd.merge(_stop_times, trips, left_on='trip_id', right_on='id').groupby(stop_id)['route_id']
        stop_routes = grouped.aggregate(lambda x: frozenset(x)).to_dict()

        def get_routes(row):
            return tuple((stop_routes[row['origin']], stop_routes[row['destination']]))

        paths = paths[(paths['origin'].isin(stop_set) & paths['destination'].isin(stop_set))]
        paths['trips'] = paths.apply(get_routes, axis=1)
        paths = paths.sort('euclidean_distance').groupby(['trips', 'dominated'], as_index=False).first()

    paths['min_transfer_time'] = paths['euclidean_distance'] / euclidean_speed
    paths = paths[paths['origin'] != paths['destination']]

    if seek_transfer_redundant_paths:
        paths['frozen'] = paths['trips'].apply(lambda a: frozenset(a[0]).union(frozenset(a[1])))
        max_length = max([len(f) for f in list(paths['frozen'])])

        to_beat = []
        for length in range(max_length + 1):
            for stop in stop_routes.values():
                for c in list(itertools.combinations(stop, length)):
                    to_beat.append(frozenset(c))

        to_beat = set(to_beat)

        paths['transfer_dominated'] = paths['frozen'].apply(lambda f: f in to_beat)
        paths['dominated'] = paths['dominated'] | paths['transfer_dominated']

    if geometry and not gtfs_only:
        paths['geometry'] = paths.apply(linestring_geometry, axis=1)

    paths['from_stop_id'] = paths['origin']
    paths['to_stop_id'] = paths['destination']
    paths['transfer_type'] = transfer_type

    if gtfs_only:
        paths = paths[~paths['dominated']]
        paths = paths[['from_stop_id', 'to_stop_id', 'transfer_type', 'min_transfer_time']]
    return paths
