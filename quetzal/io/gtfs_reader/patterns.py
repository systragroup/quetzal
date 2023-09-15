import gtfs_kit as gk
import numpy as np
import pandas as pd
from syspy.spatial import spatial


def build_stop_clusters(
    stops, distance_threshold=150, col='cluster_id', use_parent_station=False
):
    """
    Apply agglomerative clustering algorithm to stops.
    Add a column cluster_id with the cluster id.
    If use_parent_station = True: clustering based on parent stations when known
    """
    # TODO: do not work for big feeds --> suggest to use KMeans in this case
    gdf = gk.stops.geometrize_stops_0(stops, use_utm=True)
    temp = gdf.copy()
    if use_parent_station:
        if 'parent_station' not in gdf.columns:
            gdf['parent_station'] = np.nan
        gdf['dissolve'] = gdf.apply(
            lambda x: x['parent_station'] if isinstance(x['parent_station'], str) else x['stop_id'],
            1
        )
        temp = gdf.dissolve('dissolve', as_index=False)
        temp.geometry = temp.geometry.centroid

    temp[col] = spatial.agglomerative_clustering(
        temp, distance_threshold=distance_threshold
    )

    if use_parent_station:
        temp = gdf.merge(temp[['dissolve', col]], on='dissolve', how='left')
        temp.drop('dissolve', 1, inplace=True)
    return gk.stops.ungeometrize_stops_0(temp)


def build_patterns(
    feed, group=['route_id'], on='stop_id'
):
    """
    """
    trip_footprints = get_trip_footprints(feed, on=on)
    patterns = get_patterns(feed.trips, trip_footprints, group=group)
    feed.trips = feed.trips.merge(
        patterns[['trip_id', 'pattern_id']], on='trip_id'
    )


def get_trip_stop_list(stop_times, stops=None, on='stop_id'):
    trip_stops = stop_times.copy()
    if on != 'stop_id':
        s_to_c = stops.set_index('stop_id')[on].reset_index()
        trip_stops = trip_stops.merge(s_to_c)
    trip_stops = trip_stops.sort_values(['trip_id', 'stop_sequence']).groupby('trip_id').agg(
        {on: lambda x: list(x)}
    ).rename(columns={on: 'stops'})

    return trip_stops


def get_trip_footprints(feed, on='stop_id'):
    """
    Build each trip's footprint.
    The footprint is a String that will be used to derive the trip
    patterns: it must allow to identify trips that will be grouped
    and to distinguish trips that will not. Here we use the ordered
    list of stops or clusters to build each trip footprint.
    """
    trip_stops = get_trip_stop_list(feed.stop_times, feed.stops, on=on)
    trip_footprints = trip_stops.rename(columns={'stops': 'footprint'})
    trip_footprints['footprint'] = trip_footprints['footprint'].map(str)
    return trip_footprints


def get_patterns(trips, trip_footprints, group=['route_id']):  # we can add direction, it can also be on route_short_name, …
    patterns = trip_footprints.copy()
    patterns = patterns.merge(
        trips.set_index('trip_id')[group],
        left_index=True,
        right_index=True
    )
    pattern_n = patterns.drop_duplicates().set_index(['footprint'] + group).groupby(
        group,
        as_index=False
    ).cumcount()
    pattern_n.name = 'pattern_num'
    patterns = patterns.reset_index().merge(
        pattern_n,
        on=['footprint'] + group
    )
    patterns['pattern_id'] = patterns[group + ['pattern_num']].apply(
        lambda x: '_'.join(x.map(str)), 1
    )
    return patterns
