# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'

import shapely
import networkx as nx
import pandas as pd


from syspy.skims import skims


def linestring_geometry(row):
    to_return = shapely.geometry.linestring.LineString(
        [
            [row['x_origin'], row['y_origin']],
            [row['x_destination'], row['y_destination']]
        ]
    )
    return to_return


def point_geometry(row):
    return shapely.geometry.point.Point(row['stop_lon'], row['stop_lat'])


def stop_clusters(
    stops,
    longitude='stop_lon',
    latitude='stop_lat',
    reach=150,
    method='connected_components',
    geometry=False
):

    """
    Clusterizes a collection of stops to a smaller collection of centroids.
    For a given station, every trip may be linked to a different entry in the
    'stops' table of the transitfeed. This function may be used in order
    to build the parents "stations" of these stops.

    :param stops: transitfeed "stops" table where the id is used as index
    :param longitude: name of the longitude column
    :param latitude: name of the latitude column
    :param reach: maximum length of the connections of the stops of a cluster
    :param method: clustering method, connected components builds a station
        for every connected components of a graph where every stop is linked
        to its neighbors that are nearer than reach.
    :param geometry: if True : the geometry of the centroids of the clusters
        is added to the centroid table
    :return: {
        'transfers': transfers,
        'centroids': centroids,
        'clusters': cluster_list
    }
        "transfers" are the edges of the graph (the ones shorter than reach) ;
        "centroids" contains the centroids' data ;
        "clusters" joins the stops to the clusters ; 
    """

    stops[[longitude, latitude]] = stops[[longitude, latitude]].astype(float)
    euclidean = skims.euclidean(
        stops, latitude=latitude, longitude=longitude)
    transfers = euclidean[euclidean['euclidean_distance'] < reach]

    if method == 'connected_components':
        g = nx.Graph(transfers[['origin', 'destination']].values.tolist())
        components = list(nx.connected_components(g))
        cluster_list = []
        for i in range(len(components)):
            for c in components[i]:
                cluster_list.append({'stop_id': c, 'cluster_id': i})
        centroids = pd.merge(
            stops, pd.DataFrame(cluster_list),
            left_index=True, right_on='stop_id'
        ).groupby(
            'cluster_id')[[longitude, latitude]].mean()

    if geometry:
        transfers['geometry'] = transfers.apply(linestring_geometry, axis=1)
        centroids['geometry'] = centroids.apply(point_geometry, axis=1)

    return_dict = {
        'transfers': transfers,
        'centroids': centroids,
        'clusters': cluster_list
    }

    return return_dict


def stops_with_parent_station(stops, stop_cluster_kwargs={}, sort_column=None):

    clusters = pd.DataFrame(
        stop_clusters(
            stops.set_index('stop_id'),
            **stop_cluster_kwargs
        )['clusters']
    )

    if sort_column:
        sort = stops[['stop_id', sort_column]].copy()
        clusters = pd.merge(
            clusters, sort, on='stop_id').sort_values(sort_column)

    parents = clusters.groupby('cluster_id')[['stop_id']].first()

    # return parents
    to_merge = pd.merge(
        clusters,
        parents,
        left_on='cluster_id',
        right_index=True,
        suffixes=['', '_parent']
    )
    to_merge = to_merge.rename(
        columns={
            'stop_id_parent': 'parent_station',
        }
    )[['stop_id', 'parent_station']]
    _stops = pd.merge(stops, to_merge, on='stop_id', suffixes=['', '_merged'])
    try:
        _stops['parent_station'] = _stops['parent_station'].fillna(
            _stops['parent_station_merged']
        )
        # nan messing the types again... should be remove if using str
        try:
            _stops['parent_station'] = _stops['parent_station'].astype(int)
        except Exception as e:
            print('Failed to convert parent_station_id to int: \n', e)
            pass

    except KeyError:
        # KeyError: 'parent_station_merged': their was no parent station in
        # the first place
        pass
    _stops['location_type'] = (
        _stops['stop_id'] == _stops['parent_station']
    ).astype(int)

    return _stops.drop('parent_station_merged', axis=1, errors='ignore')
