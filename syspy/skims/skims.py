# -*- coding: utf-8 -*-


__author__ = 'qchasserieau'

from tqdm import tqdm
import pyproj
import shapely
import time
import json
import requests
from random import random

import pandas as pd
import numpy as np
from geopy.distance import geodesic

from syspy.spatial import spatial
from tqdm import tqdm


wgs84 = pyproj.Proj("+init=EPSG:4326")


def dist_from_row(row, projection=wgs84):
    """
    Uses vincenty formula to calculate the euclidean distances of an origin-destination pair.

    :param row: a pd.Series containing the coordinates of the origin and the destination
    :type row: pd.Series
    :param projection: projection of the zoning
    :type projection: pyproj.Proj
    :return: euclidean_distance: euclidean distance of the origin-destination
    :rtype: int
    """

    coordinates_origin = (pyproj.transform(projection, wgs84, row['x_origin'], row['y_origin']))
    coordinates_origin = (coordinates_origin[1], coordinates_origin[0])
    coordinates_destination = (pyproj.transform(projection, wgs84, row['x_destination'], row['y_destination']))
    coordinates_destination = (coordinates_destination[1], coordinates_destination[0])

    return vincenty(coordinates_origin, coordinates_destination).m


def euclidean(zones, coordinates_unit='degree', projection=wgs84, epsg=False, method='numpy', origins=False, destinations=False,
              latitude=False, longitude=False, intrazonal=False):
    """
    Calculates the euclidean distances between each origin-destination pair of a zoning.
    If the coordinates are in degree, the Vincenty formula is used.

    :param zones: a shape dataframe containing the geometries (polygons) of the zoning
    :type zones: pd.DataFrame
    :param coordinates_unit: degree or meter
    :type coordinates_unit: str
    :param origins: a list of id of the zones from which the euclidean distance is needed
    :type origins: list
    :param destinations: a list of id of the zones to which the euclidean distance is needed
    :type destination: list
    :param method: 'numpy' or 'vincenty' numpy is faster but only handles wgs84 epsg 4326
    :type method: str
    :param projection: projection of the zoning
    :type projection: pyproj.Proj
    :param epsg: epsg code of the projection, if given, the projection arg is overwritten
    :type projection: int or str
    :param intrazonal: (bool), if True a non-zero intrazonal distance is computed.
        In this case an intrazonal projection system must be provided
    :return: euclidean_distance_dataframe: a pd.DataFrame with the coordinates of the centroids
    and the euclidean distances between the zones
    :rtype: pd.DataFrame
    """

    projection = pyproj.Proj("+init=EPSG:" + str(epsg)) if epsg else projection
    if 'geometry' in zones.columns:
        z = zones[['geometry']].copy()
        z['x'] = z['geometry'].apply(lambda g: g.centroid.coords[0][0])
        z['y'] = z['geometry'].apply(lambda g: g.centroid.coords[0][1])
        z.drop(['geometry'], axis=1, inplace=True)
    elif bool(latitude) & bool(longitude):
        z = zones[[latitude, longitude]].copy()
        z['x'] = z[longitude]
        z['y'] = z[latitude]
    else:
        print('If the DataFrame has no "geometry" field, longitude and latitude should be provided')

    # zones_destination = zones_destination if zones_destination
    iterables = [zones.index]*2
    od = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=['origin', 'destination'])).reset_index()
    od = pd.merge(od, z, left_on='origin', right_index=True)
    od = pd.merge(od, z, left_on='destination', right_index=True, suffixes=['_origin', '_destination'])

    if origins:
        od = od[od['origin'].isin(origins)]
    if destinations:
        od = od[od['destination'].isin(destinations)]

    # Compute distance
    if coordinates_unit == 'degree':
        if method == 'numpy':
            columns = ['x_origin', 'y_origin', 'x_destination', 'y_destination']
            od['euclidean_distance'] = get_distance_from_lon_lat_in_m(*[od[s] for s in columns])
        else:
            od['euclidean_distance'] = od.apply(dist_from_row, axis=1, args={projection})
    elif coordinates_unit == 'meter':
        od['euclidean_distance'] = np.sqrt(
            (od['x_origin'] - od['x_destination'])**2 +\
            (od['y_origin'] - od['y_destination'])**2
        )
    else:
        raise('Invalid coordinates_unit.')

    if intrazonal:
        for i in od.index:
            if od['origin'][i] == od['destination'][i]:
                od['euclidean_distance'][i] = np.sqrt(zones['area'][od['origin'][i]]) / 2

    return od[['origin', 'destination', 'euclidean_distance', 'x_origin', 'y_origin', 'x_destination', 'y_destination']]


# google maps ################################################
def all_skim_matrix(zones=None, token=None, od_matrix=None, coordinates_unit='degree', **skim_matrix_kwargs):
    if od_matrix is not None:
        df = od_matrix.copy()
    else:
        df = euclidean(zones, coordinates_unit=coordinates_unit)
    try:
        assert token is not None
        
        if isinstance(token, str):
            token = [token, token] # il semble que l'on vide trop tôt la pile
        t = token.pop()
        print('Building driving skims matrix with Google API.')
        skim_lists = []
        rows = tqdm(list(df.iterrows()), 'skim matrix')
        for index, row in rows:
            computed = False
            while computed is False and token:
                try:
                    skim_lists.append(
                        driving_skims_from_row(
                            row, 
                            t, 
                            **skim_matrix_kwargs
                        )
                    )
                    computed = True
                except TokenError as e:
                    print(e)
                    try:
                        t = token.pop()
                        print('Popped:', t)
                    except :
                        print('Could not complete the skim matrix computation: not enough credentials.')
        df[['distance', 'duration', 'duration_in_traffic']] = pd.DataFrame(skim_lists)
        print('Done')
    except IndexError as e:
        print('Exception [%s] occured' % e)
        print('WARNING: the build of the real skim matrix has failed.')
        df[['distance', 'duration']] =df.apply(
            pseudo_driving_skims_from_row, args=[token], axis=1)
        print('A random one has been generated instead to allow testing of the next steps.')

    return df


def skim_matrix(zones, token, n_clusters, coordinates_unit='degree', skim_matrix_kwargs={}):
    clusters, cluster_series = spatial.zone_clusters(zones, n_clusters, 1e-9)
    cluster_euclidean = all_skim_matrix(
        clusters,
        token,
        coordinates_unit=coordinates_unit
        **skim_matrix_kwargs
    )

    df = euclidean(zones, coordinates_unit=coordinates_unit)

    df = pd.merge(
        df,
        pd.DataFrame(cluster_series),
        left_on='origin',
        right_index=True)

    df = pd.merge(
        df,
        pd.DataFrame(cluster_series),
        left_on='destination',
        right_index=True,
        suffixes=['_origin', '_destination'])

    df = pd.merge(
        df,
        cluster_euclidean.rename(
            columns={
                'origin': 'cluster_origin',
                'destination': 'cluster_destination',
                'distance': 'cluster_distance',
                'duration': 'cluster_duration'
            }
        ),
        on=['cluster_origin', 'cluster_destination'],
        suffixes=['', '_cluster']
    )

    df['distance_rate'] = (
        df['euclidean_distance'] / df['euclidean_distance_cluster']
        ).fillna(0)
    df['distance'] = df['cluster_distance'] * df['distance_rate']
    df['duration'] = df['cluster_duration'] * df['distance_rate']

    euclidean_to_path_length = 1 / (df['euclidean_distance_cluster'] / df['cluster_distance'] ).mean()
    euclidean_speed = (df['euclidean_distance_cluster'] / df['duration']).mean()

    df.loc[df['euclidean_distance_cluster'] == 0, 'duration'] = df['euclidean_distance'] / euclidean_speed
    df.loc[df['euclidean_distance_cluster'] == 0, 'distance'] = df['euclidean_distance'] * euclidean_to_path_length

    return df.fillna(0)


def in_url(coordinates):
    """
    :param coordinates: list of coordinates [longitude, latitude]
    :type coordinates: list
    :return: in_url_coordninates
    :rtype: str
    """
    return str(coordinates[1]) + ',' + str(coordinates[0])


class TokenError(Exception):
    def __init__(self, message='out of credentials'):

        # Call the base class constructor with the parameters it needs
        super(TokenError, self).__init__(message)


def driving_skims_from_coordinate_couple(
        origin_coordinates, 
        destination_coordinates, 
        token,
        timestamp=time.time(), 
        errors='ignore', 
        proxy=None
    ):
    """
    Requests google maps distancematrix API with a couple of coordinates. Returns the road skims of the trip.

    :param origin_coordinates: origin coordinates in wgs84 EPSG 4326
    :type origin_coordinates: list
    :param destination_coordinates: destination coordinates in wgs84 EPSG 4326
    :type destination_coordinates: list
    :param token: Google distancematrix API token (provided by Google when politely asked)
    :type token: str
    :param timestamp: timestamp of the very period to investigate
    :type timestamp: timestamp
    :return: skim_series: a pd.Series with the duration and the distance of the trip
    :rtype: pd.Series
    """

    api_url = "https://maps.googleapis.com/maps/api/distancematrix/json?"
    proto_url = api_url + "origins={0}&destinations={1}"
    proto_url += "&mode=driving&language=en-EN&sensor=false&departure_time={2}&trafic_model=pessimistic&key={3}"
    url = proto_url.format(in_url(origin_coordinates), in_url(destination_coordinates), timestamp, token)
    print(url)
    try:
        # Call to the proxy here
        if proxy is not None:
            data = {
                'latitude_origin': origin_coordinates[1],
                'longitude_origin': origin_coordinates[0],
                'latitude_destination': destination_coordinates[1],
                'longitude_destination': destination_coordinates[0],
                'timestamp': int(timestamp),
                'token': token
            }
            resp = proxy.get(**data)  # get the json string

            if proxy.get_status != 0:  # Not found in the db
                resp_json = json.loads(resp)
                if resp_json["status"] == 'OK':  # Itinerary computation done
                    proxy.populate(resp=resp, **data)
                    proxy.insert()
                    element = resp_json['rows'][0]['elements'][0]
                else : 
                    raise TokenError
            else:
                element = json.loads(resp)['rows'][0]['elements'][0]

        else:
            element = json.loads(requests.get(url).text)['rows'][0]['elements'][0]

        try: 
            duration_in_traffic = element['duration_in_traffic']['value']
        except KeyError:
            duration_in_traffic = np.nan
        return pd.Series(
            {
                'duration': element['duration']['value'],
                'distance': element['distance']['value'],
                'duration_in_traffic': duration_in_traffic,
            }
        )
    except (KeyError) as e: # Exception
        # duration_in_traffic may not be provided
        assert(errors == 'ignore'), 'Token probably out of credentials.'
        return pd.Series(
        {
            'duration': np.nan, 
            'distance': np.nan, 
            'duration_in_traffic': np.nan
        }
    )


def driving_skims_from_row(
    row,
    token,
    projection=wgs84,
    timestamp=time.time(),
    **skim_matrix_kwargs
):
    
    time.sleep(0.1)
    origin_coordinates = pyproj.transform(
        projection,
        wgs84,
        row['x_origin'],
        row['y_origin']
    )
    destination_coordinates = pyproj.transform(
        projection,
        wgs84,
        row['x_destination'],
        row['y_destination']
    )

    driving_skims = driving_skims_from_coordinate_couple(
        origin_coordinates,
        destination_coordinates,
        token,
        timestamp,
        **skim_matrix_kwargs
    )

    return driving_skims


def pseudo_driving_skims_from_row(
    row,
    token,
    projection=wgs84,
    timestamp=time.time()
):
    random_distance = 1000 * random()
    random_distance_factor = 1.3 + random() / 5
    random_duration_factor = 0.3 + random() / 20

    distance = random_distance + get_distance_from_row_in_m(row) * random_distance_factor
    duration = distance * random_duration_factor

    return pd.Series({'distance': distance, 'duration': duration})


def get_distance_from_row_in_m(row):
    return get_distance_from_lon_lat_in_m(*list(row[
        ['x_origin', 'y_origin', 'x_destination', 'y_destination']].values))


def get_distance_from_lon_lat_in_m(lon1, lat1, lon2, lat2):
    r = 6371  # Radius of the earth in km
    d_lat = deg_to_rad(lat2 - lat1)  # deg2rad user defined
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + \
        np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * \
        np.sin(d_lon / 2) * np.sin(d_lon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r * c  # Distance in km
    return d * 1000


def deg_to_rad(deg):
    return deg * (np.pi / 180)


def add_coordinates(dataframe):
    df = dataframe.copy()
    df['coords'] = df['geometry'].apply(lambda g: g.coords)
    df['x_origin'] = df['coords'].apply(lambda c: c[0][0])
    df['y_origin'] = df['coords'].apply(lambda c: c[0][1])
    df['x_destination'] = df['coords'].apply(lambda c: c[-1][0])
    df['y_destination'] = df['coords'].apply(lambda c: c[-1][1])

    return df


def drop_coordinates(dataframe):
    return dataframe.drop(
        ['coords', 'x_origin', 'x_destination', 'y_origin', 'y_destination'],
        axis=1,
        errors='ignore'
    )


def distance_from_geometry(geometry_series, projection=wgs84, method='numpy'):
    df = pd.DataFrame(geometry_series)
    df.columns = ['geometry']
    df = add_coordinates(df)
    if method == 'numpy' and projection==wgs84:
        cols = ['x_origin', 'y_origin', 'x_destination', 'y_destination']
        df['distance'] = get_distance_from_lon_lat_in_m(*[df[s] for s in cols])
    else:
        df['distance'] = df.apply(dist_from_row, axis=1)
    return df['distance']


def a_b_from_geometry(geometry):

    boundary = geometry.envelope.boundary
    a = shapely.geometry.linestring.LineString(list(boundary.coords)[0:2])
    b = shapely.geometry.linestring.LineString(list(boundary.coords)[1:3])

    return pd.Series([a, b])


def area_factor(geometry_series):
    df = pd.DataFrame(geometry_series)
    df.columns = ['geometry']
    df[['a', 'b']] = df['geometry'].apply(lambda g: a_b_from_geometry(g))
    df['la'] = distance_from_geometry(df['a'])
    df['lb'] = distance_from_geometry(df['b'])
    df['boundary_area'] = df['geometry'].apply(lambda g: g.envelope.area)
    df['actual_boundary_area'] = df['la'] * df['lb']
    df['rate'] = df['actual_boundary_area'] / df['boundary_area']

    return df['rate'].mean()





