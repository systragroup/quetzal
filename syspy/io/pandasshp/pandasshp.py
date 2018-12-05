# -*- coding: utf-8 -*-

""" pandashp: read/write shapefiles to/from special DataFrames
Offers two functions read_shp and write_shp that convert ESRI shapefiles to
pandas DataFrames that can be manipulated at will and then written back to
shapefiles. Opens up data manipulation capabilities beyond a simple GIS field
calculator.
Usage:
    import pandashp as pdshp
    # calculate population density from shapefile of cities (stupid, I know)
    cities = pdshp.read_shp('cities_germany_projected')
    cities['popdens'] = cities['population'] / cities['area']
    pdshp.write_shp(cities, 'cities_germany_projected_popdens')
"""
__all__ = ["read_shp", "write_shp", "match_vertices_and_edges"]

import requests
import numpy as np
import pandas as pd
import warnings
from shapely.geometry import LineString, Point, Polygon
import shapely.geometry.polygon
import shapely.geometry.linestring

from syspy.io.pandasshp import shapefile
from syspy.io.pandasdbf import pandasdbf
import pyproj
import shutil
from sklearn.cluster import KMeans
from tqdm import tqdm


def read_prj(filename):
    with open('.'.join(filename.split('.')[:-1]) + '.prj', 'r') as prj_file:
        return prj_file.read()

import collections
# if isinstance(e, collections.Iterable):


def convert_geometry(geometry, epsg1=None, epsg2=None, p1=None, p2=None):
    p1 = pyproj.Proj("+init=EPSG:" + str(epsg1)) if epsg1 else p1
    p2 = pyproj.Proj("+init=EPSG:" + str(epsg2)) if epsg1 else p2

    if isinstance(geometry, shapely.geometry.polygon.Polygon):
        return type(geometry)([pyproj.transform(p1, p2, longlat[0], longlat[1]) for longlat in geometry.boundary.coords])
    try:
        return type(geometry)([pyproj.transform(p1, p2, longlat[0], longlat[1]) for longlat in geometry.coords])

    except AttributeError:
        if isinstance(geometry, collections.Iterable):
            collection = geometry
            print(type(collection))
            return type(collection)([convert_geometry(g, p1=p1, p2=p2) for g in collection])


def read_shp(filename, encoding=None, type=False, strings_to_float=True):
    """Read shapefile to dataframe w/ geometry.
    if the reading fails, provid a type among ('polygon', 'polyline', 'point')

    Args:
        filename: ESRI shapefile name to be read  (without .shp extension)

    Returns:
        pandas DataFrame with column geometry, containing individual shapely
        Geometry objects (i.e. Point, LineString, Polygon) depending on
        the shapefiles original shape type

    """
    sr = shapefile.Reader(filename)

    cols = sr.fields[:]  # [:] = duplicate field list
    if cols[0][0] == 'DeletionFlag':
        cols.pop(0)
    cols = [col[0] for col in cols]  # extract field name only
    cols.append('geometry')

    records = [row for row in sr.iterRecords()]

    if sr.shapeType == shapefile.POLYGON or type == 'polygon':
        geometries = [Polygon(shape.points)
                      if len(shape.points) > 2 else np.NaN  # invalid geometry
                      for shape in sr.iterShapes()]
    elif sr.shapeType == shapefile.POLYLINE or type == 'polyline':
        geometries = [LineString(shape.points) for shape in sr.iterShapes()]
    elif sr.shapeType == shapefile.POINT or type == 'point':
        geometries = [Point(*shape.points[0]) for shape in sr.iterShapes()]
    else:
        raise NotImplementedError

    data = [r + [g] for r, g in zip(records, geometries)]

    df = pd.DataFrame(data, columns=cols)

    if strings_to_float:
        for col in df.columns:
            try:
                # the column is a string of an int, we keep it that way
                asint = df[col].astype(int)  #  do_nothing

            except (ValueError, TypeError, OverflowError):
                # invalid literal for int() with base 10:
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    pass

    if np.NaN in geometries:
        # drop invalid geometries
        df = df.dropna(subset=['geometry'])
        num_skipped = len(geometries) - len(df)
        warnings.warn('Skipped {} invalid geometrie(s).'.format(num_skipped))
    if encoding:
        return pandasdbf.convert_bytes_to_string(df, debug=False, encoding=encoding)
    return df


def write_shp(
    filename,
    dataframe,
    write_index=True,
    re_write=False,
    projection_string=None,
    projection_file=None,
    style_file=None,
    shx_file=None,
    shp_file=None,
    epsg=None,
    copy=True,
    floatlength=10,
    encoding=False,
    progress=False,
):
    """Write dataframe w/ geometry to shapefile.

    Args:
        filename: ESRI shapefile name to be written (without .shp extension)
        dataframe: a pandas DataFrame with column geometry and homogenous
                   shape types (Point, LineString, or Polygon)
        write_index: add index as column to attribute tabel (default: true)
        re_write: use pandasdbf to re_write a dbf that works with cube

    Returns:
        Nothing. test
    """

    df = dataframe.copy() if copy else dataframe

    if write_index:
        try:
            df.reset_index(inplace=True)
        except ValueError:  # cannot insert level_0, already exists:
            print('index not written')

    try:
        # split geometry column from dataframe
        first_instance = df['geometry'].iloc[0]
        geometry = list(df.pop('geometry'))
        if progress:
            geometry = tqdm(geometry, 'write geometry')

        # write geometries to shp/shx, according to geometry type
        if isinstance(first_instance, Point):
            sw = shapefile.Writer(shapefile.POINT)
            for point in geometry:
                sw.point(point.x, point.y)

        elif isinstance(first_instance, LineString):
            sw = shapefile.Writer(shapefile.POLYLINE)
            for line in geometry:
                sw.line([list(line.coords)])

        elif isinstance(first_instance, Polygon):
            sw = shapefile.Writer(shapefile.POLYGON)
            for polygon in geometry:
                try:
                    sw.poly([list(polygon.exterior.coords)])

                except (NotImplementedError, AttributeError):
                    # 'GeometryCollection' object has no attribute 'exterior'
                    # if it is not a polygon but a multipolygon,
                    # we use the convex hull
                    polygon = polygon.convex_hull
                    sw.poly([list(polygon.exterior.coords)])

        else:
            raise NotImplementedError

    except KeyError:
        # make a random shp / shx file
        sw = shapefile.Writer(shapefile.POINT)
        for point in range(len(df)):
            sw.point(0, 0)

    # add fields for dbf
    for k, column in enumerate(df.columns):
        # unicode strings freak out pyshp, so remove u'..'
        column = str(column)

        if np.issubdtype(df.dtypes[k], np.number):
            # detect and convert integer-only columns
            if (df[column] % 1 == 0).all():
                df[column] = df[column].astype(np.integer)

            # now create the appropriate fieldtype
            if np.issubdtype(df.dtypes[k], np.floating):
                sw.field(column, 'F', decimal=floatlength)
            else:
                sw.field(column, 'F', decimal=0)
        else:
            sw.field(column)

    # add records to dbf
    iterlist = list(df.itertuples())

    iterator = iterlist
    for record in iterator:
        sw.record(*record[1:])  # drop first tuple element (=index)

    sw.save(filename)

    if re_write or encoding:
        to_write = df.drop(['geometry'], axis=1, errors='ignore')
        pandasdbf.write_dbf(
            to_write,
            filename.split('.shp')[0] + '.dbf',
            encoding=encoding
        )

    if epsg:
        projection_string = requests.get(
            'http://spatialreference.org/ref/epsg/%i/prettywkt/' % epsg
        ).text

    without_extension = '.'.join(filename.split('.')[:-1])
    if projection_string:
        with open(without_extension + '.prj', 'w') as prj_file:
            prj_file.write(projection_string)
    if style_file:
        shutil.copyfile(style_file, without_extension + '.qml')

    if projection_file:
        shutil.copyfile(projection_file, without_extension+ '.prj')

    if shp_file:
        shutil.copyfile(shp_file, without_extension + '.shp')

    if shx_file:
        shutil.copyfile(shx_file, without_extension + '.shx')

def write_secondary_files(
    filename,
    epsg=None,
    style_file=None,
    projection_file=None
    ):
    if epsg:
        projection_string = requests.get(
            'http://spatialreference.org/ref/epsg/%i/prettywkt/' % epsg
        ).text

    without_extension = '.'.join(filename.split('.')[:-1])
    if projection_string:
        with open(without_extension + '.prj', 'w') as prj_file:
            prj_file.write(projection_string)
    if style_file:
        shutil.copyfile(style_file, without_extension + '.qml')

    if projection_file:
        shutil.copyfile(projection_file, without_extension+ '.prj')


def match_vertices_and_edges(vertices, edges, vertex_cols=('Vertex1', 'Vertex2')):
    """Adds unique IDs to vertices and corresponding edges.

    Identifies, which nodes coincide with the endpoints of edges and creates
    matching IDs for matching points, thus creating a node-edge graph whose
    edges are encoded purely by node ID pairs. The optional argument
    vertex_cols specifies which DataFrame columns of edges are added, default
    is 'Vertex1' and 'Vertex2'.

    Args:
        vertices: pandas DataFrame with geometry column of type Point
        edges: pandas DataFrame with geometry column of type LineString
        vertex_cols: tuple of 2 strings for the IDs numbers

    Returns:
        Nothing, the mathing IDs are added to the columns vertex_cols in
        argument edges
    """

    vertex_indices = []
    for e, line in enumerate(edges.geometry):
        edge_endpoints = []
        for k, vertex in enumerate(vertices.geometry):
            if line.touches(vertex) or line.intersects(vertex):
                edge_endpoints.append(vertices.index[k])

        if len(edge_endpoints) == 0:
            warnings.warn("edge " + str(e) +
                          " has no endpoints: " + str(edge_endpoints))
        elif len(edge_endpoints) == 1:
            warnings.warn("edge " + str(e) +
                          " has only 1 endpoint: " + str(edge_endpoints))

        vertex_indices.append(edge_endpoints)

    edges[vertex_cols[0]] = pd.Series([min(n1n2) for n1n2 in vertex_indices],
                                      index=edges.index)
    edges[vertex_cols[1]] = pd.Series([max(n1n2) for n1n2 in vertex_indices],
                                      index=edges.index)


def bounds(df):
    """Return a DataFrame of minx, miny, maxx, maxy of each geometry."""
    bounds = np.array([geom.bounds for geom in df.geometry])
    return pd.DataFrame(bounds,
                        columns=['minx', 'miny', 'maxx', 'maxy'],
                        index=df.index)


def total_bounds(df):
    """Return bounding box (minx, miny, maxx, maxy) of all geometries. """
    b = bounds(df)
    return (b['minx'].min(),
            b['miny'].min(),
            b['maxx'].max(),
            b['maxy'].max())


def add_centroid(zones_shp):
    """Returns a DataFrame with centroid attributes from a zonig pandas.DataFrame """

    inner_zones = zones_shp.copy()

    inner_zones['centroid_geometry'] = inner_zones[
        'geometry'].apply(lambda g: g.centroid)
    inner_zones['centroid_coordinates'] = inner_zones[
        'geometry'].apply(lambda g: g.centroid.coords[0])
    inner_zones['latitude'] = inner_zones[
        'geometry'].apply(lambda g: g.centroid.y)
    inner_zones['longitude'] = inner_zones[
        'geometry'].apply(lambda g: g.centroid.x)

    return inner_zones


def od_matrix(zones, centroids=False):

    _zones = zones.copy() if centroids else add_centroid(zones.copy())
    #: the {centroid: [latitude, longitude]} dictionary
    pos = _zones[['latitude', 'longitude']]
    iterate = [pos.index] * 2
    od = pd.DataFrame(index=pd.MultiIndex.from_product(iterate, names=[
                      'origin', 'destination'])).reset_index()  #: the od column matrix
    od = pd.merge(od, pos, left_on='origin', right_index=True)
    od = pd.merge(od, pos, left_on='destination', right_index=True,
                  suffixes=['_origin', '_destination'])

    od['geometry'] = od[['origin', 'destination']].apply(
        lambda r: shapely.geometry.LineString(
            [_zones.loc[r['origin'], 'centroid_geometry'],
             _zones.loc[r['destination'], 'centroid_geometry']]
        ), axis=1)
    return od


def union_geometry(geo_series):
    g = geo_series.iloc[0]
    for i in range(1, len(geo_series)):
        g = g.union(geo_series.iloc[i])
    return g


def buffer_until_polygon(g, b=1e-6):
    if type(g) == shapely.geometry.polygon.Polygon:
        return g
    else:
        return buffer_until_polygon(g.buffer(b), b * 5)


def zone_clusters(zones,  n_clusters=10, buffer=None, cluster_column=None):

    df = add_centroid(zones)

    if buffer:
        df['geometry'] = df['geometry'].apply(lambda g: g.buffer(buffer))
    x = df[['longitude', 'latitude']].values

    if cluster_column:
        cluster_series = df['cluster'] = df[cluster_column]
    else:
        y_pred = KMeans(n_clusters=n_clusters).fit_predict(x)
        cluster_series = df['cluster'] = pd.Series(y_pred, index=df.index)

    cluster_series.name = 'cluster'

    geo = df.groupby('cluster')['geometry'].agg(union_geometry)
    clusters = pd.DataFrame(geo.apply(buffer_until_polygon))

    return clusters, cluster_series


def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()

    def geometry(row):
        return shapely.geometry.linestring.LineString((point_dict[row[from_point]], point_dict[row[to_point]]))
    return df.apply(geometry, axis=1)
