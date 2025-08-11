"""
This module provides tools for spatial analysis.
"""

__author__ = 'qchasserieau'

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import geometry

import shapely.geometry.linestring
import shapely.geometry.polygon
import shapely.geometry.point
from shapely.ops import unary_union as u2
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.ops import polygonize
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import Tuple
from numba import jit, njit
import numba as nb

from pyproj import Transformer


def bounds(df):
    """Return a DataFrame of minx, miny, maxx, maxy of each geometry."""
    bounds = np.array([geom.bounds for geom in df.geometry])
    return pd.DataFrame(bounds, columns=['minx', 'miny', 'maxx', 'maxy'], index=df.index)


def total_bounds(df):
    """Return bounding box (minx, miny, maxx, maxy) of all geometries."""
    b = bounds(df)
    return (b['minx'].min(), b['miny'].min(), b['maxx'].max(), b['maxy'].max())


def add_centroid(zones_shp):
    """Returns a DataFrame with centroid attributes from a zonig pandas.DataFrame"""
    zones = zones_shp.copy()

    zones['centroid_geometry'] = zones['geometry'].apply(lambda g: g.centroid)
    zones['centroid_coordinates'] = zones['geometry'].apply(lambda g: g.centroid.coords[0])
    zones['latitude'] = zones['geometry'].apply(lambda g: g.centroid.y)
    zones['longitude'] = zones['geometry'].apply(lambda g: g.centroid.x)
    return zones


def od_matrix(zones, centroids=False):
    _zones = zones.copy() if centroids else add_centroid(zones.copy())
    pos = _zones[['latitude', 'longitude']]  #: the {centroid: [latitude, longitude]} dictionary
    iterate = [pos.index] * 2

    #: the od column matrix
    od = pd.DataFrame(index=pd.MultiIndex.from_product(iterate, names=['origin', 'destination'])).reset_index()
    od = pd.merge(od, pos, left_on='origin', right_index=True)
    od = pd.merge(od, pos, left_on='destination', right_index=True, suffixes=['_origin', '_destination'])

    od['geometry'] = od[['origin', 'destination']].apply(
        lambda r: shapely.geometry.LineString(
            [_zones.loc[r['origin'], 'centroid_geometry'], _zones.loc[r['destination'], 'centroid_geometry']]
        ),
        axis=1,
    )
    return od


def union_geometry(geo_series):
    # todo: implement speedup with
    # shapely.geometry.multilinestring.MultiLineString etc...
    g = geo_series.iloc[0]
    for i in range(1, len(geo_series)):
        g = g.union(geo_series.iloc[i])
    return g


def simplify_then_buffer(geometry, buffer):
    return geometry.simplify(buffer / 5).buffer(buffer)


def buffer_until_polygon(g, b=1e-6, step=5):
    if type(g) == shapely.geometry.polygon.Polygon:
        return g
    else:
        return buffer_until_polygon(simplify_then_buffer(g, b), b * step, step)


def zone_clusters(
    zones,
    n_clusters=10,
    buffer=None,
    cluster_column=None,
    geo_union_method=union_geometry,
    geo_join_method=lambda g: g.convex_hull,
):
    n_clusters = min(n_clusters, len(zones))

    df = gpd.GeoDataFrame(add_centroid(zones))

    if buffer:
        df['geometry'] = df['geometry'].apply(lambda g: g.buffer(buffer))
    x = df[['longitude', 'latitude']].values

    if cluster_column:
        cluster_series = df['cluster'] = df[cluster_column]
    else:
        y_pred = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(x)
        cluster_series = df['cluster'] = pd.Series(y_pred, index=df.index)

    cluster_series.name = 'cluster'

    geo = df.dissolve('cluster')['geometry']  # .agg(union_geometry)

    clusters = pd.DataFrame(geo.apply(geo_join_method))
    return clusters, cluster_series


def agglomerative_clustering(stops, distance_threshold=150, **kwargs):
    """
    Stops must be in a metric cartesian coordinate system.
    """
    df = gpd.GeoDataFrame(stops).copy()
    df['x'] = df.geometry.x
    df['y'] = df.geometry.y
    c = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, **kwargs).fit(
        df[['x', 'y']].values
    )
    return c.labels_


def DBSCAN_sclustering(stops, distance_threshold=150, min_samples=1, metric='euclidean', **kwargs):
    df = gpd.GeoDataFrame(stops).copy()
    df['x'] = df.geometry.x
    df['y'] = df.geometry.y

    db = DBSCAN(eps=distance_threshold, min_samples=min_samples, metric=metric, **kwargs)
    labels = db.fit_predict(df[['x', 'y']].values)
    return labels


def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()

    def geometry(row):
        return shapely.geometry.linestring.LineString((point_dict[row[from_point]], point_dict[row[to_point]]))

    return df.apply(geometry, axis=1)


def _join_geometry(link_row, one, many):
    return shapely.geometry.LineString([one[link_row['ix_one']], many[link_row['ix_many']]])


def add_geometry_coordinates(df, columns=['x_geometry', 'y_geometry'], to_crs=None):
    """
    get centroid and split geometry into 2 columns ['x_geometry', 'y_geometry'].
    parameters
    to_crs: (int)
        convert geometry to this crs. (this is faster than gpd.to_crs() method.)
    """
    df = df.copy()
    centroids = df['geometry']
    # if the geometry is not a point... get centroids.
    # need this try for pandas and geopandas type.
    try:
        all_points = all(centroids.type == 'Point')
    except:
        all_points = all(centroids.apply(type) == shapely.geometry.Point)
    if not all_points:
        centroids = centroids.apply(lambda g: g.centroid)

    df[columns[0]], df[columns[1]] = zip(*centroids.apply(lambda g: g.coords[0]))

    if to_crs is not None:
        from_crs = df.crs
        if from_crs != to_crs:
            # pyproj takes [y,x] and return [x,y]. however. if from_crs == to_crs. it return [y,x].
            # so we do not apply this function if to_crs == from_crs.
            transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
            df[columns[0]], df[columns[1]] = transformer.transform(df[columns[0]].values, df[columns[1]].values)

    return df


def nearest(one, many, geometry=False, n_neighbors=1, to_crs=None):
    try:
        assert many.index.is_unique
        assert one.index.is_unique
    except AssertionError:
        msg = 'index of one and many should not contain duplicates'
        print(msg)
        warnings.warn(msg)
    n_neighbors = int(n_neighbors)
    df_many = add_geometry_coordinates(many.copy(), columns=['x_geometry', 'y_geometry'], to_crs=to_crs)
    df_one = add_geometry_coordinates(one.copy(), columns=['x_geometry', 'y_geometry'], to_crs=to_crs)

    x = df_many[['x_geometry', 'y_geometry']].values
    y = df_one[['x_geometry', 'y_geometry']].values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(y)

    index_one = pd.DataFrame(df_one.index.values, columns=['ix_one'])
    index_many = pd.DataFrame(df_many.index.values, columns=['ix_many'])

    to_concat = []
    for i in range(n_neighbors):
        links = pd.merge(
            index_one, pd.DataFrame(indices[:, i], columns=['index_nn']), left_index=True, right_index=True
        )

        links = pd.merge(links, index_many, left_on='index_nn', right_index=True)

        links = pd.merge(links, pd.DataFrame(distances[:, i], columns=['distance']), left_index=True, right_index=True)
        links['rank'] = i
        to_concat.append(links)

    links = pd.concat(to_concat)

    one_dict = one['geometry'].to_dict()
    many_dict = many['geometry'].to_dict()
    if geometry:
        links['geometry'] = links.apply(lambda r: _join_geometry(r, one_dict, many_dict), axis=1)

    return links


def nearest_radius(one, many, radius=1, to_crs=None):
    try:
        assert many.index.is_unique
        assert one.index.is_unique
    except AssertionError:
        msg = 'index of one and many should not contain duplicates'
        print(msg)

    df_many = add_geometry_coordinates(many.copy(), columns=['x_geometry', 'y_geometry'], to_crs=to_crs)
    df_one = add_geometry_coordinates(one.copy(), columns=['x_geometry', 'y_geometry'], to_crs=to_crs)

    x = df_many[['x_geometry', 'y_geometry']].values
    y = df_one[['x_geometry', 'y_geometry']].values

    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.radius_neighbors(y, sort_results=True)

    index_one = pd.DataFrame(df_one.index.values, columns=['ix_one'])
    index_many = pd.DataFrame(df_many.index.values, columns=['ix_many'])

    df = pd.merge(index_one, pd.Series(indices, name='index_nn'), left_index=True, right_index=True)
    df = pd.merge(df, pd.Series(distances, name='distance'), left_index=True, right_index=True)
    df = df.set_index('ix_one').apply(pd.Series.explode).reset_index()
    df['rank'] = df.groupby('ix_one').cumcount().astype(int)
    df['distance'] = df['distance'].astype(float)

    df = pd.merge(df, index_many, left_on='index_nn', right_index=True)

    return df[['ix_one', 'ix_many', 'distance', 'rank']]


def nearest_geometry(one, many, geometry=False, n_neighbors=1, n_neighbors_centroid=10):
    one = pd.DataFrame(one).copy()
    many = pd.DataFrame(many).copy()
    one_centroid = pd.DataFrame(one).copy()
    many_centroid = pd.DataFrame(many).copy()

    one_centroid['geometry'] = one_centroid['geometry'].apply(lambda g: g.centroid)
    many_centroid['geometry'] = many_centroid['geometry'].apply(lambda g: g.centroid)

    actual_nearest = nearest(one_centroid, many_centroid, n_neighbors=n_neighbors_centroid, geometry=geometry)

    one_geometry_dict = one['geometry'].to_dict()
    many_geometry_dict = many['geometry'].to_dict()

    def actual_distance(dict_a, dict_b, ix_a, ix_b):
        return dict_a[ix_a].distance(dict_b[ix_b])

    actual_nearest['actual_distance'] = [
        actual_distance(one_geometry_dict, many_geometry_dict, ix_one, ix_many)
        for ix_one, ix_many in tqdm(actual_nearest[['ix_one', 'ix_many']].values, 'nearest_link')
    ]

    actual_nearest.sort_values(['ix_one', 'actual_distance'], inplace=True)

    ranks = list(range(n_neighbors_centroid)) * len(one)
    actual_nearest['actual_rank'] = ranks
    return actual_nearest.loc[actual_nearest['actual_rank'] < n_neighbors]


def zones_in_influence_area(zones, area=None, links=None, cut_buffer=0.02):
    if not area:
        union_links = union_geometry(links['geometry'])
        area = union_links.buffer(cut_buffer)

    zone_dict = zones.to_dict(orient='index')
    keep = {key: value for key, value in zone_dict.items() if value['geometry'].intersects(area)}
    return pd.DataFrame(keep).T.reset_index(drop=True)


def voronoi_diagram_dataframes(points: pd.Series) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """take series of geometry (ex: zones['geometry'])."""
    assert isinstance(points, pd.Series), 'points should be a series.'
    assert points.index.duplicated().sum() == 0, 'Index must not be duplicated'

    items = list(dict(points).items())
    key_dict = {}
    key_list = []
    values = []

    for i in range(len(items)):
        key_dict[i] = items[i][0]
        key_list.append(items[i][0])
        values.append(items[i][1])

    # if not, we have less polygons than centroids,
    # centroids may be really close
    assert len(key_list) == len(values)

    polygons, ridges = voronoi_diagram(values)
    polygon_dataframe = gpd.GeoDataFrame(geometry=polygons)
    index_dict = nearest(polygon_dataframe, gpd.GeoDataFrame(points)).set_index('ix_one')['ix_many'].to_dict()
    polygon_dataframe.index = polygon_dataframe.index.map(index_dict.get)

    ridge_dataframe = pd.DataFrame(ridges, columns=['a', 'b', 'geometry'])

    ridge_dataframe['a'] = ridge_dataframe['a'].apply(lambda x: key_dict[x])
    ridge_dataframe['b'] = ridge_dataframe['b'].apply(lambda x: key_dict[x])
    return polygon_dataframe, ridge_dataframe


def voronoi_diagram(points, plot=False, size=None, method='box'):
    multi = shapely.geometry.multipoint.MultiPoint(points)
    if method == 'box':
        g = shapely.geometry.box(*multi.bounds)
    elif method == 'convex_hull':
        g = multi.convex_hull

    size = size if size else pow(g.area, 0.5)
    buffer = g.buffer(size)
    points_and_bound = points + [shapely.geometry.point.Point(c) for c in buffer.boundary.coords][:-1]

    vor = Voronoi([list(g.coords)[0] for g in points_and_bound])

    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

    polygons = [poly.intersection(g.buffer(size / 10)) for poly in polygonize(lines)]

    if plot:
        voronoi_plot_2d(vor)

    ridges = pd.DataFrame(vor.ridge_points, columns=['a', 'b'])
    ridges = ridges[(ridges['a'] < len(points)) & (ridges['b'] < len(points))]
    ridges['geometry'] = ridges.apply(lambda r: shapely.geometry.LineString([points[r['a']], points[r['b']]]), axis=1)
    return polygons, ridges[['a', 'b', 'geometry']].values.tolist()


@jit(nopython=True)
def fast_point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in nb.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def fast_points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    D = np.empty(len(points), dtype=nb.boolean)
    for i in nb.prange(0, len(points)):
        D[i] = fast_point_in_polygon(points[i, 0], points[i, 1], polygon)
    return np.where(D)[0]


def points_in_polygon(points: np.ndarray, polygon: gpd.GeoDataFrame) -> np.ndarray:
    """
    return a list of point in the polygon. values are the index in the points array.

    points:np.array[np.array[float,float]]
        list of all the points coords (x,y)
    polygon: gpd.GeoDataFrame
        geodataframe of multiples polygons.
    """
    try:
        poly = np.array([*polygon.exterior.coords])
        return fast_points_in_polygon(points, poly)
    except:
        res = np.array([])
        # polygon = polygon.geoms
        for i in range(len(polygon)):
            poly = np.array([*polygon[i].exterior.coords])
            val = fast_points_in_polygon(points, poly)
            res = np.append(res, val)
        return res


def _coords(g):
    ls = list(g.coords)
    a, b = ls[0], ls[-1]
    xa, ya = a
    xb, yb = b
    return [xa, xb, ya, yb]


def get_links_hull(links: gpd.GeoDataFrame, tolerance: float = 1e-2) -> geometry.Polygon:
    # return tight envelop around links (not convex hull)
    # NOTE: cluster before.
    # sm.preparation_clusterize_nodes(distance_threshold=n * tolerance)
    links = links.loc[links['a'] != links['b']]
    df = pd.DataFrame(
        data=list(links['geometry'].apply(_coords).values), index=links.index, columns=['xa', 'xb', 'ya', 'yb']
    )
    df['length'] = np.sqrt(np.power(df['yb'] - df['ya'], 2) + np.power(df['xb'] - df['xa'], 2))

    df['nx'] = (df['yb'] - df['ya']) / df['length'] * tolerance
    df['ny'] = (df['xb'] - df['xa']) / df['length'] * tolerance
    df['xc'] = df['xa'] + df['nx']
    df['yc'] = df['ya'] + df['ny']
    df['xd'] = df['xb'] + df['nx']
    df['yd'] = df['yb'] + df['ny']
    df['xe'] = df['xa'] - df['nx']
    df['ye'] = df['ya'] - df['ny']
    df['xf'] = df['xb'] - df['nx']
    df['yf'] = df['yb'] - df['ny']
    df[['xc', 'yc', 'xd', 'yd', 'xe', 'ye', 'xf', 'yf']]
    df['c'] = [tuple(v) for v in df[['xc', 'yc']].values]
    df['d'] = [tuple(v) for v in df[['xd', 'yd']].values]
    df['e'] = [tuple(v) for v in df[['xe', 'ye']].values]
    df['f'] = [tuple(v) for v in df[['xf', 'yf']].values]
    df['rectangle'] = [geometry.Polygon(points) for points in df[['c', 'd', 'f', 'e']].values]
    united = u2(list(df['rectangle']))

    return (
        united.simplify(tolerance)
        .buffer(tolerance)
        .buffer(2 * tolerance)
        .buffer(-2 * tolerance)
        .simplify(tolerance / 2)
    )


def plot_lineStrings(gdf, ax, **kwargs):
    # quickly plot a gpd.GeoDataframe. given an ax (fig, ax = plt.subplots())
    # kwargs for matplotlib plot classic kwargs.
    from matplotlib.collections import LineCollection

    geometries = gdf['geometry'].apply(lambda g: [pts for pts in g.coords]).values
    line_segments = LineCollection(geometries, **kwargs)
    ax.add_collection(line_segments)
    ax.autoscale()


def plot_points(gdf, ax, **kwargs):
    # quickly plot a gpd.GeoDataframe. given an ax (fig, ax = plt.subplots())
    # kwargs for matplotlib plot classic kwargs.
    xy = gdf['geometry'].map(lambda point: point.xy)
    x, y = zip(*xy)
    ax.scatter(x, y, **kwargs)
