import geopandas as gpd
import json
import osmnx as ox
import pandas as pd
from quetzal.engine import gps_tracks
from shapely.geometry import Point
# from syspy.spatial.graph import graphbuilder as gb


def _merge_reversed_geometries_dict(geojson_dict):
    features = {
        hash(tuple(tuple(p) for p in feature['geometry']['coordinates'])):
        feature for feature in geojson_dict['features']
    }
    counts = {}  # {geohash : 2 if the reverse geohash is in the geometries, 1 otherwise}
    drop = set()  # for each direct/indirect geometry pair, countain the lesser geohash
    for _, feature in features.items():
        geo_tuple = tuple(tuple(p) for p in feature['geometry']['coordinates'])
        reversed_geo_tuple = reversed(geo_tuple)
        k = hash(tuple(geo_tuple))
        kr = hash(tuple(reversed_geo_tuple))
        drop.add(min(k, kr))
        counts[k] = counts.get(k, 0) + 1
        counts[kr] = counts.get(kr, 0) + 1
        if k == kr:
            print('hey')

    # keep only the feature with the higher geohash
    drop = {d for d in drop if counts[d] > 1}
    features = {k: v for k, v in features.items() if k not in drop}
    for k, v in features.items():
        v['properties']['oneway'] = int(counts[k] == 1)

    result = dict(geojson_dict)
    result['features'] = list(features.values())
    return result


def _merge_reversed_geometries(geometries):
    try:
        return _merge_reversed_geometries_dict(geometries)
    except KeyError:  # 'features'
        geojson_dict = json.loads(geometries.to_json())
        temp = _merge_reversed_geometries_dict(geojson_dict)
        return gpd.read_file(json.dumps(temp))


hierarchical_drive_types = [
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'road', 'unclassified', 'residential', 'other'
]
not_driving_types = [
    'footway', 'pedestrian', 'path', 'track', 'steps',
    'service', 'living_street', 'cycleway'
]
hierarchical_types = hierarchical_drive_types + not_driving_types


def _get_type(x):
    for t in hierarchical_types:
        try:
            if isinstance(x['highway'], list):
                for h in x['highway']:
                    if t in h:
                        return t
            else:
                if t in x['highway']:
                    return t
        except KeyError:
            return 'other'
    return 'other'


def clean_osm_links(road_links, road_nodes):
    links = road_links.copy()
    decimal_threshold = 1
    ng = road_nodes['geometry'].to_dict()

    def directed(row):
        a = tuple([round(x, decimal_threshold) for x in list(ng[row['a']].coords)[0]])
        b = tuple([round(x, decimal_threshold) for x in list(row['geometry'].coords)[0]])
        return a == b

    def return_directed(row):
        a = tuple([round(x, decimal_threshold) for x in list(ng[row['a']].coords)[0]])
        b = tuple([round(x, decimal_threshold) for x in list(row['geometry'].coords)[-1]])
        return a == b

    import shapely

    def reversed_polyline(polyline):
        coords = list(polyline.coords)
        return shapely.geometry.LineString(reversed(coords))

    reverse = links.apply(
        return_directed,
        axis=1
    )
    loc = (reverse, 'geometry')
    links.loc[loc] = links.loc[loc].apply(
        reversed_polyline
    )
    return links


def get_osm_data(north, south, east, west, network_type, epsg, output_folder):
    """
    Download OSM road data and project into new coordinate system.
    params:
        - north: latitude of northern end
        - south: latitude of southern end
        - east: longitude of eastern end
        - west: longituted of western end
        - epsg: epsg code defining projection coordinate system
        - output_folder: folder in which to save data

    returns: None
    """

    # download
    osm_graph = ox.graph_from_bbox(north, south, east, west, network_type)
    road_nodes, road_links = ox.graph_to_gdfs(osm_graph)

    # process
    road_links.reset_index(drop=False, inplace=True)
    road_links.rename(columns={'u': 'a', 'v': 'b'}, inplace=True)
    try:
        road_nodes['osmid'] = road_nodes['osmid'].astype(str)
    except KeyError:
        road_nodes['osmid'] = [str(i) for i in road_nodes.index]

    road_nodes.index = 'osm_node_' + road_nodes['osmid'].astype(str)
    road_nodes = road_nodes[['geometry']]

    road_links[['a', 'b']] = 'osm_node_' + road_links[['a', 'b']].astype(str)

    # get type
    road_links['type'] = road_links.apply(_get_type, 1)

    # clean
    road_links = _merge_reversed_geometries(road_links)

    # filter
    road_links = road_links[['a', 'b', 'oneway', 'type', 'length', 'geometry']]
    # add reverse links
    road_links_reversed = road_links.copy()
    road_links_reversed['a'] = road_links['b']
    road_links_reversed['b'] = road_links['a']
    road_links = pd.concat([road_links, road_links_reversed])
    road_links = road_links.drop_duplicates()
    road_links['oneway'] = 1

    # change projection
    road_links.crs = {'init': 'epsg:4326'}
    road_links = road_links.to_crs({'init': 'epsg:{}'.format(epsg)})
    road_nodes.crs = {'init': 'epsg:4326'}
    road_nodes = road_nodes.to_crs({'init': 'epsg:{}'.format(epsg)})

    # fix reversed geometries
    road_links = clean_osm_links(road_links, road_nodes)

    # save
    road_links.to_file(output_folder + 'road_links.shp')
    road_nodes.to_file(output_folder + 'road_nodes.shp')


def _get_shape_coordinates(osm_road_links, road_links):
    shape_coordinates = []

    def append_coordinates(row):
        for x in row.coords[:]:
            shape_coordinates.append(x)

    road_links.loc[osm_road_links].to_crs(epsg=4326).geometry.apply(
        lambda x: append_coordinates(x), 1)

    return shape_coordinates


def _create_shape_df(osm_road_links, road_links, road_nodes, shape_id):

    shape_coordinates = _get_shape_coordinates(osm_road_links, road_links, road_nodes)

    shapes = pd.DataFrame(columns=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'])
    shapes = pd.DataFrame(shape_coordinates, columns=['shape_pt_lat', 'shape_pt_lon'])
    shapes['shape_id'] = shape_id
    shapes['shape_pt_sequence'] = shapes.index + 1

    return shapes


def _gps_pts_to_gdf(pts, epsg):

    points = gpd.GeoDataFrame(columns=['x', 'y', 'z'], data=pts)
    points['geometry'] = points.apply(lambda x: Point([x['x'], x['y']]), 1)
    points.crs = {'init': 'epsg:4326'}
    points = points.to_crs(epsg=epsg)

    return points


def _get_osm_links(points, road_links, road_nodes, buffer, penalty_factor):
    # TODO: improve with networkcaster

    path = gps_tracks.get_path(points, road_links, road_nodes, buffer=buffer, penalty_factor=penalty_factor)
    p = path[1:-1]

    osm_road_links = road_links.reset_index().set_index(['a', 'b']).loc[
        list(zip(p[:-1], p[1:]))]['index'].values

    return osm_road_links


def get_shape_and_osm_links(points, road_links, road_nodes, epsg, buffer=20, penalty_factor=2):
    """
    Match a list of gps points on a road network (road_links, road_nodes)
    Returns the output shape coordinates and list of road_links.
    params:
        - points: list of gps coordinates (lon, lat, ele)
        - road_links: geodataframe with network links
        - road_nodes: geodataframe with network nodes
        - epsg: coordinate system to use

    returns:
        - shape_coordinates: list of road coordinates (lat, lon)
        - osm_road_links: list of road_link ids
    """
    points = _gps_pts_to_gdf(points, epsg)
    osm_road_links = _get_osm_links(points, road_links, road_nodes, buffer=buffer, penalty_factor=penalty_factor)
    shape_coordinates = _get_shape_coordinates(osm_road_links, road_links)

    # shapes = _create_shape_df(osm_road_links, road_links, shape_id)

    return shape_coordinates, osm_road_links
