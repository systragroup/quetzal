import geopandas as gpd
import json
import osmnx as ox
import pandas as pd
from . import gps_tracks


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


def get_osm_data(north, south, east, west, network_type, epsg, output_folder):

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


def _create_shape_df(osm_road_links, road_links, shape_id):

    shape_coordinates = _get_shape_coordinates(osm_road_links, road_links)

    shapes = pd.DataFrame(columns=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'])
    shapes = pd.DataFrame(shape_coordinates, columns=['shape_pt_lat', 'shape_pt_lon'])
    shapes['shape_id'] = shape_id
    shapes['shape_pt_sequence'] = shapes.index + 1

    return shapes


def _get_osm_links(points, road_links, road_nodes, buffer=50, penalty_factor=2):
    # TODO: improve with networkcaster

    path = gps_tracks.get_path(points, road_links, road_nodes, buffer=50, penalty_factor=2)
    p = path[1:-1]

    osm_road_links = road_links.reset_index().set_index(['a', 'b']).loc[
        list(zip(p[:-1], p[1:]))]['index'].values

    return osm_road_links


def get_shape_and_osm_links(points, road_links, road_nodes, buffer=50, penalty_factor=2):

    osm_road_links = _get_osm_links(points, road_links, road_nodes, buffer=50, penalty_factor=2)
    shape_coordinates = _get_shape_coordinates(osm_road_links, road_links)

    # shapes = _create_shape_df(osm_road_links, road_links, shape_id)

    return shape_coordinates, osm_road_links
