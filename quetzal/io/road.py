import json
from copy import deepcopy
import geopandas as gpd
from shapely import geometry
import pandas as pd

def merged_reversed_geometries_dict(geojson_dict):
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


def merge_reversed_geometries(geometries):
    try:
        return merged_reversed_geometries_dict(geometries)
    except KeyError:  # 'features'
        geojson_dict = json.loads(geometries.to_json())
        temp = merged_reversed_geometries_dict(geojson_dict)
        return gpd.read_file(json.dumps(temp))


def get_intersections(geojson_dict):
    count = {}
    for feature in geojson_dict['features']:
        for p in feature['geometry']['coordinates']:
            count[tuple(p)] = count.get(tuple(p), 0) + 1
    return {k for k, v in count.items() if v > 1}


def get_nodes(geojson_dict):
    nodes = set()
    for feature in geojson_dict['features']:
        nodes.add(tuple(feature['geometry']['coordinates'][0]))
        nodes.add(tuple(feature['geometry']['coordinates'][-1]))
    return nodes.union(get_intersections(geojson_dict))


def split_feature(feature, nodes=(), start=0):
    length = len(feature['geometry']['coordinates'])
    if length == 2:
        return [feature]
    if length > 2:
        cut = 0
        for p in feature['geometry']['coordinates'][start + 1: length - 1]:
            
            cut += 1
            if tuple(p) in nodes:
                left = deepcopy(feature)
                right = deepcopy(feature)
                left['geometry']['coordinates'] = left['geometry']['coordinates'][start: cut + 1]
                right['geometry']['coordinates'] = right['geometry']['coordinates'][cut:]
                return [left] + split_feature(right, nodes=nodes, start=0)
    return [feature]


def get_split_features(geojson_dict):
    features = []
    intersections = get_intersections(geojson_dict)
    for feature in geojson_dict['features']:
        features += split_feature(feature, nodes=intersections) 
    return features


def split_features(geojson_dict):
    geojson_dict['features'] = get_split_features(geojson_dict)

def split_directions(geojson_dict):
    features = geojson_dict['features']
    reversed_features = [deepcopy(f) for f in features if not f['properties']['oneway']]
    for f in reversed_features:
        f['geometry']['coordinates'] = list(reversed(f['geometry']['coordinates']))
    geojson_dict['features'] = features + reversed_features

def get_links_and_nodes(geojson_file):
    with open(geojson_file, 'r') as file:
        text = file.read()
    
    road =  json.loads(text)
    split_features(road)
    
    node_coordinates = list(get_nodes(road))
    node_index = dict(
        zip(
            node_coordinates, 
            ['road_node_%i' % i for i in range(len(node_coordinates))]
        )
    )
    df = pd.DataFrame(node_index.items(), columns=['coordinates', 'index'])
    df['geometry'] = df['coordinates'].apply(lambda t: geometry.Point(t))
    nodes = gpd.GeoDataFrame(df.set_index(['index'])[['geometry']])
    


    split_directions(road)

    for f in road['features']:
        first = tuple(f['geometry']['coordinates'][0])
        last = tuple(f['geometry']['coordinates'][-1])
        f['properties']['a'] = node_index[first]
        f['properties']['b'] = node_index[last]

    links = gpd.read_file(json.dumps(road))
    links.index = ['road_link_%i' % i for i in range(len(links))]
    return links, nodes