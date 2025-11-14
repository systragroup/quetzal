import geopandas as gpd
import pandas as pd
from shapely import geometry
from shapely.ops import linemerge
from copy import deepcopy
import json

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

def get_links_and_nodes(geojson_file, text=None):
    if text is None:
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


def merged_reversed_geometries_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge reversed duplicate LineString geometries (A→B and B→A become one).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing LineString geometries.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame where reversed duplicates are merged.
        Adds a 'oneway' column: 1 if unique, 0 if reversed pair exists.
    """
    # Hash each geometry and its reversed version
    def geom_hash(geom):
        return hash(tuple(tuple(p) for p in geom.coords))

    def geom_hash_rev(geom):
        return hash(tuple(reversed(tuple(tuple(p) for p in geom.coords))))

    gdf = gdf.copy()
    gdf["_hash"] = gdf.geometry.apply(geom_hash)
    gdf["_hash_rev"] = gdf.geometry.apply(geom_hash_rev)

    # Count occurrences of each hash (direct and reversed)
    counts = pd.Series(gdf["_hash"].tolist() + gdf["_hash_rev"].tolist()).value_counts()

    # Determine which geometries to drop (keep only one of each reversed pair)
    drop_hashes = {min(h, hr) for h, hr in zip(gdf["_hash"], gdf["_hash_rev"]) if counts[h] > 1}

    mask_keep = ~gdf["_hash"].isin(drop_hashes)
    gdf = gdf[mask_keep].copy()
    gdf["oneway"] = gdf["_hash"].map(lambda h: int(counts[h] == 1))

    return gdf.drop(columns=["_hash", "_hash_rev"])


def get_intersections_gdf(gdf: gpd.GeoDataFrame) -> set:
    """
    Find all intersection coordinates between LineStrings.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing LineStrings.

    Returns
    -------
    set of tuple
        Coordinates (x, y) that appear in more than one LineString.
    """
    # Flatten all coordinates into a single Series
    coords = pd.Series([p for geom in gdf.geometry for p in geom.coords])
    return set(coords.value_counts()[coords.value_counts() > 1].index)


def get_nodes_gdf(gdf: gpd.GeoDataFrame) -> set:
    """
    Collect all unique nodes (start, end, and intersection points).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    Returns
    -------
    set of tuple
        Node coordinates (x, y).
    """
    start_nodes = gdf.geometry.apply(lambda g: tuple(g.coords[0]))
    end_nodes = gdf.geometry.apply(lambda g: tuple(g.coords[-1]))
    intersection_nodes = get_intersections_gdf(gdf)
    return set(start_nodes).union(end_nodes).union(intersection_nodes)


def split_feature_rec(row, nodes):
    """
    Recursively split a LineString at node coordinates.

    Parameters
    ----------
    row : pandas.Series
        A row from the GeoDataFrame (with a geometry).
    nodes : set
        Set of coordinates to split at.

    Returns
    -------
    list of pandas.Series
        The resulting split features as rows.
    """
    coords = list(row.geometry.coords)
    if len(coords) <= 2:
        return [row]

    for i, p in enumerate(coords[1:-1], start=1):
        if tuple(p) in nodes:
            left = deepcopy(row)
            right = deepcopy(row)
            left.geometry = geometry.LineString(coords[:i + 1])
            right.geometry = geometry.LineString(coords[i:])
            return [left] + split_feature_rec(right, nodes)
    return [row]


def split_features_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Split LineStrings at intersection and node points.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame with split LineStrings.
    """
    nodes = get_intersections_gdf(gdf)
    features = []
    for _, row in gdf.iterrows():
        features.extend(split_feature_rec(row, nodes))
    return gpd.GeoDataFrame(features, geometry="geometry", crs=gdf.crs)


def split_directions_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Duplicate two-way (non-oneway) LineStrings in reversed direction.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    Returns
    -------
    gpd.GeoDataFrame
        Original + reversed LineStrings for non-oneway edges.
    """
    gdf = gdf.copy()
    reversed_rows = gdf[gdf["oneway"] == 0].copy()
    reversed_rows["geometry"] = reversed_rows.geometry.apply(
        lambda geom: geometry.LineString(list(reversed(geom.coords)))
    )
    return pd.concat([gdf, reversed_rows], ignore_index=True)


def get_links_and_nodes_gdf(gdf: gpd.GeoDataFrame):
    """
    Build a link-node topology from a road network GeoDataFrame.

    This function:
      1. Merges reversed geometries
      2. Splits lines at intersections
      3. Creates node GeoDataFrame
      4. Duplicates bidirectional edges
      5. Adds 'a'/'b' columns with node IDs

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input road network (LineStrings)

    Returns
    -------
    links : gpd.GeoDataFrame
        Road links with columns ['geometry', 'oneway', 'a', 'b']
    nodes : gpd.GeoDataFrame
        Node points with index IDs
    """
    # 1. Merge reversed geometries
    gdf = merged_reversed_geometries_gdf(gdf)

    # 2. Split at intersections
    gdf = split_features_gdf(gdf)

    # 3. Build nodes GeoDataFrame
    node_coords = list(get_nodes_gdf(gdf))
    node_index = dict(zip(node_coords, [f"road_node_{i}" for i in range(len(node_coords))]))

    df_nodes = pd.DataFrame(node_index.items(), columns=["coordinates", "index"])
    df_nodes["geometry"] = df_nodes["coordinates"].apply(lambda t: geometry.Point(t))
    nodes = gpd.GeoDataFrame(df_nodes.set_index("index")[["geometry"]], geometry="geometry", crs=gdf.crs)

    # 4. Add reversed directions
    gdf = split_directions_gdf(gdf)

    # 5. Add node references
    gdf["a"] = gdf.geometry.apply(lambda geom: node_index[tuple(geom.coords[0])])
    gdf["b"] = gdf.geometry.apply(lambda geom: node_index[tuple(geom.coords[-1])])

    # 6. Clean index and output
    gdf.index = [f"road_link_{i}" for i in range(len(gdf))]

    return gdf, nodes
