import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point, MultiPoint
from shapely import ops
from syspy.spatial import spatial
from syspy.syspy_utils.syscolors import linedraft_shades, rainbow_shades
import json

def from_linedraft(
    links,
    nodes,
    zones,
    recolor,
    cut_buffer,
    set_emission
):
    links = links.copy()
    nodes = nodes.copy()
    # tous les identifiants de noeuds sont des str !
    nodes.index = [str(i) for i in nodes.index]
    # la fonction d'export de linedraft utilise le champ color,
    # quetzal fonctionne avec le champ line_color
    links['line_color'] = links['color']

    if recolor:
        colordict = dict(
            zip(linedraft_shades, rainbow_shades[: len(linedraft_shades)]))
        links['line_color'] = links['line_color'].apply(lambda c: colordict[c])

    if cut_buffer:
        zones = spatial.zones_in_influence_area(
            zones,
            area=None,
            links=links,
            cut_buffer=cut_buffer
        )

    try:
        # cross fillna are made in order to avoid zero values
        zones['emission'] = zones['pop'].fillna(zones['emp'] / 100)
        zones['attraction'] = zones['emp'].fillna(zones['pop'] / 100)

    except KeyError:  # pop and emp are not in columns
        zones['emission'] = 1
        zones['attraction'] = 1

    scale = zones['emission'].sum() / zones['attraction'].sum()
    zones['attraction'] = zones['attraction'] * scale

    if set_emission:
        grow = set_emission / zones['emission'].sum()
        zones[['emission', 'attraction']] *= grow

    zones['emission_rate'] = zones['emission'] \
        / (zones['emission'] + zones['attraction'])
    zones['weight'] = zones['emission'] + zones['attraction']

    return links, nodes, zones


def links_and_nodes(linestring, node_index=0):
    nodes = []
    for c in linestring.coords:
        g = Point(c)
        nodes.append((node_index, g))
        node_index += 1

    links = []
    sequence = 0
    node_index_a, node_a = nodes[0]
    for node_index_b, node_b in nodes[1:]:
        g = LineString([node_a, node_b])
        links.append((node_index_a, node_index_b, sequence, 0, g))
        node_index_a = node_index_b
        node_a = node_b
        sequence += 1

    nodes = list(reversed(nodes))
    sequence = 0
    node_index_a, node_a = nodes[0]
    for node_index_b, node_b in nodes[1:]:
        g = LineString([node_a, node_b])
        links.append((node_index_a, node_index_b, sequence, 1, g))
        node_index_a = node_index_b
        node_a = node_b
        sequence += 1

    nodes = pd.DataFrame(nodes, columns=['n', 'geometry']).set_index('n')
    links = pd.DataFrame(
        links,
        columns=['a', 'b', 'link_sequence', 'direction_id', 'geometry']
    )
    return links, nodes


def from_lines(lines, node_index=0, add_return=True, to_keep=[]):
    """Import public transport lines to Quetzal format from geodataframe
    containing the pt lines (as one per row).
    Creates the dataframe links and nodes defined in the stepmodel class.

    Parameters
    ----------
    lines : geodataframe
        Name of DataFrame describing the alignements as LineSring in a *geometry* column.
    node_index : int, optional, default 0
        number on which to start indexing nodes
    add_return : bool, optional, default True
        if True, return lines are created.
        Use False if the two directions of the line are in the geodataframe.
    to_keep : list, optional, default []
        columns of lines geodataframe to keep in links 

    Returns
    -------
    links
        Links of the public transport system and pt routes caracteristics.
        Each line of the geodataframe correspond to a section of a PT route between two nodes

    nodes
        Public transport stations.
    
    """

    lines = lines.copy()
    lines['temp_index'] = lines.index
    to_concat_links = []
    to_concat_nodes = []

    for line, geometry in lines[['temp_index', 'geometry']].values:
        links, nodes = links_and_nodes(geometry, node_index=node_index)
        node_index += len(nodes)
        links['line'] = line
        links['trip_id'] = str(line)
        links['route_id'] = str(line)
        for c in to_keep:
            links[c] = lines.loc[line, c]        
        if not add_return:
            links = links[links['direction_id'] == 0]
        else:
            links.loc[links['direction_id'] == 0, 'trip_id'] += '_bis'

        to_concat_nodes.append(nodes)
        to_concat_links.append(links)

    links = pd.concat(to_concat_links)
    nodes = pd.concat(to_concat_nodes)
    return links.reset_index(drop=True), nodes

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return [None, LineString(line)]
    if distance >= line.length:
        return [LineString(line), None]
    coords = list(line.coords)
    pd = 0
    for i, p in enumerate(coords):
        if i == 0:
            continue
        pd += euclidean_distance(p, coords[i - 1])
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def cut_inbetween(geom, d_a, d_b):
    geom1 = cut(geom, d_a)[1]
    return cut(geom1, d_b - d_a)[0]


def from_line_and_nodes(lines, nodes):
    all_links = pd.DataFrame()
    all_nodes = nodes.copy()
    for trip_id in lines['trip_id'].unique():
        line = lines.loc[lines['trip_id']==trip_id]
        geom = line['geometry'].values[0]

        # filter stops
        if trip_id in nodes.columns:
            nodes_loc = nodes[trip_id]==1
        else:
            buffer = line.buffer(100).geometry.values[0]
            nodes_loc = nodes.within(buffer)
        stops = nodes.loc[nodes_loc][['id', 'geometry']]

        # sort stops
        stops['seq'] = stops.apply(lambda x: line.project(x['geometry']), 1)
        stops = stops.sort_values('seq').reset_index(drop=True)
        stops['link_sequence'] = stops.index + 1

        # build links
        links_a = stops[['id', 'seq', 'link_sequence']].iloc[:-1].rename(columns={'id': 'a'})
        links_b = stops[['id', 'seq', 'link_sequence']].iloc[1:].rename(columns={'id': 'b'})
        links_b['link_sequence'] -= 1
        links = links_a.merge(links_b, on='link_sequence')
        links['geometry'] = links.apply(
            lambda x: cut_inbetween(geom, x['seq_x'], x['seq_y']), 1
        )
        links['length'] = links['geometry'].apply(lambda x: x.length)
        
        links['trip_id'] = trip_id
        links.drop(['seq_x', 'seq_y'], 1, inplace=True)

        all_links = pd.concat([all_links, links])
        all_nodes.drop(trip_id, 1, errors='ignore', inplace=True)

    all_links = all_links.reset_index(drop=True)

    return gpd.GeoDataFrame(all_links).set_crs(lines.crs), gpd.GeoDataFrame(all_nodes).set_crs(lines.crs)

def from_lines_and_stations(lines, stations, buffer=1e-3, og_geoms=True, **kwargs):
    """Convert a set of alignement and station into a table of links.

    Parameters
    ----------
    lines : pd.DataFrame (or gpd.GeoDataFrame)
        DataFrame describing the alignements as LineSring in a *geometry* column.
    stations : pd.DataFrame (or gpd.GeoDataFrame)
        DataFrame describing the stations as Point in a *geometry* column.
    buffer : Float, optional
        Buffer for station detection near each alignement, by default 1e-3
    og_geoms : bool, optional
        If True (by default), the original geometry will be split between stations.
        If False, returned geometry will be a simplified geometry (st1 -> st2)

    Returns
    -------
    pd.DataFrame
        Table of links. As per, :func:`from_lines` output.

    """
    stations = stations.copy()
    lines = lines.copy()
    links_concat = []
    for index, line in lines.iterrows():
        linestring = line['geometry']
        buffered = linestring.buffer(buffer)

        # Filter stations using the buffer and project those stations
        stations['keep'] = stations['geometry'].apply(lambda g: buffered.contains(g))
        near = stations[stations['keep']].copy()
        near['proj'] = [linestring.project(pt, normalized=True) 
                            for pt in near['geometry'].to_list()]
        near = near.sort_values(by='proj')
        stations.drop(columns=['keep'], inplace=True)

        # Create simplified geometry (st1 -> st2 -> ...)
        nodes = [linestring.interpolate(d, normalized=True) for d in near['proj']]
        lines.loc[index, 'geometry'] = LineString(nodes)

        # Get links table from simplified geometry
        links, nodes = from_lines(lines.loc[[index]], **kwargs)
        index_dict = near.reset_index()['index'].to_dict()
        links['a'] = links['a'].apply(lambda x: index_dict.get(x))
        links['b'] = links['b'].apply(lambda x: index_dict.get(x))

        # Split original geometry with stations to add orignal geometry at each links 
        if og_geoms:
            split_pts = MultiPoint(nodes['geometry'].to_list())
            i1 = 0 if near.iloc[0]['proj'] == 0 else 1
            i2 = None if near.iloc[-1]['proj'] == 1.0 else -1
            og_geoms = list(split_line_by_point(linestring, split_pts).geoms)[i1:i2]
            if kwargs.get('add_return', True):
                og_geoms += og_geoms[::-1]
        
            links['geometry'] = og_geoms

        links_concat.append(links)

    return pd.concat(links_concat), og_geoms

def split_line_by_point(line, point, tolerance: float=1.0e-9):
    return ops.split(ops.snap(line, point, tolerance), point)


def read_geojson(file_name,list_columns=['road_link_list']):
    '''
    read with geopandas and manualy add list properties from list_columns.
    '''
    def get_property_values(json_data,column='road_link_list',default = []):
        ls = []
        for features in json_data['features']:
            properties = features['properties']
            ls.append(properties.get(column, default))
        return ls
    gdf = gpd.read_file(file_name)

    with open(file_name) as f:
        json_data = json.load(f)
        
    for col in list_columns:
        ls = get_property_values(json_data, col)
        gdf[col] = ls
    return gdf