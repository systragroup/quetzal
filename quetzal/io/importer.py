import pandas as pd
import numpy as np
import shapely
from syspy.spatial import spatial
from syspy.syspy_utils.syscolors import linedraft_shades, rainbow_shades


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
        g = shapely.geometry.Point(c)
        nodes.append((node_index, g))
        node_index += 1

    links = []
    sequence = 0
    node_index_a, node_a = nodes[0]
    for node_index_b, node_b in nodes[1:]:
        g = shapely.geometry.LineString([node_a, node_b])
        links.append((node_index_a, node_index_b, sequence, 0, g))
        node_index_a = node_index_b
        node_a = node_b
        sequence += 1

    nodes = list(reversed(nodes))
    sequence = 0
    node_index_a, node_a = nodes[0]
    for node_index_b, node_b in nodes[1:]:
        g = shapely.geometry.LineString([node_a, node_b])
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
    """
    lines index is used, links and nodes are returned
    if add_return = True, return lines are created
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
        lines.loc[index, 'geometry'] = shapely.geometry.LineString(nodes)

        # Get links table from simplified geometry
        links, nodes = from_lines(lines.loc[[index]], **kwargs)
        index_dict = near.reset_index()['index'].to_dict()
        links['a'] = links['a'].apply(lambda x: index_dict.get(x))
        links['b'] = links['b'].apply(lambda x: index_dict.get(x))

        # Split original geometry with stations to add orignal geometry at each links 
        if og_geoms:
            split_pts = shapely.geometry.MultiPoint(nodes['geometry'].to_list())
            i1 = 0 if near.iloc[0]['proj'] == 0 else 1
            i2 = None if near.iloc[-1]['proj'] == 1.0 else -1
            og_geoms = list(split_line_by_point(linestring, split_pts).geoms)[i1:i2]
            if kwargs.get('add_return', True):
                og_geoms += og_geoms[::-1]
        
            links['geometry'] = og_geoms

        links_concat.append(links)

    return pd.concat(links_concat), og_geoms

def split_line_by_point(line, point, tolerance: float=1.0e-9):
    return shapely.ops.split(shapely.ops.snap(line, point, tolerance), point)