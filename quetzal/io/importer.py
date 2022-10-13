import geopandas as gpd
import pandas as pd
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


def from_lines(lines, node_index=0, add_return=True):
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
        if not add_return:
            links = links[links['direction_id'] == 0]
        else:
            links.loc[links['direction_id'] == 0, 'trip_id'] += '_bis'

        to_concat_nodes.append(nodes)
        to_concat_links.append(links)

    links = pd.concat(to_concat_links)
    nodes = pd.concat(to_concat_nodes)
    return links.reset_index(drop=True), nodes


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return [shapely.geometry.LineString(), shapely.geometry.LineString(line)]
    elif distance >= line.length:
        return [shapely.geometry.LineString(line), shapely.geometry.LineString()]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(shapely.geometry.Point(p))
        if pd == distance:
            return [
                shapely.geometry.LineString(coords[:i+1]),
                shapely.geometry.LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                shapely.geometry.LineString(coords[:i] + [(cp.x, cp.y)]),
                shapely.geometry.LineString([(cp.x, cp.y)] + coords[i:])]

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
