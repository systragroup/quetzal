import bz2
import os
import xml
import geopandas as gpd
import pandas as pd
from shapely import geometry
import string


class OSMContentHandler(xml.sax.handler.ContentHandler):
    """
    SAX content handler for OSM XML.

    Used to build an Overpass-like response JSON object in self.object. For format
    notes, see http://wiki.openstreetmap.org/wiki/OSM_XML#OSM_XML_file_format_notes
    and http://overpass-api.de/output_formats.html#json
    """

    def __init__(self):
        self._element = None
        self.object = {'elements': []}

    def startElement(self, name, attrs):
        if name == 'osm':
            self.object.update({k: attrs[k] for k in attrs.keys()
                                if k in ('version', 'generator')})
        elif name in ('node', 'way'):
            self._element = dict(type=name, tags={}, nodes=[], **attrs)
            self._element.update({k: float(attrs[k]) for k in attrs.keys()
                                  if k in ('lat', 'lon')})
            self._element.update({k: int(attrs[k]) for k in attrs.keys()
                                  if k in ('id', 'uid', 'version', 'changeset')})
        elif name == 'tag':
            self._element['tags'].update({attrs['k']: attrs['v']})
        elif name == 'member':
            if attrs['type'] == 'node':
                self._element['nodes'].append(int(attrs['ref']))
            elif attrs['type'] == 'way':
                self._element['ways'].append(int(attrs['ref']))
        elif name == 'nd':
            self._element['nodes'].append(int(attrs['ref']))

        elif name == 'relation':
            self._element = dict(type=name, tags={}, nodes=[], ways=[], **attrs)
            self._element.update({k: attrs[k] for k in attrs.keys()})

    def endElement(self, name):
        if name in ('node', 'way', 'relation'):
            self.object['elements'].append(self._element)


def overpass_json_from_file(filename):
    """
    Read OSM XML from input filename and return Overpass-like JSON.

    Parameters
    ----------
    filename : string
        name of file containing OSM XML data

    Returns
    -------
    OSMContentHandler object
    """
    _, ext = os.path.splitext(filename)

    if ext == '.bz2':
        # Use Python 2/3 compatible BZ2File()
        def opener(filename):
            bz2.BZ2File(filename)
    else:
        # Assume an unrecognized file extension is just XML
        def opener(filename):
            open(filename, mode='rb')

    with opener(filename) as file:
        handler = OSMContentHandler()
        xml.sax.parse(file, handler)
        return handler.object


def routes_links_nodes(data):
    data = dict(data)
    # ROUTES
    route_data = [
        dict(e) for e in data['elements']
        if e['type'] == 'relation'
        and e['tags']['type'] == 'route'
        and e['tags']['route'] == 'bus'
    ]

    for route in route_data:
        route.update(route['tags'])
        route.pop('tags')

    routes = pd.DataFrame(route_data)

    # NODES
    stops = set()
    for route in route_data:
        stops = stops.union(route['nodes'])

    node_data = [
        dict(e) for e in data['elements']
        if e['type'] == 'node'
        and e['id'] in stops
    ]

    for node in node_data:
        node.update(node['tags'])
        node['geometry'] = geometry.Point([node['lon'], node['lat']])
        node.pop('tags')
        node.pop('lat')
        node.pop('lon')
    nodes = gpd.GeoDataFrame(node_data)
    node_geometries = nodes.set_index('id')['geometry'].to_dict()

    # LINKS
    to_concat = []
    for trip_id, node_list in routes[['id', 'nodes']].values:
        ab = pd.DataFrame({'a': node_list[:-1], 'b': node_list[1:]})
        ab = pd.DataFrame({'a': node_list[:-1], 'b': node_list[1:]})
        if len(ab):
            ab['geometry'] = ab.apply(
                lambda r: geometry.LineString(
                    [
                        node_geometries[r['a']],
                        node_geometries[r['b']],
                    ]
                ),
                axis=1
            )
            ab['trip_id'] = route['id']
            to_concat.append(ab)

    links = pd.concat(to_concat)
    links = pd.merge(links, routes, left_on='trip_id', right_on='id', suffixes=['_link', '_route'])
    links = gpd.GeoDataFrame(links)
    links[['a', 'b']] = links[['a', 'b']].astype(int)
    return routes, links, nodes


highway_order = [
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential',
    'living_street', 'service', 'pedestrian', 'track', 'bus_guideway', 'escape', 'raceway', 'road',
    'footway', 'bridleway', 'steps', 'corridor', 'path',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
]


def min_lanes(value):
    try:
        return min(int(s) for s in value)
    except TypeError:
        return value


def highest_highway_order(value, highway_order=highway_order):
    if value in highway_order:
        return value

    for match in highway_order:
        if match in value:
            return match

    return 'unknown'


def first_item(value):
    try:
        hash(value)
        return value
    except TypeError:
        return value[0]


def printable(value):
    try:
        return ''.join([s for s in value if s in string.printable])

    except TypeError:  # nan
        return value


def clean_road_links(road_links, copy=False):

    if copy:
        road_links = road_links.copy()

    road_links['highway'] = road_links['highway'].apply(highest_highway_order)
    road_links['lanes'] = road_links['lanes'].fillna(1).apply(min_lanes).astype(int)

    for c in set(road_links.columns) - {'geometry'}:
        road_links[c] = road_links[c].apply(first_item)

    road_links['name'] = road_links['name'].apply(printable)

    return road_links
