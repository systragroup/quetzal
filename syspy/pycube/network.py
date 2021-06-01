__author__ = 'qchasserieau'

import pandas as pd
import shapely
from IPython.display import display
from IPython.html.widgets import FloatProgress
from syspy.spatial import spatial
from syspy.syspy_utils import syscolors


def zone_stops(zones, nodes, stop_list, leg_type='contains'):

    if leg_type == 'contains':
        progress = FloatProgress(
            min=0, max=len(list(zones.iterrows())), width=975, height=10, color=syscolors.rainbow_shades[1], margin=5)
        progress.value = 0
        display(progress)
        zone_stops = {}
        for zone_id, zone in zones.iterrows():
            zone_stops[zone_id] = []
            for stop_id, stop in nodes.loc[stop_list].iterrows():
                if zone['geometry'].contains(stop['geometry']):
                    zone_stops[zone_id].append(stop_id)
            progress.value += 1

    if leg_type == 'nearest':
        centroids = zones.copy()
        centroids['geometry'] = zones['geometry'].apply(lambda g: g.centroid)
        stops = nodes.loc[stop_list]

        links_a = spatial.nearest(stops, centroids).rename(columns={'ix_many': 'zone', 'ix_one': 'stop'})
        links_b = spatial.nearest(centroids, stops).rename(columns={'ix_one': 'zone', 'ix_many': 'stop'})
        links = pd.concat([links_a, links_b]).drop_duplicates()
        zone_stops = dict(links.groupby('zone')['stop'].agg(lambda s: list(s)))
    return zone_stops


def nontransitleg_geometries(nontransitleg_list, zones, nodes):
    df_a = pd.DataFrame(nontransitleg_list, columns=['a', 'b'])

    def geometry(row):
        return shapely.geometry.LineString(
            [nodes.loc[row['a'], 'geometry'], zones.loc[row['b'], 'geometry'].centroid]
        )

    df_a['geometry'] = df_a.apply(geometry, axis=1)
    df_b = df_a.rename(columns={'a': 'b', 'b': 'a'})
    return pd.concat([df_a, df_b])


def nontransitleg_list(zone_stops):
    nontransitlegs = []
    for zone in zone_stops.keys():
        for stop in zone_stops[zone]:
            nontransitlegs.append((stop, zone))

    return list(set(nontransitlegs))


def nontransitlegs(zones, nodes, stop_list, leg_type='contains'):
    _zone_stops = zone_stops(zones, nodes, stop_list, leg_type)
    _list = nontransitleg_list(_zone_stops)
    return nontransitleg_geometries(_list, zones, nodes)


def reindex_nodes(links, nodes, start_from=0, reindex_node=None):
    if reindex_node is None:
        nodes = nodes.copy()
        links = links.copy()

        index = nodes.index
        rename_dict = {}
        current = start_from
        for n in index:
            rename_dict[n] = str(current)
            current += 1

        def reindex_node(node):
            return rename_dict[node]

    nodes.index = [reindex_node(n) for n in nodes.index]
    links['a'] = links['a'].apply(reindex_node)
    links['b'] = links['b'].apply(reindex_node)
    return links, nodes
