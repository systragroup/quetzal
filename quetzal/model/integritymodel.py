import collections

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from syspy.spatial.graph import network as networktools
from syspy.transitfeed import feed_links
from tqdm import tqdm
from functools import wraps
import warnings



def deprecated_method(method):
    @wraps(method)
    def decorated(self, *args, **kwargs):
        message = 'Deprecated: replaced by %s' % method.__name__
        warnings.warn(
            message,
            DeprecationWarning
        )
        print(message)
        return method(self, *args, **kwargs)

    decorated.__doc__ = 'deprecated! ' + str(decorated.__doc__)
    return decorated


def label_links(links, node_prefixe):
    links = links.copy()
    links['a'] = [node_prefixe + str(i).split(node_prefixe)[-1] for i in links['a']]
    links['b'] = [node_prefixe + str(i).split(node_prefixe)[-1] for i in links['b']]
    return links


def list_duplicates(l):
    return [x for x, y in collections.Counter(l).items() if y > 1]


def geodataframe_place_holder(geom_type, prefix=None):
    if geom_type == 'LineString':
        geo = LineString([(0, 0), (0, 0)])
    if geom_type == 'Polygon':
        geo = Polygon([(0, 0), (0, 0), (0, 0)])
    if geom_type == 'Point':
        geo = Point(0, 0)
    if prefix is not None:
        i = '{}0'.format(prefix)
    else:
        i = 0
    return gpd.GeoDataFrame(pd.DataFrame([geo], columns=['geometry'], index=[i]))

def rename_duplicate_stops(g, sep='_circular-fix_'):
    stop_set = {}
    u_ab = []
    for a, b in g[['a', 'b']].values:
        # a
        ab = [a, b]
        if b not in stop_set.keys():
            ab[1] = b
        else:
            ab[1] = b + sep + str(stop_set[b])
            count = stop_set[b]
        
        if a not in stop_set.keys():
            ab[0] = a
            stop_set.update({a: 0})
        else:
            ab[0] = a + sep + str(stop_set[a])
            stop_set.update({a: count+1})

        u_ab.append(ab)

    g[['a', 'b']] = u_ab

    return g

def fix_circular_lines(links, nodes, on='trip_id', sep='_circular-fix_'):
    links = links.groupby(on).apply(rename_duplicate_stops, sep='_circular-fix_')
    # add new nodes
    c_nodes = links[links['a'].apply(lambda x: sep in x )]['a']
    c_nodes = c_nodes.append(links[links['b'].apply(lambda x: sep in x )]['b']).drop_duplicates()

    c_nodes_geom = nodes.loc[c_nodes.apply(lambda x: x.split(sep)[0]).values]
    c_nodes_geom.index = c_nodes.values
    nodes = nodes.append(c_nodes_geom)

    return links, nodes


class IntegrityModel:

    def __init__(self, debug=False, walk_on_road=False, epsg=None, coordinates_unit=None, **kwargs):

        self.parameters = {}
        self.debug = debug
        self.walk_on_road = walk_on_road
        self.coordinates_unit = coordinates_unit
        self.epsg = epsg
        self.segments = ['all']

    def integrity_test_collision(
        self,
        sets=('links', 'nodes', 'zones', 'road_links', 'road_nodes')
    ):
        """
            * requires: links, nodes, zones
        """
        tuples = [(key, list(self.__getattribute__(key).index)) for key in sets]

        for left, left_list in tuples:
            try:
                duplicates = list_duplicates(left_list)
                assert len(duplicates) <= 0
            except AssertionError:
                message = """
                %i duplicates in %s.index: %s
                """ % (len(duplicates), left, str(duplicates))
                raise AssertionError(message)

            left_set = set(left_list)
            for right, right_list in tuples:
                right_set = set(right_list)
                if left != right:
                    try:
                        intersection = left_set.intersection(right_set)
                        ilength = len(intersection)
                        assert ilength == 0
                    except AssertionError:
                        message = """
                        %i values are shared between %s and %s indexes : %s
                        """ % (ilength, left, right, str(intersection))
                        raise AssertionError(message)

    def integrity_fix_collision(
        self,
        prefixes={'nodes': 'node_', 'links': 'link_', 'zones': 'zone_'}
    ):
        """
            * requires: links, nodes, zones
            * builds: links, nodes, zones
        """
        try:
            self.integrity_test_collision()
        except AssertionError:
            if len(set(self.links.index)) != len(self.links):
                self.links = self.links.reset_index(drop=True)
            try:
                self.integrity_test_collision()
            except AssertionError:
                self._add_type_prefixes(prefixes)

    def _add_type_prefixes(
        self,
        prefixes={'nodes': 'node_', 'links': 'link_', 'zones': 'zone_'}
    ):
        """
            * requires: links, nodes, zones
            * builds: links, nodes, zones
        """
        for key in prefixes.keys():
            attribute = self.__getattribute__(key)
            prefixe = prefixes[key]
            attribute.index = [prefixe + str(i).split(prefixe)[-1] for i in attribute.index]

        if 'nodes' in prefixes.keys():
            for key in ['links', 'footpaths']:
                try:
                    link_like = self.__getattribute__(key)
                    self.__setattr__(key, label_links(link_like, prefixes['nodes']))
                except (AttributeError, KeyError):  # KeyError: 'a'
                    print('can not add prefixes on table: ', key)

    def integrity_test_sequences(self):
        """
            * requires: links
            * builds: broken_sequences
        """
        links = self.links.copy().sort_values('link_sequence')
        broken_sequences = []
        for trip_id in set(links['trip_id']):
            subset = links.loc[links['trip_id'] == trip_id]
            broken = tuple(subset['a'])[1:] != tuple(subset['b'])[:-1]
            if broken:
                broken_sequences.append(trip_id)
            message = "some lines have a broken pattern \n"
            message += "ex : the following pattern is broken: a->b, b->c, d->e "
            message += " because it misses the c->d link \n"
            message += "broken lines: " + str(broken_sequences)

        self.broken_sequences = broken_sequences
        assert len(broken_sequences) == 0, message

        count_series = self.links.groupby(['trip_id'])['link_sequence'].count()
        max_series = self.links.groupby(['trip_id'])['link_sequence'].max()
        message = 'link_sequences are not continuous (1, 2, 3, 5, 6) for example'
        assert (count_series == max_series).all(), message

    def integrity_fix_sequences(self):
        """
            * requires: links
            * builds: links
        """
        try:
            self.integrity_test_sequences()
        except AssertionError:
            # a - b
            to_drop = self.links['trip_id'].isin(self.broken_sequences)
            self.links = self.links.loc[~to_drop]
            print('dropped broken sequences: ' + str(self.broken_sequences))

            # link_sequence
            l = self.links.copy()
            l['index'] = l.index
            l = feed_links.clean_sequences(l, sequence='link_sequence', group_id='trip_id')
            self.links = l.set_index('index')

    def integrity_test_circular_lines(self):
        """
        The model does not work with circular lines
            * requires: links
            * builds: circular_lines
        """
        links = self.links.copy().sort_values('link_sequence')
        circular_lines = []
        for trip_id in set(links['trip_id']):
            subset = links.loc[links['trip_id'] == trip_id]

            start = subset['a'].value_counts()
            end = subset['b'].value_counts()

            if start.max() > 1 or end.max() > 1:
                circular_lines.append(trip_id)

            message = "some lines stop many time at the same stop (circular) \n"
            message += "ex : the following pattern is circular : a->b, b->c, c->d, d->b, b->f "
            message += " because it stops twice in b \n"
            message += "circular lines: " + str(circular_lines)

        self.circular_lines = circular_lines
        assert len(circular_lines) == 0, message

    def integrity_fix_circular_lines(self, method='drop', sep='_circular-fix_'):
        """
            * requires: links, nodes
            * builds: links, nodes
            Either drop circular lines (method='drop') or create node duplicates (method='duplicate')
        """
        try:
            self.integrity_test_circular_lines()
        except AssertionError:
            if method == 'drop':
                is_circular = self.links['trip_id'].isin(self.circular_lines)
                self.links = self.links.loc[~is_circular]
                print('dropped circular lines: ' + str(self.circular_lines))
            elif method=='duplicate':
                self.links, self.nodes = fix_circular_lines(self.links, self.nodes, sep=sep)
    

    def integrity_test_isolated_roads(self):
        """
            * requires: road_links
        """
        g = nx.Graph()
        g.add_edges_from(self.road_links[['a', 'b']].values)
        ncc = nx.number_connected_components(g)
        msg = 'the road graph have %i connected components (> 1)' % ncc
        assert nx.number_connected_components(g) == 1, msg

    def integrity_test_dead_ends(self, cutoff=5):
        """
        look for dead-ends in the road network
        only the dead-ends with a dead-rank lower than the cutoff
        will be identified
            * requires: road_links
        """
        road_graph = nx.DiGraph()
        road_graph.add_edges_from(self.road_links[['a', 'b']].values)

        a = nx.all_pairs_shortest_path_length(road_graph, cutoff=cutoff)

        dead_ends = []
        for node, path_dict in tqdm(a):
            if len(path_dict) < cutoff:
                dead_ends.append(node)

        message = "some directional road_links are dead-ends \n"
        message += "ex 1 : if their is an a->b link and their is no link leaving b; b is a dead-nodes \n"
        message += "ex 2 : in the given network (a->b, b->c, c->d, b->a) both c and d are dead-nodes \n"
        message += "because you can not reach 'a' from them \n"
        message += "the dead-rank of d is 0, the dead rank of c is 1 \n"
        message += "dead-nodes: " + str(dead_ends)

        self.dead_ends = dead_ends
        assert len(dead_ends) == 0, message

    def integrity_test_nodeset_consistency(self):
        """
            * requires: nodes, links
        """

        try:
            missing_nodes = self.link_nodeset() - self.nodeset()
            msg = 'some nodes are referenced in links but not in nodes \n'

            # [:1000] we do not want to raise a heavy error (huge string)
            msg += 'missing nodes: ' + str(missing_nodes)[:1000]

            self.missing_nodes = missing_nodes
            assert len(missing_nodes) == 0, msg

            orphan_nodes = self.nodeset() - self.link_nodeset()
            msg = 'some nodes are referenced in nodes but not in links \n'

            # [:1000] we do not want to raise a heavy error (huge string)
            msg += 'orphan nodes: ' + str(orphan_nodes)[:1000]

            self.orphan_nodes = orphan_nodes
            assert len(orphan_nodes) == 0, msg

        except AttributeError:  # no links or nodes
            print('no links or nodes')
            pass

        try:
            missing_road_nodes = self.road_link_nodeset() - self.road_nodeset()

            msg = 'some nodes are referenced in links but not in nodes \n'
            msg += 'missing road_nodes' + str(missing_road_nodes)[:1000]

            self.missing_road_nodes = missing_road_nodes
            assert len(missing_road_nodes) == 0, msg

            orphan_road_nodes = self.road_nodeset() - self.road_link_nodeset()
            msg = 'some nodes are referenced in nodes but not in links \n'

            # [:1000] we do not want to raise a heavy error (huge string)
            msg += 'orphan road_nodes: ' + str(orphan_road_nodes)[:1000]

            self.orphan_nodes = orphan_nodes
            assert len(orphan_nodes) == 0, msg
        except (AttributeError, KeyError):
            # no road_links or road_nodes, KeyError if they are place_holders
            print('no road_links or road_nodes')

    def integrity_test_road_nodeset_consistency(self):
        """
            * requires: nodes, links
        """
        missing_road_nodes = self.road_link_nodeset() - self.road_nodeset()

        msg = 'some nodes are referenced in links but not in nodes \n'
        msg += 'missing road_nodes' + str(missing_road_nodes)[:1000]

        self.missing_road_nodes = missing_road_nodes
        assert len(missing_road_nodes) == 0, msg

    def integrity_fix_nodeset_consistency(self):
        self.links = self.links.loc[self.links['a'].isin(self.nodeset())]
        self.links = self.links.loc[self.links['b'].isin(self.nodeset())]
        self.nodes = self.nodes.loc[self.link_nodeset()]
        try:
            self.integrity_fix_road_nodeset_consistency()
        except KeyError:  # 'a'
            pass

    def integrity_fix_road_nodeset_consistency(self):
        self.road_links = self.road_links.loc[
            self.road_links['a'].isin(self.road_nodeset())
        ]
        self.road_links = self.road_links.loc[
            self.road_links['b'].isin(self.road_nodeset())
        ]
        self.road_nodes = self.road_nodes.loc[self.road_link_nodeset()]

    def integrity_test_road_network(self, cutoff=10):
        self.integrity_test_isolated_roads()
        self.integrity_test_dead_ends(cutoff=cutoff)
        self.integrity_test_road_nodeset_consistency()

    def integrity_fix_road_network(self, cutoff=10, recursive_depth=1):
        """
        clean road_network
            * requires: road_links, road_nodes
            * builds: road_links, road_nodes
        """
        if recursive_depth < 1:
            print('Reached max recursive_depth')
            return

        self.road_links = networktools.drop_secondary_components(self.road_links)
        self.road_links = networktools.drop_deadends(self.road_links, cutoff=cutoff)
        self.integrity_fix_road_nodeset_consistency()
        try:
            self.integrity_test_road_network()
        except AssertionError:
            self.integrity_fix_road_network(
                cutoff=cutoff,
                recursive_depth=recursive_depth - 1
            )

    def integrity_test_all(self, errors='raise', verbose=True):
        """
        errors='ignore' can be passed
        """
        integrity_test_methods = [
            m for m in list(dir(self))
            if ('integrity_test_' in m) and ('integrity_test_all' not in m)
        ]
        for name in integrity_test_methods:
            try:
                try:
                    self.__getattribute__(name)()
                    if verbose:
                        print('passed:', name)
                except AssertionError as e:
                    if verbose:
                        print('failed:', name)
                    if errors != 'ignore':
                        raise e
            except Exception as e:  # broad exception
                if verbose:
                    print('not performed:', name)
                if errors != 'ignore':
                    raise e

    def link_nodeset(self):
        return set(self.links['a']).union(set(self.links['b']))

    def nodeset(self):
        return set(self.nodes.index)

    def road_link_nodeset(self):
        return set(self.road_links['a']).union(set(self.road_links['b']))

    def road_nodeset(self):
        return set(self.road_nodes.index)
