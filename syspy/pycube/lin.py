# -*- coding: utf-8 -*-

"""

**This package provides tools for editing and analysing CUBE .LIN files**



example:
::


    import pycube

    links = pandasshp.read_shp('Q:/my_links.shp').set_index('n')
    nodes = pandasshp.read_shp('Q:/network/my_nodes.shp').set_index('n')
    with open('Q:/my_lines.LIN', 'r') as lin:
        text = lin.read()

    hubs = pycube.lin.find_hubs(zones, nodes, text)
    pruned_text = pycube.lin.prune_text(text, hubs)

    with open('Q:/my_lines_pruned.LIN', 'w') as lin:
        lin.write(pruned_text)
"""

__author__ = 'qchasserieau'



from IPython.html.widgets import FloatProgress
from IPython.display import display
import itertools
import re
import pandas as pd
import networkx as nx
import shapely
import numpy as np

from syspy.pycube._line import Line
from syspy.pycube.dijkstra import DijkstraMonkey
from syspy.syspy_utils import syscolors
from syspy.io.pandasshp import pandasshp
from syspy.spatial import spatial


class Lin:

    """
    Joins a .LIN to a zoning and a network.

    example:
    ::
        import pycube

        links = pandasshp.read_shp('Q:/my_links.shp').set_index('n')
        nodes = pandasshp.read_shp('Q:/network/my_nodes.shp').set_index('n')
        lines = pycube.lin.Lin(zones, nodes, file=r'Q:/my_lines.LIN')


    """
    def __init__(
        self,
        zones=None,
        nodes=None,
        text=None,
        file=None,
        edges=None,
        build_geometries=False,
        build_graph=False,
        sep='line name',
        leg_type='nearest',
        prj=None
    ):

        progress = FloatProgress(min=0, max=5, width=975, height=10, color=syscolors.rainbow_shades[1], margin=5)
        progress.value = 1
        display(progress)

        if not text and file:
            with open(file, 'r') as lin_file:
                text = lin_file.read()

        equal = re.compile('[ ]*[=]+[ ]*')
        coma = re.compile('[ ]*[,]+[ ]*')
        lower_text = text.lower().replace('n=', 'N=').replace('rt=', 'RT=').replace('<<pt>>','<<pt>>')
        self.text = coma.sub(', ', equal.sub('=', lower_text.replace('"', "'")))  #: raw text of the .LIN (str)

        stop_list = _stop_list(self.text)
        self.lin_chunks = self.text.split(sep)
        self.sep = sep

        self._to_dict()
        self.line_names = [line_name(c) for c in self.lin_chunks]
        self.line_names = [name for name in self.line_names if name != 'not_a_line']


        if zones is not None:

            zone_stops = _zone_stops(zones, nodes, stop_list, leg_type)
            stop_lines = _stop_lines(stop_list, self.lin_chunks)
            zone_lines = _zone_lines(zone_stops, stop_list, stop_lines)
            hubs = _hubs(zone_stops, stop_lines, zone_lines)

            self.zone_stops = zone_stops  #: dictionary of the stops of each zone {zone: [stops that are in the zone]}
            self.stop_lines = stop_lines  #: dictionary of the lines of each stop {stop: [lines that stop]}
            self.zone_lines = zone_lines  #: dictionary of the lines of each zone {zone: [lines that stop in the zone]}
            self.hubs = hubs  #:  minimal set of nodes that are necessary to keep the keep zone_lines stable while pruning zone_stops
            self.hubs_and_terminus = self.hubs.union(self.find_endpoints())
            self.transitlegs = _transitlegs(self.stop_lines)  #: list of stop<->line links (based on self.stop_lines)
            self.nontransitlegs = _nontransitlegs(self.zone_stops)  #: list of zone<->stop links (based on self.zone_stops)




        self.stop_list = stop_list
        self.zones = zones  #: GeoDataFrame of the zones : str
        self.prj = prj
        self.nodes = nodes  #: GeoDataFrame of the nodes
        self.line_count = _line_count(text) #: line count by node
        self.data = self.geo_dataframe(geometry=False)  #: data organized as a dataframe

        progress.value += 1

        if build_graph:
            self.connection_graph = nx.Graph(self.transitlegs + self.nontransitlegs) #: nx.Graph built with self.nontransitlegs and self.transitlegs
            self.path_matrix = _path_matrix(self.connection_graph, self.zones)  #: OD matrix that contains the path and the skims of each OD pair in the zoning (base on path_matrix)

        progress.value += 1

        if build_geometries:
            geometries = pandasshp.od_matrix(zones)
            self.path_matrix_geometries = pd.merge(self.path_matrix, geometries, on=['origin', 'destination'])  #: OD matrix that contains the path and the skims of each OD pair in the zoning + the geometry

        progress.value += 1

        if edges is not None:
            self.dijkstra = DijkstraMonkey(edges.values)

        progress.value += 1

    def _to_dict(self):
        self.line_dict = {line_name(c):  Line(c) for c in self.text.split(self.sep)}

    def _to_text(self, format_chunks=True):
        _lines = [self.line_dict[n] for n in self.line_names]

        if format_chunks:
            self.text = self.sep.join([';;<<PT>>;;\n\n'] + [l.formated_chunk() for l in _lines])
        else:
            self.text = self.sep.join([';;<<PT>>;;\n\n'] + [l.chunk for l in _lines])
        self.line_count = _line_count(self.text)

    def to_text(self):
        self._to_text()
        return self.text

    def change_time(self, to_change, factor=1, offset=0, inplace=True):
        """
        Changes the Route Times of a line

        :param inplace: edit the Lin object if True, return an edited text chunk if False
        :param to_change name of the line to change (may be a list of names)
        :param factor: multiplicative factor to multiply by the Route Times
        :param offset: time to add to the Route Times
        """

        if type(to_change) in [set, list]:
            for entity in to_change:
                self.change_time(entity, factor, offset, inplace)
        else:
            self.line_dict[to_change].change_time(factor, offset)
            self._to_text()

    def cut_at_node(self, name, n, keep='left'):

        """
        Sets a new terminus to a line

        :param name: name of the line to cut
        :param n: id of the new terminus of the line (must be stop of the line in the fist place)
        :param keep: part of the line to keep if left, keep the nodes with lesser Route Times than n in the .LIN file

        """
        self.line_dict[name].cut_at_node(n, keep=keep)
        self._to_text()

    def cut_between(self, to_cut, from_node, to_node, inplace=True):

        """
        Shortens a line

        :param to_cut: name of the line to cut or list of names
        :param from_node: first terminus of the new line (must be stop of the line in the fist place)
        :param to_node: second terminus of the new line (must be stop of the line in the fist place)
        :param inplace: if true the shortened line replaces the former, if false, it is returned.
        """
        if type(to_cut) in [set, list]:
            for entity in to_cut:
                self.cut_between(entity, from_node, to_node)
        else:
            self.line_dict[to_cut].cut_between(from_node, to_node)
            self._to_text(format_chunks=False)

    def copy_line(self, name, copy, from_node=None, to_node=None):
        """
        Copies a line

        :param name: name of the line to copy
        :param copy: name of the copy

        keeping the main corridor of a fork line in Monterrey:
        ::
            lines.copy_line('ligne orange_est', 'ligne orange')  #  builds a line (ligne orange) from a branch (ligne orange_est)
            lines.drop_line('ligne orange_est')                  #  drops the first branch
            lines.drop_line('ligne orange_ouest')                #  drops the other branch
            lines.cut_between('ligne orange',6293, 53191)        #  reduces the copy to the main corridor
        """

        self.line_dict[copy] = Line(self.line_dict[name].chunk.replace("'" + name + "'", "'" + copy + "'"))
        self.line_names.append(copy)
        self._to_text(format_chunks=False)

        if from_node and to_node:
            self.cut_between(copy, from_node, to_node)

    def add_line(self, to_add):

        """
        Adds a line to a Lin object

        :param to_add: the Line object to add (one of the objects of Lin.line_dict for example)

        Adding feeders from a distinct file to a Lin
        ::
            lines = pycube.lin.Lin(nodes=nodes, file = data_path + r'/lineas/bus_2015_metro_2045.lin')
            feeders = pycube.lin.Lin(nodes=nodes, file = data_path + r'/lineas/lin_2045_net_2045_dijkstra.lin')

            to_add = [feeders.line_dict[name] for name in ['ruta_bis 67', 'ruta 67', 'ruta_bis 1', 'ruta 1']]
            lines.add_line(to_add)
        """

        if type(to_add) in [set, list]:
            for entity in to_add:
                self.add_line(entity)
        else:
            self.line_names.append(to_add.name)
            self.line_dict[to_add.name] = to_add
            self._to_text(format_chunks=False)

    def drop_line(self, to_drop):

        """
        Drops a line or a collection of lines

        :param to_drop: the name or a collection of names of the lines to drop
        """
        if type(to_drop) in [set, list]:
            for entity in to_drop:
                self.drop_line(entity)

        else:
            self.line_names = [n for n in self.line_names if n != to_drop]
            self._to_text(format_chunks=False)

    def new_line(self, name, header='', node_list=None, node_time_dict=None, nodes=None, speed=None):
        chunk = header
        if node_time_dict is not None:
            for node, time in node_time_dict.items():
                chunk += 'N=%s, RT=%s, ' % (str(node), str(time))
        elif node_list is not None:
            chunk_length = 0
            for node in node_list:
                chunk += 'N=%s, ' %(str(node))
                chunk_length += 1
                chunk += 'RT=%s, ' % (str(round(line_length(chunk, nodes)/speed, 2)) if chunk_length > 1 else '0')


        self.line_names.append(name)
        self.line_dict[name] = Line(chunk[:-2])
        self._to_text(format_chunks=False)

    def drop_mode(self, to_drop=None, all_but=None):
        """
        Drops a mode or a collection of modes

        :param to_drop: the name or a collection of names of the modes to drop
        """

        geo = self.geo_dataframe(geometry=False)

        to_drop = to_drop if type(to_drop) in [set, list] else ([to_drop] if to_drop else to_drop)
        all_but = all_but if type(all_but) in [set, list] else ([all_but] if all_but else all_but)

        modes = list(set(geo['mode'].unique())-set(all_but) if all_but else to_drop)
        line_names = list(geo[ geo['mode'].isin(modes)]['name'])
        self.drop_line(line_names)
        self._to_text()

    def merge_lines(self, left_name, right_name, delete_right=True, start='left'):
        """

        :param left_name: name of the line to edit
        :param right_name: name of the line to merge on the left line
        :param delete_right: if True, the right line will be deleted
        :param start: 'left' or 'right', 'left' means that the merged line starts with the node of the left line

        Let's merge the 'ruta 67' (which is a bus line) on the 'ligne violette' which is a twice faster metro line
        ::
            lines.change_time('ruta_bis 67', 0.5, 0)             #  changes the Route Times of 'ruta_bis 67'
            lines.merge_lines('ligne violette', 'ruta_bis 67',)  #  merge the lines and deletes 'ruta_bis 67'
            lines.change_time('ruta 67', 0.5, 0)
            lines.merge_lines('ligne violette_bis', 'ruta 67', start='right')


        """
        self.line_dict[left_name].add_line(self.line_dict[right_name], start=start)
        if delete_right:
            self.line_names = [i for i in self.line_names if i != right_name]
        self._to_text()

    def set_parameter(self, to_set, parameter, value):
        """
        Set parameters such as mode or headway for a line or a set of lines

        :param to_set: the name or the list of names of the lines to edit
        :param parameter: the name of the parameter to edit
        :param value: the value to set for the parameter
        """
        if type(to_set) in [set, list]:
            for entity in to_set:
                self.set_parameter(entity, parameter, value)
        else:
            self.line_dict[to_set].set_parameter(parameter, value)
            self._to_text(format_chunks=False)

    def set_direct(self, to_set, from_stop, to_stop):
        """
        remove the stops between from_stop and to_stop (and reverse)

        """
        if type(to_set) in [set, list]:
            for entity in to_set:
                self.set_direct(entity, from_stop, to_stop)
        else:
            self.line_dict[to_set].set_direct(from_stop, to_stop)
            self._to_text(format_chunks=False)


    def drop_checkpoints(self, to_drop):

        if type(to_drop) in [set, list]:
            for entity in to_drop:
                self.drop_checkpoints(entity)
        else:
            self.line_dict[to_drop].drop_checkpoints()
            self._to_text(format_chunks=False)

    def change_stop(self, to_change, from_stop, to_stop, to_text=True):
        if type(to_change) in [set, list]:
            for entity in to_change:
                self.change_stop(entity, from_stop, to_stop)
            self.to_text()
        else:
            self.line_dict[to_change].change_stop(from_stop, to_stop)
            if to_text:
                self.to_text()


    def find_endpoints(self, mode=None):
        """
        Returns a set of terminus

        :param mode: the mode
        :return: the set of terminus
        """

        return _find_endpoints(self.text, mode=mode, sep=self.sep)

    def find_mode_stops(self, mode, regex='N=[0-9]{4,6}'):
        """
        Returns a set of stops for a given mode

        :param mode: the mode
        :param regex: the regex that defines the stops
        :return: the set of stops for this mode
        """
        return _mode_stops(mode, self.text, regex=regex, sep=self.sep)

    def prune_text(self, stop_list=None, inplace=False):
        stop_list = stop_list if stop_list else self.hubs_and_terminus
        if inplace:
            self.text = _prune_text(self.text, stop_list)
        else:
            return _prune_text(self.text, stop_list)

    def geo_dataframe(self, mode_split_string='mode=', geometry=True, all_nodes=True):

        """
        Returns a pd.DataFrame that contains the name, the mode, the headway and the geometry of the lines. It may be
        used to dump the Lin to a .shp

        :param mode_split_string: 'mode=' for example
        :return: a pd.DataFrame that contains the name, the mode, the headway and the geometry of the lines

        Saving a Lin as a .shp file::
            geo = lines.geo_dataframe()
            geo['color'] = geo['name'].apply(syscolors.in_string)
            pandasshp.write_shp(sig_path+'lin_2045', geo, projection_string=epsg_32614)
        """
        chunks = self.text.split(self.sep)

        if geometry :
            df = pd.DataFrame({'name': pd.Series([line_name(c) for c in chunks ]),
                             'mode': pd.Series([line_mode(c, split_string=mode_split_string) for c in chunks]),
                             'headway': pd.Series([line_headway(c) for c in chunks]),
                             'time': pd.Series([line_time(c) for c in chunks]),
                             'geometry': pd.Series([line_geometry(c, self.nodes, all_nodes) for c in chunks]),
                             'stops':pd.Series([str(_stop_list(c)) for c in chunks]),
                             'nodes':pd.Series([str(_node_list(c)) for c in chunks]),
                             'nstops': pd.Series([len(_stop_list(c)) for c in chunks])})
            df.dropna(inplace=True)
            df['length'] = df['geometry'].apply(lambda g: g.length)
            df['speed'] = df['length']/df['time']
        else:

            df = pd.DataFrame({'name': pd.Series([line_name(c) for c in chunks ]),
                                 'mode': pd.Series([line_mode(c, split_string=mode_split_string) for c in chunks]),
                                 'headway': pd.Series([line_headway(c) for c in chunks]),
                                 'time': pd.Series([line_time(c) for c in chunks]),
                             'stops':pd.Series([str(_stop_list(c)) for c in chunks]),
                             'nodes':pd.Series([str(_node_list(c)) for c in chunks]),
                             'nstops': pd.Series([len(_stop_list(c)) for c in chunks])})

        return df.dropna()

    def to_shape(self, stop_file=None, link_file=None, all_nodes=True):
        """
        Saves the geometry of the Lin as a .shp that contains the points (stops) and the polylines (links)

        :param node_file: name of the file that contains the points
        :param link_file: name ot the file that contains the polylines
        """
        if bool(link_file):
            pandasshp.write_shp(link_file, self.geo_dataframe(all_nodes=True), projection_string=self.prj)
        if bool(stop_file):
            stops = pd.merge(self.nodes, self.line_count, left_index=True, right_index=True).reset_index()
            pandasshp.write_shp(stop_file, stops, projection_string=self.prj)


    def nontransitleg_geometries(self):
        """
        returns a pd.DataFrame of the connectors that link the zones to the traffic nodes, it includes their
        geometry.
        :return: a pd.DataFrame of the connectors that link the zones to the traffic nodes
        """
        df_a = pd.DataFrame(self.nontransitlegs, columns=['a', 'b'])

        def geometry(row):
            return shapely.geometry.LineString(
                [self.nodes.loc[row['a'], 'geometry'], self.zones.loc[row['b'], 'geometry'].centroid]
            )

        df_a['geometry'] = df_a.apply(geometry, axis=1)
        df_b = df_a.rename(columns={'a': 'b', 'b': 'a'})
        return pd.concat([df_a, df_b])

    def line_links(self, to_links=None, mode=None):
        if mode:
            return self.line_links(list(self.data[self.data['mode'] == mode]['name']))
        if type(to_links) in [set, list]:
            return pd.concat([self.line_links(entity) for entity in to_links]).drop_duplicates()
        else:
            line_stops = _stop_list(self.line_dict[to_links].chunk)
            _line_links = [[line_stops[i], line_stops[i + 1]] for i in range(len(line_stops) - 1)]
            return  pd.DataFrame(_line_links, columns=['a', 'b']).drop_duplicates()



regex_time = 'RT=[0-9]{1,6}[.]?[0-9]{0,6}'
time_re = re.compile(regex_time)

def find_hubs(zones, nodes, text):

    stop_list = _stop_list(text)
    lin_chunks = ['LINE Name' + chunk for chunk in text.split('LINE Name')]

    zone_stops = _zone_stops(zones, nodes, stop_list)
    stop_lines = _stop_lines(stop_list, lin_chunks)
    zone_lines = _zone_lines(zone_stops, stop_list, stop_lines)
    hubs = _hubs(zone_stops, stop_lines, zone_lines)

    return hubs


def line_length(chunk, nodes):
    return line_geometry(chunk, nodes).length


def line_speed(chunk, nodes):
    return line_length(chunk, nodes)/line_time(chunk)


def line_geometry(chunk, nodes, all_nodes=True):
    point_list = _node_list(chunk) if all_nodes else _stop_list(chunk)
    try :
        return shapely.geometry.LineString([nodes.loc[node, 'geometry'] for node in point_list])
    except :
        return np.nan


def line_time(chunk):
    try:
        return float(time_re.findall(chunk)[-1].split('RT=')[1])
    except:
        return np.nan


def line_stops(chunk):
    return len(_stop_list(chunk))


def line_name(lin_chunk):
    try :
        return lin_chunk.split("'")[1]
    except:
        return 'not_a_line'


def line_mode(lin_chunk, split_string='mode='):
    try :
        return int(lin_chunk.split(split_string)[1].split(',')[0])
    except:
        return 'not_a_line'

def line_headway(lin_chunk, split_string='headway='):
    try :
        return float(lin_chunk.lower().replace(' ', '').split(split_string)[1].split(',')[0])
    except:
        return 'not_a_line'



def _zone_stops(zones, nodes, stop_list, leg_type='contains'):

    if leg_type == 'contains':
        progress = FloatProgress(
            min=0, max=len(list(zones.iterrows())), width=975, height = 10, color=syscolors.rainbow_shades[1], margin=5)
        progress.value=0
        display(progress)
        zone_stops = {}
        for zone_id, zone in zones.iterrows():
            zone_stops[zone_id] = []
            for stop_id, stop in nodes.loc[stop_list].iterrows():
                if zone['geometry'].contains(stop['geometry']):
                    zone_stops[zone_id].append(stop_id)
            progress.value +=1

    if leg_type == 'nearest':
        centroids = zones.copy()
        centroids['geometry'] = zones['geometry'].apply(lambda g: g.centroid)
        stops = nodes.loc[stop_list]

        links_a = spatial.nearest(stops, centroids).rename(columns={'ix_many': 'zone', 'ix_one': 'stop'})
        links_b = spatial.nearest(centroids, stops).rename(columns={'ix_one': 'zone', 'ix_many': 'stop'})
        links = pd.concat([links_a, links_b]).drop_duplicates()
        zone_stops = dict(links.groupby('zone')['stop'].agg(lambda s: list(s)))

    return zone_stops



def _stop_lines(stop_list, lin_chunks):
    progress = FloatProgress(
        min=0, max=len(stop_list), width=975, height=10, color=syscolors.rainbow_shades[1], margin=5)
    progress.value=0
    display(progress)
    stop_lines = {}
    for stop in stop_list:
        stop_lines[stop] = set()
        for lin_chunk in lin_chunks[1:]:
            if 'N=' + str(stop) in lin_chunk:
                stop_lines[stop] = stop_lines[stop].union([line_name(lin_chunk)])
        progress.value += 1
    return stop_lines


def _zone_lines(zone_stops, stop_list, stop_lines):
    zone_lines = {}
    for zone, zone_stop_list in zone_stops.items():
        zone_lines[zone] = set()
        for stop in set(stop_list).intersection(zone_stop_list):
            zone_lines[zone] = zone_lines[zone].union(stop_lines[stop])
    return zone_lines


def _hubs(zone_stops, stop_lines, zone_lines):
    pop_zone_lines = dict(zone_lines)
    to_keep = {}
    for zone in list(zone_lines.keys()):
        to_keep[zone] = []
        while len(pop_zone_lines[zone]):

            dict_intersection = {len(pop_zone_lines[zone].intersection(stop_lines[stop])): stop for stop in zone_stops[zone]}
            max_intersection = sorted(dict_intersection.keys())[-1]
            max_stop = dict_intersection[max_intersection]
            to_keep[zone].append(max_stop)
            pop_zone_lines[zone] = pop_zone_lines[zone]-stop_lines[max_stop]

    hubs = set(itertools.chain(*list(to_keep.values())))
    return hubs


def _stop_list(text, regex='N=[0-9]{4,6}'):
    stop_re = re.compile(regex)
    return [int(f[2:]) for f in stop_re.findall(text)]


def _node_list(text, regex='N=[-]?[0-9]{4,6}'):
    node_re = re.compile(regex)
    return [int(f[2:].replace('-', '')) for f in node_re.findall(text)]


def _endpoints(lin_chunk):
    return [_node_list(lin_chunk)[0], _node_list(lin_chunk)[-1]]


def _nontransitlegs(zone_stops):

    nontransitlegs = []
    for zone in zone_stops.keys():
        for stop in zone_stops[zone]:
            nontransitlegs.append((stop, zone))

    return list(set(nontransitlegs))


def _transitlegs(stop_lines):
    transitlegs = []
    for stop in stop_lines.keys():
        for line in stop_lines[stop]:
            transitlegs.append((stop, line))

    return list(set(transitlegs))


def _line_count(text):
    return pd.DataFrame(pd.Series(_stop_list(text)).value_counts(), columns=['lines'])


def _mode_stops(mode, text, regex='N=[0-9]{4,6}', sep='LINE NAME'):
    stop_re = re.compile(regex)
    lin_chunks = [sep + chunk for chunk in text.split(sep)]
    mode_chunks = [chunk for chunk in lin_chunks if 'mode='+str(mode) in chunk]
    mode_find = stop_re.findall(''.join(mode_chunks))
    return [int(f[2:]) for f in mode_find]


def _find_endpoints(text, mode=None, sep='LINE NAME'):
    lin_chunks = [sep + chunk for chunk in text.split(sep)[1:]]
    lin_chunks = [chunk for chunk in lin_chunks if 'mode='+str(mode) in chunk] if mode else lin_chunks
    return list(itertools.chain(*[_endpoints(chunk) for chunk in lin_chunks]))


def _prune_text(text, stops):

    pruned_text = text.replace('N=', 'N=-')
    pruned_text = re.sub('[-]+', '-', pruned_text)

    for stop in stops:
        before = 'N=-' + str(stop)
        after = 'N=' + str(stop)
        pruned_text = pruned_text.replace(before, after)

    return pruned_text


# path matrix


def connection_graph(zones, nodes, text):

    lin_chunks = ['LINE Name' + chunk for chunk in text.split('LINE Name')]
    stop_list = _stop_list(text)
    zone_stops = _zone_stops(zones, nodes, stop_list)
    stop_lines = _stop_lines(stop_list, lin_chunks)

    nontransitlegs = []
    for zone in zone_stops.keys():
        for stop in zone_stops[zone]:
            nontransitlegs.append((stop, zone))

    transitlegs = []
    for stop in stop_lines.keys():
        for line in stop_lines[stop]:
            transitlegs.append((stop, line))

    g = nx.Graph(transitlegs + nontransitlegs)

    return g


def path_matrix(zones, nodes, text):
    """

    :param zones:
    :param nodes:
    :param text:
    :return:

    display ODs that require more than 2 transfers:
    ::
        # using direct access to the lin attribute
        skims = lines.path_matrix_geometries['connections']
        to_shape = skims[skims['connections'] > 3]
        pandasshp.write_shp(sig_path + 'Q:/paths.shp', to_shape)

        # using path_matrix on it's own
        skimps = path_matrix(zones, nodes, text)
        to_shape = skims[skims['connections'] > 3]
        pandasshp.write_shp(sig_path + 'Q:/paths.shp', to_shape


    .. figure:: ./pictures/path_lin.png
        :width: 25cm
        :align: center
        :alt: path lin
        :figclass: align-center

        OD that require 2 transfers in Monterrey (Mexico)
    """

    g = connection_graph(zones, nodes, text)
    paths = nx.shortest_path(g)

    od_list = []
    for o in list(zones.index):
        for d in list(zones.index):
            try:
                od_list.append({'origin':o, 'destination':d, 'path_len':len(paths[o][d]), 'path':paths[o][d]})
            except:
                pass

    od = pd.DataFrame(od_list)
    od['lines'] = od['path'].apply(lambda path: [leg for leg in path if type(leg) == str])
    od['connections'] = od['lines'].apply(lambda l: len(l))

    return od


def _path_matrix(g, zones):
    paths = nx.shortest_path(g)

    od_list = []
    for o in list(zones.index):
        for d in list(zones.index):
            try:
                od_list.append({'origin':o, 'destination':d, 'path_len':len(paths[o][d]), 'path':paths[o][d]})
            except:
                pass

    od = pd.DataFrame(od_list)
    od['lines'] = od['path'].apply(lambda path: [leg for leg in path if type(leg) == str])
    od['connections'] = od['lines'].apply(lambda l: len(l))

    return od









