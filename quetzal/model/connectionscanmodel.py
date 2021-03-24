import bisect
import ntpath
import pathlib
import re
import shutil
import time
import uuid

import numpy as np
import pandas as pd
import shapely
from quetzal.engine import csa, parallelization
from quetzal.model import timeexpandedmodel
from tqdm import tqdm

LIN_HEADER_LINE = 'LINE NAME="{}", ONEWAY=T,'


def read_hdf(filepath, *args, **kwargs):
    m = ConnectionScanModel(hdf_database=filepath, *args, **kwargs)
    return m


def read_zip(filepath, *args, **kwargs):
    try:
        m = ConnectionScanModel(zip_database=filepath, *args, **kwargs)
        return m
    except Exception:
        # the zip is a zipped hdf and can not be decompressed
        return read_zipped_hdf(filepath, *args, **kwargs)


def read_zipped_hdf(filepath, *args, **kwargs):
    filedir = ntpath.dirname(filepath)
    tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
    shutil.unpack_archive(filepath, tempdir)
    m = read_hdf(tempdir + r'/model.hdf', *args, **kwargs)
    shutil.rmtree(tempdir)
    return m


def read_json(folder):
    m = ConnectionScanModel(json_folder=folder)
    return m


def read_zippedpickles(folder, *args, **kwargs):
    m = ConnectionScanModel(zippedpickles_folder=folder, *args, **kwargs)
    return m


def get_alighting_links(link_path, link_trip_dict):
    try:
        if len(link_path) == 0:
            return []
    except TypeError:  # object of type 'float' has no len()
        return []
    alighting_links = []
    trip = link_trip_dict[link_path[0]]
    former_link = link_path[0]
    for link in link_path[0:]:
        link_trip = link_trip_dict[link]
        if link_trip != trip:
            trip = link_trip
            alighting_links.append(former_link)
        former_link = link
    alighting_links.append(link)
    return alighting_links


def links_to_nodes_departures_lin(trip_links: pd.DataFrame, link_columns: list = None) -> str:
    def add_link_columns_to_lin(link: dict, link_columns: list = None) -> str:
        result = ''
        if link_columns is not None:
            for link_col in link_columns:
                if link_col in link:
                    result += f", {link_col}={link[link_col]}"
        return result

    result = ''
    for idx, (_, link) in enumerate(trip_links.iterrows()):
        result += 'N=' if not result else ', N='
        result += f"{link['a']}, T={time.strftime('%H%M%S', time.gmtime(int(link['departure_time'])))}"
        result += add_link_columns_to_lin(link, link_columns)
        if idx == len(trip_links) - 1:
            result += f", N={link['b']}, T={time.strftime('%H%M%S', time.gmtime(int(link['arrival_time'])))}"
            result += add_link_columns_to_lin(link, link_columns)
    return result


def load_links_from_lin(lin_filepath: str) -> list:
    """
    Loads trips links from lin file format.
    Returns a list of trips with all links parameters.
    Ex.:
    [
        {
            'trip_id': '12345',
            'links': [{'a': 'node_a', 'departure_time': 55800, 'b': 'node_b', 'arrival_time': 55980', link_sequence': 1}, ...]
        }, ...
    ]
    """
    file_path = pathlib.Path(lin_filepath)
    result = []
    protected_value = r'"[^"]+"'
    any_word = r'[^, ]+'
    unprotected_value = fr'({any_word}( +{any_word})*)'
    regex = fr' *([^=]+) *= *({protected_value}|{unprotected_value}) *,'
    line_regex = r'N *='

    if file_path.exists():
        with open(lin_filepath, 'r') as file:
            lines = file.readlines()
        for line in lines:
            dict_to_add = {'trip_id': '', 'links': []}
            line_split = re.split(line_regex, line)
            if not line_split:
                continue
            line_split[-1] = line_split[-1].strip('\n')
            line_split[-1] += ','
            # Trip properties
            trip_properties = line_split[0]
            for m in re.finditer(regex, trip_properties):
                key, value = m.group(1).strip(), m.group(2).strip('"')
                if key == 'LINE NAME':
                    # trip_id
                    dict_to_add['trip_id'] = value
                else:
                    # Other trip attribute
                    dict_to_add[key] = value
            # Trip nodes and departures
            nodes_departures = {'N': [], 'T': [], 'attributes': []}
            trip_links = line_split[1:]
            for trip_link in trip_links:
                trip_link = 'N=' + trip_link
                trip_link_n, trip_link_t = None, None
                trip_link_other = {}
                for m in re.finditer(regex, trip_link):
                    key, value = m.group(1).strip(), m.group(2).strip('"').strip('\n')
                    if 'N' == key:
                        trip_link_n = value
                    elif 'T' == key:
                        try:
                            trip_link_t = f"{int(value):06}"
                        except ValueError:
                            trip_link_t = f"{0:06}"
                    else:
                        # Other link attribute
                        trip_link_other[key] = value
                if trip_link_n is None:
                    continue
                nodes_departures['N'].append(trip_link_n)
                nodes_departures['T'].append(trip_link_t)
                nodes_departures['attributes'].append(trip_link_other)
            # Build links with departure and arrival times
            crt_link = {}
            if len(nodes_departures['N']) != len(nodes_departures['T']):
                # Nb departures != Nb nodes, something is wrong
                continue
            link_sequence = 0
            for idx, node in enumerate(nodes_departures['N']):
                nb_seconds = None
                if nodes_departures['T'][idx] is not None:
                    tm = time.strptime(nodes_departures['T'][idx], '%H%M%S')
                    nb_seconds = tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec
                if idx == 0:
                    crt_link['a'] = node
                    crt_link['departure_time'] = nb_seconds
                    for other_attribute in nodes_departures['attributes'][idx]:
                        crt_link[other_attribute] = nodes_departures['attributes'][idx][other_attribute]
                else:
                    crt_link['b'] = node
                    crt_link['arrival_time'] = nb_seconds
                    link_sequence += 1
                    crt_link['link_sequence'] = link_sequence
                    dict_to_add['links'].append(crt_link)
                    if idx < len(nodes_departures['N']) - 1:
                        crt_link = {'a': node, 'departure_time': nb_seconds}
                        for other_attribute in nodes_departures['attributes'][idx]:
                            crt_link[other_attribute] = nodes_departures['attributes'][idx][other_attribute]
            result.append(dict_to_add)
    return result


class ConnectionScanModel(timeexpandedmodel.TimeExpandedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'time_interval' not in dir(self):
            self.time_interval = [0, 24 * 3600 - 1]

    def to_lin(self, trip_ids: list = None, link_columns: list = None, filepath: str = None) -> str:
        """
        Exports links in lin file format
        """
        links = self.links.copy()
        if trip_ids:
            trip_ids_str = list(map(str, trip_ids))
            links = links.loc[links['trip_id'].isin(trip_ids_str)]
        links = links.loc[links['a'] != links['b']].copy()
        unique_trip_ids = sorted(links.trip_id.unique())
        lin_buffer = ''
        for trip_id in tqdm(unique_trip_ids):
            trip_links = links.loc[links['trip_id'] == trip_id].sort_values('link_sequence', ascending=True)
            # TODO: add optional trip attributes ?
            current_header = LIN_HEADER_LINE.format(trip_id)
            lin_buffer += f'{current_header} {links_to_nodes_departures_lin(trip_links, link_columns)}\n'
        if filepath:
            with open(filepath, 'w') as file:
                file.write(lin_buffer)
        return lin_buffer

    def update_trips_from_lin(self, lin_filepath: str, trip_ids_to_remove: list = None, duplicates: list = None):
        """
        Update trips links from lin file.
        Optional: trip_ids_to_remove, duplicates parameters.
        trip_ids_to_remove: trip ids list. Ex.: ['12345', '67890']
        duplicates: list of tuples - [('trip_id', [<departure_time1>, <departure_time2>, ...]), ...]
        lin_filepath: path to lin file.
        """
        trips_links = load_links_from_lin(lin_filepath)
        for trip_links in trips_links:
            trip_links_df = pd.DataFrame(columns=self.links.columns)
            trip_id = trip_links['trip_id']
            for index, link in enumerate(trip_links['links']):
                trip_links_df.loc[index, 'trip_id'] = trip_id
                trip_links_df.loc[index, 'index'] = index
                for link_key in link:
                    if link_key in trip_links_df.columns:
                        trip_links_df.loc[index, link_key] = link[link_key]
            trip_links_df['time'] = trip_links_df['arrival_time'] - trip_links_df['departure_time']
            if all([link['a'] in self.nodes.index, link['b'] in self.nodes.index]):
                node_a = self.nodes.loc[link['a']]['geometry']
                node_b = self.nodes.loc[link['b']]['geometry']
                trip_links_df['geometry'] = shapely.geometry.LineString([node_a, node_b])
            self.links = pd.concat([self.links, trip_links_df], ignore_index=True)
        self.remove_trips(trip_ids=trip_ids_to_remove)
        self.duplicate_trips(duplicates)
        if trips_links:
            self.links.drop_duplicates(subset=['a', 'b', 'departure_time', 'arrival_time'], inplace=True, keep='last')
            self.links.rename('link_{}'.format, inplace=True)

    def remove_trips(self, trip_ids: list = None, fix_nodeset_consistency: bool = True):
        """
        Remove trip_ids in links
        """
        if trip_ids is None:
            return
        trip_ids_str = list(map(str, trip_ids))
        if trip_ids_str:
            self.links = self.links.loc[~self.links['trip_id'].isin(trip_ids_str)]
        if fix_nodeset_consistency:
            self.integrity_fix_nodeset_consistency()

    def duplicate_trips(self, duplicates: list = None):
        """
        Duplicate trips. duplicates parameter has the following format:
        [('trip_id', [<departure_time1>, <departure_time2>, ...]), ...]
        Example:
        [('trip_1', [18000, 19000]), ('trip_2', [17500])]
        It returns the number of duplicated trips.
        """
        if duplicates is None:
            return
        # Assertions
        assert(isinstance(duplicates, list))
        for trip_tuple in duplicates:
            assert(isinstance(trip_tuple, tuple))
            assert(len(trip_tuple) == 2)
            assert(isinstance(trip_tuple[1], list))
        # Build unique trip - departures from duplicates list
        trip_departures = {}
        for trip_tuple in duplicates:
            trip_id = str(trip_tuple[0])
            departures = trip_tuple[1]
            if trip_id not in trip_departures:
                trip_departures[trip_id] = set()
            for departure in departures:
                trip_departures[trip_id].add(departure)
        # Duplicate trips
        nb_duplicates = 0
        for trip_id in trip_departures:
            links_to_copy = self.links.loc[self.links['trip_id'] == trip_id].copy().sort_values('link_sequence', ascending=True)
            if links_to_copy.empty:
                continue
            for departure_time in trip_departures[trip_id]:
                # Compute time offset to apply to new links
                time_offset = departure_time - links_to_copy.iloc[0]['departure_time']
                for idx, (index, link) in enumerate(links_to_copy.iterrows()):
                    links_to_copy.loc[index, 'trip_id'] = f"{trip_id}_{departure_time}"
                    links_to_copy.loc[index, 'departure_time'] = link['departure_time'] + time_offset
                    links_to_copy.loc[index, 'arrival_time'] = link['arrival_time'] + time_offset
                    links_to_copy.loc[index, 'link_sequence'] = idx + 1
                self.links = pd.concat([self.links, links_to_copy], ignore_index=True)
                nb_duplicates += 1
        if trip_departures:
            self.links.drop_duplicates(subset=['a', 'b', 'departure_time', 'arrival_time'], inplace=True, keep='last')
            self.links.rename('link_{}'.format, inplace=True)
        return nb_duplicates

    def lighten_pt_los(self):
        super(ConnectionScanModel, self).lighten_pt_los()
        to_drop = [
            'connection_path', 'path', 'first_connections',
            'last_connection', 'ntransfers'
        ]
        self.pt_los = self.pt_los.drop(to_drop, axis=1, errors='ignore')

    def lighten_pseudo_connections(self):
        self.pseudo_connections = self.pseudo_connections[
            ['csa_index', 'link_index', 'model_index', 'trip_id']
        ]

    def lighten(self):
        super(ConnectionScanModel, self).lighten()
        try:
            self.lighten_pseudo_connections()
        except AttributeError:
            pass

    def preparation_build_connection_dataframe(
        self, min_transfer_time=0,
        time_interval=None, cutoff=np.inf,
        reindex=True,
    ):
        time_interval = time_interval if time_interval is not None else self.time_interval

        links = self.links.loc[self.links['a'] != self.links['b']]
        links = links.loc[links['departure_time'] >= time_interval[0]]
        links = links.loc[links['departure_time'] <= time_interval[1] + cutoff]

        try:
            links = pd.merge(
                links, self.nodes[['transfer_duration']],
                how='left', left_on='a',
                right_index=True
            )
            links['min_transfer_time'] = links['transfer_duration'].fillna(0) + min_transfer_time
        except KeyError:
            links['min_transfer_time'] = min_transfer_time

        pseudo_links = links[
            [
                'a', 'b', 'departure_time', 'arrival_time',
                'min_transfer_time', 'trip_id', 'link_sequence',
            ]
        ].copy()
        pseudo_links['link_index'] = pseudo_links['model_index'] = pseudo_links.index
        pseudo_links['actual_departure_time'] = pseudo_links['departure_time']
        pseudo_links['departure_time'] -= links['min_transfer_time']
        zone_to_transit = csa.time_zone_to_transit(pseudo_links, self.zone_to_transit)
        footpaths = csa.time_footpaths(pseudo_links, self.footpaths)
        self.time_expended_footpaths = footpaths
        self.time_expended_zone_to_transit = zone_to_transit

        pseudo_connections = pd.concat([pseudo_links, footpaths, zone_to_transit])
        pseudo_connections = pseudo_connections[pseudo_connections['a'] != pseudo_connections['b']]

        # connections of each trip are consecutive
        pseudo_connections.sort_values(by=['trip_id', 'link_sequence'], inplace=True)
        if reindex:
            pseudo_connections['csa_index'] = range(len(pseudo_connections))  # use int as index
        else:
            pseudo_connections['csa_index'] = pseudo_connections['csa_index'].fillna(
                pseudo_connections['model_index']
            )
        pseudo_connections['actual_departure_time'].fillna(
            pseudo_connections['departure_time'], inplace=True)
        pseudo_connections.sort_values('actual_departure_time', ascending=False, inplace=True)
        self.pseudo_connections = pseudo_connections

    def step_pt_pathfinder(
        self,
        min_transfer_time=0,
        time_interval=None,
        cutoff=np.inf,
        build_connections=True,
        targets=None,
        od_set=None,
        workers=1,
        reindex=True
    ):
        time_interval = time_interval if time_interval is not None else self.time_interval

        if build_connections:
            self.preparation_build_connection_dataframe(
                min_transfer_time=min_transfer_time,
                time_interval=time_interval,
                cutoff=cutoff,
                reindex=reindex,
            )
        seta = set(self.time_expended_zone_to_transit['a'])
        setb = set(self.time_expended_zone_to_transit['b'])
        zone_set = set(self.zones.index).intersection(seta).intersection(setb)
        targets = zone_set if targets is None else set(targets).intersection(zone_set)

        self.pt_los = csa.pathfinder(
            time_expended_zone_to_transit=self.time_expended_zone_to_transit,
            pseudo_connections=self.pseudo_connections,
            zone_set=zone_set,
            targets=targets,
            od_set=od_set,
            min_transfer_time=min_transfer_time,
            time_interval=time_interval,
            cutoff=cutoff,
            workers=workers
        )

    def analysis_paths(
        self, workers=1,
        alighting_links=True, alightings=True,
        keep_connection_path=False
    ):
        pseudo_connections = self.pseudo_connections
        clean = pseudo_connections[['csa_index', 'trip_id']].dropna()
        clean.sort_values(by='csa_index', inplace=True)
        trip_connections = {}
        for trip, connection in clean[['trip_id', 'csa_index']].values:
            try:
                trip_connections[trip].append(connection)
            except KeyError:
                trip_connections[trip] = [connection]

        connection_trip = clean.set_index('csa_index')['trip_id'].to_dict()
        df = self.pt_los
        paths = list(df['csa_path'])
        kwargs = {
            'trip_connections': trip_connections,
            'connection_trip': connection_trip,
        }
        values = parallelization.parallel_map_kwargs(
            csa.path_to_boarding_links_and_boarding_path,
            paths, workers=workers, show_progress=True, **kwargs

        )
        df['connection_path'] = [v[0] for v in values]
        df['first_connections'] = [v[1] for v in values]
        del values

        # model
        pool = pseudo_connections[['model_index', 'csa_index']].dropna()
        d = pool.set_index('csa_index')['model_index'].to_dict()
        df['path'] = [
            [origin] + [d[i] for i in p if i in d] + [destination]
            for origin, destination, p in
            df[['origin', 'destination', 'connection_path']].values
        ]

        # links
        pool = pseudo_connections[['link_index', 'csa_index']].dropna()
        d = pool.set_index('csa_index')['link_index'].to_dict()
        df['link_path'] = [
            [d[i] for i in p if i in d]
            for p in df['connection_path']
        ]
        if not keep_connection_path:
            del df['connection_path']

        df['boarding_links'] = [
            [d[i] for i in p if i in d]
            for p in df['first_connections']
        ]
        if not keep_connection_path:
            del df['first_connections']

        df['ntransfers'] = df['boarding_links'].apply(lambda b: len(b) - 1)
        df['ntransfers'] = np.clip(df['ntransfers'], 0, a_max=None)
        linka = self.links['a'].to_dict()
        df['boardings'] = [[linka[b] for b in bl] for bl in df['boarding_links']]

        if alighting_links or alightings:
            link_trip_dict = self.links['trip_id'].to_dict()

            df['alighting_links'] = df['link_path'].apply(
                get_alighting_links, link_trip_dict=link_trip_dict
            )
        if alightings:
            linkb = self.links['b'].to_dict()
            df['alightings'] = [[linkb[a] for a in al] for al in df['alighting_links']]
        self.pt_los = df
