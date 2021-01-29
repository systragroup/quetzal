
from quetzal.model import timeexpandedmodel
from quetzal.engine import csa
import pandas as pd
import numpy as np
import bisect
import shutil
import ntpath
import uuid
from tqdm import tqdm
from quetzal.engine import parallelization

def read_hdf(filepath, *args, **kwargs):
    m = ConnectionScanModel(hdf_database=filepath, *args, **kwargs)
    return m


def read_zip(filepath, *args, **kwargs):
    try:
        m = ConnectionScanModel(zip_database=filepath, *args, **kwargs)
        return m
    except:
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


class ConnectionScanModel(timeexpandedmodel.TimeExpandedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'time_interval' not in dir(self):
            self.time_interval = [0, 24*3600-1]

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
        time_interval=None, cutoff=np.inf
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
            ['a', 'b', 'departure_time', 'arrival_time',
            'min_transfer_time', 'trip_id', 'link_sequence',
            ]
        ].copy()
        pseudo_links['link_index'] = pseudo_links['model_index'] = pseudo_links.index
        pseudo_links['departure_time'] -= links['min_transfer_time']
        zone_to_transit = csa.time_zone_to_transit(pseudo_links, self.zone_to_transit)
        footpaths = csa.time_footpaths(pseudo_links, self.footpaths)
        self.time_expended_footpaths = footpaths
        self.time_expended_zone_to_transit = zone_to_transit
        
        pseudo_connections = pd.concat([pseudo_links, footpaths, zone_to_transit])
        pseudo_connections = pseudo_connections[pseudo_connections['a'] != pseudo_connections['b']]

        # connections of each trip are consecutive
        pseudo_connections.sort_values(by=['trip_id', 'link_sequence'], inplace=True)
        pseudo_connections['csa_index'] = range(len(pseudo_connections)) # use int as index
        pseudo_connections.sort_values('departure_time', ascending=False, inplace=True)
        
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
    ):

        time_interval = time_interval if time_interval is not None else self.time_interval

        if build_connections:
            self.preparation_build_connection_dataframe(
                min_transfer_time=min_transfer_time
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
        
    def analysis_paths(self, workers=1):
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
        del df['connection_path']

        df['boarding_links'] = [
            [d[i] for i in p if i in d] 
            for p in df['first_connections']
        ]
        del df['first_connections']

        df['ntransfers'] = df['boarding_links'].apply(lambda b: len(b)-1)
        df['ntransfers'] = np.clip(df['ntransfers'], 0, a_max=None)
        linka = self.links['a'].to_dict()
        df['boardings'] = [[linka[b] for b in bl] for bl in df['boarding_links']]

        self.pt_los = df

    

