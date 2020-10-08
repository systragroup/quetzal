# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from quetzal.engine import engine,  connectivity, gps_tracks
from quetzal.engine.add_network import NetworkCaster
from quetzal.model import model, cubemodel

from syspy.skims import skims
from syspy.renumber import renumber

from tqdm import tqdm


def read_hdf(filepath):
    m = PreparationModel(hdf_database=filepath)
    return m


def read_json(folder):
    m = PreparationModel()
    m.read_json(folder)
    return m


track_args = model.track_args
log = model.log


class PreparationModel(model.Model, cubemodel.cubeModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 



    @track_args
    def preparation_footpaths(
        self, 
        road=False, 
        speed=3, 
        max_length=None, 
        n_clusters=None, 
        **kwargs
    ):
        """
            * requires: nodes
            * builds: footpaths

        """
        try:
            self.footpaths = connectivity.build_footpaths(
                self.nodes,
                speed=speed,
                max_length=max_length,
                n_clusters=n_clusters,
                coordinates_unit=self.coordinates_unit,
                **kwargs
            )
        except ValueError as e: #Shape of passed values is (1, 3019), indices imply (1, 5847)
            print('an error has occured: ', e)
            n_clusters = int(int(str(e).split('1, ')[1].split(')')[0])  * 0.9)
            print('now trying to run the method with n_cluster = ' + str(n_clusters))
            self.footpaths = connectivity.build_footpaths(
                self.nodes,
                speed=speed,
                max_length=max_length,
                n_clusters=n_clusters,
                coordinates_unit=self.coordinates_unit,
                **kwargs
            )

        if road:
            v = kwargs['speed'] * 1000 / 3600 # en m/s
            self.road_links['walk_time'] = self.road_links['length'] / v

    @track_args
    def preparation_ntlegs(
        self, 
        short_leg_speed=2, 
        long_leg_speed=10,
        threshold=1000,
        n_ntlegs=5, 
        max_ntleg_length=5000,
        zone_to_transit=True,
        zone_to_road=False
    ):
        """
        Builds the centroids and the ntlegs
            * requires: zones, nodes
            * builds: centroids, zone_to_transit, zone_to_road

        :param ntleg_speed: in km/h
        :param n_ntlegs: int
        :param max_ntleg_length: in m

        example:
        ::
            sm.step_ntlegs(
                n_ntlegs=5,
                walk_speed=2,
                short_leg_speed=3,
                long_leg_speed=15,
                threshold=500,
                max_ntleg_length=5000
            )
        """
        self.centroids = self.zones.copy()
        self.centroids['geometry'] = self.centroids['geometry'].apply(
            lambda g: g.centroid)

        length = max_ntleg_length

        if zone_to_transit:
            self.integrity_test_collision(sets=('nodes', 'zones'))
            ntlegs = engine.ntlegs_from_centroids_and_nodes(
                self.centroids,
                self.nodes,
                short_leg_speed=short_leg_speed,
                long_leg_speed=long_leg_speed,
                threshold=threshold,
                n_neighbors=n_ntlegs,
                coordinates_unit=self.coordinates_unit
            )
            self.zone_to_transit = ntlegs.loc[ntlegs['distance'] < length].copy()

        if zone_to_road:
            self.integrity_test_collision(sets=('road_nodes', 'zones'))

            ntlegs = engine.ntlegs_from_centroids_and_nodes(
                self.centroids,
                self.road_nodes,
                short_leg_speed=short_leg_speed,
                long_leg_speed=long_leg_speed,
                threshold=threshold,
                n_neighbors=n_ntlegs,
                coordinates_unit=self.coordinates_unit
            )
            self.zone_to_road = ntlegs.loc[ntlegs['distance'] < length].copy()

            ntlegs = engine.ntlegs_from_centroids_and_nodes(
                self.nodes,
                self.road_nodes,
                short_leg_speed=short_leg_speed,
                long_leg_speed=long_leg_speed,
                threshold=threshold,
                n_neighbors=n_ntlegs,
                coordinates_unit=self.coordinates_unit
            )

            self.road_to_transit = ntlegs.loc[ntlegs['distance'] < length].copy()
            

    @track_args
    def preparation_cast_network(
        self,
        nearest_method='nodes',
        weight='length',
        penalty_factor=1,
        speed=3,
        replace_nodes=False,
        dumb_cast=False,
        **nc_kwargs
    ):
        """
        Finds a path for the transport lines in an actual road network
            * requires: nodes, links, road_nodes, road_links
            * builds: links
        :param nearest_method: if 'links', looks for the nearest link to a stop
          in road_links and links the stop to its end_node (b). If 'node' looks
          for the actual nearest node in road_nodes.
        :param nodes_checkpoints: mandatory transit nodes
        :param penalty factor: ...
        :ng_kwargs: ...

        """


        if dumb_cast:
            nc = NetworkCaster(
                self.nodes, 
                self.links, 
                self.road_nodes
            )
            nc.dumb_cast()

        else:
            try :
                dump = self.road_links[weight] + 1
            except TypeError:
                raise TypeError(str(weight) + ' should be an int or a float')
 
            nc = NetworkCaster(
                self.nodes, 
                self.links, 
                self.road_nodes, 
                self.road_links,
                weight=weight
            )
            nc.build(
                nearest_method=nearest_method, 
                penalty_factor=penalty_factor,
                coordinates_unit=self.coordinates_unit,
                geometry=True,
                **nc_kwargs
            )

        self.networkcaster = nc
        self.links = nc.links

        if not dumb_cast:
            self.networkcaster_neighbors = nc.neighbors.reset_index(drop=True)
            self.networkcaster_road_access = nc.road_access.reset_index(drop=True)

        if replace_nodes:
            self.links[['a', 'b']] = self.links[['road_a', 'road_b']]
            self.links = self.links.loc[self.links['a'] != self.links['b']]
            self.nodes = self.road_nodes.loc[list(self.link_nodeset())].copy()
            self.road_to_transit = None

        # if we do not replace the nodes by the road_nodes,
        # we have to provide road to transit legs...
        elif not dumb_cast:  

            rc = nc.road_access['geometry'].reset_index()
            if self.coordinates_unit == 'degree':
                rc['length'] = skims.distance_from_geometry(rc['geometry'])
            elif self.coordinates_unit == 'meter':
                rc['length'] = rc['geometry'].apply(lambda x: x.length)
            else:
                raise('Invalid coordinates_unit.')

            rc['time'] = (rc['length'] / 1000 / speed * 3600)
            to_road = rc.rename(columns={'node': 'a', 'road_node': 'b'})
            from_road = rc.rename(columns={'node': 'b', 'road_node': 'a'})
            to_road['direction'] = 'to_road'
            from_road['direction'] = 'from_road'
            concatenated = pd.concat([from_road, to_road])
            self.road_to_transit = concatenated.reindex().reset_index(drop=True)

    @track_args
    def preparation_logit(
        self, 
        mode=1, 
        pt_mode=1, 
        pt_path=1, 
        segments=[],
        time=-1,
        price=-1,
        transfers=-1,
        time_shift=None # for time expanded model
    ):
        """
        * requires: 
        * builds: mode_utility, mode_nests, logit_scales, utility_values

        :param mode: phi parameter used in the logit choice between modes
        :param pt_mode: phi parameter used in the logit choice between pt modes
        :param pt_mode: phi parameter used in the logit choice between pt paths
        :param segments: the demand segments that will have to chose ex ['car', 'nocar']
        :param time: number of utility units by seconds
        :param price: number of utility units by currency unit
        :param transfers: number of utility units by transfer

        we should have 1 >= mode >= pt_mode >= pt_path > 0
        if the three of them are equal to 1 the nested logit will be equivalent to a flat logit
        """
        #TODO : move to preparation
        
        # utility values
        if time_shift is None:
            self.utility_values = pd.DataFrame(
                {'root': pd.Series( {'time': time, 'price': price, 'ntransfers': transfers, 'mode_utility': 1})}
            )
        else:
            self.utility_values = pd.DataFrame(
                {'root': pd.Series( {'time': time, 'price': price, 'ntransfers': transfers, 'mode_utility': 1, 'time_shift': time_shift})}
            )
        self.utility_values.index.name = 'value'
        self.utility_values.columns.name = 'segment'


        route_types = set(self.links['route_type'].unique()).union({'car', 'walk', 'root'})

        # mode_utility
        self.mode_utility = pd.DataFrame(
            {'root': pd.Series({rt: 0 for rt in route_types})}
        )

        self.mode_utility.index.name = 'route_type'
        self.mode_utility.columns.name = 'segment'


        # mode nests
        self.mode_nests =  pd.DataFrame(
            {'root': pd.Series({rt: 'pt' for rt in route_types})}
        )

        self.mode_nests.loc['pt', 'root'] = 'root'
        self.mode_nests.loc[['car', 'walk'], 'root'] = 'root'
        self.mode_nests.loc[['root'], 'root'] = np.nan
        self.mode_nests.index.name = 'route_type'
        self.mode_nests.columns.name = 'segment'

        # logit_scales
        self.logit_scales = self.mode_nests.copy()
        self.logit_scales['root'] = pt_path
        self.logit_scales.loc[['car', 'walk'], 'root'] = 0
        self.logit_scales.loc[['pt'], 'root'] = pt_mode
        self.logit_scales.loc[['root'], 'root'] = mode

        for segment in segments:
            for df in (self.mode_utility, self.mode_nests, self.logit_scales, self.utility_values):
                df[segment] = df['root']

    @track_args
    def preparation_clusterize_zones(
        self, max_zones=500, cluster_column=None,
        is_od_stack=False, **kwargs
        ):
        """
        clusterize zones
            * requires: zones, volumes
            * builds: zones, volumes, cluster_series
        """
        zones = self.zones
        zones['geometry'] = zones['geometry'].apply(lambda g: g.buffer(1e-9))
        self.micro_zones = zones.copy()
        self.micro_volumes = self.volumes.copy()
        if is_od_stack:
            self.micro_od_stack = self.od_stack.copy()
            self.zones, self.volumes, self.cluster_series, self.od_stack = renumber.renumber_quetzal(
                self.micro_zones,
                self.micro_volumes,
                self.micro_od_stack,
                max_zones,
                cluster_column,
                **kwargs
            )
        else:
            self.zones, self.volumes, self.cluster_series = renumber.renumber(
                self.micro_zones,
                self.micro_volumes,
                max_zones,
                cluster_column,
                **kwargs
            )
            
    @track_args
    def preparation_clusterize_nodes(self, n_clusters, **kwargs):
        """
        clusterize nodes
            * requires: nodes
            * builds: links, nodes
        """
        self.disaggregated_nodes = self.nodes.copy()
        if len(self.nodes) <= n_clusters:
            return
        
        self.links, self.nodes,  self.node_clusters, self.node_parenthood = connectivity.node_clustering(
        self.links, self.nodes, n_clusters, **kwargs)

        self.node_parenthood = self.node_parenthood[['cluster', 'geometry']]
        self.node_clusters['geometry'] = self.node_clusters[
            'geometry'
        ].apply(lambda g: g.buffer(1e-9))


    def preparation_map_tracks(
        self, 
        agg={'gps_speed': lambda s: s.mean() * 3.6}, 
        buffer=50, 
        smoothing_span=100, 
        *args, **kwargs
    ):
        # agg = ['mean', 'min', 'max', 'std', list] for extensive description of speeds
        to_concat = []
        iterator = tqdm(self.track_points['trip_id'].unique())
        for trip_id in iterator:
            iterator.desc = str(trip_id)
            points = self.track_points.loc[self.track_points['trip_id'] == trip_id]
            times = gps_tracks.get_times(
                points, 
                road_links=self.road_links, 
                buffer=buffer, 
                road_nodes=self.road_nodes, 
                smoothing_span=smoothing_span
            )
            times['trip_id'] = trip_id
            to_concat.append(times)
            
        # INDEX
        self.road_links.drop(['index'], axis=1, errors='ignore', inplace=True)
        indexed = self.road_links.reset_index().set_index(['a', 'b'])['index'].to_dict()
        concatenated = pd.concat(to_concat)
        concatenated['road_link'] = concatenated.apply(lambda r: indexed[(r['a'], r['b'])], axis=1) 
        aggregated = concatenated.groupby(['road_link'])['speed'].agg(agg)
        
        for c in aggregated.columns:
            self.road_links[c] = aggregated[c]
            
        self.track_links = concatenated.drop(['a', 'b'], axis=1)
