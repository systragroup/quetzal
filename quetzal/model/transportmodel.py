# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.road_pathfinder import RoadPathFinder
from quetzal.engine import nested_logit
from quetzal.model import model, preparationmodel

from syspy.assignment import raw as raw_assignment
from syspy.skims import skims
from tqdm import tqdm
import networkx as nx


def read_hdf(filepath):
    m = TransportModel()
    m.read_hdf(filepath)
    return m


def read_json(folder):
    m = TransportModel()
    m.read_json(folder)
    return m

track_args = model.track_args
log = model.log

class TransportModel(preparationmodel.PreparationModel):

    @track_args
    def step_distribution(
        self,
        impedance_matrix=None,
        **od_volume_from_zones_kwargs
    ):
        """
        * requires: zones
        * builds: volumes

        :param impedance_matrix: an OD unstaked friction dataframe
            used to compute the distribution.
        :param od_volume_from_zones_kwargs: if the friction matrix is not
            provided, it will be automatically computed using a gravity distribution which
            uses the following parameters:
            * param power: (int) the gravity exponent
            * param intrazonal: (bool) set the intrazonal distance to 0 if False,
                compute a characteristic distance otherwise.
        """
        self.volumes = engine.od_volume_from_zones(
            self.zones,
            impedance_matrix,
            coordinates_unit=self.coordinates_unit,
            **od_volume_from_zones_kwargs
        )

    @track_args
    def step_pathfinder(
        self,
        walk_on_road=False,
        complete=True,
        **kwargs
    ):
        """
        * requires: links, footpaths, zone_to_transit, zone_to_road
        * builds: pt_los
        """

        assert self.links['time'].isnull().sum() == 0

        self.links = engine.graph_links(self.links)
        self.walk_on_road = walk_on_road

        if walk_on_road:
            footpaths = self.road_links.copy()
            footpaths['time'] = footpaths['walk_time']
            ntlegs = self.zone_to_road
            nodes = self.road_nodes
        else: 
            footpaths = self.footpaths
            ntlegs = self.zone_to_transit
            nodes = self.nodes

        #TODO even with walk on road, transit nodes may not belong to road_nodes
        self.pt_los, self.graph = engine.path_and_duration_from_links_and_ntlegs(
            self.links,
            ntlegs=ntlegs,
            pole_set=set(self.zones.index),
            footpaths=footpaths,
            **kwargs
        )

        if complete:        
            self.pt_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pt_los, 
                links=self.links,
                nodes=nodes,
                centroids=self.centroids,
            )
  
    @track_args
    def step_road_pathfinder(self, maxiters=1, *args, **kwargs):
        """
        * requires: zones, road_links, zone_to_road
        * builds: car_los, road_links
        """
        roadpathfinder = RoadPathFinder(self)
        roadpathfinder.frank_wolfe(maxiters=maxiters, *args, **kwargs)
        self.car_los = roadpathfinder.car_los
        self.road_links = roadpathfinder.road_links

    @track_args
    def step_pt_pathfinder(
        self,
        broken_routes=True, 
        broken_modes=True, 
        route_column='route_id',
        mode_column='route_type',
        speedup=False, 
        **kwargs):
        """
        * requires: zones, links, footpaths, zone_to_road, zone_to_transit
        * builds: pt_los
        """
        self.links = engine.graph_links(self.links)
        publicpathfinder = PublicPathFinder(self)
        publicpathfinder.find_best_paths(
            broken_routes=broken_routes, 
            broken_modes=broken_modes, 
            route_column=route_column,
            mode_column=mode_column,
            speedup=speedup,
            **kwargs
        )
        self.pt_los = publicpathfinder.paths.copy()
        
        self.pt_los = analysis.path_analysis_od_matrix(
            od_matrix=self.pt_los,
            links=self.links,
            nodes=self.nodes,
            centroids=self.centroids,
        )

    @track_args
    def step_concatenate_los(self):
        """
        * requires: pt_los, car_los
        * builds: los
        """

    @track_args
    def step_build_los(
         self,
         build_car_skims=True,
         token=None,
         nb_clusters=20,
         skim_matrix_kwargs={}
        ):
        """
        * requires: pt_los, car_los
        * builds: los

        :param build_car_skims: if True, the car_los matrix is build using
            Google API (if a valid token is given, a random matrix is
            generated otherwise). If False the current car_los matrix is used.
        :param token: a token or list of tokens
        :param nb_clusters: the number of clusters that will be build from the
            zoning. A single token allows only 2500 itineraries so 50 zones.
        """
        if build_car_skims:
            skim_columns = [
                'origin', 'destination', 'euclidean_distance',
                'distance', 'duration'
            ]
            self.car_los = skims.skim_matrix(
                self.zones,
                token,
                nb_clusters,
                coordinates_unit=self.coordinates_unit,
                skim_matrix_kwargs=skim_matrix_kwargs
            )[skim_columns]

        self.los = pd.merge(  # Weird: we lose the los for which one of the mode is missing?
            self.car_los,
            self.pt_los,
            on=['origin', 'destination'],
            suffixes=['_car', '_pt']
        )

    @track_args
    def step_modal_split(self, build_od_stack=True, **modal_split_kwargs):
        """
        * requires: volumes, los
        * builds: od_stack, shared

        :param modal_split_kwargs: kwargs of engine.modal_split

        example:
        ::
            sm.step_modal_split(
                time_scale=1/1800,
                alpha_car=2,
                beta_car=600
            )
        """
        shared = engine.modal_split_from_volumes_and_los(
            self.volumes,
            self.los,
            **modal_split_kwargs
        )
        # shared['distance_car'] = shared['distance']
        if build_od_stack:
            self.od_stack = analysis.volume_analysis_od_matrix(shared)

        self.shared = shared

    @track_args
    def step_pt_assignment(
        self,
        volume_column='volume_pt',
        road=False,
        **loaded_links_and_nodes_kwargs
        ):
        """
        Assignment step
            * requires: links, nodes, pt_los, road_links, volumes, path_probabilities
            * builds: loaded_links, loaded_nodes, loaded_road_links

        :param loaded_links_and_nodes_kwargs: kwargs of engine.loaded_links_and_nodes

        example:
        ::
            sm.step_assignment(
                loaded_links_and_nodes_kwargs={
                    'boardings': True,
                    'alightings': True,
                    'transfers': True
                }
            )
        """

        self.loaded_links, self.loaded_nodes = engine.loaded_links_and_nodes(
            self.links,
            self.nodes,
            volumes=self.volumes,
            path_finder_stack=self.pt_los,
            volume_column=volume_column,
            **loaded_links_and_nodes_kwargs
        )
        
        if road:
            self.road_links[volume_column] = raw_assignment.assign(
                volume_array=list(self.loaded_links[volume_column]), 
                paths=list(self.loaded_links['road_link_list'])
            )
            # todo remove 'load' from analysis module: 
            self.road_links['load'] = self.road_links[volume_column]

    #TODO move all utility features to another object / file
    def build_logit_parameters(
        self, 
        mode=1, 
        pt_mode=1, 
        pt_path=1, 
        segments=[],
        time=-1,
        price=-1,
        transfers=-1
    ):
        """
        * requires: 
        * builds: mode_utility, mode_nests, logit_scales, utility_values
        
        """
        #TODO : move to preparation
        
        # utility values
        self.utility_values = pd.DataFrame(
            {'root': pd.Series( {'time':time,'price':price,'ntransfers':transfers,'mode_utility':1})}
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
        self.logit_scales.loc[['car', 'walk'], 'root'] = 0.01
        self.logit_scales.loc[['pt'], 'root'] = pt_mode
        self.logit_scales.loc[['root'], 'root'] = mode

        for segment in segments:
            for df in (self.mode_utility, self.mode_nests, self.logit_scales, self.utility_values):
                df[segment] = df['root']

    def analysis_mode_utility(self, how='min', segment='root'):
        """
        * requires: mode_utility, los, utility_values
        * builds: los
        """
        mode_utility = self.mode_utility[segment].to_dict()
        route_types = self.los['route_types'].unique()
        route_types = pd.DataFrame(route_types, columns=['route_types'])
        route_types['mode_utility'] = route_types['route_types'].apply(
            get_combined_mode_utility, how=how, mode_utility=mode_utility)
        
        los = self.los.copy()
        los['index'] = los.index
        merged = pd.merge(
            los, 
            route_types, 
            on=['route_types'], 
        ).set_index('index')
        
        los['mode_utility'] = merged['mode_utility']
        
        utility_values = self.utility_values[segment].to_dict()
        u = 0
        for key, value in utility_values.items():
            u += value * los[key]
            
        self.los[(segment, 'utility')] = u
        
    def analysis_utility(self, segment='root'):
        utility_values = self.utility_values[segment].to_dict()
        u = 0
        for key, value in utility_values.items():
            u += value * self.los[key]
        self.los[(segment, 'utility')] = u

    def initialize_logit(self):
        od = pd.DataFrame(
            index=self.volumes.set_index(['origin', 'destination']).index
        )
        self.od_probabilities = od.copy()
        self.od_utilities = od.copy()

    def step_logit(self, segment='root', *args, **kwargs):
        """
        * requires: mode_nests, logit_scales, los
        * builds: los, od_utilities, od_probabilities, path_utilities, path_probabilities
        """
        
        mode_nests = self.mode_nests.reset_index().groupby(segment)['route_type'].agg(
            lambda s: list(s)).to_dict()
        nls = self.logit_scales[segment].to_dict()
        
        paths = self.los.copy()
        paths['utility'] = paths[(segment, 'utility')]
        
        l, u, p = nested_logit.nested_logit_from_paths(
            paths, 
            mode_nests=mode_nests,
            phi=nls,
            *args,
            **kwargs
        )
        
        l = l[['utility', 'probability']]
        u = u.set_index(['origin', 'destination'])
        p = p.set_index(['origin', 'destination'])
        
        for df in l, u, p:
            df.columns = [(segment, c) for c in df.columns]
        
        self.los[l.columns] = l 
        self.od_utilities[u.columns] = u
        self.od_probabilities[p.columns] = p


def get_combined_mode_utility(route_types, mode_utility,  how='min',):
    utilities = [mode_utility[mode] for mode in route_types]
    if not len(utilities):
        return 0
    if how=='min': # worse mode
        return min(utilities)
    elif how=='max': # best mode
        max(utilities)
    elif how=='sum':
        sum(utilities)
    elif how=='mean':
        sum(utilities) / len(utilities)