# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.park_and_ride_pathfinder import ParkRidePathFinder 
from quetzal.engine.road_pathfinder import RoadPathFinder
from quetzal.engine import nested_logit
from quetzal.model import model, preparationmodel, optimalmodel

from syspy.assignment import raw as raw_assignment
from syspy.assignment.raw import fast_assign as assign
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

class TransportModel(optimalmodel.OptimalModel):

    @track_args
    def step_distribution(
        self,
        deterrence_matrix=None,
        **od_volume_from_zones_kwargs
    ):
        """
        * requires: zones
        * builds: volumes

        :param deterrence_matrix: an OD unstaked dataframe representing the disincentive to 
            travel as distance/time/cost increases.
        :param od_volume_from_zones_kwargs: if the friction matrix is not
            provided, it will be automatically computed using a gravity distribution which
            uses the following parameters:
            * param power: (int) the gravity exponent
            * param intrazonal: (bool) set the intrazonal distance to 0 if False,
                compute a characteristic distance otherwise.
        """
        self.volumes = engine.od_volume_from_zones(
            self.zones,
            deterrence_matrix,
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
    def step_pr_pathfinder(
        self,
        force=False,
        path_analysis=True,
        **kwargs
    ):
        if not force:
            sets = ['nodes', 'links', 'zones', 'road_nodes', 'road_links']
            self.integrity_test_collision(sets)
        self.links = engine.graph_links(self.links)
        parkridepathfinder = ParkRidePathFinder(self)
        parkridepathfinder.find_best_path(**kwargs)

        self.pr_los = parkridepathfinder.paths
        
        if path_analysis:
            analysis_nodes = pd.concat([self.nodes, self.road_nodes])
            analysis_links = pd.concat([self.links, self.road_links])
            self.pr_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pr_los,
                links=self.links,
                nodes=analysis_nodes,
                centroids=self.centroids,
            ) # analyse non vérifiée, prise directement depuis pt_los

    @track_args
    def step_pt_pathfinder(
        self,
        broken_routes=True, 
        broken_modes=True, 
        route_column='route_id',
        mode_column='route_type',
        boarding_time=0,
        speedup=False,
        walk_on_road=False, 
        keep_graph=False,
        keep_pathfinder=False,
        force=False,
        path_analysis=True,
        **kwargs):
        """
        * requires: zones, links, footpaths, zone_to_road, zone_to_transit
        * builds: pt_los
        """
        sets = ['nodes', 'links', 'zones']
        if walk_on_road:
            sets += ['road_nodes', 'road_links']

        if not force:
            self.integrity_test_collision(sets)

        self.links = engine.graph_links(self.links)

        publicpathfinder = PublicPathFinder(self, walk_on_road=walk_on_road)
        publicpathfinder.find_best_paths(
            broken_routes=broken_routes, 
            broken_modes=broken_modes, 
            route_column=route_column,
            mode_column=mode_column,
            speedup=speedup,
            boarding_time=boarding_time,
            **kwargs
        )

        if keep_graph:
            self.nx_graph=publicpathfinder.nx_graph
        
        if keep_pathfinder:
            self.publicpathfinder = publicpathfinder
        
        self.pt_los = publicpathfinder.paths
        analysis_nodes = pd.concat([self.nodes, self.road_nodes]) if walk_on_road else self.nodes
        
        if path_analysis:
            self.pt_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pt_los,
                links=self.links,
                nodes=analysis_nodes,
                centroids=self.centroids,
            )

    @track_args
    def step_concatenate_los(self):
        """
        * requires: pt_los, car_los
        * builds: los
        """

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

    def compute_los_volume(self):
        segments = self.segments
        probabilities = [(segment, 'probability') for segment in segments] 
        on = ['origin', 'destination']
        left = self.los[on + probabilities]
        left['index'] = left.index
        df = pd.merge(left, self.volumes, on=on).set_index('index')
        values = df[probabilities].values * df[segments].values
        right = pd.DataFrame(values, index=df.index, columns=segments)
        for segment in segments:
            self.los[segment] = right[segment]
        self.los['volume'] = right.T.sum() 

    def step_assignment(
        self, 
        road=False, 
        boardings=False, 
        alightings=False, 
        transfers=False,
        segmented=False
        ):
        self.compute_los_volume()
        column = 'link_path'
        l = self.los.dropna(subset=[column])
        self.links['volume'] = assign(l['volume'], l[column])
        
        if road:
            self.road_links[('volume', 'car')] = assign(l['volume'], l[column])
            self.road_links[('volume', 'pt')] = assign(
                self.links['volume'], 
                self.links['road_link_list']
            )
            
        if boardings:
            column = 'boarding_links'
            l = self.los.dropna(subset=[column])
            self.links[ 'boardings'] = assign(l[segment], l[column])
            
            column = 'boardings'
            l = self.los.dropna(subset=[column])
            self.nodes['boardings'] = assign(l[segment], l[column])
            
        if alightings:
            column = 'alighting_links'
            l = self.los.dropna(subset=[column])
            self.links['alightings'] = assign(l[segment], l[column])
            
            column = 'alightings'
            l = self.los.dropna(subset=[column])
            self.nodes['alightings'] = assign(l[segment], l[column])
            
        if transfers:
            column = 'transfers'
            l = self.los.dropna(subset=[column])
            self.nodes[ 'transfers'] = assign(l[segment], l[column])
        if segmented:
            self.segmented_assigment(
                road=road, 
                boardings=boardings, alightings=alightings, transfers=transfers
            )
        
    def segmented_assigment(
        self, 
        road=False, 
        boardings=False, 
        alightings=False, 
        transfers=False
        ):
        
        for segment in self.segments:

            column = 'link_path'
            l = self.los.dropna(subset=[column])
            self.links[segment] = assign(l[segment], l[column])
            if road:
                self.road_links[(segment, 'car')] = assign(l[segment], l[column])
                self.road_links[(segment, 'pt')] = assign(
                    self.links[segment], 
                    self.links['road_link_list']
                )
            if boardings:
                column = 'boarding_links'
                l = self.los.dropna(subset=[column])
                self.links[(segment, 'boardings')] = assign(l[segment], l[column])

                column = 'boardings'
                l = self.los.dropna(subset=[column])
                self.nodes[(segment, 'boardings')] = assign(l[segment], l[column])

            if alightings:
                column = 'alighting_links'
                l = self.los.dropna(subset=[column])
                self.links[(segment, 'alightings')] = assign(l[segment], l[column])

                column = 'alightings'
                l = self.los.dropna(subset=[column])
                self.nodes[(segment, 'alightings')] = assign(l[segment], l[column])

            if transfers:
                column = 'transfers'
                l = self.los.dropna(subset=[column])
                self.nodes[(segment, 'transfers')] = assign(l[segment], l[column])

    @track_args
    def step_pt_assignment(
        self,
        volume_column=None,
        road=False,
        **kwargs
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

        if volume_column is None:
            self.segmented_pt_assignment(road=road, **kwargs)
            return 

        self.loaded_links, self.loaded_nodes = engine.loaded_links_and_nodes(
            self.links,
            self.nodes,
            volumes=self.volumes,
            path_finder_stack=self.pt_los,
            volume_column=volume_column,
            **kwargs
        )
        
        if road:
            self.road_links[volume_column] = raw_assignment.assign(
                volume_array=list(self.loaded_links[volume_column]), 
                paths=list(self.loaded_links['road_link_list'])
            )
            # todo remove 'load' from analysis module: 
            self.road_links['load'] = self.road_links[volume_column]

    def segmented_pt_assignment(self, *args, **kwargs):
        segments = self.segments
        index_columns = ['pathfinder_session', 'route_types', 'path']

        msg = 'duplicated paths'
        assert self.pt_los.duplicated(subset=index_columns).sum() == 0, msg


        iterator = tqdm(segments)
        for segment in iterator:
            iterator.desc = str(segment)
            self.step_pt_assignment(
                volume_column=segment,
                path_pivot_column=(segment, 'probability'),
                road=kwargs['road'],
                boardings=kwargs['boardings'],
                alightings=kwargs['alightings'],
                transfers=kwargs['transfers'],
            )

            for column in ['boardings', 'alightings', 'transfers']:
                try:
                    self.loaded_links[(segment, column)] = self.loaded_links[column]
                    self.loaded_nodes[(segment, column)] = self.loaded_nodes[column]
                except:
                    pass

            self.links = self.loaded_links
            self.nodes = self.loaded_nodes

        for column in ['boardings', 'alightings', 'transfers']:
            try:
                columns = [(segment, column) for segment in segments]
                self.loaded_links[column] = self.loaded_links[columns].T.sum()
                self.loaded_nodes[column] = self.loaded_nodes[columns].T.sum()
            except:
                pass
            
        self.loaded_links['all_pt'] = self.loaded_links[
            [segment for segment in segments]].T.sum()
        self.loaded_nodes['all_pt'] = self.loaded_nodes[
            [segment for segment in segments]].T.sum()

    
    def step_car_assignment(self, volume_column=None):
        """
        Assignment step
            * requires: road_links, car_los, road_links, volumes, path_probabilities
            * builds: loaded_road_links
        """
        if volume_column is None:
            self.segmented_car_assignment()

    def segmented_car_assignment(self):

        segments = self.segments
        iterator = tqdm(segments)
        for segment in iterator:
            iterator.desc = str(segment)
            merged = pd.merge(self.car_los, self.volumes, on=['origin', 'destination'])
            merged['to_assign'] = merged[(segment ,'probability')] * merged[segment].fillna(0)
            assigned = raw_assignment.assign(merged['to_assign'], merged['link_path']).fillna(0)
            self.road_links[(segment, 'car')] = assigned

        columns = [(segment, 'car') for segment in self.segments]
        self.road_links[('all', 'car')] = self.road_links[columns].T.sum()



    

    #TODO move all utility features to another object / file

    def analysis_mode_utility(self, how='min', segment=None, segments=None):
        """
        * requires: mode_utility, los, utility_values
        * builds: los
        """
        if segment is None:
            for segment in self.segments:
                print(segment)
                self.analysis_mode_utility(how=how, segment=segment)
            return 
        mode_utility = self.mode_utility[segment].to_dict()
        route_types = self.los['route_types'].unique()
        route_types = pd.DataFrame(route_types, columns=['route_types'])
        route_types['mode_utility'] = route_types['route_types'].apply(
            get_combined_mode_utility, how=how, mode_utility=mode_utility)
        
        route_types['rt_string'] = route_types['route_types'].astype(str)
        los = self.los.copy()
        los['rt_string'] = los['route_types'].astype(str)
        los['index'] = los.index
        
        merged = pd.merge(
            los[['rt_string', 'index']], 
            route_types[['rt_string', 'mode_utility']], 
            on=['rt_string'], 
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
        zones = list(self.zones.index)
        od = pd.DataFrame(index=pd.MultiIndex.from_product([zones, zones]))
        self.od_probabilities = od.copy()
        self.od_utilities = od.copy()

    def _unique_model_segmented_logit(self):
        # assert all logit scales are the same and pick one
        logit_scales = self.logit_scales.T.drop_duplicates().T
        assert len(logit_scales.columns) == 1
        logit_scales.columns = ['root']
        nls = logit_scales['root'].to_dict()

        # assert all mode_nests are the same and pick one
        mode_nests = self.mode_nests.T.drop_duplicates().T
        assert len(mode_nests.columns) == 1
        mode_nests.columns = ['root']
        nests = mode_nests.reset_index().groupby('root')['route_type'].agg(
            lambda s: list(s)).to_dict()

        # concatenate paths
        to_concat = []
        for segment in self.segments:
            paths = self.los.copy()
            paths['utility'] = paths[(segment, 'utility')]
            paths['segment'] = segment
            to_concat.append(paths[['origin', 'destination', 'route_type', 'utility', 'segment']])
        segmented_paths = pd.concat(to_concat)

        l, u, p = nested_logit.nested_logit_from_paths(
            segmented_paths, 
            mode_nests=nests,
            phi=nls,
            verbose=False
        )

        pool = l.reset_index().set_index(['segment', 'index'])
        for segment in self.segments:
            self.los[(segment, 'probability')] = pool.loc[segment]['probability']
        
        self.probabilities = p
        self.utilities = u

    def step_logit(self, segment=None, probabilities=None, utilities=None, *args, **kwargs):
        """
        * requires: mode_nests, logit_scales, los
        * builds: los, od_utilities, od_probabilities, path_utilities, path_probabilities
        """
        try:
            self._unique_model_segmented_logit()
            return 
        except AssertionError:
            print('segment specific logit models')
            pass

        if segment is None:
            probabilities = []
            utilities = []
            for segment in self.segments:
                self.step_logit(
                    segment=segment, 
                    probabilities=probabilities, utilities=utilities,
                    *args, **kwargs
                )
            self.od_probabilities = pd.concat(probabilities, axis=1)
            self.od_utilities = pd.concat(utilities, axis=1)
            return 

        mode_nests = self.mode_nests.reset_index().groupby(segment)['route_type'].agg(
            lambda s: list(s)).to_dict()
        nls = self.logit_scales[segment].to_dict()

        paths = self.los
        paths['utility'] = paths[(segment, 'utility')]

        l, u, p = nested_logit.nested_logit_from_paths(
            paths, 
            mode_nests=mode_nests,
            phi=nls,
            *args,
            **kwargs
        )

        l = l[['probability']]
        u = u.set_index(['origin', 'destination'])
        p = p.set_index(['origin', 'destination'])

        for df in l, u, p:
            df.columns = [(segment, c) for c in df.columns]

        self.los.drop(l.columns, axis=1, errors='ignore', inplace=True)
        self.los = pd.concat([self.los, l], axis=1)
        probabilities.append(p)
        utilities.append(u)


def get_combined_mode_utility(route_types, mode_utility,  how='min',):
    utilities = [mode_utility[mode] for mode in route_types]
    if not len(utilities):
        return 0
    if how=='min': # worse mode
        return min(utilities)
    elif how=='max': # best mode
        return max(utilities)
    elif how=='sum':
        return sum(utilities)
    elif how=='mean':
        return sum(utilities) / len(utilities)