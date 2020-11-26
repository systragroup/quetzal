# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.park_and_ride_pathfinder import ParkRidePathFinder 
from quetzal.engine.road_pathfinder import RoadPathFinder
from quetzal.engine import nested_logit
from quetzal.model import model, optimalmodel, parkridemodel

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


class TransportModel(optimalmodel.OptimalModel, parkridemodel.ParkRideModel):

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
        boarding_time=None,
        speedup=False,
        walk_on_road=False, 
        # keep_graph=False,
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

        # if keep_graph:
        #     self.nx_graph=publicpathfinder.nx_graph
        
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

    def compute_los_volume(self, time_expanded=False):

        los = self.los if not time_expanded else self.te_los
    
        segments = self.segments
        probabilities = [(segment, 'probability') for segment in segments]

        shared_cols = list(set(self.volumes.columns).intersection(set(los.columns)))
        on = [col for col in shared_cols if col in ['origin', 'destination', 'wished_departure_time']]

        left = los[on + probabilities]
        left['index'] = left.index
        df = pd.merge(left, self.volumes, on=on).set_index('index')
        values = df[probabilities].values * df[segments].values
        right = pd.DataFrame(values, index=df.index, columns=segments)
        for segment in segments:
            los[segment] = right[segment]
        los['volume'] = right.T.sum() 

        if time_expanded:
            los_volumes = self.te_los.groupby('path_id')[['volume'] + segments].sum()
            self.los.drop(['volume'] + segments, axis=1, inplace=True, errors='ignore')
            self.los = self.los.merge(los_volumes, on='path_id')

    def step_assignment(
        self, 
        road=False, 
        boardings=False, 
        alightings=False,
        transfers=False,
        segmented=False,
        time_expanded=False
        ):
        self.compute_los_volume(time_expanded=time_expanded)
        los = self.los.copy()

        column = 'link_path'
        l = los.dropna(subset=[column])
        l = los.loc[los['volume'] > 0]
        self.links['volume'] = assign(l['volume'], l[column])
        
        if road:
            self.road_links[('volume', 'car')] = assign(l['volume'], l[column])
            to_assign = self.links.dropna(subset=['volume'])
            self.road_links[('volume', 'pt')] = assign(
                to_assign['volume'], 
                to_assign['road_link_list']
            )
            
        if boardings:
            column = 'boarding_links'
            l = los.dropna(subset=[column])
            self.links[ 'boardings'] = assign(l['volume'], l[column])
            
            column = 'boardings'
            l = los.dropna(subset=[column])
            self.nodes['boardings'] = assign(l['volume'], l[column])
            
        if alightings:
            column = 'alighting_links'
            l = los.dropna(subset=[column])
            self.links['alightings'] = assign(l['volume'], l[column])
            
            column = 'alightings'
            l = los.dropna(subset=[column])
            self.nodes['alightings'] = assign(l['volume'], l[column])
            
        if transfers:
            column = 'transfers'
            l = los.dropna(subset=[column])
            self.nodes[ 'transfers'] = assign(l['volume'], l[column])

        if segmented:
            self.segmented_assigment(
                road=road, 
                boardings=boardings, alightings=alightings, transfers=transfers,
                aggregated_los=los
            )
        
    def segmented_assigment(
        self, 
        road=False, 
        boardings=False, 
        alightings=False, 
        transfers=False,
        aggregated_los=None
        ):
        los = aggregated_los if aggregated_los is not None else self.los
    
        for segment in self.segments:

            column = 'link_path'
            l = los.dropna(subset=[column])
            self.links[segment] = assign(l[segment], l[column])
            if road:
                self.road_links[(segment, 'car')] = assign(l[segment], l[column])
                self.road_links[(segment, 'pt')] = assign(
                    self.links[segment], 
                    self.links['road_link_list']
                )
            if boardings:
                column = 'boarding_links'
                l = los.dropna(subset=[column])
                self.links[(segment, 'boardings')] = assign(l[segment], l[column])

                column = 'boardings'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'boardings')] = assign(l[segment], l[column])

            if alightings:
                column = 'alighting_links'
                l = los.dropna(subset=[column])
                self.links[(segment, 'alightings')] = assign(l[segment], l[column])

                column = 'alightings'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'alightings')] = assign(l[segment], l[column])

            if transfers:
                column = 'transfers'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'transfers')] = assign(l[segment], l[column])

    @track_args
    def step_pt_assignment(
        self,
        volume_column=None,
        on_road_links=False,
        split_by=None,
        **kwargs
        ):
        """
        Assignment step
            * requires: links, nodes, pt_los, road_links, volumes, path_probabilities
            * builds: loaded_links, loaded_nodes, add load to road_links

        :param volume_column: volume column of self.volumes to assign. If none, all columns will be assigned
        :param on_road_links: if True, performs pt assignment on road_links as well
        :param split_by: path categories to be tracked in the assignment. Must be a column of self.pt_los

        example:
        ::
            sm.step_assignment(
                    volume_column=None,
                    on_road_links=False,
                    split_by='route_type',
                    boardings=True,
                    alightings=True,
                    transfers=True
                }
            )
        """

        if volume_column is None:
            self.segmented_pt_assignment(
                on_road_links=on_road_links,
                split_by=split_by,
                **kwargs
            )
            return 

        # When split_by is not None, this call could be replaced by a sum, provided 
        # prior dumb definition of loaded_links and loaded_nodes 
        self.loaded_links, self.loaded_nodes = engine.loaded_links_and_nodes(
            self.links,
            self.nodes,
            volumes=self.volumes,
            path_finder_stack=self.pt_los,
            volume_column=volume_column,
            **kwargs
        )

        # Rename columns
        self.loaded_links.rename(columns={volume_column: ('load', volume_column)}, inplace=True)
        self.loaded_nodes.rename(columns={volume_column: ('load', volume_column)}, inplace=True)
        for col in list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys())):
            self.loaded_links.rename(columns={col: (col, volume_column)}, inplace=True)
            self.loaded_nodes.rename(columns={col: (col, volume_column)}, inplace=True)

        # Group assignment
        if split_by is not None:
            groups = self.pt_los[split_by].unique()
            for group in groups:
                # TODO remove rows with empty link_path
                group_pt_los = self.pt_los.loc[self.pt_los[split_by]==group]
                group_loaded_links, group_loaded_nodes = engine.loaded_links_and_nodes(
                    self.links,
                    self.nodes,
                    volumes=self.volumes,
                    path_finder_stack=group_pt_los,
                    volume_column=volume_column,
                    **kwargs
                )
                # Append results columns
                self.loaded_links[('load', volume_column, group)] = group_loaded_links[volume_column]
                self.loaded_nodes[('load', volume_column, group)] = group_loaded_nodes[volume_column]
                for col in list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys())):
                    self.loaded_links[(col, volume_column, group)] = group_loaded_links[col]
                    self.loaded_nodes[(col, volume_column, group)] = group_loaded_nodes[col]

        # Assignment on road_links
        if on_road_links:
            if not 'road_link_path' in self.pt_los.columns:
                # create road_link_path column from networkcasted linkss if not already defined
                self._analysis_road_link_path()

            merged = pd.merge(self.pt_los, self.volumes, on=['origin', 'destination'])
            merged['to_assign'] = merged[(volume_column ,'probability')] * merged[volume_column].fillna(0)
            
            if split_by is not None:
                def assign_group(g):
                    x = g.reset_index()
                    result = raw_assignment.assign(x['to_assign'], x['road_link_path'])
                    return result
                group_assigned = merged.groupby(split_by).apply(assign_group)
                assigned = group_assigned.unstack().T.loc['volume'].fillna(0)
                # Add empty groups
                for empty in list(set(groups).difference(set(assigned.columns))):
                    assigned[empty] = 0
                self.road_links[[(volume_column, col) for col in groups]] = assigned[[col for col in groups]]
                self.road_links[volume_column] =  assigned.T.sum()

            else: # no groups
                assigned = raw_assignment.assign(merged['to_assign'], merged['road_link_path'])
                self.road_links[volume_column] = assigned['volume']

            # todo remove 'load' from analysis module: 
            self.road_links['load'] = self.road_links[volume_column]

    def segmented_pt_assignment(self, split_by=None, on_road_links=False, *args, **kwargs):
        """
        Performs pt assignment for all demand segments.
        Requires computed path probabilities in pt_los for each segment.
        """
        segments = self.segments
	# Merge conflict: should we modify pt_los here 
	# or let the modeller check
        # index_columns = ['pathfinder_session', 'route_types', 'path']
        # for segment in segments:
            # self.pt_los.drop((segment, 'probability'), axis=1, inplace=True, errors='ignore')
            # right = self.los.set_index(index_columns)[[(segment, 'probability')]]
            # msg = 'self.los cannot have several rows with the same combination of ' 
            # msg += str(index_columns)
            # assert right.index.is_unique, msg
            # self.pt_los = pd.merge(
            #    self.pt_los, 
            #     right,
            #     left_on=index_columns,
            #     right_index=True
            # )

        iterator = tqdm(segments)
        for segment in iterator:
            iterator.desc = str(segment)
            # Assign demand segment
            self.step_pt_assignment(
                volume_column=segment,
                path_pivot_column=(segment, 'probability'),
                split_by=split_by,
                on_road_links=on_road_links,
                **kwargs
            )
            # Update links and nodes to keep results as loaded links and nodes
            # are erased at each call of step_pt_assignment
            self.links = self.loaded_links
            self.nodes = self.loaded_nodes

        # Group assignment results: sum over demand segments
        try:
            groups = self.pt_los[split_by].unique()
        except KeyError as e:
            groups = []
        cols = ['load']
        # Add boardings, alightings and transfers if processed
        cols += list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys()))
        for col in cols:
            for g in groups:
                columns = [tuple([col, s, g]) for s in segments]
                name = tuple([col, g])
                self.loaded_links[name] = self.loaded_links[columns].T.sum()
                self.loaded_links.drop(columns, 1, inplace=True)

                self.loaded_nodes[name] = self.loaded_nodes[columns].T.sum()
                self.loaded_nodes.drop(columns, 1, inplace=True)

            columns = [tuple([col, s]) for s in segments]
            self.loaded_links[col] = self.loaded_links[columns].T.sum()
            self.loaded_links.drop(columns, 1, inplace=True)

            self.loaded_nodes[col] = self.loaded_nodes[columns].T.sum()
            self.loaded_nodes.drop(columns, 1, inplace=True)

        if on_road_links:
            for group in groups:
                self.road_links[('all', group)] = self.road_links[[(s, group) for s in segments]].T.sum()
                self.road_links.drop([(s, group) for s in segments], 1, inplace=True)

            self.road_links['load'] = self.road_links[[s for s in segments]].T.sum()
            self.road_links.drop([s for s in segments], 1, inplace=True)

    
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
	#TODO Merge conflict: TO CHECK WITH ACCRA        
	# self.road_links.drop(columns, 1, inplace=True)
        # if not 'load' in self.road_links.columns:
        #    self.road_links['load'] = 0
        # self.road_links['load'] += self.road_links[('all','car')]
        

    #TODO move all utility features to another object / file

    def analysis_mode_utility(self, how='min', segment=None, segments=None, time_expanded=False):
        """
        * requires: mode_utility, los, utility_values
        * builds: los
        """
        if segment is None:

            for segment in self.segments:

                self.analysis_mode_utility(how=how, segment=segment, time_expanded=time_expanded)
            return 
        if time_expanded:
            logit_los = self.te_los
        else:
            logit_los = self.los
        mode_utility = self.mode_utility[segment].to_dict()
        route_types = logit_los['route_types'].unique()
        route_types = pd.DataFrame(route_types, columns=['route_types'])
        route_types['mode_utility'] = route_types['route_types'].apply(
            get_combined_mode_utility, how=how, mode_utility=mode_utility)
        
        route_types['rt_string'] = route_types['route_types'].astype(str)
        los = logit_los.copy()
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

        logit_los[(segment, 'utility')] = u
        logit_los[(segment, 'utility')] = logit_los[(segment, 'utility')].astype(float)
        
    def analysis_utility(self, segment='root', time_expanded=False):
        """
        * requires: mode_utility, los, utility_values
        * builds: los
        """
        if segment is None:
            for segment in self.segments:
                print(segment)
                self.analysis_mode_utility(how=how, segment=segment, time_expanded=time_expanded)
            return 
        if time_expanded:
            los = self.te_los
        else:
            los = self.los
        
        utility_values = self.utility_values[segment].to_dict()
        u = 0
        for key, value in utility_values.items():
            u += value * los[key]

        los[(segment, 'utility')] = u
        los[(segment, 'utility')] = los[(segment, 'utility')].astype(float)

    def initialize_logit(self):
        zones = list(self.zones.index)
        od = pd.DataFrame(index=pd.MultiIndex.from_product([zones, zones]))
        self.od_probabilities = od.copy()
        self.od_utilities = od.copy()

    def _unique_model_segmented_logit(
        self, 
        time_expanded=False,
        decimals=None, 
        n_paths_max=None,
        nchunks=100,
        workers=1
        ):
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
        od_cols = ['origin', 'destination']
        if time_expanded:
            od_cols.append('wished_departure_time')
        to_concat = []
        for segment in self.segments:
            paths = self.los.copy() if not time_expanded else self.te_los.copy()
            paths['utility'] = paths[(segment, 'utility')]
            paths['segment'] = segment
            to_concat.append(paths[od_cols + ['route_type', 'utility', 'segment']])
        segmented_paths = pd.concat(to_concat)

        l, u, p = nested_logit.nested_logit_from_paths(
            segmented_paths, 
            od_cols,
            mode_nests=nests, phi=nls,
            verbose=False,
            decimals=decimals, n_paths_max=n_paths_max,
            nchunks=nchunks, workers=workers,
        )

        pool = l.reset_index().set_index(['segment', 'index'])
        for segment in self.segments:
            if time_expanded:
                self.te_los[(segment, 'probability')] = pool.loc[segment]['probability']
            else:
                self.los[(segment, 'probability')] = pool.loc[segment]['probability']
        
        self.probabilities = p
        self.utilities = u

    def step_logit(
        self, segment=None, probabilities=None, 
        utilities=None, time_expanded=False, 
        decimals=None, n_paths_max=None, nchunks=100, workers=1,
        *args, **kwargs):
        """
        * requires: mode_nests, logit_scales, los
        * builds: los, od_utilities, od_probabilities, path_utilities, path_probabilities
        """
        try:
            self._unique_model_segmented_logit(
                time_expanded=time_expanded,
                decimals=decimals, n_paths_max=n_paths_max,
                nchunks=nchunks, workers=workers
                )
            return 
        except AssertionError:
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
