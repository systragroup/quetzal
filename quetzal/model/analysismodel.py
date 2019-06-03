# -*- coding: utf-8 -*-

from quetzal.analysis import analysis
from quetzal.engine import engine, linearsolver_utils
from quetzal.model import model, transportmodel, summarymodel
from quetzal.io import export
from syspy.syspy_utils import neighbors
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from syspy.spatial import spatial, geometries
from quetzal.engine import nested_logit

def read_hdf(filepath):
    m = AnalysisModel()
    m.read_hdf(filepath)
    return m


def read_json(folder):
    m = AnalysisModel()
    m.read_json(folder)
    return m


track_args = model.track_args
log = model.log

class AnalysisModel(summarymodel.SummaryModel):
    
    def _aggregate(self, nb_clusters):
        """
        Aggregates a model (in order to perform optimization)
            * requires: nb_clusters, cluster_series, od_stack, indicator
            * builds: cluster_series, aggregated model, reduced indicator
        """
        self.agg = self.copy()
        self.agg.renumber(nb_clusters, is_od_stack=True)
        self.cluster_series = self.agg.cluster_series
        self.agg.indicator = linearsolver_utils.reduce_indicator(
            self.indicator,
            self.cluster_series,
            self.od_stack
        )

    def _disaggregate(self):
        self.pivot_stack_matrix, self.od_stack = linearsolver_utils.extrapolate(
                self.agg.pivot_stack_matrix,
                self.od_stack,
                self.cluster_series
        )

    def _build_pivot_stack_matrix(self, constrained_links, linprog_kwargs):
        """
        Builds the pivot_stack_matrix. Performs the optimization.
            * requires: constrained_links, od_stack, indicator
            * builds: pivot_stack_matrix
        """
        self.pivot_stack_matrix = linearsolver_utils.linearsolver(
            self.indicator,
            constrained_links,
            self.od_stack,
            **linprog_kwargs
        )

    def analysis_linear_solver(
        self,
        constrained_links,
        nb_clusters=20,
        linprog_kwargs={
            'bounds_A': [0.75, 1.5],
            'bounds_emissions': [0.8, 1.2],
            'bounds_tot_emissions': [0.95, 1.05],
            'pas_distance': 200,
            'maxiter': 3000,
            'tolerance': 1e-5
        }
        ):
        """
        To perform the optimization on a model object once it is built and run,
        in order to match the observed volumes.
            * requires: od_stack, constrained_links
            * builds: aggregated model, pivot_stack_matrix
        Le but de linear_solver est de modifier la matrice des volumes par OD
        en la multipliant par un pivot, afin de coller aux observations
        recueillies sur certains nœuds/liens du réseau.
        Etapes:
        0. Construction de l'indicatrice (matrice qui indique la présence des
            liens contraints dans chaque OD)
        1. Agrégation du modèle.
        2. Résolution du problème d'optimisation linéaire pour construire
            pivot_stack_matrix (mztrice pivot). Plus de détails dans
            linersolver_utils
        3. Désagrégation de la matrice pivot pour revenir au modèle de base.
        """
        self.indicator = linearsolver_utils.build_indicator(
            self.od_stack,
            constrained_links)
        if len(self.zones) < nb_clusters:
            self._build_pivot_stack_matrix(constrained_links, linprog_kwargs)
        else:
            self._aggregate(nb_clusters)
            self.agg._build_pivot_stack_matrix(constrained_links, linprog_kwargs)
            self._disaggregate()

    def analysis_pt_route_type(self, hierarchy):
        route_type_dict = self.links['route_type'].to_dict()
        self.pt_los['route_types'] = self.pt_los['link_path'].apply(
            lambda p: tuple({route_type_dict[l] for l in p})
        )

        def higher_route_type(route_types):
            for mode in hierarchy:
                if mode in route_types:
                    return mode
            return hierarchy[-1]

        self.pt_los['route_type'] = self.pt_los['route_types'].apply(higher_route_type)

    def analysis_car_route_type(self):
        self.car_los['route_types'] = [tuple(['car']) for i in self.car_los.index]
        self.car_los['route_type'] = 'car'


    def analysis_pt_time(self, boarding_time=0, walk_on_road=False):
        footpaths = self.footpaths
        access = self.zone_to_transit

        if walk_on_road:
            road_links = self.road_links.copy()
            road_links['time'] = road_links['walk_time']
            road_to_transit = self.road_to_transit.copy()
            road_to_transit['length'] = road_to_transit['distance']
            footpaths = pd.concat([road_links, road_to_transit])
            access = self.zone_to_road

        d = access.set_index(['a', 'b'])['time'].to_dict()
        self.pt_los['access_time'] = self.pt_los['ntlegs'].apply(
            lambda l: sum([d[t] for t in l]))

        d = footpaths.set_index(['a', 'b'])['time'].to_dict()
        self.pt_los['footpath_time'] = self.pt_los['footpaths'].apply(
            lambda l: sum([d[t] for t in l]))

        d = self.links['time'].to_dict()
        self.pt_los['in_vehicle_time'] = self.pt_los['link_path'].apply(
            lambda l: sum([d[t] for t in l]))
        d = self.links['headway'].to_dict()
        self.pt_los['waiting_time'] = self.pt_los['boarding_links'].apply(
            lambda l: sum([d[t] / 2 for t in l]))
        self.pt_los['boarding_time'] = self.pt_los['boarding_links'].apply(
            lambda t: len(t)*boarding_time)
        self.pt_los['time'] = self.pt_los[
            ['access_time', 'footpath_time', 'waiting_time', 'boarding_time', 'in_vehicle_time']
        ].T.sum()

    def analysis_pt_length(self, walk_on_road=False):

        footpaths = self.footpaths
        access = self.zone_to_transit

        if walk_on_road:
            road_links = self.road_links.copy()
            road_links['time'] = road_links['walk_time']
            road_to_transit = self.road_to_transit.copy()
            road_to_transit['length'] = road_to_transit['distance']
            footpaths = pd.concat([road_links, road_to_transit])
            access = self.zone_to_road

        d = access.set_index(['a', 'b'])['distance'].to_dict()
        self.pt_los['access_length'] = self.pt_los['ntlegs'].apply(
            lambda l: sum([d[t] for t in l]))
        d = footpaths.set_index(['a', 'b'])['length'].to_dict()
        self.pt_los['footpath_length'] = self.pt_los['footpaths'].apply(
            lambda l: sum([d[t] for t in l]))
        d = self.links['length'].to_dict()
        self.pt_los['in_vehicle_length'] = self.pt_los['link_path'].apply(
            lambda l: sum([d[t] for t in l]))
        self.pt_los['length'] = self.pt_los[
            ['access_length', 'footpath_length',  'in_vehicle_length']
        ].T.sum()

    def analysis_car_time(self):
        d = self.zone_to_road.set_index(['a', 'b'])['time'].to_dict()
        self.car_los['access_time'] = self.car_los['ntlegs'].apply(
            lambda l: sum([d[t] for t in l]))   
        d = self.road_links['time'].to_dict()
        self.car_los['in_vehicle_time'] = self.car_los['link_path'].apply(
            lambda l: sum([d[t] for t in l]))
        self.car_los['time'] = self.car_los[
            ['access_time', 'in_vehicle_time']
        ].T.sum()
    
    def analysis_car_length(self):
        d = self.zone_to_road.set_index(['a', 'b'])['distance'].to_dict()
        self.car_los['access_length'] = self.car_los['ntlegs'].apply(
            lambda l: sum([d[t] for t in l]))
        
        d = self.road_links['length'].to_dict()
        self.car_los['in_vehicle_length'] = self.car_los['link_path'].apply(
            lambda l: sum([d[t] for t in l]))


    def analysis_pt_fare(self):

        # fare_rules 
        route_dict = self.links['route_id'].to_dict()
        fare_dict = self.fare_rules.set_index('route_id')['fare_id'].to_dict()
        def fare_id_list(path):
            return [fare_dict[route] for route in {route_dict[link] for link in path}]

        # fare_attributes
        transfers = self.fare_attributes.set_index('fare_id')['transfers'].to_dict()
        price = self.fare_attributes.set_index('fare_id')['price'].to_dict()

        def fare(count, allowed_transfers, price):
            return max(np.ceil(count / (allowed_transfers + 1))  , 1) * price

        def price_breakdown(fare_id_list):
            return {
                f: fare(
                    count=fare_id_list.count(f),
                    allowed_transfers=transfers[f],
                    price=price[f]
                )
                for f in set(fare_id_list)
            }

        fare_id_list_series= self.pt_los['link_path'].apply(fare_id_list)
        self.pt_los['fare_id_list'] = fare_id_list_series
        self.pt_los['price_breakdown'] = fare_id_list_series.apply(price_breakdown)
        self.pt_los['price'] = self.pt_los['price_breakdown'].apply(lambda d: sum(d.values()))


    @track_args
    def analysis_summary(self):
        """
        To perform on a model object once it is built and run,
        aggregate and analyses results.
            * requires: shared, zones, loaded_links, od_stack
            * builds: aggregated_shares, lines, economic_series
        """
        try: 
            self.aggregated_shares = engine.aggregate_shares(
                self.shared, self.zones)
        except AttributeError: 
            pass
        self.lines = analysis.tp_summary(self.loaded_links, self.od_stack)
        self.lines = analysis.analysis_tp_summary(self.lines)
        self.economic_series = analysis.economic_series(self.od_stack, self.lines)

    @track_args
    def analysis_desire(self, store_shp=False, **to_shp_kwarg):
        """
        Builds the desire matrix
            * requires: zones, shares
            * builds: neighborhood, macro_neighborhood
        """
        self.neighborhood = neighbors.Neighborhood(
            self.zones,
            self.volumes,
            volume_columns=['volume'],
            display_progress=False
        )
        zones = self.zones.copy()
        zones['geometry'] = zones['geometry'].apply(lambda g: g.buffer(1e-9))

        self.macro_neighborhood = neighbors.Neighborhood(
            zones,
            self.volumes,
            volume_columns=['volume'],
            display_progress=False,
            n_clusters=min(25, len(zones)),
            od_geometry=True)
        
        if store_shp:
            columns_to_keep = ['origin', 'destination', 'volume', 'volume_transit', 'geometry']
            self.desire_lines = self.neighborhood.volume[columns_to_keep].dropna(subset=['geometry'])

    @track_args
    def analysis_checkpoints(
        self,
        link_checkpoints=(),
        node_checkpoints=(),
        **loaded_links_and_nodes_kwargs
        ):

        """
        tree analysis (arborescences)
        :param link_checkpoints: mandatory transit links collection (set)
        :param nodes_checkpoints: mandatory transit nodes
        :param volume column: column of self.od_stack to assign
        :loaded_links_and_nodes_kwargs: ...

        example:
        ::
            sm.checkpoints(link_checkpoints = {}, node_checkpoints={41})
            export.assigned_links_nodes_to_shp(
                sm.checkpoint_links,
                sm.checkpoint_nodes,
                gis_path=gis_path,
                link_name='links_test.shp',
                node_name='nodes_test.shp'
        )
        """

        selected = engine.loaded_links_and_nodes(
            self.links,
            self.nodes,
            volumes=self.volumes,
            path_finder_stack=self.pt_los,
            link_checkpoints=set(link_checkpoints),
            node_checkpoints=set(node_checkpoints),
            **loaded_links_and_nodes_kwargs
        )

        self.checkpoint_links = selected[0]
        self.checkpoint_nodes = selected[1]

    def analysis_lines(self, line_columns='all', group_id='trip_id', *args, **kwargs):
        self.lines = export.build_lines(
            self.links, 
            line_columns=line_columns, 
            group_id=group_id,
            *args, **kwargs
        )

    def get_road_links(self, trip_id='trip_id'):
        l = self.links.copy()
        flat = []
        for key, links in l['road_link_list'].to_dict().items():
            flat += [(key, link) for link in links]

        core = pd.DataFrame(flat, columns=['transit', 'road'])

        merged = pd.merge(self.links, core, left_index=True, right_on='transit')
        merged = pd.merge(merged, self.road_links, left_on='road', right_index=True, suffixes=['_transit', ''])
        return merged[['a', 'b', 'transit', 'geometry', 'road', trip_id]]


    def get_lines_with_offset(self, width=1, trip_id='trip_id'):
        # get road_links
        l = self.get_road_links()
        l['ab'] = l.apply(lambda r: tuple(sorted([r['a'], r['b']])), axis=1)

        # line_tuples geometry
        line_tuples = l.groupby(['road'])[trip_id].agg(lambda s: tuple(sorted(tuple(s))))
        road_links = gpd.GeoDataFrame(self.road_links)
        road_links['line_tuple'] = line_tuples
        road_links = road_links.dropna(subset=['line_tuple']).copy()

        line_tuples = list(set(line_tuples))
        line_tuple_geometries = dict()
        for line_tuple in tqdm(line_tuples):
            
            # build sorted_edges
            edges = road_links.loc[road_links['line_tuple'] == line_tuple]
            sorted_road_links = []
            selected = self.links.loc[self.links['trip_id'] == line_tuple[0]]
            for road_link_list in selected.sort_values('link_sequence')['road_link_list']:
                for road_link in road_link_list:
                    sorted_road_links.append(road_link)
            sorted_edges = edges.loc[sorted_road_links].dropna(subset=['a', 'b'])
            
            line_tuple_geometries[line_tuple] = geometries.connected_geometries(sorted_edges)
        
        return geometries.geometries_with_side(line_tuple_geometries, width=width)
