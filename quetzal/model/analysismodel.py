# -*- coding: utf-8 -*-

from quetzal.analysis import analysis
from quetzal.engine import engine, linearsolver_utils
from quetzal.model import model, transportmodel
from quetzal.io import export

from syspy.syspy_utils import neighbors


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


class AnalysisModel(transportmodel.TransportModel):

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
