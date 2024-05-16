import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from quetzal.engine import engine
from quetzal.engine.graph_utils import combine_edges, expand_path
from quetzal.engine.pathfinder_utils import los_from_graph, adjacency_matrix
from quetzal.engine.pathfinder_utils import paths_from_edges, link_edge_array, sparse_matrix_with_access_penalty
from quetzal.engine.pathfinder_utils import get_first_and_last, get_all, pruned_matrix,efficient_od_sets


class PublicPathFinder:
    def __init__(self, model, walk_on_road=False):
        self.zones = model.zones.copy()
        self.links = engine.graph_links(model.links.copy())
        self.csgraph = None
        self.node_index = None

        if walk_on_road:
            road_links = model.road_links.copy()
            road_links['time'] = road_links['walk_time']
            to_concat = [road_links, model.road_to_transit]
            try:
                to_concat.append(model.footpaths)
            except AttributeError:
                pass
            self.footpaths = pd.concat([df[['a','b', 'time']] for df in to_concat])

            to_concat = [model.zone_to_road]
            try:
                to_concat.append(model.zone_to_transit)
            except AttributeError:
                pass
            self.ntlegs = pd.concat([df[['a','b', 'time']] for df in to_concat])

        else:
            self.footpaths = model.footpaths.copy()
            self.ntlegs = model.zone_to_transit.copy()

        try:
            self.centroids = model.centroids.copy()
        except AttributeError:
            self.centroids = self.zones.copy()
            self.centroids['geometry'] = self.centroids['geometry'].apply(
                lambda g: g.centroid
            )

    def first_link(self, path):
        for n in path:
            if n in self.links.index:
                return n

    def last_link(self, path):
        for n in reversed(path):
            if n in self.links.index:
                return n
            
    def build_best_paths_sets(self, column='route_id', first_and_last_only=False):
        link_dict = self.links[column].to_dict()
        getter = get_first_and_last if first_and_last_only else get_all
        self.best_paths[column + '_set'] = [
            getter(path, link_dict) 
            for path in self.best_paths['path']
        ]

    def build_route_id_sets(self, first_and_last_only=False):
        self.build_best_paths_sets(column='route_id', first_and_last_only=first_and_last_only)

    def build_route_type_sets(self, first_and_last_only=False):
        self.build_best_paths_sets(column='route_type', first_and_last_only=first_and_last_only)


    def build_od_sets(self, split_factor=0, verbose=False, drop_empty_sets=True):
        self.combinations = {
            column: {frozenset(broken) for broken in combinations} 
            for column, combinations in self.combinations.items()
        }
        self.od_sets = dict()
        self.splitted_od_sets = dict()
        for column, combinations in self.combinations.items():

            od_set = {
                combination: {
                    (o, d)
                    for o, d, s in self.best_paths[['origin', 'destination', column + '_set']].values 
                    if s.intersection(combination)
                } for combination in self.combinations[column]
            }

            if drop_empty_sets:
                # Drop combinations if OD set is empty 
                relevant_combinations = []
                for combination in combinations :
                    if len(od_set[combination]) > 0:
                        relevant_combinations.append(combination)
                self.combinations[column] = relevant_combinations
                od_set = {k:v for k,v in od_set.items() if len(v)}

            self.od_sets[column] = od_set
            self.splitted_od_sets[column] = {k: (v, set()) for k, v in od_set.items()}

            self.splitted_od_sets[column] = {
                c : efficient_od_sets(od_set, factor=split_factor, verbose=verbose) 
                for c, od_set in self.od_sets[column].items()
            }


    def build_route_zones(self, route_column):
        """
        find origin zones that are likely to be affected by the removal
        each one of the routes
        """

        los = self.best_paths.copy()
        los['first_link'] = los['path'].apply(self.first_link)
        los['last_link'] = los['path'].apply(self.last_link)

        right = self.links[[route_column]]

        merged = pd.merge(los, right, left_on='first_link', right_index=True)
        merged = pd.merge(merged, right, left_on='last_link', right_index=True,
            suffixes = ['_first', '_last'])

        first = merged[['origin', route_column + '_first']]
        first.columns = ['zone', 'route']
        last = merged[['destination', route_column + '_last']]
        last.columns = ['zone', 'route']
        zone_route = pd.concat([first, last]).drop_duplicates(subset=['zone', 'route'])

        route_zone_sets = zone_route.groupby('route')['zone'].apply(set)
        self.route_zones = route_zone_sets.to_dict()

    def build_route_breaker(self, route_column='route_id'):
        self.build_route_zones(route_column=route_column)

    def build_mode_breaker(self, mode_column='route_type'):
        self.build_mode_combinations(mode_column='route_type')

    def build_mode_combinations(self, mode_column='route_type'):
        mode_list = sorted(list(set(self.links[mode_column])))
        boolean_array = list(itertools.product((True, False), repeat=len(mode_list)))
        mode_combinations = []
        for booleans in boolean_array:
            combination = {
                mode_list[i]
                for i in range(len(mode_list))
                if booleans[i]
            }
            mode_combinations.append(combination)
        self.mode_combinations = mode_combinations

    def find_best_path(
        self,
        od_set=None,
        cutoff=np.inf,
        boarding_time=None,
        build_shortcuts=False,
        keep_matrix=False,
        verbose=False,
    ):

        link_e = link_edge_array(self.links, boarding_time)
        footpaths_e = self.footpaths[['a', 'b', 'time']].values
        ntlegs_e = self.ntlegs[['a', 'b', 'time']].values
        edges = np.concatenate([link_e, footpaths_e, ntlegs_e])
        if keep_matrix:
            self.csgraph, self.node_index = sparse_matrix_with_access_penalty(edges, sources=self.zones.index)

        if build_shortcuts:
            keep = {o for o, d in od_set}.union({d for o, d in od_set})
            e, s = combine_edges(edges, keep=keep)
            los = paths_from_edges(edges=e, od_set=od_set, cutoff=cutoff, log=True)
            los['path'] = [expand_path(p, shortcuts=s) for p in los['path']]
        else:
            los = paths_from_edges(edges=edges, od_set=od_set, cutoff=cutoff, csgraph=self.csgraph, node_index=self.node_index, log=verbose)

        los['pathfinder_session'] = 'best_path'
        los['reversed'] = False
        self.best_paths = los.rename(columns={'length': 'gtime'})
    
    def find_broken_combination_paths(
        self, column=None, prune=True,
        cutoff=np.inf, build_shortcuts=False,
        boarding_time=None, od_set=None, reuse_matrix=True,
        log=False, keep_matrix=False
    ):

        def get_task(column, combination, od_set=None):
            od_sets = self.splitted_od_sets[column][combination] if od_set is None else (od_set, set())
            return (column, combination, od_sets)
        
        csgraph, node_index, pcsgraph, pnode_index, edges = None, None, None, None, None

        if reuse_matrix :
            if self.csgraph is None:
                link_e = link_edge_array(self.links, boarding_time)
                footpaths_e = self.footpaths[['a', 'b', 'time']].values
                ntlegs_e = self.ntlegs[['a', 'b', 'time']].values
                edges = np.concatenate([link_e, footpaths_e, ntlegs_e])
                csgraph, node_index = sparse_matrix_with_access_penalty(edges, sources=self.zones.index)
            else :
                csgraph, node_index = self.csgraph, self.node_index
            
        if column is not None:
            combinations = [(column, combination) for combination in self.combinations[column]]
        else:
            combinations = []
            for column, _combinations in self.combinations.items():
                for combination in _combinations:
                    combinations.append((column, combination))
        iterator = tqdm([get_task(column, combination, od_set) for column, combination in combinations])
        to_concat = []
        for column, combination, od_sets in iterator:
            iterator.desc = column + ' ' + str(set(combination))
            
            to_prune = set(self.links.index[self.links[column].isin(combination)])
            links = self.links.drop(to_prune)
            footpaths = self.footpaths
            ntlegs = self.ntlegs

            if prune:
                removed_nodes = set(self.links['a']).union(self.links['b']) - set(links['a']).union(links['b'])
                removed_footpaths = set(footpaths.index[
                    (~footpaths['a'].isin(removed_nodes)) & (~footpaths['b'].isin(removed_nodes))
                ])
                footpaths = footpaths.drop(removed_footpaths)
                removed_ntlegs = set(ntlegs.index[
                    (~ntlegs['a'].isin(removed_nodes)) & (~ntlegs['b'].isin(removed_nodes))
                ])
                ntlegs = ntlegs.drop(removed_ntlegs)
                to_prune = to_prune.union(removed_nodes).union(removed_footpaths).union(removed_ntlegs)

            if not reuse_matrix:
                link_e = link_edge_array(links, boarding_time)
                footpaths_e = footpaths[['a', 'b', 'time']].values
                ntlegs_e = ntlegs[['a', 'b', 'time']].values
                edges = np.concatenate([link_e, footpaths_e, ntlegs_e])

            o_od_set, d_od_set = od_sets
            od_set = o_od_set.union(d_od_set)
            if reuse_matrix:
                # we remove the edges from the initial matrix
                pcsgraph, pnode_index = pruned_matrix(csgraph, node_index, to_prune)

            if len(o_od_set) == 0 and len(d_od_set) == 0:
                continue
            # OLOS forward search
            if len(o_od_set):
                if build_shortcuts:
                    assert not reuse_matrix, 'set reuse_matrix to false to if build_shortcuts is true'
                    keep = {o for o, d in od_set}.union({d for o, d in od_set})
                    e, s = combine_edges(edges, keep=keep)
                    o_los = paths_from_edges(edges=e, od_set=o_od_set, cutoff=cutoff, log=log)
                    o_los['path'] = [expand_path(p, shortcuts=s) for p in o_los['path']]
                else:
                    o_los = paths_from_edges(edges=edges, od_set=o_od_set, cutoff=cutoff, csgraph=pcsgraph, node_index=pnode_index, log=log)
                o_los['reversed'] = False
                
            # DLOS backward search
            if len(d_od_set):
                if build_shortcuts:
                    d_los = paths_from_edges(edges=e, od_set=d_od_set, cutoff=cutoff, log=log)
                    d_los['path'] = [expand_path(p, shortcuts=s) for p in d_los['path']]
                else:
                    d_los = paths_from_edges(edges=edges, od_set=d_od_set, cutoff=cutoff, csgraph=pcsgraph, node_index=pnode_index, log=log)
                d_los['reversed'] = True
                # CONCAT
                los = pd.concat([o_los, d_los])
            else:
                los = o_los

            los['broken_column'] = column
            los['broken_' + column] = [combination for i in range(len(los))]
            to_concat.append(los.rename(columns={'length': 'gtime'}))
        self.broken_combination_paths = pd.concat(to_concat)
        if keep_matrix:
            self.csgraph = csgraph
            self.node_index = node_index

    def find_best_paths(
        self,
        route_column='route_id',
        mode_column='route_type',
        broken_routes=False,
        broken_modes=False,
        drop_duplicates=True,
        cutoff=np.inf,
        od_set=None,
        boarding_time=None,
        verbose=True,
        **kwargs
    ):
        if od_set is None:
            pole_set = set(self.zones.index)
            od_set = set([(i, j) for i in pole_set for j in pole_set])

        self.find_best_path(
            boarding_time=boarding_time,
            cutoff=cutoff,
            od_set=od_set,
            verbose=verbose,
            keep_matrix=True,
            **kwargs
        )  # builds the graph

        self.combinations = dict()
         
        # BUILD OD SETS
        if broken_modes:
            # BUILD ALL MODE COMBINATIONS
            mode_combinations = [set()]
            modes = set(self.links[mode_column])
            for mode in modes:
                mode_combinations += [s.union({mode}) for s in mode_combinations] 
            self.combinations[mode_column] = mode_combinations[1:] # remove empty set
            self.build_best_paths_sets(column=mode_column, first_and_last_only=False)

        if broken_routes:
            # BUILD ROUTE COMBINATIONS | ONLY ONE ROUTE BROKEN IN EACH COMBINATION
            broken_route_set = set(self.links[route_column])
            self.combinations[route_column] = [{route} for route in broken_route_set]
            self.build_best_paths_sets(column=route_column, first_and_last_only=False)

        if broken_routes or broken_modes:
            self.build_od_sets(split_factor=0.5, verbose=verbose)

        # FIND BROKEN ROUTES
        self.broken_route_paths = pd.DataFrame()
        if broken_routes:
            self.find_broken_combination_paths(column='route_id', cutoff=cutoff, build_shortcuts=False, prune=False, reuse_matrix=True, keep_matrix=True, log=verbose)
            self.broken_route_paths = self.broken_combination_paths
            self.broken_route_paths['pathfinder_session'] = 'route_breaker' 
            self.broken_route_paths['broken_route'] = self.broken_route_paths['broken_' + route_column].apply(
                lambda s: list(s)[0]
            ) # we assume only one route is broken at a time

        # FIND BROKEN PATHS
        self.broken_mode_paths = pd.DataFrame()
        if broken_modes:
            self.find_broken_combination_paths(column='route_type', cutoff=cutoff, build_shortcuts=False, prune=False, reuse_matrix=True, log=verbose, keep_matrix=True)
            self.broken_mode_paths = self.broken_combination_paths
            self.broken_mode_paths['pathfinder_session'] = 'mode_breaker'
            self.broken_mode_paths['broken_modes'] = self.broken_mode_paths['broken_' + mode_column].apply(set)

        self.paths = pd.concat([
            self.best_paths,
            self.broken_mode_paths, 
            self.broken_route_paths, 
            ]
        )
        self.paths['path'] = [tuple(p) for p in self.paths['path']]
        self.paths.loc[self.paths['origin'] == self.paths['destination'], ['gtime']] = 0.0

        if drop_duplicates:
            self.paths.drop_duplicates(subset=['path'], inplace=True)
