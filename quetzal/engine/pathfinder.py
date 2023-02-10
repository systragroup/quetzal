import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from quetzal.engine import engine
from quetzal.engine.graph_utils import combine_edges, expand_path
from quetzal.engine.pathfinder_utils import los_from_graph, adjacency_matrix
from quetzal.engine.pathfinder_utils import paths_from_edges, link_edge_array
from quetzal.engine.pathfinder_utils import get_first_and_last, get_all

class PublicPathFinder:
    def __init__(self, model, walk_on_road=False):
        self.zones = model.zones.copy()
        self.links = engine.graph_links(model.links.copy())

        if walk_on_road:
            road_links = model.road_links.copy()
            road_links['time'] = road_links['walk_time']
            to_concat = [road_links, model.road_to_transit]
            try:
                to_concat.append(model.footpaths)
            except AttributeError:
                pass
            self.footpaths = pd.concat(to_concat)

            to_concat = [model.zone_to_road]
            try:
                to_concat.append(model.zone_to_transit)
            except AttributeError:
                pass
            self.ntlegs = pd.concat(to_concat)

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

    def build_route_id_sets(self, first_and_last_only=False):
        link_dict = self.links['route_id'].to_dict()
        getter = get_first_and_last if first_and_last_only else get_all
        self.best_paths['route_id_set'] = [
            getter(path, link_dict) 
            for path in self.best_paths['path']
        ]
        
    def build_route_type_sets(self, first_and_last_only=False):
        link_dict = self.links['route_type'].to_dict()
        getter = get_first_and_last if first_and_last_only else get_all
        self.best_paths['route_type_set'] = [
            getter(path, link_dict)
            for path in self.best_paths['path']
        ]

    def build_od_sets(self):
        self.combinations = {
            column: {frozenset(broken) for broken in combinations} 
            for column, combinations in self.combinations.items()
        }
        self.od_sets = dict()
        self.splitted_od_sets = dict()
        for column in self.combinations.keys():
            od_set = {
                combination : {
                    (o, d) 
                    for o, d, s in self.best_paths[['origin', 'destination', column + '_set']].values 
                    if s.intersection(combination)
                } for combination in self.combinations[column]
            }
            self.od_sets[column] = od_set
            self.splitted_od_sets[column] = {k: (v, set()) for k, v in od_set.items()}

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
            suffixes=['_first', '_last'])

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

        links = self.links.copy()
        links.drop('index', axis=1, inplace=True, errors='ignore')
        links.index.name = 'index'

        self.mode_links = links.reset_index().groupby(
            [mode_column]
        )['index'].apply(lambda s: set(s)).to_dict()

    def find_best_path(
        self,
        od_set=None,
        cutoff=np.inf,
        boarding_time=None,
        ntlegs_penalty=1e9,
        engine='b',
        build_shortcuts=False,
        **kwargs
    ):
        if engine == 'a':
            self._find_best_path_a(
                od_set=od_set, cutoff=cutoff,
                ntlegs_penalty=ntlegs_penalty, boarding_time=boarding_time, 
                **kwargs)
        elif engine == 'b':
            self._find_best_path_b(
                od_set=od_set, cutoff=cutoff,
                build_shortcuts=build_shortcuts, boarding_time=boarding_time)

    def _find_best_path_a(
        self,
        od_set=None,
        cutoff=np.inf,
        ntlegs_penalty=1e9,
        boarding_time=None,
        **kwargs
    ):
        pole_set = set(self.zones.index)
        matrix, node_index = adjacency_matrix(
            links=self.links, ntlegs=self.ntlegs, footpaths=self.footpaths,
            ntlegs_penalty=ntlegs_penalty, boarding_time=boarding_time,
            **kwargs)

        los = los_from_graph(
            csgraph=matrix,
            node_index=node_index, pole_set=pole_set, od_set=od_set,
            cutoff=cutoff, ntlegs_penalty=ntlegs_penalty)

        los['pathfinder_session'] = 'best_path'
        los['reversed'] = False
        self.best_paths = los

    def _find_best_path_b(
        self,
        od_set=None,
        cutoff=np.inf,
        boarding_time=None,
        build_shortcuts=False,
    ):
        link_e = link_edge_array(self.links, boarding_time)
        footpaths_e = self.footpaths[['a', 'b', 'time']].values
        ntlegs_e = self.ntlegs[['a', 'b', 'time']].values
        edges = np.concatenate([link_e, footpaths_e, ntlegs_e])

        if build_shortcuts:
            keep = {o for o, d in od_set}.union({d for o, d in od_set})
            e, s = combine_edges(edges, keep=keep)
            los = paths_from_edges(edges=e, od_set=od_set, cutoff=cutoff, log=True)
            los['path'] = [expand_path(p, shortcuts=s) for p in los['path']]
        else:
            los = paths_from_edges(edges=edges, od_set=od_set, cutoff=cutoff)

        los['pathfinder_session'] = 'best_path'
        los['reversed'] = False
        self.best_paths = los.rename(columns={'length': 'gtime'})

    def find_broken_route_paths(
        self,
        od_set=None,
        cutoff=np.inf,
        route_column='route_id',
        ntlegs_penalty=1e9,
        boarding_time=None,
        speedup=True,
        prune=True,
        **kwargs
    ):
        pole_set = set(self.zones.index)
        do_set = {(d, o) for o, d in od_set} if od_set is not None else None
        to_concat = []
        iterator = tqdm(self.route_zones.items())
        for route_id, zones in iterator:
            if not speedup:
                zones = set(self.zones.index).intersection(set(self.ntlegs['a']))
            
            route_od_set = {(o, d) for o, d in od_set if o in zones}
            route_do_set = {(d, o) for d, o in do_set if d in zones}
            iterator.desc = 'breaking route: ' + str(route_id) + ' '
            links=self.links.loc[self.links[route_column] != route_id]
            footpaths = self.footpaths
            ntlegs = self.ntlegs 
            try:
                if prune:
                    removed_nodes = set(self.links['a']).union(self.links['b']) - set(links['a']).union(links['b'])
                    removed_nodes = removed_nodes.union(self.zones)
                    footpaths = footpaths.loc[(~footpaths['a'].isin(removed_nodes)) & (~footpaths['b'].isin(removed_nodes))]
                    ntlegs  = ntlegs.loc[(~ntlegs['a'].isin(removed_nodes)) & (~ntlegs['b'].isin(removed_nodes))]

                matrix, node_index = adjacency_matrix(
                    links=links,
                    ntlegs=ntlegs,
                    footpaths=footpaths,
                    ntlegs_penalty=ntlegs_penalty,
                    boarding_time=boarding_time,
                    **kwargs
                )

                los = los_from_graph(
                    csgraph=matrix,
                    node_index=node_index,
                    pole_set=pole_set,
                    od_set=od_set,
                    cutoff=cutoff,
                    ntlegs_penalty=ntlegs_penalty
                )
            except:
                if prune:
                    tqdm.write('Pathfinder failed with prune=True. Trying with prune=False')
                footpaths = self.footpaths
                ntlegs = self.ntlegs

                matrix, node_index = adjacency_matrix(
                    links=links,
                    ntlegs=ntlegs,
                    footpaths=footpaths,
                    ntlegs_penalty=ntlegs_penalty,
                    boarding_time=boarding_time,
                    **kwargs
                )

                los = los_from_graph(
                    csgraph=matrix,
                    node_index=node_index,
                    pole_set=pole_set,
                    od_set=od_set,
                    cutoff=cutoff,
                    ntlegs_penalty=ntlegs_penalty)
            los['reversed'] = False
            los['broken_route'] = route_id
            los['pathfinder_session'] = 'route_breaker'

            to_concat.append(los)
            los = los_from_graph(
                csgraph=matrix.transpose(),
                node_index=node_index,
                pole_set=pole_set,
                od_set=route_do_set,
                sources=zones,
                cutoff=cutoff,
                ntlegs_penalty=ntlegs_penalty
            )
            los[['origin', 'destination']] = los[['destination', 'origin']]
            los['path'] = los['path'].apply(lambda p: list(reversed(p)))
            los['reversed'] = True
            los['broken_route'] = route_id
            los['pathfinder_session'] = 'route_breaker'
            to_concat.append(los)
        self.broken_route_paths = pd.concat(to_concat)

    def find_broken_mode_paths(
        self,
        od_set=None,
        cutoff=np.inf,
        mode_column='mode_type',
        ntlegs_penalty=1e9,
        boarding_time=None,
        prune=True,
        **kwargs
    ):
        pole_set = set(self.zones.index)
        to_concat = []
        iterator = tqdm(self.mode_combinations)
        for combination in iterator:
            iterator.desc = 'breaking modes: ' + str(combination) + ' '

            links=self.links.loc[~self.links[mode_column].isin(combination)]
            footpaths = self.footpaths
            ntlegs = self.ntlegs

            try:
                if prune:
                    removed_nodes = set(self.links['a']).union(self.links['b']) - set(links['a']).union(links['b'])
                    removed_nodes = removed_nodes.union(self.zones)
                    footpaths = footpaths.loc[(~footpaths['a'].isin(removed_nodes)) & (~footpaths['b'].isin(removed_nodes))]
                    ntlegs  = ntlegs.loc[(~ntlegs['a'].isin(removed_nodes)) & (~ntlegs['b'].isin(removed_nodes))]
                print('starting ajacency_matrix')
                matrix, node_index = adjacency_matrix(
                    links=links,
                    ntlegs=ntlegs,
                    footpaths=footpaths,
                    ntlegs_penalty=ntlegs_penalty,
                    boarding_time=boarding_time,
                    **kwargs
                )
                print('starting los_from_graph')
                los = los_from_graph(
                    csgraph=matrix,
                    node_index=node_index,
                    pole_set=pole_set,
                    od_set=od_set,
                    cutoff=cutoff,
                    ntlegs_penalty=ntlegs_penalty
                )
            except Exception as e:
                print(e)
                if prune:
                    tqdm.write('Pathfinder failed with prune=True. Trying with prune=False')
                footpaths = self.footpaths
                ntlegs = self.ntlegs

                matrix, node_index = adjacency_matrix(
                    links=links,
                    ntlegs=ntlegs,
                    footpaths=footpaths,
                    ntlegs_penalty=ntlegs_penalty,
                    boarding_time=boarding_time,
                    **kwargs
                )

                los = los_from_graph(
                    csgraph=matrix,
                    node_index=node_index,
                    pole_set=pole_set,
                    od_set=od_set,
                    cutoff=cutoff,
                    ntlegs_penalty=ntlegs_penalty)

            los['reversed'] = False
            los['pathfinder_session'] = 'mode_breaker'
            los['broken_modes'] = [combination for i in range(len(los))]
            to_concat.append(los)
        self.broken_mode_paths = pd.concat(to_concat)
    
    def find_broken_combination_paths(
        self, column=None, prune=True, 
        cutoff=np.inf, build_shortcuts=False,
        boarding_time=None
        ):
    
        if column is not None:
            iterator = tqdm(
                [
                    (column, combination, self.splitted_od_sets[column][combination])
                    for combination in self.combinations[column]
                ],
            )
        else:
            flat_combinations = []
            for column, combinations in self.combinations.items():
                for combination in combinations:
                    flat_combinations.append((column,combination))
                    
            iterator = tqdm(
                [
                    (column, combination, self.splitted_od_sets[column][combination])
                    for column, combination in flat_combinations
                ]
            )
        to_concat = []
        for column, combination, od_sets in iterator:
            iterator.desc = column + ' ' + str(set(combination))
            
            links=self.links.loc[~self.links[column].isin(combination)]
            footpaths = self.footpaths
            ntlegs = self.ntlegs

            if prune:
                removed_nodes = set(self.links['a']).union(self.links['b']) - set(links['a']).union(links['b'])
                footpaths = footpaths.loc[(~footpaths['a'].isin(removed_nodes)) & (~footpaths['b'].isin(removed_nodes))]
                ntlegs  = ntlegs.loc[(~ntlegs['a'].isin(removed_nodes)) & (~ntlegs['b'].isin(removed_nodes))]

            pole_set = set(self.zones.index)
            link_e = link_edge_array(links, boarding_time)
            footpaths_e = footpaths[['a', 'b', 'time']].values
            ntlegs_e = ntlegs[['a', 'b', 'time']].values
            edges = np.concatenate([link_e, footpaths_e, ntlegs_e])

            o_od_set, d_od_set = od_sets
            od_set = o_od_set.union(d_od_set)

            if len(o_od_set) == 0 and len(d_od_set) == 0:
                continue
            # OLOS forward search
            if len(o_od_set):
                if build_shortcuts:
                    keep = {o for o, d in od_set}.union({d for o, d in od_set})
                    e, s = combine_edges(edges, keep=keep)
                    o_los = paths_from_edges(edges=e, od_set=o_od_set, cutoff=cutoff)
                    o_los['path'] = [expand_path(p, shortcuts=s) for p in o_los['path']]
                else:
                    o_los = paths_from_edges(edges=edges, od_set=o_od_set, cutoff=cutoff)
                o_los['reversed'] = False
                
            # DLOS backward search
            if len(d_od_set):
                if build_shortcuts:
                    d_los = paths_from_edges(edges=e, od_set=d_od_set, cutoff=cutoff)
                    d_los['path'] = [expand_path(p, shortcuts=s) for p in d_los['path']]
                else:
                    d_los = paths_from_edges(edges=edges, od_set=d_od_set, cutoff=cutoff)
                d_los['reversed'] = True
                # CONCAT
                los = pd.concat([o_los, d_los])
            else:
                los = o_los

            los['broken_column'] = column
            los['broken_' + column] = [combination for i in range(len(los))]
            to_concat.append(los.rename(columns={'length': 'gtime'}))
        self.broken_combination_paths = pd.concat(to_concat)

    def find_best_paths(
        self,
        route_column='route_id',
        mode_column='route_type',
        broken_routes=False,
        broken_modes=False,
        drop_duplicates=True,
        speedup=True,
        cutoff=np.inf,
        od_set=None,
        boarding_time=None,
        **kwargs
    ):
        if od_set is None:
            pole_set = set(self.zones.index)
            od_set = set([(i, j) for i in pole_set for j in pole_set])

        to_concat = []
        if broken_routes:
            self.find_best_path(
                boarding_time=boarding_time,
                cutoff=cutoff,
                od_set=od_set,
                **kwargs
            )  # builds the graph
            to_concat.append(self.best_paths)
            self.build_route_breaker(route_column=route_column)
            self.find_broken_route_paths(
                speedup=speedup,
                od_set=od_set,
                boarding_time=boarding_time,
                cutoff=cutoff,
                route_column=route_column,
                **kwargs
            )
            to_concat.append(self.broken_route_paths)

        if broken_modes:
            # self.build_graph(**kwargs)
            print('build_mode_combinations')
            self.build_mode_combinations(mode_column=mode_column)
            print('find_broken_mode_paths')
            self.find_broken_mode_paths(
                od_set=od_set,
                cutoff=cutoff,
                boarding_time=boarding_time,
                mode_column=mode_column,
                **kwargs
            )
            to_concat.append(self.broken_mode_paths)

        if not (broken_modes or broken_routes):
            self.find_best_path(
                cutoff=cutoff,
                od_set=od_set,
                boarding_time=boarding_time,
                **kwargs
            )
            to_concat.append(self.best_paths)

        self.paths = pd.concat(to_concat)
        self.paths['path'] = self.paths['path'].apply(tuple)
        self.paths.loc[self.paths['origin'] == self.paths['destination'], ['gtime']] = 0.0

        if drop_duplicates:
            self.paths.drop_duplicates(subset=['path'], inplace=True)
