import numpy as np
import pandas as pd
from quetzal.engine import pathfinder_utils
from quetzal.model import preparationmodel


class ParkRideModel(preparationmodel.PreparationModel):
    def node_transit_zone_edges(
        self,
        pr_nodes,
        reverse=False,
        boarding_time=None,
        alighting_time=None
    ):
        # link edges
        edges = pathfinder_utils.link_edges(
            self.links,
            boarding_time=boarding_time,
            alighting_time=alighting_time
        )

        # connectors and footpaths
        zn = 'a' if reverse else 'b'
        ztr = self.zone_to_road.copy()
        ztr = ztr.loc[ztr[zn].isin(self.zones.index)]
        ztr = ztr[['a', 'b', 'time']].values.tolist()
        rtt = self.road_to_transit[['a', 'b', 'time']].values.tolist()
        footpaths = self.road_links[['a', 'b', 'walk_time']].values.tolist()
        return edges + ztr + footpaths + rtt

    def get_node_transit_zone(
        self, pr_nodes, reverse=False,
        boarding_time=None, alighting_time=None,
        cutoff=np.inf
    ):
        ntz_edges = self.node_transit_zone_edges(
            pr_nodes=pr_nodes, reverse=reverse,
            boarding_time=boarding_time, alighting_time=alighting_time
        )

        matrix, node_index = pathfinder_utils.sparse_matrix(ntz_edges)

        zn = 'a' if reverse else 'b'

        zones = set(self.zones.index).intersection(self.zone_to_road[zn])
        sources, targets = (zones, pr_nodes) if reverse else (pr_nodes, zones)

        node_transit_zone = pathfinder_utils.paths_from_graph(
            csgraph=matrix,
            node_index=node_index,
            sources=sources,
            targets=targets,
            cutoff=cutoff
        )

        node_transit_zone['reverse'] = reverse
        return node_transit_zone

    def zone_road_node_edges(
        self,
        pr_nodes=None,
        reverse=False,
        zrn_access_time='time',
    ):
        # zn = 'a' keeps zone->road zn='b' keeps road->zone
        zn, pn = ('b', 'a') if reverse else ('a', 'b')
        ztr = self.zone_to_road.copy()
        ztr['time'] = ztr[zrn_access_time]
        ztr = ztr.loc[ztr[zn].isin(self.zones.index)][['a', 'b', 'time']]
        rtt = self.road_to_transit.copy()
        rtt['time'] = rtt[zrn_access_time]
        if pr_nodes is not None:
            rtt = rtt.loc[rtt[pn].isin(pr_nodes)]

        edges = ztr[['a', 'b', 'time']].values.tolist()
        edges += rtt[['a', 'b', 'time']].values.tolist()
        edges += self.road_links[['a', 'b', 'time']].values.tolist()
        return edges

    def get_zone_road_node(
        self, pr_nodes=None, reverse=False,
        zrn_access_time='time',
        cutoff=np.inf,
    ):
        zn = 'b' if reverse else 'a'
        zones = set(self.zones.index).intersection(self.zone_to_road[zn])

        zrt_edges = self.zone_road_node_edges(
            zrn_access_time=zrn_access_time,
            pr_nodes=pr_nodes, reverse=reverse
        )
        matrix, node_index = pathfinder_utils.sparse_matrix(zrt_edges)

        sources, targets = (pr_nodes, zones) if reverse else (zones, pr_nodes)
        zone_road_node = pathfinder_utils.paths_from_graph(
            csgraph=matrix,
            node_index=node_index,
            sources=sources,
            targets=targets,
            cutoff=cutoff
        )
        zone_road_node['reverse'] = reverse
        return zone_road_node

    def build_park_ride_shortcuts(
        self, pr_nodes,
        zrn_access_time='time',
        boarding_time=None, alighting_time=None,
        reverse=False,
        cutoff=np.inf
    ):
        # MORNING
        self.node_transit_zone = self.get_node_transit_zone(
            pr_nodes=pr_nodes, reverse=reverse,
            boarding_time=boarding_time, alighting_time=alighting_time,
            cutoff=cutoff
        )
        self.zone_road_node = self.get_zone_road_node(
            zrn_access_time=zrn_access_time,
            pr_nodes=pr_nodes, reverse=reverse,
            cutoff=cutoff
        )

    def combine_shortcuts(
        self,
        pr_nodes=None,
        ntlegs_penalty=1e9,
        cutoff=np.inf,
        od_set=None,
        parking_times=None,
        reverse=False,
    ):
        zrn, ntz = self.zone_road_node, self.node_transit_zone
        ntz.reset_index(drop=True, inplace=True)
        ntz.index = ['ntz_' + str(i) for i in ntz.index]
        zrn.reset_index(drop=True, inplace=True)
        zrn.index = ['zrn_' + str(i) for i in zrn.index]

        ntz = ntz.copy()
        ntz['length'] += ntlegs_penalty

        shortcuts = pd.concat([ntz, zrn])
        if reverse is not None:
            shortcuts = shortcuts.loc[shortcuts['reverse'] == reverse]

        # ADD PARKING TIME
        if parking_times is not None:
            right = pd.DataFrame(
                parking_times.items(),
                columns=['origin', 'parking_time']
            )
            shortcuts = pd.merge(shortcuts, right, on='origin', how='left')
            shortcuts['parking_time'].fillna(0, inplace=True)
            shortcuts['length'] += shortcuts['parking_time']

        if pr_nodes is not None:
            shortcuts.loc[
                shortcuts['origin'].isin(pr_nodes)
                | shortcuts['destination'].isin(pr_nodes)
            ]
        edges = shortcuts[['origin', 'destination', 'length']].values

        matrix, node_index = pathfinder_utils.sparse_matrix(edges)

        s_path = shortcuts.set_index(['origin', 'destination'])['path'].to_dict()

        def concatenate_path(path):
            try:
                return s_path[(path[0], path[1])] + s_path[(path[1], path[2])]
            except IndexError:
                return path

        zone_set = set(self.zones.index)
        paths = pathfinder_utils.paths_from_graph(
            csgraph=matrix,
            node_index=node_index,
            sources=zone_set,
            targets=zone_set,
            cutoff=cutoff + ntlegs_penalty,
            od_set=od_set
        )
        paths['shortcut_path'] = paths['path']
        paths['path'] = [concatenate_path(p) for p in paths['shortcut_path']]
        paths = paths.loc[paths['origin'] != paths['destination']]
        paths['time'] = paths['length'] - ntlegs_penalty
        paths = paths.loc[paths['time'] <= cutoff]
        paths['parking_node'] = [path[1] for path in paths['shortcut_path']]
        self.pr_los = paths

    def get_pr_paths(self, od_set, pr_nodes, length_spread_max=0, reverse=False):
        parking_times = self.nodes.loc[pr_nodes, 'parking_time'].to_dict()

        zrn = self.zone_road_node.loc[self.zone_road_node['reverse'] == reverse]
        ntz = self.node_transit_zone.loc[self.node_transit_zone['reverse'] == reverse]

        c = ['origin', 'destination', 'length', 'path']
        left, right = (ntz[c], zrn[c]) if reverse else (zrn[c], ntz[c])
        left = pd.merge(
            left, pd.Series(parking_times, name='parking_time'),
            left_on='destination', right_index=True
        )

        left.rename(columns={'destination': 'parking_node'}, inplace=True)
        right.rename(columns={'origin': 'parking_node'}, inplace=True)

        hinge = pd.DataFrame(od_set, columns=['origin', 'destination'])
        left = pd.merge(left, hinge, on=['origin'])
        paths = pd.merge(
            left, right, on=['parking_node', 'destination'],
            suffixes=['_left', '_right']
        )

        paths['length'] = paths['length_left'] + paths['length_right'] + paths['parking_time']
        lengths = paths.groupby(['origin', 'destination'], as_index=False)['length'].min()
        paths = pd.merge(
            paths, lengths, on=['origin', 'destination'],
            suffixes=['', '_min']
        )
        paths['length_spread'] = paths['length'] - paths['length_min']
        paths = paths.loc[paths['length_spread'] <= length_spread_max]
        paths['path'] = paths['path_left'] + paths['path_right']
        paths.drop(['path_left', 'path_right'], axis=1, inplace=True)
        paths['reverse'] = reverse
        paths['gtime'] = paths['length']
        return paths

    def lighten_pr_los(self, los_attributes=['pr_los'], keep_summary_columns=False):

        time_columns = [
            'access_time',  'footpath_time',
            'waiting_time', 'boarding_time',  'gtime'
        ]
        if not keep_summary_columns:
            time_columns += ['time', 'in_vehicle_time']
        length_columns = ['access_length']
        if not keep_summary_columns:
            length_columns += ['length', 'in_vehicle_length']
        path_columns = ['path', 'link_path', 'node_path', 'ntlegs', 'footpaths']
        pt_columns = [
            'boardings', 'alightings', 'boarding_links', 'alighting_links', 'footpaths'
        ]
        to_drop = []
        for clist in [time_columns, length_columns, path_columns, pt_columns]:
            for c in clist:
                to_drop.append(c)
                to_drop.append(c + '_car')
                to_drop.append(c + '_transit')
                to_drop = [td for td in to_drop if td != 'path']
        for los in los_attributes:
            self.__getattribute__(los).drop(to_drop, axis=1, errors='ignore', inplace=True)



    def analysis_pr_los(
        self,
        reverse=False,
        analysis_time=False,
        analysis_length=False,
        boarding_time=None,
        zrn_access_time='time',
    ):
        time_columns = [
            'access_time', 'in_vehicle_time', 'footpath_time',
            'waiting_time', 'boarding_time', 'time', 'gtime']
        length_columns = ['access_length', 'in_vehicle_length', 'length']
        path_columns = ['path', 'link_path', 'node_path', 'ntlegs', 'footpaths']
        pt_columns = [
            'boardings', 'alightings', 'boarding_links', 'alighting_links', 'footpaths', 'transfers'
        ]

        # node_transit_zone
        s = self.copy()
        s.lighten_pr_los()
        s.pt_los = s.node_transit_zone.rename(columns={'length': 'gtime'})
        s.car_los = s.zone_road_node.rename(columns={'length': 'gtime'})
        s.pt_los['path'] = [['to_strip'] + p + ['to_strip'] for p in s.pt_los['path']]
        # s.car_los['path'] = [['to_strip'] + p for p in s.car_los['path']]

        s.analysis_pt_los(walk_on_road=True)

        s.zone_to_road = pd.concat([s.zone_to_road, s.road_to_transit])
        s.analysis_car_los()

        if analysis_time:
            s.analysis_pt_time(boarding_time=boarding_time, walk_on_road=True)
            s.analysis_car_time(access_time=zrn_access_time)

        if analysis_length:
            s.analysis_pt_length(walk_on_road=True)
            s.analysis_car_length()

        s.car_los['parking_node'] = s.car_los['destination']
        s.car_los.loc[s.car_los['reverse'] == True, 'parking_node'] = s.car_los['origin']

        s.pt_los['parking_node'] = s.pt_los['origin']
        s.pt_los.loc[s.pt_los['reverse'] == True, 'parking_node'] = s.pt_los['destination']

        on = ['destination' if reverse else 'origin', 'parking_node', 'reverse']
        columns = on + time_columns + length_columns + path_columns
        columns = [c for c in set(columns) if c in s.car_los]
        right = s.car_los[columns]
        merged = pd.merge(s.pr_los, right, on=on, suffixes=['_total', ''])

        on = ['origin' if reverse else 'destination', 'parking_node', 'reverse']

        columns = on + time_columns + length_columns + path_columns + pt_columns
        columns = [c for c in set(columns) if c in s.pt_los]
        right = s.pt_los[columns]
        pr_los = pd.merge(merged, right, on=on, suffixes=['_car', '_transit'])
        s.pt_los['path'] = [p[1:-1] for p in s.pt_los['path']]
        for c in set(time_columns + length_columns + path_columns):
            ca, cb = c + '_car', c + '_transit'
            if reverse:
                ca, cb = cb, ca
            try:
                pr_los[c] = pr_los[ca] + pr_los[cb]
            except KeyError:
                pass
        self.pr_los = pr_los
