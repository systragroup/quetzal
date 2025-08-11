import numpy as np
import pandas as pd
from quetzal.engine import connectivity, engine, gps_tracks
from quetzal.engine.add_network import NetworkCaster
from quetzal.engine.add_network_mapmatching import (
    RoadLinks,
    get_gps_tracks,
    Multi_Mapmatching,
    Parallel_Mapmatching,
    duplicate_nodes,
)
from quetzal.model import cubemodel, model, integritymodel
from syspy.spatial import spatial
from syspy.renumber import renumber
from syspy.skims import skims
from tqdm import tqdm
import networkx as nx
import warnings

from shapely.geometry import LineString, Point


def read_hdf(filepath):
    m = PreparationModel(hdf_database=filepath)
    return m


def read_json(folder, **kwargs):
    m = PreparationModel()
    m.read_json(folder, **kwargs)
    return m


track_args = model.track_args
log = model.log


class PreparationModel(model.Model, cubemodel.cubeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """This class contains fonctions that can be applied to models
    for preparation of :
            - connectors (zone to transit, to road)
            - pedestrian footpaths

        
        """

    @track_args
    def preparation_footpaths(self, road=False, speed=3, max_length=None, clusters_distance=None, **kwargs):
        """Create the footpaths : pedestrian links between stations (nodes),
        that will allow transfers between stations.

        Requires
        ----------
        self.nodes

        Parameters
        ----------
        road : bool, optional, default False
                        If True, compute walk_time on road_links based on parameter speed
        speed : int, optional, default 3
                        Speed of walk on footpaths. Smaller than real walk speed
                        because the footpaths do not follow roads
        max_length : int, optional, default None
                        Maximal length of footpaths
        clusters_distance : float, optional, default None
                        distance to clusterize nodes. The nodes footpath are from those cluster and
                        all the clusterized node will be connected to the cluster one with 0 length.
                        It will agregate nodes based on their distance to build "stop areas".
                        Works in increment of 10 (0.01, 0.1,0, 10, 100, etc)


        Builds
        ----------
        self.footpaths
        self.road_links :
                        add columns walk_time if road=True

        """
        try:
            self.footpaths = connectivity.build_footpaths(
                self.nodes,
                speed=speed,
                max_length=max_length,
                clusters_distance=clusters_distance,
                coordinates_unit=self.coordinates_unit,
                **kwargs,
            )
        except ValueError as e:  # Shape of passed values is (1, 3019), indices imply (1, 5847)
            print('an error has occured: ', e)
            if self.coordinates_unit == 'degree':
                clusters_distance = 0.00001  # deg (approx 1meters. to increase if error persist)
            else:
                clusters_distance = 1  # meter (to increate if error persist)
            print('now trying to run the method with clusters_distance = ' + str(clusters_distance))
            print(
                'this error is usually because some nodes are really close or at the same exact position.\
                You should do a node clustering first.'
            )
            self.footpaths = connectivity.build_footpaths(
                self.nodes,
                speed=speed,
                max_length=max_length,
                clusters_distance=clusters_distance,
                coordinates_unit=self.coordinates_unit,
                **kwargs,
            )

        if road:
            v = kwargs['speed'] * 1000 / 3600  # en m/s
            self.road_links['walk_time'] = self.road_links['length'] / v

    @track_args
    def preparation_ntlegs(
        self,
        short_leg_speed=2,
        long_leg_speed=10,
        threshold=1000,
        n_ntlegs=5,
        keep_centroids=False,
        max_ntleg_length=5000,
        zone_to_transit=True,
        road_to_transit=True,
        zone_to_road=False,
        prefix=False,
    ):
        """Builds the centroids and the non-transit links/legs (ntlegs), also known as connectors.
        Pameters short_leg_speed and long_leg_speed allow to model diferent types of access to the network (PT/private):
        for short connectors, the short_leg_speed is used - it represents a walk speed. For long connectors,
        which will occur for large zones at the edge of the study area, we may want to consider that the access
        to the network is made by car/taxi, and hence at a larger speed, the long_leg_speed. Function integrates a
        curve to smoothly go from short_leg_speed to long_leg_speed (can be understood as probability to access by foot or car).

        Requires
        ----------
        self.nodes
        self.zones

        Parameters
        ----------
        short_leg_speed : int, optional, default 2
                        Speed of the short legs, in km/h
        long_leg_speed : int, optional, default 10
                        Speed of the short legs, in km/h
        threshold : int, optional, default 1000
                        Threshold for the definition of the short and long legs
        n_ntlegs : int, optional, default 5
                        Number of ntlegs to create per zone (and per type)
        max_ntleg_length : int, optional, default 5000
                        maximal length of the ntlegs, in m
        zone_to_transit : bool, optional, default True
                        True to create links between zones and transit stops (nodes)
        road_to_transit : bool, optional, default True
                        True to create links between road_nodes and transit stops (nodes)
        zone_to_road : bool, optional, default False
                        True to create links between zones and road_nodes
        prefix : bool, optional, default False
                        If True, add prefixes to the index of the ntlegs
                        ztt_ (zone_to_transit), ztr_ (zone_to_road), rtt_ (road_to_transit)


        Builds
        ----------
        self.centroids
        self.zone_to_transit
        self.zone_to_road
        self.road_to_transit


        Examples
        --------
        ::
                                                                                                            sm.step_ntlegs(
                                                                                                                n_ntlegs=5,
                                                                                                                walk_speed=2,
                                                                                                                short_leg_speed=3,
                                                                                                                long_leg_speed=15,
                                                                                                                threshold=500,
                                                                                                                max_ntleg_length=5000,
                                                                                                            )
        """
        if keep_centroids:
            if not hasattr(self, 'centroids') or self.centroids.equals(
                integritymodel.geodataframe_place_holder('Point')
            ):
                self.centroids = self.zones.copy()
                self.centroids['geometry'] = self.centroids['geometry'].apply(lambda g: g.centroid)
        else:
            self.centroids = self.zones.copy()
            self.centroids['geometry'] = self.centroids['geometry'].apply(lambda g: g.centroid)
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
                coordinates_unit=self.coordinates_unit,
            )
            ntlegs['walk_time'] = ntlegs['time']
            self.zone_to_transit = ntlegs.loc[ntlegs['distance'] < length].copy()
            if prefix:
                self.zone_to_transit.index = 'ztt_' + pd.Series(self.zone_to_transit.index).astype(str)
            else:
                warnings.warn(
                    (
                        'zone_to_transit indexes does not have prefixes. This may cause collisions.'
                        'Consider using the option prefix=True. Prefixes will be added by default in'
                        'a future update'
                    ),
                    FutureWarning,
                )

        if zone_to_road:
            self.integrity_test_collision(sets=('road_nodes', 'zones'))

            ntlegs = engine.ntlegs_from_centroids_and_nodes(
                self.centroids,
                self.road_nodes,
                short_leg_speed=short_leg_speed,
                long_leg_speed=long_leg_speed,
                threshold=threshold,
                n_neighbors=n_ntlegs,
                coordinates_unit=self.coordinates_unit,
            )
            ntlegs['walk_time'] = ntlegs['time']
            self.zone_to_road = ntlegs.loc[ntlegs['distance'] < length].copy()
            if prefix:
                self.zone_to_road.index = 'ztr_' + pd.Series(self.zone_to_road.index).astype(str)
            else:
                warnings.warn(
                    (
                        'zone_to_road indexes does not have prefixes. This may cause collisions.'
                        'Consider using the option prefix=True. Prefixes will be added by default in'
                        'a future update'
                    ),
                    FutureWarning,
                )
        if road_to_transit:
            ntlegs = engine.ntlegs_from_centroids_and_nodes(
                self.nodes,
                self.road_nodes,
                short_leg_speed=short_leg_speed,
                long_leg_speed=long_leg_speed,
                threshold=threshold,
                n_neighbors=n_ntlegs,
                coordinates_unit=self.coordinates_unit,
            )
            ntlegs['walk_time'] = ntlegs['time']
            self.road_to_transit = ntlegs.loc[ntlegs['distance'] < length].copy()
            if prefix:
                self.road_to_transit.index = 'rtt_' + pd.Series(self.road_to_transit.index).astype(str)
            else:
                warnings.warn(
                    (
                        'road_to_transit indexes does not have prefixes. This may cause collisions.'
                        'Consider using the option prefix=True. Prefixes will be added by default in'
                        'a future update'
                    ),
                    FutureWarning,
                )

    def preparation_drop_redundant_zone_to_transit(self):
        """Keeps only relevant zone_to_transit connectors : verifies if the nodes linked to the zones
        exist in the self.links, and drops redundants

        Requires
        ----------
        self.links
        self.zone_to_transit
        self.zones

        Builds
        ----------
        self.zone_to_transit :
                        update zone_to_transit dataframe
        """
        self.zone_to_transit.sort_values('time', inplace=True)
        trips = self.links.groupby('a')['trip_id'].agg(set)
        df = self.zone_to_transit

        keep = []
        # access
        zones = set(df.loc[df['direction'] == 'access']['a'])
        for zone in tqdm(set(zones)):
            ztt = self.zone_to_transit.loc[self.zone_to_transit['a'] == zone]
            if len(ztt):
                ztt = ztt[['b']].reset_index()
                ztt['trips'] = [trips.get(n, set()) for n in ztt['b']]
                n = 1
                t = set()
                while len(ztt) > 0:
                    ztt['trips'] = [trips - t for trips in ztt['trips']]
                    ztt['n_trips'] = [len(trips) for trips in ztt['trips']]
                    index, t, n = ztt.iloc[0][['index', 'trips', 'n_trips']].values
                    ztt = ztt.loc[ztt['n_trips'] > 0]
                    keep.append(index)

        # egress
        zones = set(df.loc[df['direction'] != 'access']['b'])
        for zone in tqdm(set(zones)):
            ztt = self.zone_to_transit.loc[self.zone_to_transit['b'] == zone]
            if len(ztt):
                ztt = ztt[['a']].reset_index()
                ztt['trips'] = [trips.get(n, set()) for n in ztt['a']]
                n = 1
                t = set()
                while n > 0:
                    ztt['trips'] = [trips - t for trips in ztt['trips']]
                    ztt['n_trips'] = [len(trips) for trips in ztt['trips']]
                    ztt.sort_values('n_trips', ascending=False, inplace=True)
                    index, t, n = ztt.iloc[0][['index', 'trips', 'n_trips']].values
                    ztt = ztt.loc[ztt['n_trips'] > 0]
                    keep.append(index)
        self.zone_to_transit = self.zone_to_transit.loc[list(set(keep))]

    def preparation_drop_redundant_footpaths(self, access_time='time', log=False):
        """Reduce number of footpaths to optimize computation by performing
        a shortest path algorithm in the graph made of footpaths, road_links and road_to_transit.

        Requires
        ----------
        self.road_links
        self.road_to_transit
        self.footpaths

        Parameters
        ----------
        access_time : str, optional, default 'time'
                        Time column in road_to_transit and footpaths
        log : bool, optional, default False
                        If true, returns the old and new numbers of footpaths

        Builds
        ----------
        self.footpaths :
                        update footpaths dataframe

        """
        a = len(self.footpaths)
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.road_links[['a', 'b', 'walk_time']].values)
        g.add_weighted_edges_from(self.road_to_transit[['a', 'b', access_time]].values)
        g.add_weighted_edges_from(self.footpaths[['a', 'b', access_time]].values)
        self.footpaths['dijkstra_time'] = [
            nx.dijkstra_path_length(g, a, b) for a, b in self.footpaths[['a', 'b']].values
        ]
        self.footpaths = self.footpaths.loc[self.footpaths[access_time] <= self.footpaths['dijkstra_time']]
        self.footpaths.drop('dijkstra_time', axis=1, inplace=True)
        if log:
            print('reduced number of footpaths from', a, 'to', len(self.footpaths))

    def preparation_drop_redundant_zone_to_road(self, access_time='time', log=False):
        """Reduce number of zone_to_road connectors to optimize computation by performing
        a shortest path algorithm in the graph made of road_links and zone_to_road.

        Requires
        ----------
        self.road_links
        self.zone_to_road

        Parameters
        ----------
        access_time : str, optional, default 'time'
                        Time column in zone_to_road
        log : bool, optional, default False
                        If true, returns the old and new numbers of zone_to_road

        Builds
        ----------
        self.zone_to_road :
                        update zone_to_road dataframe

        """
        a = len(self.zone_to_road)
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.road_links[['a', 'b', 'walk_time']].values)
        g.add_weighted_edges_from(self.zone_to_road[['a', 'b', access_time]].values)

        self.zone_to_road['dijkstra_time'] = [
            nx.dijkstra_path_length(g, a, b) for a, b in self.zone_to_road[['a', 'b']].values
        ]
        self.zone_to_road = self.zone_to_road.loc[self.zone_to_road[access_time] <= self.zone_to_road['dijkstra_time']]
        self.zone_to_road.drop('dijkstra_time', axis=1, inplace=True)
        if log:
            print('reduced number of zone_to_road from', a, 'to', len(self.zone_to_road))

    def preparation_drop_redundant_road_to_transit(self, access_time='time', log=False):
        """Reduce number of road_to_transit connectors to optimize computation by performing
        a shortest path algorithm in the graph made of road_links and road_to_transit.

        Requires
        ----------
        self.road_links
        self.road_to_transit

        Parameters
        ----------
        access_time : str, optional, default 'time'
                        Time column in road_to_transit
        log : bool, optional, default False
                        If true, returns the old and new numbers of road_to_transit


        Builds
        ----------
        self.road_to_transit :
                        update road_to_transit dataframe

        """
        a = len(self.road_to_transit)
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.road_links[['a', 'b', 'walk_time']].values)
        g.add_weighted_edges_from(self.road_to_transit[['a', 'b', access_time]].values)

        self.road_to_transit['dijkstra_time'] = [
            nx.dijkstra_path_length(g, a, b) for a, b in self.road_to_transit[['a', 'b']].values
        ]
        self.road_to_transit = self.road_to_transit.loc[
            self.road_to_transit[access_time] <= self.road_to_transit['dijkstra_time']
        ]
        self.road_to_transit.drop('dijkstra_time', axis=1, inplace=True)
        if log:
            print('reduced number of road_to_transit from', a, 'to', len(self.road_to_transit))

    @track_args
    def preparation_cast_network(
        self,
        nearest_method='nodes',
        weight='length',
        penalty_factor=1,
        speed=3,
        replace_nodes=False,
        dumb_cast=False,
        **nc_kwargs,
    ):
        """Finds a path for the transport lines in the actual road network,
        to know on which roads a bus line is going, because the public transport links are defined
        between two stops, without road information.
        It will evaluate the best combination of nodes in the road network between the two stops.
        The evaluation is done with the distance.
        The results will be found as a list of road_links for each public transport link.
        If the transport network has modes on dedicated infrastructure (train, metro...), create two submodels
        and use a dumbcast on the dedicated infrastructure modes.

        Requires
        ----------
        self.nodes
        self.links
        self.road_nodes
        self.road_links

        Parameters
        ----------
        nearest_method :  ['nodes'|'links'], optional, default 'nodes'
                        Options are:

                        'nodes'   --(default) looks for the actual nearest node in road_nodes.
                        'links'   --looks for the nearest link to a stop in road_links and links the stop to its end_node (b)

        weight : str, optional, default 'length'
                        Column of road_links containing road_links length
        penalty_factor : int, optional, default 1
                        Multiplicative penality of weight
        speed : int, optional, default 3
                        Walk speed
        replace_nodes : bool, optional, default False
                        If True replaces nodes by road_nodes. If False, model will use road_to_transit ntlegs
        dumb_cast : bool, optional, default False
                        If True, the network is casted on himself (cast links on links and not on road_links).
                        It will still return the closest road_node for the stops.
        nodes_checkpoints :
                        mandatory transit nodes

        Builds
        ----------
        self.links :
                        add columns road_a,	road_b,	road_node_list,	road_link_list, road_length


        """
        if dumb_cast:
            nc = NetworkCaster(self.nodes, self.links, self.road_nodes)
            nc.dumb_cast()
        else:
            try:
                self.road_links[weight] + 1
            except TypeError:
                raise TypeError(str(weight) + ' should be an int or a float')

            nc = NetworkCaster(self.nodes, self.links, self.road_nodes, self.road_links, weight=weight)
            nc.build(
                nearest_method=nearest_method,
                penalty_factor=penalty_factor,
                coordinates_unit=self.coordinates_unit,
                geometry=True,
                **nc_kwargs,
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
        # if we do not replace the nodes by the road_nodes,
        # we have to provide road to transit legs...
        elif not dumb_cast:
            rc = nc.road_access['geometry'].reset_index()
            if self.coordinates_unit == 'degree':
                rc['length'] = skims.distance_from_geometry(rc['geometry'])
            elif self.coordinates_unit == 'meter':
                rc['length'] = rc['geometry'].apply(lambda x: x.length)
            else:
                raise ('Invalid coordinates_unit.')

            rc['time'] = rc['length'] / 1000 / speed * 3600
            to_road = rc.rename(columns={'node': 'a', 'road_node': 'b'})
            from_road = rc.rename(columns={'node': 'b', 'road_node': 'a'})
            to_road['direction'] = 'to_road'
            from_road['direction'] = 'from_road'
            concatenated = pd.concat([from_road, to_road])
            self.road_to_transit = concatenated.reindex().reset_index(drop=True)

    @track_args
    def preparation_map_matching(
        self,
        by: str = 'trip_id',
        sequence: str = 'link_sequence',
        n_neighbors_centroid: int = 10,
        radius_search: int = 500,
        on_centroid: bool = False,
        n_neighbors: int = 20,
        distance_max: int = 3000,
        nearest_method: str = 'radius',
        speed_limit: bool = False,
        turn_penalty: bool = False,
        routing: bool = True,
        overwrite_geom: bool = False,
        overwrite_nodes: bool = False,
        remove_duplicated_links_per_trips: bool = True,
        num_cores=1,
        **kwargs,
    ):
        """Mapmatch each trip_id in self.links to the road_network (self.road_links)

        Parameters
        ----------
        by : str,
                        links column name for each mapmaptching. they are group according to this column
        sequence : str,
                        links column giving the sequence of point for a given by (trip_id)
        routing : bool,
                        if True return the complete routing from the first to the last point on the road network.
        n_neighbors_centroid : int,
                        number of kneighbor in the first rough KNN on links centroid. if using on_centroid: you can go real high (ex: 2000)
        radius_search : int,
                        radius of kneighbor in the first rough KNN on links. markers are added to links every radius_search meters to be sure we find
                        them if on_centroid = False.
        on_centroid :bool,
                        if false add points along line for the KNN. so we actually find a really long highway with a centroid really far away.
                        Using True here is not recommended, it is the old behavior of this function wich caused problems.
        n_neighbors : int,
                        number of possible links for each point in the mapmatching. 10 top closest links
        distance_max : int
                        max radius to search candidat links for each gps point
        nearest_method: str [radius, knn, both]:
                        finding candidats with the radius, the knn or both. If radius is used, any point with 0 candidat will use the knn.
                        This occur when a point is really far away. we still want the closest roads so knn is use as it is fail proof.
        overwrite_geom : bool, optional
                        by default True
        overwrite_nodes: bool
                        'overwrite nodes: nodes shares between different trips will be duplicated and rename'
        remove_duplicated_links_per_trips: bool
                        for a trip (multiple links) remove rlinks in road_link_list if its duplicated between adjacent links.
                        ex: links 1 et 2 are [rlink_1, rlink_2] and [rlink_2, rlink_3] => [rlink_1, rlink_2] and [rlink_3]
                        this make sure that we dont count a road link multiple time qhen asigning load!
        num_cores : int,
                        parallelize.
        ----------
        Builds


        """
        if 'road_link_list' in self.links.columns:
            self.links = self.links.drop(columns=['road_link_list'])
        if 'road_node_list' in self.links.columns:
            self.links = self.links.drop(columns=['road_node_list'])
        if overwrite_nodes:
            print('overwrite nodes: nodes shares between different trips will be duplicated and rename')
            if 'index' in self.nodes.columns:
                self.nodes.set_index('index', inplace=True)
            elif self.nodes.index.name != 'index':
                self.nodes.index.name = 'index'
            self.links, self.nodes = duplicate_nodes(self.links, self.nodes)

        road_links = RoadLinks(
            self.road_links,
            n_neighbors_centroid=n_neighbors_centroid,
            radius_search=radius_search,
            on_centroid=on_centroid,
        )
        gps_tracks = get_gps_tracks(self.links, self.nodes, by=by, sequence=sequence)
        if num_cores == 1:
            matched_links, links_mat, _ = Multi_Mapmatching(
                gps_tracks,
                road_links,
                routing=routing,
                n_neighbors=n_neighbors,
                distance_max=distance_max,
                by=by,
                nearest_method=nearest_method,
                speed_limit=speed_limit,
                turn_penalty=turn_penalty,
                **kwargs,
            )
        else:
            matched_links, links_mat, _ = Parallel_Mapmatching(
                gps_tracks,
                road_links,
                routing=routing,
                n_neighbors=n_neighbors,
                distance_max=distance_max,
                by=by,
                nearest_method=nearest_method,
                speed_limit=speed_limit,
                turn_penalty=turn_penalty,
                num_cores=num_cores,
                **kwargs,
            )
        # we added a first node to to the mapmatching. we need to reshift the index
        # a link is 2 node. if we have 5 links, there are 6 points in mapmatching.
        matched_links = matched_links.shift(1)[1:]
        links_mat = links_mat.shift(1)[1:]

        matched_links['road_id_a'] = matched_links['road_id_a'].apply(lambda x: road_links.links_index_dict.get(x))
        matched_links['road_id_b'] = matched_links['road_id_b'].apply(lambda x: road_links.links_index_dict.get(x))
        road_a_dict = matched_links['road_id_a'].to_dict()
        road_b_dict = matched_links['road_id_b'].to_dict()
        offset_a_dict = matched_links['offset_a'].to_dict()
        offset_b_dict = matched_links['offset_b'].to_dict()
        length_dict = matched_links['length'].to_dict()
        self.links['road_a'] = self.links.index.map(road_a_dict.get)
        self.links['road_b'] = self.links.index.map(road_b_dict.get)
        self.links['offset_a'] = self.links.index.map(offset_a_dict.get)
        self.links['offset_b'] = self.links.index.map(offset_b_dict.get)
        self.links['length'] = self.links.index.map(length_dict.get)
        self.links = self.links.merge(
            links_mat[['road_node_list', 'road_link_list']], left_index=True, right_index=True, how='left'
        )

        if overwrite_nodes:
            # get nodes
            node_dict_a = self.links['a'].to_dict()
            node_dict_b = self.links['b'].to_dict()
            matched_nodes = matched_links
            matched_nodes = matched_nodes.reset_index()
            matched_nodes['node_index'] = matched_nodes['index'].apply(lambda x: node_dict_a.get(x))
            matched_nodes_a = matched_nodes.set_index('road_id_a')[['node_index', 'offset_a', 'trip_id']].rename(
                columns={'offset_a': 'offset'}
            )
            matched_nodes_b = (
                matched_nodes.groupby('trip_id')[['road_id_b', 'offset_b', 'index']]
                .agg('last')
                .reset_index()
                .set_index('road_id_b')
            )
            matched_nodes_b['node_index'] = matched_nodes_b['index'].apply(lambda x: node_dict_b.get(x))
            matched_nodes_b = matched_nodes_b.drop(columns=['index'])

            matched_nodes_b = matched_nodes_b.rename(columns={'offset_b': 'offset'})
            matched_nodes = pd.concat([matched_nodes_a, matched_nodes_b])
            matched_nodes = matched_nodes.rename(columns={'matched_nodes': 'road_id'})

            def matched(line, offset):
                return line.interpolate(offset)

            geom_dict = road_links.links.set_index('index')['geometry'].to_dict()
            matched_nodes['geometry'] = matched_nodes.index.map(geom_dict.get)
            matched_nodes['geometry'] = matched_nodes.apply(lambda x: matched(x['geometry'], x['offset']), axis=1)
            geom_dict = matched_nodes.set_index('node_index')['geometry'].to_dict()
            self.nodes['new_geometry'] = self.nodes.index.map(geom_dict.get)
            self.nodes['geometry'] = self.nodes['new_geometry'].combine_first(self.nodes['geometry'])
            self.nodes.drop(columns=['new_geometry'], inplace=True)

        if overwrite_geom:
            links_geom_dict = road_links.links.set_index('index')['geometry'].to_dict()

            def get_geom(ls, geom_dict):
                if type(ls) is not list:
                    return None
                # transform road_links index to linetring
                ls = [*map(geom_dict.get, ls)]
                new_line = []
                # for each linestring
                for link in ls:
                    ## init on first iteration
                    if len(new_line) == 0:
                        # get list of point instead of linetring
                        new_line = [Point(x, y) for x, y in zip(link.coords.xy[0], link.coords.xy[1])]
                    else:
                        # get list of point instead of linetring
                        link = [Point(x, y) for x, y in zip(link.coords.xy[0], link.coords.xy[1])]
                        # the link geom is reverse (B-A instead of A-B), reverse it
                        if link[0] != new_line[-1]:
                            link.reverse()
                        # append other points to the linetring
                        new_line.extend(link[1:])
                # return a linetring of all road points.
                return LineString(new_line)

            self.links['new_geometry'] = self.links['road_link_list'].apply(lambda x: get_geom(x, links_geom_dict))
            self.links['geometry'] = self.links['new_geometry'].combine_first(self.links['geometry'])
            self.links.drop(columns=['new_geometry'], inplace=True)

        if overwrite_geom and overwrite_nodes:
            from syspy.spatial.geometries import cut

            def crops(links, nodes):
                # cut linestring at nodes a and b
                nodes_dict = nodes['geometry'].to_dict()
                nodes_a = [*map(nodes_dict.get, links['a'])]
                nodes_b = [*map(nodes_dict.get, links['b'])]
                geometries = links['geometry'].values
                res = []
                for g, a, b in zip(geometries, nodes_a, nodes_b):
                    offset_a = g.project(a)
                    g = cut(g, offset_a)[1]
                    if g is None:
                        res.append(LineString([a, b]))
                        continue
                    offset_b = g.project(b)
                    g = cut(g, offset_b)[0]
                    if g is None:
                        res.append(LineString([a, b]))
                    else:
                        res.append(g)
                return res

            self.links['geometry'] = crops(self.links, self.nodes)

        def remove_dup_in_road_link_list(links):
            """
            ex: links 1 et 2 are [rlink_1, rlink_2] and [rlink_2, rlink_3] => [rlink_1, rlink_2] and [rlink_3]
            this make sure that we dont count a road link multiple time qhen asigning load!
            """
            res = {}
            for t in links['trip_id'].unique():
                trip = links[links['trip_id'] == t]
                visited_links = set()
                for i, ls in trip['road_link_list'].items():
                    if type(ls) is not list:
                        ls = []
                    new_ls = [el for el in ls if el not in visited_links]
                    visited_links.update(new_ls)
                    res[i] = new_ls
            links['road_link_list'] = links.index.map(res)
            return links

        if remove_duplicated_links_per_trips:
            self.links = remove_dup_in_road_link_list(self.links)

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
        time_shift=None,
        route_types=None,
    ):
        """Builds the necessary tables to perform analysis_mode_utility and step_logit.
        They contain the parameters of the nested logit.
        For the neste logit we should have 1 >= mode >= pt_mode >= pt_path > 0.
        If the three of them are equal to 1 the nested logit will be equivalent to a flat logit.

        Does not require specific attributes in self.

        Parameters
        ----------
        mode : int, optional, default 1
                        phi parameter used in the logit choice between modes
        pt_mode : int, optional, default 1
                        phi parameter used in the logit choice between pt modes
        pt_path : int, optional, default 1
                        phi parameter used in the logit choice between pt paths
        segments : list, optional, default []
                        Demand segments we want to use in the logit
        time : int, optional, default -1
                        number of utility units by seconds
        price : int, optional, default -1
                        number of utility units by currency unit
        transfers : int, optional, default -1
                        number of utility units by transfer
        time_shift : int, optional, default None
                        Used with timetable (time expanded) models. Number of utility units by time_shift

        Builds
        ----------
        self.mode_utility :
                        Modal constants, per mode and per segment
        self.mode_nests :
                        Structure of the nested logit per segment
        self.logit_scales :
                        Scales of the nested logit per segment (parameters phi)
        self.utility_values
                        Utility values of time, price, transfers and time_shift per segment


        """
        # TODO : move to preparation
        # utility values
        if time_shift is None:
            self.utility_values = pd.DataFrame(
                {'root': pd.Series({'time': time, 'price': price, 'ntransfers': transfers, 'mode_utility': 1})}
            )
        else:
            self.utility_values = pd.DataFrame(
                {
                    'root': pd.Series(
                        {
                            'time': time,
                            'price': price,
                            'ntransfers': transfers,
                            'mode_utility': 1,
                            'time_shift': time_shift,
                        }
                    )
                }
            )
        self.utility_values.index.name = 'value'
        self.utility_values.columns.name = 'segment'

        if route_types is None:
            link_rt = set(self.links['route_type'].unique())
            route_types = link_rt.union({'car', 'walk', 'root'})

        # mode_utility
        self.mode_utility = pd.DataFrame({'root': pd.Series({rt: 0 for rt in route_types})})

        self.mode_utility.index.name = 'route_type'
        self.mode_utility.columns.name = 'segment'

        # mode nests
        self.mode_nests = pd.DataFrame({'root': pd.Series({rt: 'pt' for rt in route_types})})

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
    def preparation_clusterize_zones(self, max_zones=500, cluster_column=None, is_od_stack=False, **kwargs):
        """Clusterize zones to optimize computation time.

        Requires
        ----------
        self.zones
        self.volumes

        Parameters
        ----------
        max_zones : int, optional, default 500
                        _description_, by
        cluster_column : string, optional, default None
                        cluster column in self.zones if clusters are already defined
        is_od_stack : bool, optional, default False
                        If True, requires table od_stack

        Builds
        ----------
        self.zones
        self.volumes
        self.cluster_series

        """

        zones = self.zones
        zones['geometry'] = zones['geometry'].apply(lambda g: g.buffer(1e-9))
        self.micro_zones = zones.copy()
        self.micro_volumes = self.volumes.copy()
        if is_od_stack:
            self.micro_od_stack = self.od_stack.copy()
            self.zones, self.volumes, self.cluster_series, self.od_stack = renumber.renumber_quetzal(
                self.micro_zones, self.micro_volumes, self.micro_od_stack, max_zones, cluster_column, **kwargs
            )
        else:
            self.zones, self.volumes, self.cluster_series = renumber.renumber(
                self.micro_zones, self.micro_volumes, max_zones, cluster_column, **kwargs
            )

    @track_args
    def preparation_clusterize_nodes(
        self, n_clusters=None, adaptive_clustering=False, distance_threshold=150, prefix=None, fast=False, **kwargs
    ):
        """Create nodes clusters to optimize computation time.
        It will agregate nodes based on their relative distance to build "stop areas"

        Requires
        ----------
        self.nodes

        Parameters
        ----------
        n_clusters : int, optional, default None
                        Number of nodes clusters
        adaptive_clustering : bool, optional, default False
                        If True, will define itself the number of clusters.
        distance_threshold : int, optional, default 150
                        If n_cluster and adaptive_clustering are None,
                        clustering will be done using a distance threshold
        fast : bool, optional, default False
                        uses DBSCAN. faster and more memory efficient
                        for distance clustering

        Builds
        ----------
        self.links :
                        contain recomputed links with the clusterized nodes
        self.nodes :
                        contain the clusterized nodes
        self.disaggregated_nodes :
                        contain the former nodes

        """
        self.disaggregated_nodes = self.nodes.copy()

        if adaptive_clustering:
            if 'clustering_zones' not in self.__dict__.keys():
                self.clustering_zones = self.zones.copy()
            self.nodes = connectivity.adaptive_clustering(self.nodes, self.clustering_zones, **kwargs)
            self.links, self.nodes, self.node_clusters, self.node_parenthood = connectivity.node_clustering(
                self.links, self.nodes, n_clusters, group_id='adaptive_cluster_id'
            )
        elif n_clusters:
            if len(self.nodes) <= n_clusters:
                return
            self.links, self.nodes, self.node_clusters, self.node_parenthood = connectivity.node_clustering(
                self.links, self.nodes, n_clusters, **kwargs
            )
        else:
            if fast:
                self.nodes['cluster'] = spatial.DBSCAN_sclustering(
                    self.nodes, distance_threshold=distance_threshold, **kwargs
                )
            else:
                self.nodes['cluster'] = spatial.agglomerative_clustering(
                    self.nodes, distance_threshold=distance_threshold, **kwargs
                )
            self.links, self.nodes, self.node_clusters, self.node_parenthood = connectivity.node_clustering(
                self.links, self.nodes, n_clusters, group_id='cluster'
            )
        self.node_parenthood = self.node_parenthood[['cluster', 'geometry']]
        self.node_clusters['geometry'] = self.node_clusters['geometry'].apply(lambda g: g.buffer(1e-9))

        # add prefixes
        if prefix:
            self._add_type_prefixes({'nodes': prefix})
            func = lambda x: prefix + str(x).split(prefix)[-1]
            self.node_parenthood['cluster'] = self.node_parenthood['cluster'].apply(func)

    def preparation_map_tracks(
        self, agg={'speed': lambda s: s.mean() * 3.6}, buffer=50, smoothing_span=100, *args, **kwargs
    ):
        """Grand mystère

        Requires
        ----------
        self.nodes

        Parameters
        ----------
        agg : dict, optional
                        _description_, by default {'speed': lambda s: s.mean() * 3.6}
        buffer : int, optional
                        _description_, by default 50
        smoothing_span : int, optional
                        _description_, by default 100

        Builds
        ----------

        """
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
                smoothing_span=smoothing_span,
            )
            times['trip_id'] = trip_id
            to_concat.append(times)

        # INDEX
        self.road_links.drop(['index'], axis=1, errors='ignore', inplace=True)
        indexed = self.road_links.reset_index().set_index(['a', 'b'])['index'].to_dict()
        concatenated = pd.concat(to_concat)
        concatenated['road_link'] = concatenated.apply(lambda r: indexed[(r['a'], r['b'])], axis=1)
        aggregated = concatenated.groupby(['road_link']).agg(agg)

        for c in aggregated.columns:
            self.road_links[c] = aggregated[c]
        self.track_links = concatenated.drop(['a', 'b'], axis=1)
