import pandas as pd

from quetzal.analysis import analysis
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
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
    def step_road_pathfinder(self, **kwargs):
        """
        * requires: zones, road_links, zone_to_road
        * builds: road_paths
        """
        print('FutureWarning: ça va changer')
        road_links = self.road_links
        road_links['index'] = road_links.index
        indexed = road_links.set_index(['a', 'b']).sort_index()
        ab_indexed_dict = indexed['index'].to_dict()

        def node_path_to_link_path(road_node_list, ab_indexed_dict):
            tuples = [
                (road_node_list[i], road_node_list[i+1]) 
                for i in range(len(road_node_list)-1)
            ]
            road_link_list = [ab_indexed_dict[t] for t in tuples]
            return road_link_list

        road_graph = nx.DiGraph()
        road_graph.add_weighted_edges_from(
            self.road_links[['a', 'b', 'time']].values.tolist()
        )
        road_graph.add_weighted_edges_from(
            self.zone_to_road[['a', 'b', 'time']].values.tolist()
        )

        l = []
        for origin in tqdm(list(self.zones.index)):
            lengths, paths = nx.single_source_dijkstra(road_graph, origin)
            for destination in list(self.zones.index):
                try:
                    length = lengths[destination]
                    path = paths[destination]
                    node_path = path[1:-1]
                    link_path = node_path_to_link_path(node_path, ab_indexed_dict)
                    try:
                        ntlegs = [(path[0], path[1]), (path[-2], path[-1])]
                    except IndexError:
                        ntlegs = []
                    l.append( [origin, destination, path, node_path, link_path, ntlegs, length])
                except KeyError:
                    l.append( [origin, destination, path, node_path, link_path, ntlegs, length])
                    
        self.car_los = pd.DataFrame(
            l, 
            columns=['origin', 'destination', 'path','node_path', 'link_path', 'ntlegs', 'time']
        )

    @track_args
    def step_pt_pathfinder(self, **kwargs):
        """
        * requires: zones, links, footpaths, zone_to_road, zone_to_transit
        * builds: pt_paths
        """
        self.links = engine.graph_links(self.links)
        publicpathfinder = PublicPathFinder(self)
        publicpathfinder.find_best_paths(**kwargs)
        self.pt_los = publicpathfinder.paths.copy()
        self.pt_los = analysis.path_analysis_od_matrix(
            od_matrix=self.pt_los,
            links=self.links,
            nodes=self.nodes,
            centroids=self.centroids,
        )

    @track_args
    def step_evaluation(self, **kwargs):
        """
        * requires: pt_paths, road_paths, volumes
        * builds: shares
        """
        pass

    @track_args
    def step_build_los(
         self,
         build_car_skims=True,
         token=None,
         nb_clusters=20,
         skim_matrix_kwargs={}
        ):
        """
        * requires: pt_los
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
    def step_assignment(
        self,
        volume_column='volume_pt',
        road=False,
        **loaded_links_and_nodes_kwargs
        ):
        """
        Assignment step
            * requires: links, pt_paths, road_links, road_paths, shares, nodes
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

