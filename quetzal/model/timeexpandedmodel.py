
from quetzal.model import stepmodel
from quetzal.engine import time_expanded_utils as te_utils
import pandas as pd
import networkx as nx
import numpy as np
from shapely import geometry
from syspy.spatial import spatial
from tqdm import tqdm
import warnings
from functools import wraps
import shutil
import ntpath
import uuid

def read_hdf(filepath, *args, **kwargs):
    m = TimeExpandedModel(hdf_database=filepath, *args, **kwargs)
    return m

def read_zip(filepath, *args, **kwargs):
    try:
        m = TimeExpandedModel(zip_database=filepath, *args, **kwargs)
        return m
    except : 
        # the zip is a zipped hdf and can not be decompressed
        return read_zipped_hdf(filepath, *args, **kwargs)

def read_zipped_hdf(filepath, *args, **kwargs):
    filedir = ntpath.dirname(filepath)
    tempdir = filedir + '/quetzal_temp' + '-' + str(uuid.uuid4())
    shutil.unpack_archive(filepath, tempdir)
    m = read_hdf(tempdir + r'/model.hdf', *args, **kwargs)
    shutil.rmtree(tempdir)
    return m


def read_json(folder):
    m = TimeExpandedModel(json_folder=folder)
    return m

class TimeExpandedModel(stepmodel.StepModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 'time_interval' in dir(self):
            self.time_interval = [0, 24*3600-1]

    def preparation_clusterize_stops(self, distance_threshold=1000, walking_speed=3):
        """
        Create node clusters with an agglomerative clustering approach and a distance threshold.
        For each cluster, compute the average transfer length and corresponding walking duration.
        """
        #TODO faster transfer length
        # Create clusters
        self.nodes['cluster_id'] = spatial.agglomerative_clustering(
            self.nodes,
            distance_threshold=distance_threshold
        )
        # Save nodes
        self.disaggregated_nodes = self.nodes.copy()

        # Replace in model
        stop_id_to_cluster = self.nodes['cluster_id'].to_dict()
        self.links['disaggregated_a'] = self.links['a']
        self.links['disaggregated_b'] = self.links['b']
        self.links['a'] = self.links['a'].apply(lambda x: stop_id_to_cluster[x])
        self.links['b'] = self.links['b'].apply(lambda x: stop_id_to_cluster[x])
        self.links = self.links.loc[self.links['a']!=self.links['b']]

        # Compute cluster characteristics
        self.nodes['transfer_length'] = self.nodes['geometry'].copy()
        g = self.nodes.groupby('cluster_id').get_group(1)
        def transfer_length(g):
            """
            For a list of stops, compute the average length between each pair of stops
            """
            if len(g)==1:
                return 0
            else:
                lengths = []
                for i in range(len(g)):
                    for j in range(i+1, len(g)):
                        lengths.append(geometry.LineString(g.iloc[[i,j]].values).length)
                return sum(lengths) / len(lengths)

        self.nodes = self.nodes.groupby('cluster_id').agg(
            {
                'geometry': lambda x: geometry.MultiPoint(x).centroid,
                'transfer_length': lambda x: transfer_length(x),
            }
        )
        self.nodes['transfer_duration'] = self.nodes['transfer_length'] / (walking_speed / 3.6)

    def preparation_dense_footpaths(self, max_length=1000, walking_speed=3):
        self.footpaths = te_utils.build_dense_footpaths(
            self.nodes, max_length=max_length, walking_speed=walking_speed
        )

    def _preparation_create_graph_edges(self, boarding_time=0, min_transfer_time=0):

        # PT edges and transfer
        edges, graph_nodes = te_utils.pt_edges_and_nodes_from_links(
            self.links, self.nodes,
            time_interval=self.time_interval,
            boarding_time=boarding_time
        )
        # access / egress
        access_edges = te_utils.create_access_edges(
            graph_nodes,
            self.zone_to_transit, 
            time_interval=self.time_interval
        )

        egress_edges = te_utils.create_egress_edges(
            graph_nodes,
            self.zone_to_transit, 
            time_interval=self.time_interval
        )

        footpath_edges = te_utils.create_footpath_edges(
            graph_nodes,
            self.footpaths
        )

        self.edges = pd.concat([edges, access_edges, egress_edges, footpath_edges])
        self.graph_nodes = graph_nodes

    def step_pt_pathfinder(self, boarding_time, min_transfer_time=0, ntlegs_penalty=1e9):
        # WARNING: all walk paths are not computed (one must take a trip to reach an alighting node)
        #TODO: solve this issue (adding an all walk pathfinder?)

        # create edges
        self._preparation_create_graph_edges(boarding_time=boarding_time, min_transfer_time=min_transfer_time)
        
        # build graph
        edge_list = [tuple([a, b, w]) for a,b,w in self.edges[['a', 'b', 'weight']].values.tolist()]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edge_list)

        # Build pole set
        zone_dep_nodes = self.edges.loc[self.edges['type'] == 'access', 'a'].drop_duplicates().values.tolist()
        zone_arr_nodes = self.edges.loc[self.edges['type']=='zone_to_zone', 'b'].drop_duplicates().values.tolist()
        zone_nodes = zone_dep_nodes + zone_arr_nodes

        # compute LOS
        los = te_utils.sparse_los_from_nx_graph(
            G, zone_nodes, sources=zone_dep_nodes, cutoff=np.inf+ntlegs_penalty
        )

        los = los.loc[los['destination'].apply(len)==2] # remove destinations that are departure nodes

        los = los.rename(columns={'path': 'node_path'})
        los['departure_time'] = los['origin'].apply(lambda x: x[2])
        los['arrival_time'] = los['node_path'].apply(lambda x: x[-2][2])  # before last node
        los['origin'] = los['origin'].apply(lambda x: x[0])
        los['destination'] = los['destination'].apply(lambda x: x[0])
        los = los[los['origin']!=los['destination']]
        
        # build paths, boardings, transfers
        los = te_utils.get_edge_path(los)
        los = te_utils.get_model_link_path(los, self.edges)
        los = te_utils.get_boarding_links(los, self.edges)

        # transfers
        los['ntransfers'] = los['boarding_links'].apply(len) - 1
        los['ntransfers'] = los['ntransfers'].clip(0)

        self.pt_los = los

    def no_step_logit(
        self, 
        decimals=None, 
        n_paths_max=None,
        nchunks=100, 
        keep_utililities=False, 
        keep_probabilities=False
    ):

        worker = type(self)()
        planner = self.te_los.set_index('origin', drop=False)
        planner.index.name = 'index'
        worker.mode_nests = self.mode_nests
        worker.logit_scales = self.logit_scales
        worker.segments = self.segments
        
        index = list(set(planner.index))
        groups = [
            [
                index[i + nchunks*n ]
                for n in range(len(index) // nchunks + 1)
                if i+nchunks*n < len(index)
            ]
            for i in range(nchunks)
        ]
        
        # call logit
        utility_chunks = []
        probability_chunks = []
        los_chunks = []
        for group in tqdm(groups) :
            worker.te_los = planner.loc[group].reset_index(drop=True)
            worker._unique_model_segmented_logit(
                time_expanded=True, 
                decimals=decimals, 
                n_paths_max=n_paths_max
            )
            
            # keep data
            los_chunks.append(worker.te_los)
            
            # odt arrays
            if keep_utililities:
                utility_chunks.append(worker.utilities)
            if keep_probabilities:
                probability_chunks.append(worker.probabilities)

        self.te_los = pd.concat(los_chunks)
        
        # odt arrays
        if keep_utililities:
            self.utilities = pd.concat(utility_chunks)
        if keep_probabilities:
            self.probabilities = pd.concat(probability_chunks)


    def analysis_paths(
        self, 
        typed_edges=True, 
        boardings=True, 
        alightings=False, 
        transfers=False, 
        kpis=True, 
        ntlegs_penalty=1e9
    ):
        self.pt_los = te_utils.analysis_paths(
            self.pt_los, self.edges, typed_edges=typed_edges
        )
        
        if boardings:
            self.pt_los = te_utils.get_boarding_links(self.pt_los, self.edges)

            # transfers
            self.pt_los['ntransfers'] = self.pt_los['boarding_links'].apply(len) - 1
            self.pt_los['ntransfers'] = self.pt_los['ntransfers'].clip(0)

        if kpis:
            self.pt_los = te_utils.analysis_lengths(
                self.pt_los, self.links, self.footpaths, self.zone_to_transit
            )
            self.pt_los = te_utils.analysis_durations(self.pt_los, self.edges, ntlegs_penalty)
    
    def build_te_los(self):
        """
        Requires time expanded volumes.
        Expand los as in time expanded volumes
        """
        # index paths
        self.los.index = self.los.index.set_names(['path_id'])
        self.los.drop('path_id', axis=1, inplace=True, errors='ignore')
        self.los.reset_index(inplace=True)
        # Create te_los
        columns_to_keep = {'origin', 'destination', 'path_id', 'departure_time', 'arrival_time',
            'transfers', 'route_types', 'route_type', 'ntransfers', 'time', 'price'}
        columns = list(set(self.los.columns).intersection(columns_to_keep))
        self.te_los = self.volumes[['origin', 'destination', 'wished_departure_time']].merge(
            self.los[columns],
            on=['origin', 'destination']
        )

    def analysis_time_shift(self, threshold=0):
        def time_shift(departure_time, wished_time, threshold=threshold):
            return np.maximum((abs(departure_time - wished_time) - threshold), 0)

        temp = self.te_los[['departure_time', 'wished_departure_time']].values.T
        self.te_los['time_shift'] = time_shift(temp[0], temp[1])