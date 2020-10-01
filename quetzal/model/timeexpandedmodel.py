
from quetzal.model import stepmodel
from quetzal.engine import time_expanded_pathfinder as te_pathfinder
import pandas as pd
import networkx as nx
import numpy as np

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

    def __init__(self, time_interval=[0,24*3600-1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_interval = time_interval

    def _preparation_create_graph_edges(self, boarding_cost=300):

        # PT edges and transfer
        edges, graph_nodes = te_pathfinder.pt_edges_and_nodes_from_links(
            self.links,
            time_interval=self.time_interval,
            boarding_cost=boarding_cost
        )
        # access / egress
        access_edges = te_pathfinder.create_access_edges(
            graph_nodes,
            self.zone_to_transit, 
            time_interval=self.time_interval
        )

        egress_edges = te_pathfinder.create_egress_edges(
            graph_nodes,
            self.zone_to_transit, 
            time_interval=self.time_interval
        )

        footpath_edges = te_pathfinder.create_footpath_edges(
            graph_nodes,
            self.footpaths
        )

        self.edges = pd.concat([edges, access_edges, egress_edges, footpath_edges])

    def step_pt_pathfinder(self, boarding_cost, ntlegs_penalty=1e9):
        # create edges
        self._preparation_create_graph_edges(boarding_cost=boarding_cost)
        
        # build graph
        edge_list = [tuple([a, b, w]) for a,b,w in self.edges[['a', 'b', 'weight']].values.tolist()]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edge_list)

        # Build pole set
        zone_dep_nodes = self.edges.loc[self.edges['type'] == 'access', 'a'].drop_duplicates().values.tolist()
        zone_arr_nodes = self.edges.loc[self.edges['type']=='zone_to_zone', 'b'].drop_duplicates().values.tolist()
        zone_nodes = zone_dep_nodes + zone_arr_nodes

        # compute LOS
        los = te_pathfinder.sparse_los_from_nx_graph(
            G, zone_nodes, sources=zone_dep_nodes, cutoff=np.inf+ntlegs_penalty
        )

        los = los.loc[los['destination'].apply(len)==2] # remove destinations that are departure nodes

        los = los.rename(columns={'path': 'node_path'})
        los['departure_time'] = los['origin'].apply(lambda x: x[2])
        los['arrival_time'] = los['node_path'].apply(lambda x: x[-2][2])  # before last node
        los['origin'] = los['origin'].apply(lambda x: x[0])
        los['destination'] = los['destination'].apply(lambda x: x[0])
        los = los[los['origin']!=los['destination']]

        self.pt_los = los

    
    def analysis_paths(self, typed_edges=True, kpis=True, ntlegs_penalty=1e9):
        self.pt_los = te_pathfinder.analysis_paths(
            self.pt_los, self.edges, typed_edges=typed_edges
        )
        if kpis:
            self.pt_los = te_pathfinder.analysis_lengths(
                self.pt_los, self.links, self.footpaths, self.zone_to_transit
            )
            self.pt_los = te_pathfinder.analysis_transfers(self.pt_los)
            self.pt_los = te_pathfinder.analysis_durations(self.pt_los, self.edges, ntlegs_penalty)