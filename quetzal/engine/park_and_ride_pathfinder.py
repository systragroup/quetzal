import numpy as np
import pandas as pd
from quetzal.engine import engine
from quetzal.engine.pathfinder_utils import path_and_duration_from_graph


class ParkRidePathFinder:
    def __init__(self, model, walk_on_road=False):
        self.zones = model.zones.copy()
        self.links = engine.graph_links(model.links.copy())
        self.road_links = model.road_links
        self.pr_nodes = set(model.nodes.loc[model.nodes['parking_spots'].astype(bool)].index)
        self.road_to_transit = model.road_to_transit.loc[
            model.road_to_transit['b'].isin(self.pr_nodes)]
        self.zone_to_road = model.zone_to_road.loc[
            model.zone_to_road['direction'] == 'access']
        self.transit_to_zone = model.zone_to_transit.loc[
            model.zone_to_transit['direction'] != 'access']
        self.footpaths = model.footpaths

    def build_graph(self, **kwargs):
        self.nx_graph = engine.multimodal_graph(
            self.links,
            ntlegs=pd.concat([self.zone_to_road, self.transit_to_zone]),
            footpaths=pd.concat([self.footpaths, self.road_to_transit, self.road_links]),
            pole_set=set(self.zones.index),
            **kwargs
        )

    def find_best_path(self, cutoff=np.inf, od_set=None, **kwargs):
        self.build_graph(**kwargs)
        pr_los = path_and_duration_from_graph(
            self.nx_graph,
            pole_set=set(self.zones.index),
            cutoff=cutoff,
            od_set=od_set
        )

        pr_los['pathfinder_session'] = 'park_and_ride'
        pr_los['path'] = [tuple(p) for p in pr_los['path']]
        self.paths = pr_los
