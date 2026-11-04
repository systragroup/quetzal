import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from syspy.spatial.spatial import add_geometry_coordinates
from quetzal.os.parallel_call import parallel_executor
from quetzal.engine.pathfinder_utils import get_path, get_edge_path, sparse_matrix, fast_dijkstra


def get_nodes_list(links: gpd.GeoDataFrame, by: str = 'trip_id', sequence: str = 'link_sequence'):
    """
    format links to a format used by the Multi Mapmatching
    """
    links = links[[by, sequence, 'geometry']].sort_values([by, sequence]).copy()
    if 'index' not in links.columns:
        links = links.reset_index()

    last = links.groupby(by).last().reset_index()

    last['geometry'] = last['geometry'].apply(lambda g: Point(g.coords[-1]))  # type: ignore
    links['geometry'] = links['geometry'].apply(lambda g: Point(g.coords[0]))  # type: ignore

    links['order'] = links.groupby(by).cumcount()
    last_order = links.groupby(by)['order'].last() + 1
    last['order'] = last[by].apply(last_order.get)

    points_list = pd.concat([links, last]).sort_values([by, 'order'])
    points_list = gpd.GeoDataFrame(points_list)
    points_list = points_list.rename(columns={'index': 'link_index'}).reset_index(drop=True)

    return points_list


def emission_logprob(distance, SIGMA, p):
    return 0.5 * (distance / SIGMA) ** p


def transition_logprob(dijkstra_dist, points_dist, BETA, diff):
    if diff:
        return abs(dijkstra_dist - points_dist) / BETA
    else:
        return dijkstra_dist / BETA


def nearest(one, nbrs, n_neighbors=1):
    try:
        # Assert df_many.index.is_unique
        assert one.index.is_unique
    except AssertionError:
        msg = 'Index of one and many should not contain duplicates'
        print(msg)

    df_one = add_geometry_coordinates(one.copy())

    y = df_one[['x_geometry', 'y_geometry']].values

    indices = nbrs.kneighbors(y, n_neighbors=n_neighbors, return_distance=False)
    indices = pd.DataFrame(indices).stack().reset_index()
    indices = indices.rename(columns={'level_0': 'ix_one', 'level_1': 'rank', 0: 'index_nn'})
    return indices


class RoadNetwork:
    def __init__(self, road_links, road_nodes, n_neighbors=4, linestring_length=True):
        self.road_links = road_links
        if linestring_length:
            self.road_links['length'] = self.road_links.length

        self.csgraph, self.node_index = sparse_matrix(self.road_links[['a', 'b', 'length']].values)
        self.index_node = {v: k for k, v in self.node_index.items()}

        road_nodes = road_nodes.reset_index()

        road_nodes['sparse_index'] = road_nodes['index'].map(self.node_index)
        road_nodes = road_nodes[~np.isnan(road_nodes['sparse_index'])]
        road_nodes['sparse_index'] = road_nodes['sparse_index'].astype(np.int32)
        self.road_nodes = road_nodes.set_index('sparse_index')

        x = add_geometry_coordinates(self.road_nodes.copy())[['x_geometry', 'y_geometry']].values
        self.n_neighbors = n_neighbors
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(x)

        # origins = list(self.node_index.values())
        # self.dist_matrix, _ = fast_dijkstra(csgraph=self.csgraph, indices=origins, limit=np.inf)

        self.road_links['sparse_a'] = self.road_links['a'].map(self.node_index.get)
        self.road_links['sparse_b'] = self.road_links['b'].map(self.node_index.get)
        self.ab_dict = self.road_links.reset_index().set_index(['sparse_a', 'sparse_b'])['index'].to_dict()

    def shortest_path_distance(self, origins: np.ndarray, destinations: np.ndarray, limit=np.inf) -> np.ndarray:
        unique_origins = list(set(origins))
        origin_index = {o: i for i, o in enumerate(unique_origins)}
        origins_index = np.array([*map(origin_index.get, origins)], dtype=np.int32)
        dist_matrix, _ = fast_dijkstra(csgraph=self.csgraph, indices=unique_origins, limit=limit)
        return dist_matrix[origins_index, destinations]


def viterbi(log_emission: np.ndarray, log_transition: csc_matrix) -> list[int]:
    """
    log_emission[t,candidats]:
    log_transition[t*candidats,candidats]: sparse_matrix, flatten to n_candicats*n_points
    """
    n = len(log_emission)
    n_candidates = len(log_transition.indptr) - 1
    scores = log_emission[0].copy()
    predecessors = []

    for t in range(n - 1):
        start = t * n_candidates
        end = (t + 1) * n_candidates
        candidates = scores[:, None] + log_transition[start:end].toarray()
        # Best predecessor for each destination
        pred = np.argmin(candidates, axis=0)
        scores = candidates[pred, np.arange(len(pred))] + log_emission[t + 1]
        predecessors.append(pred)
    # Best final candidate
    state = np.argmin(scores)
    # Backtrack
    path = [state]
    for pred in reversed(predecessors):
        state = pred[state]
        path.append(state)

    return path[::-1]  # type: ignore


def get_candidates(
    nodes_list: gpd.GeoDataFrame,
    road_network: RoadNetwork,
    by='trip_id',
    SIGMA: float = 4.07,
    BETA: float = 3,
    POWER: float = 2,
    DIFF=True,
):
    nodes_list = nodes_list.reset_index(drop=True)
    if 'order' not in nodes_list.columns:
        nodes_list['order'] = nodes_list.groupby(by).cumcount()

    candidats = nearest(nodes_list, road_network.nbrs, n_neighbors=road_network.n_neighbors).drop(columns=['rank'])
    candidats = candidats.rename(columns={'index_nn': 'candidat'})
    candidats['candidat'] = candidats['candidat'].astype(np.int32)
    candidats = candidats.merge(nodes_list[['order', by]], left_on='ix_one', right_index=True)
    # emission
    emissions = candidats.copy()
    point_geometry = nodes_list['geometry'].to_dict()
    candidat_geometry = road_network.road_nodes['geometry'].to_dict()
    candidat_geom_arr = candidats['candidat'].map(candidat_geometry.get).to_numpy()
    gps_geom_arr = candidats['ix_one'].map(point_geometry.get).to_numpy()

    emissions['distance'] = [p1.distance(p2) for p1, p2 in zip(candidat_geom_arr, gps_geom_arr)]
    emissions['logprob'] = emission_logprob(emissions['distance'], SIGMA, POWER)
    emissions = emissions.drop(columns=['ix_one'])

    next_dist_dict = nodes_list['geometry'].distance(nodes_list.shift(-1)).to_dict()
    candidats['distance'] = candidats['ix_one'].apply(lambda x: next_dist_dict.get(x))
    candidats = candidats.drop(columns=['ix_one'])
    transitions = candidats.groupby([by, 'order'], sort=False).agg(list)
    shifted = transitions.groupby(by).shift(-1)[['candidat']]
    transitions = transitions.merge(shifted, on=[by, 'order'], suffixes=['_a', '_b'])
    transitions = transitions.explode(['candidat_a', 'distance']).explode(['candidat_b'])
    transitions = transitions.dropna()  # last node as no transition. is NaN because we groupby(by).shift before merge
    transitions = transitions.reset_index()

    transitions['candidat_a'] = transitions['candidat_a'].astype(np.int32)
    transitions['candidat_b'] = transitions['candidat_b'].astype(np.int32)
    origins = transitions['candidat_a'].to_numpy()
    destinations = transitions['candidat_b'].to_numpy()
    transitions['routing_distance'] = road_network.shortest_path_distance(origins, destinations)

    # probabilities
    transitions['logprob'] = transition_logprob(transitions['routing_distance'], transitions['distance'], BETA, DIFF)
    transitions['logprob'] = transitions['logprob'].fillna(np.inf)
    return emissions, transitions


def single_viterbi(emission: pd.DataFrame, transition: pd.DataFrame):
    # reindex
    unique_candidats = emission['candidat'].unique()
    candidat_index = {k: i for i, k in enumerate(unique_candidats)}
    reversed_candidat_index = {v: k for k, v in candidat_index.items()}
    emission['_candidat'] = emission['candidat'].map(candidat_index.get)
    transition['_candidat_a'] = transition['candidat_a'].map(candidat_index.get)
    transition['_candidat_b'] = transition['candidat_b'].map(candidat_index.get)

    # make matrix
    n_point = len(emission['order'].unique())
    n_candidates = len(unique_candidats)
    # create a dense matrix for emission
    emission_mat = np.full((n_point, n_candidates), np.inf)
    emission_mat[emission['order'].to_numpy(), emission['_candidat'].to_numpy()] = emission['logprob']
    # create a flat sparse matrix for transition
    order = transition['order'].to_numpy(dtype=np.int32)
    row = order * n_candidates + transition['_candidat_a'].to_numpy(dtype=np.int32)
    col = transition['_candidat_b'].to_numpy(dtype=np.int32)
    data = transition['logprob'].to_numpy()
    transition_mat = csc_matrix((data, (row, col)), shape=(n_point * n_candidates, n_candidates))
    # find best path
    path = viterbi(emission_mat, transition_mat)
    path = [*map(reversed_candidat_index.get, path)]
    return path


def batch_viterbi(emissions: pd.DataFrame, transitions: pd.DataFrame, by='trip_id'):
    emissions_batch = [batch for _, batch in emissions.groupby(by)]
    transitions_batch = [batch for _, batch in transitions.groupby(by)]

    paths = []
    for emission, transition in zip(emissions_batch, transitions_batch):
        path = single_viterbi(emission, transition)
        paths.extend(path)
    return paths


def parallel_batch_viterbi(emissions: pd.DataFrame, transitions: pd.DataFrame, by='trip_id', num_workers=1):
    trips = emissions['trip_id'].unique()
    chunks = np.array_split(trips, num_workers)
    emissions_batch = [emissions[emissions[by].isin(ls)] for ls in chunks]
    transitions_batch = [transitions[transitions[by].isin(ls)] for ls in chunks]
    results = parallel_executor(
        batch_viterbi,
        num_workers=num_workers,
        method='pipe',
        parallel_kwargs={'emissions': emissions_batch, 'transitions': transitions_batch},
        by=by,
    )
    return [item for sublist in results for item in sublist]  # flatten


def viterbi_mapmatching(nodes_list: gpd.GeoDataFrame, road_network: RoadNetwork, by='trip_id', num_workers=1, **kwargs):
    """
    nodes_list: columns [by,'order','geometry'] where by is an argument here (default=trip_id)
    return
        node_list with ['road_node'] column
    """
    emissions, transitions = get_candidates(nodes_list=nodes_list, road_network=road_network, by=by, **kwargs)
    if num_workers > 1:
        paths = parallel_batch_viterbi(emissions, transitions, by, num_workers=num_workers)
    else:
        paths = batch_viterbi(emissions, transitions, by)
    nodes_list['road_node'] = paths
    nodes_list['road_node'] = nodes_list['road_node'].map(road_network.index_node.get)
    return nodes_list


def route_between_points(nodes_list: pd.DataFrame, road_network: RoadNetwork, by='trip_id', min_nodes=True):
    """
    nodes_list: columns [by,'road_node'] where by is an argument here (default=trip_id)
    min_nodes: get the min number of nodes in between a,b. not the shortest path
    return
        node_list with ['road_node_list' ,'road_link_list'] column

    """

    node_index = road_network.node_index
    index_node = road_network.index_node
    ab_dict = road_network.ab_dict

    routing_df = nodes_list.copy()
    routing_df['sparse_id'] = routing_df['road_node'].map(node_index.get)
    shifted = routing_df.groupby(by)[['sparse_id']].shift(-1)
    routing_df = routing_df.merge(shifted, left_index=True, right_index=True, suffixes=['_a', '_b'])
    routing_df = routing_df[~routing_df['sparse_id_b'].isna()]
    routing_df['sparse_id_b'] = routing_df['sparse_id_b'].astype(np.int32)

    origins = list(routing_df['sparse_id_a'].unique())
    origin_dict = {index: i for i, index in enumerate(origins)}
    if min_nodes:
        road_network.road_links['_val'] = 1
        csgraph, _ = sparse_matrix(road_network.road_links[['a', 'b', '_val']].values, node_index)
    else:
        csgraph = road_network.csgraph
    _, pred = fast_dijkstra(csgraph=csgraph, indices=origins, return_predecessors=True, limit=np.inf)

    routing_df['origin'] = routing_df['sparse_id_a'].map(origin_dict)
    links_paths = []
    nodes_paths = []
    for ori, dest in routing_df[['origin', 'sparse_id_b']].values:
        path = get_path(pred, ori, dest)
        nodes_paths.append([*map(index_node.get, path)])
        links_path = get_edge_path(path)
        links_paths.append([*map(ab_dict.get, links_path)])
    routing_df['road_node_list'] = nodes_paths
    routing_df['road_link_list'] = links_paths

    routing_df = routing_df.drop(columns=['sparse_id_a', 'sparse_id_b', 'origin'])

    return routing_df
