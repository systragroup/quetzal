import warnings
import logging
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from shapely import get_coordinates
from sklearn.neighbors import NearestNeighbors
from quetzal.engine.pathfinder_utils import sparse_matrix, get_path, fast_dijkstra
from syspy.spatial.spatial import add_geometry_coordinates
from quetzal.os.parallel_call import parallel_executor
from quetzal.engine.road_pathfinder import links_to_expanded_links

from numba import njit

log = logging.getLogger(__name__)


@njit
def get_points_along_line(p1, p2, distance=20):
    direction = p2 - p1
    # Calculate the magnitude of the direction vector
    length = np.linalg.norm(direction)
    # ge number of points to interp
    points_count = int(length // distance)
    resp = np.empty((points_count, 2))
    # Normalize the direction vector to get the unit vector
    unit_vector = direction / length

    # Compute the point at distance t meters along the vector
    for i in range(points_count):
        pt = p1 + distance * (i + 1) * unit_vector
        resp[i] = pt

    return resp


def get_points_along_multiline(multiline, distance=10):
    resp = multiline
    num_pts = len(multiline)
    for i in range(num_pts - 1):
        new_pts = get_points_along_line(multiline[i], multiline[i + 1], distance)
        resp = np.concatenate([resp, new_pts])
    return resp


@njit
def _point_to_segment_distance(point, segment) -> float:
    """
    Calculate the distance between a point and a line segment.

    :param point: Tuple representing the point (x0, y0)
    :param segment: Tuple containing two points that define the segment ((x1, y1), (x2, y2))
    :return: The shortest distance from the point to the segment
    """
    (x0, y0) = point
    (x1, y1), (x2, y2) = segment

    # Vector from the first point of the segment to the given point
    p1_to_p = np.array([x0 - x1, y0 - y1])

    # Vector along the segment
    p1_to_p2 = np.array([x2 - x1, y2 - y1])

    # Squared length of the segment
    segment_length_squared = p1_to_p2.dot(p1_to_p2)

    if segment_length_squared == 0:
        # The segment is actually a point (x1, y1)
        return np.linalg.norm(p1_to_p)

    # Projection of point onto the segment, normalized by the segment's length squared
    t = p1_to_p.dot(p1_to_p2) / segment_length_squared
    if t < 0:
        # The projection falls before the segment's start point
        nearest_point = np.array([x1, y1])
    elif t > 1:
        # The projection falls after the segment's end point
        nearest_point = np.array([x2, y2])
    else:
        # The projection falls on the segment
        nearest_point = np.array([x1, y1]) + t * p1_to_p2

    # Distance from the point to the nearest point on the segment
    distance = np.sqrt((x0 - nearest_point[0]) ** 2 + (y0 - nearest_point[1]) ** 2)
    return distance


@njit
def _point_to_multiLine_distance(point, line) -> float:
    num_pts = len(line)
    best = np.inf
    for i in range(num_pts - 1):
        res = _point_to_segment_distance(point, [line[i], line[i + 1]])
        best = min(best, res)
    return best


def point_to_line_distance(points: np.ndarray, lines: np.ndarray) -> list[float]:
    """
    gives the distance of a point to a line
    points should be an array of pts and not a shapely Point :[[x, y],[x, y]]
    lines should be an array of lines and not a shapely lines :[[[x1, y1], [x2, y2]], ...]

    This new version is up to 16X faster than shapely distance function

    transform geom with :
    points -> geom.coords[0]
    lines -> np.array(geom.coords)
    """
    return [_point_to_multiLine_distance(p, l) for p, l in zip(points, lines)]


def project(A, B, normalized=False):
    return [a.project(b, normalized=normalized) for a, b in zip(A, B)]


def nearest(one, links_model, radius=False):
    try:
        # Assert df_many.index.is_unique
        assert one.index.is_unique
    except AssertionError:
        msg = 'Index of one and many should not contain duplicates'
        print(msg)
        warnings.warn(msg)

    df_one = add_geometry_coordinates(one.copy())

    y = df_one[['x_geometry', 'y_geometry']].values
    if radius:
        indices = links_model.r_nbrs.radius_neighbors(y, radius=links_model.radius_search, return_distance=False)
    else:
        indices = links_model.nbrs.kneighbors(y, n_neighbors=links_model.n_neighbors_centroid, return_distance=False)

    indices = pd.DataFrame(indices)
    indices = (
        pd.DataFrame(indices.stack(), columns=['index_nn'])
        .reset_index()
        .rename(columns={'level_0': 'ix_one', 'level_1': 'rank'})
    )
    if radius:
        indices = indices.explode('index_nn')

    indices['index_nn'] = indices['index_nn'].apply(lambda x: links_model.knn_dict.get(x))

    return indices


def emission_logprob(distance, SIGMA, p):
    # c = 1 / (SIGMA * np.sqrt(2 * np.pi))
    # return c*np.exp(-0.5*(distance/SIGMA)**2)
    # return -np.log10(np.exp(-0.5*(distance/SIGMA)**2))
    return 0.5 * (distance / SIGMA) ** p  # Drop constant with log. its the same for everyone.


def transition_logprob(dijkstra_dist, gps_dist, BETA, diff):
    c = 1 / BETA
    delta = abs(dijkstra_dist - gps_dist)
    # return c * np.exp(-c * delta)
    if diff:
        return c * delta
    else:
        return c * dijkstra_dist


def turning_penalty_logprob(angle, BETA, ALPHA=50):
    c = 1 / BETA
    angle = (angle + 180) % 360 - 180
    turning_penalty = ALPHA / (1 + np.exp(-0.09 * (abs(angle) - 90)))
    # return c*np.exp(-c*delta)
    return c * turning_penalty


def get_candidat_links(gps_track, links_model, method):
    if method == 'knn':
        candidat_links = nearest(gps_track, links_model, radius=False).drop(columns=['rank'])

    elif method == 'both':
        candidat_links = nearest(gps_track, links_model, radius=True).drop(columns=['rank'])
        candidat_links = candidat_links.dropna()
        temp_candidat = nearest(gps_track, links_model, radius=False).drop(columns=['rank'])
        candidat_links = pd.concat([candidat_links, temp_candidat]).sort_values('ix_one').reset_index(drop=True)

    elif method == 'radius':
        candidat_links = nearest(gps_track, links_model, radius=True).drop(columns=['rank'])
        unfound_points = candidat_links[candidat_links['index_nn'].isnull()]['ix_one'].values
        candidat_links = candidat_links.dropna()

        if len(unfound_points) > 0:
            print(len(unfound_points), 'unfound with radius. use KNN for those')
            index_dict = {i: k for i, k in enumerate(unfound_points)}
            temp_candidat = nearest(gps_track.loc[unfound_points], links_model, radius=False).drop(columns=['rank'])
            temp_candidat['ix_one'] = temp_candidat['ix_one'].apply(lambda x: index_dict.get(x))
            candidat_links = pd.concat([candidat_links, temp_candidat]).sort_values('ix_one').reset_index(drop=True)

    return candidat_links.drop_duplicates(['ix_one', 'index_nn'])


def convert_timestamp(gps_track):
    try:
        gps_track['timestamp']
    except KeyError:
        raise Exception('must inclide timestamp in columns for the speed_limit=True flag')

    # need pandas timestamp of ms. time in ms!!
    # convert to ms timestamp
    if type(gps_track['timestamp'][0]) == pd.Timestamp:
        gps_track['timestamp'] = gps_track['timestamp'].apply(lambda x: int(x.timestamp() * 1000))
    elif type(gps_track['timestamp'][0]) == str:
        gps_track['timestamp'] = (
            gps_track['timestamp'].apply(lambda x: pd.Timestamp(x)).apply(lambda x: int(x.timestamp() * 1000))
        )
    else:  # it's float or int, must be ms tho.
        pass


def duplicate_nodes(original_links, original_nodes):
    nodes = original_links[['a', 'b', 'trip_id']]
    nodes_a = nodes[['a', 'trip_id']].set_index('a')
    nodes_b = nodes.groupby('trip_id')[['b']].agg('last').reset_index().set_index('b')
    nodes = pd.concat([nodes_a, nodes_b])
    nodes = nodes.reset_index()
    # get only the trips after the first one (first is not changing)
    nodes_dup_list = nodes.groupby('index')[['trip_id']].agg(list)
    nodes_dup_list['trip_id'] = nodes_dup_list['trip_id'].apply(lambda x: x[:-1])
    nodes_dup_list['len'] = nodes_dup_list['trip_id'].apply(len)
    nodes_dup_list = nodes_dup_list[nodes_dup_list['len'] > 0]

    if len(nodes_dup_list) > 0:  # skip if no duplicated nodes
        # explode and split tuple (uuid) in original trip_id  .
        nodes_dup_list = nodes_dup_list.explode('trip_id').reset_index()
        # new name!
        nodes_dup_list['new_index'] = nodes_dup_list['index'].astype(str) + '-' + nodes_dup_list['trip_id'].astype(str)
        # create dict (trip_id, index) : new_stop_name for changing the links
        # for b. remove 1 in  as we used link sequence for a. last node we did add 1 in link sequence.
        new_index_dict = nodes_dup_list.set_index(['trip_id', 'index'])['new_index'].to_dict()
        new_index_dict = nodes_dup_list.set_index(['trip_id', 'index'])['new_index'].to_dict()
        if len(nodes_dup_list[nodes_dup_list['index'].duplicated()]) > 0:
            print('there is at least a node with duplicated (index, trip_id), will not be split in different nodes')
            print(nodes_dup_list[nodes_dup_list['new_index'].duplicated()]['new_index'])

        # duplicate nodes and concat them to the existing nodes.
        new_nodes = nodes_dup_list[['index', 'new_index']].merge(original_nodes, left_on='index', right_on='index')
        new_nodes = new_nodes.drop(columns=['index']).rename(columns={'new_index': 'index'})
        new_nodes = new_nodes.set_index('index')
        original_nodes = pd.concat([original_nodes, new_nodes])

        # change nodes stop_id with new ones in links

        original_links['new_a'] = original_links.set_index(['trip_id', 'a']).index.map(new_index_dict.get)
        original_links['a'] = original_links['new_a'].combine_first(original_links['a'])

        original_links['new_b'] = original_links.set_index(['trip_id', 'b']).index.map(new_index_dict.get)
        original_links['b'] = original_links['new_b'].combine_first(original_links['b'])

        original_links = original_links.drop(columns=['new_a', 'new_b'])
    return original_links, original_nodes


class RoadLinks:
    """
    Link Object for mapmatching

    parameters
    ----------
    links (gpd.GeoDataFrame): links in a metre projection (not 4326 or 3857)
    gps_track (gpd.GeoDataFrame): [['index','geometry']] ordered list of geometry Points. only needed if links is None (to import the links from osmnx)
    n_neighbors_centroid (int) : number of neighbor using the links centroid. first quick pass to find all good candidat road.
    radius_search (int) : radius (meters) to search neighbor. first quick pass to find all good candidat road.
    on_centroid (bool) : if False. add points on links every search_radius * 2 meters. so we are sure to find close roads.
                        else: use links centroid for the neighbor search.

    returns
    ----------
    RoadLinks object for mapmatching
    """

    def __init__(self, links, n_neighbors_centroid=10, radius_search=250, on_centroid=False, precompute_routing=False):
        self.links = links
        assert self.links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        self.crs = links.crs
        self.n_neighbors_centroid = n_neighbors_centroid
        self.n_neighbors_centroid = min(self.n_neighbors_centroid, len(links))
        self.radius_search = radius_search
        self.dist_matrix = None

        try:
            self.links['length']
        except Exception:
            self.links['length'] = self.links.length

        if 'index' not in self.links.columns:
            self.links = self.links.reset_index()

        self.get_sparse_matrix()

        if precompute_routing:
            origins = list(self.node_index.values())
            self.dist_matrix = fast_dijkstra(csgraph=self.mat, indices=origins, return_predecessors=False, limit=np.inf)

        self.get_dict()
        if on_centroid:
            self.fit_nearest_centroid()
        else:
            self.fit_nearest_model()

    def get_sparse_matrix(self):
        self.mat, self.node_index = sparse_matrix(self.links[['a', 'b', 'length']].values)
        self.index_node = {v: k for k, v in self.node_index.items()}

    def get_dict(self):
        # create dict of road network parameters
        self.dict_node_a = self.links['a'].to_dict()
        self.dict_node_b = self.links['b'].to_dict()
        self.links_index_dict = self.links['index'].to_dict()

        self.length_dict = self.links['length'].to_dict()
        self.geom_dict = self.links['geometry'].to_dict()
        self.geom_dict_arr = {key: get_coordinates(item) for key, item in self.geom_dict.items()}
        if 'bearing' in self.links.columns:
            self.bearing_dict = self.links['bearing'].to_dict()

    def fit_nearest_model(self):
        # Fit Nearest neighbors model
        # add points along each links at every self.radius_search
        # do the knn on those points. this will make sure that every links is found in the search radius,
        # even very long links with centroid realy far away.

        dist = self.radius_search  # * 1.41 could divide by 2 or srt2. as we seach in diameter of double de radius
        interp_points_dict = {key: get_points_along_multiline(item, dist) for key, item in self.geom_dict_arr.items()}

        indexes = []
        points = []
        for key, item in interp_points_dict.items():
            indexes += [key] * len(item)
            points.append(item)
        points = np.concatenate(points, axis=0)
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(points)
        self.r_nbrs = NearestNeighbors(radius=self.radius_search, algorithm='ball_tree').fit(points)
        # we added points to links. this give us: knn_index:link_index
        self.knn_dict = {i: j for i, j in enumerate(indexes)}

    def fit_nearest_centroid(self):
        # Fit Nearest neighbors model with links centroid
        links = add_geometry_coordinates(self.links, columns=['x_geometry', 'y_geometry'])
        x = links[['x_geometry', 'y_geometry']].values
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(x)
        self.r_nbrs = NearestNeighbors(radius=self.radius_search, algorithm='ball_tree').fit(x)
        # we added points to links. this give us: knn_index:link_index
        self.knn_dict = {i: i for i in range(len(x))}


def get_gps_tracks(links, nodes, by='trip_id', sequence='link_sequence'):
    """
    format links to a format used by the Multi Mapmatching
    """

    # Format links to a "gps track". keep node a,b of first links and node b of evey other ones.
    gps_tracks = links[['a', 'b', by, sequence]]
    gps_tracks = gps_tracks.sort_values([by, sequence])
    node_dict = nodes['geometry'].to_dict()
    gps_tracks['node_seq'] = gps_tracks['a']
    # counter = gps_tracks.groupby(by).agg(len)['b'].values
    # order = [i for j in range(len(counter)) for i in range(counter[j])]
    # gps_tracks[sequence] = order
    # for trip with single links, duplicate them to have a mapmatching between a and b.

    single_points = gps_tracks.reset_index().groupby('trip_id').last()
    single_points = single_points.reset_index().set_index('index')
    single_points['node_seq'] = single_points['b']
    single_points[sequence] += 1
    single_points.index = 'node_b_' + single_points.index.map(str)

    gps_tracks = pd.concat([gps_tracks, single_points])

    # remove single links that a==b.
    gps_tracks = gps_tracks[gps_tracks['a'] != gps_tracks['b']]
    gps_tracks['geometry'] = gps_tracks['node_seq'].apply(lambda x: node_dict.get(x))

    gps_tracks = gps_tracks.sort_values([by, sequence])
    gps_tracks = gpd.GeoDataFrame(gps_tracks)
    gps_tracks = gps_tracks.drop(columns=['a', 'b', sequence])

    return gps_tracks


def Parallel_Mapmatching(
    gps_tracks: pd.DataFrame,
    road_links: RoadLinks,
    by: str = 'trip_id',
    num_cores: int = 1,
    routing: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    **kwargs : see Mapmatching args
    this method only work (and faster) if we precomputed the dijkstra (road_links.dist_matrix)
    """
    if (road_links.dist_matrix is None) | (num_cores < 2):
        # cannot run fast_dijktra on parallel.
        return Multi_Mapmatching(gps_tracks, road_links, by=by, routing=routing, **kwargs)
    # parallelize
    trip_list = gps_tracks[by].unique()
    if num_cores > len(trip_list):
        num_cores = max(len(trip_list), 1)
    chunk_length = round(len(trip_list) / num_cores)
    # Split the list into four sub-lists
    chunks = [trip_list[j : j + chunk_length] for j in range(0, len(trip_list), chunk_length)]
    chunk_gps_tracks = [gps_tracks[gps_tracks[by].isin(trips)] for trips in chunks]

    kwargs = {'road_links': road_links, 'by': by, **kwargs}
    results = parallel_executor(
        Multi_Mapmatching,
        num_workers=len(chunks),
        method='pipe',
        parallel_kwargs={'gps_tracks': chunk_gps_tracks},
        routing=False,  # we run routing after. with all threads available
        **kwargs,
    )

    df = pd.concat([res[0] for res in results])
    route_lists = gpd.GeoDataFrame()
    if routing:
        print('routing')
        route_lists = route_mapmatched_points(df, road_links, by)
    unmatched_trip = []
    return df, route_lists, unmatched_trip


def Multi_Mapmatching(
    gps_tracks: pd.DataFrame, road_links: RoadLinks, by: str = 'trip_id', routing: bool = False, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    gps_track: use get_gps_tracks
    links: RoadLinks object
    **kwargs : see Mapmatching args
    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009
    """

    final_df = gpd.GeoDataFrame()
    unmatched_trip = []
    trip_id_list = gps_tracks[by].unique()
    it = 0
    for trip_id in trip_id_list:
        if it % max((len(trip_id_list) // 5), 5) == 0:  # print 5 time
            print(f'{it} / {len(trip_id_list)}')
        it += 1
        gps_track = gps_tracks[gps_tracks[by] == trip_id].drop(columns=by)
        # format index. keep dict to reindex after the mapmatching
        gps_track = gps_track.reset_index()
        gps_index_dict = gps_track['index'].to_dict()
        gps_track = gps_track.drop(columns=['index'])

        if len(gps_track) < 2:  # cannot mapmatch less than 2 points.
            unmatched_trip.append(trip_id)
        else:
            df = Mapmatching(gps_track, road_links, **kwargs)
            if len(df) == 0:
                unmatched_trip.append(trip_id)
                continue

            df[by] = trip_id
            df.index = df.index.map(gps_index_dict.get)
            final_df = pd.concat([final_df, df])

    print(f'{len(trip_id_list)} / {len(trip_id_list)}')
    route_lists = gpd.GeoDataFrame()
    if routing:
        print('routing')
        route_lists = route_mapmatched_points(final_df, road_links, by)

    return final_df, route_lists, unmatched_trip


def add_distance_to_road(
    candidat_links: pd.DataFrame, links_dict_arr: dict, point_dict_arr: dict, n_neighbors: int, distance_max: float
):
    road_geom_arr = candidat_links['index_nn'].map(links_dict_arr.get).to_numpy()
    gps_geom_arr = candidat_links['ix_one'].map(point_dict_arr.get).to_numpy()
    candidat_links['distance'] = point_to_line_distance(gps_geom_arr, road_geom_arr)

    candidat_links.sort_values(['ix_one', 'distance'], inplace=True)
    candidat_links['rank'] = candidat_links.groupby('ix_one').cumcount()

    candidat_links = candidat_links.loc[candidat_links['rank'] < n_neighbors]
    candidat_links = candidat_links[candidat_links['distance'] < distance_max]
    candidat_links = candidat_links.reset_index(drop=True).drop(columns=['rank'])
    return candidat_links


def add_road_offset(candidat_links: pd.DataFrame, links_dict: dict, point_dict: dict):
    road_geom = candidat_links['index_nn'].map(links_dict.get)
    gps_geom = candidat_links['ix_one'].map(point_dict.get)
    candidat_links['offset'] = project(road_geom, gps_geom, normalized=False)
    return candidat_links


def get_routing_distance(candidat_links: pd.DataFrame, links: RoadLinks, dijkstra_limit):
    if links.dist_matrix is not None:  # precomputed dijkstra on all the network
        ori = candidat_links['node_a'].map(links.node_index.get)
        dest = candidat_links['node_b'].map(links.node_index.get)
        candidat_links['routing_distance'] = links.dist_matrix[ori, dest]
    else:
        origins = list(candidat_links['node_a'].unique())
        origin_sparse = [links.node_index[x] for x in origins]

        dist_matrix = fast_dijkstra(
            csgraph=links.mat, indices=origin_sparse, return_predecessors=False, limit=dijkstra_limit
        )

        origin_dict = {index: i for i, index in enumerate(origins)}
        ori = candidat_links['node_a'].map(origin_dict.get)
        dest = candidat_links['node_b'].map(links.node_index.get)
        candidat_links['routing_distance'] = dist_matrix[ori, dest]
    return candidat_links


# self.dist_matrix
def _links_path_to_nodes_path(path: list[str], dict_a: dict[str, str], dict_b: dict[str, str]):
    nodes = []
    for link_id in path:
        node_a = dict_a.get(link_id)
        nodes.append(node_a)
    nodes.append(dict_b.get(link_id))
    return nodes


def route_mapmatched_points(df: pd.DataFrame, road_links: RoadLinks, by='trip_id'):

    expanded_links = links_to_expanded_links(road_links.links.set_index('index')[['a', 'b', 'length']], u_turns=False)
    csr_matrix, node_index = sparse_matrix(expanded_links[['from_link', 'to_link', 'length']].values)

    index_node = {v: k for k, v in node_index.items()}

    routing_df = df.copy()[['road_id', by]]
    routing_df['sparse_id'] = routing_df['road_id'].map(node_index.get)
    routing_df = routing_df.merge(routing_df.shift(-1), on='index', suffixes=['_a', '_b']).iloc[:-1]

    origins = list(routing_df['sparse_id_a'].unique())
    origin_dict = {index: i for i, index in enumerate(origins)}
    _, pred = fast_dijkstra(csgraph=csr_matrix, indices=origins, return_predecessors=True, limit=np.inf)

    dict_node_a = road_links.links.set_index('index')['a'].to_dict()
    dict_node_b = road_links.links.set_index('index')['b'].to_dict()

    routing_df['origin'] = routing_df['sparse_id_a'].map(origin_dict)
    cols = ['origin', 'sparse_id_b', by + '_a', by + '_b']
    paths = []
    nodes_paths = []
    for ori, dest, trip_a, trip_b in routing_df[cols].values:
        if trip_a != trip_b:  # dont route between trips
            paths.append([])
            nodes_paths.append([])
        else:
            path = get_path(pred, ori, dest)
            path = [*map(index_node.get, path)]
            paths.append(path)
            nodes_paths.append(_links_path_to_nodes_path(path, dict_node_a, dict_node_b))

    routing_df['road_link_list'] = paths
    routing_df['road_node_list'] = nodes_paths
    # road_id_a	sparse_id_a	road_id_b	sparse_id_b	origin
    routing_df = routing_df.drop(columns=['road_id_a', 'road_id_b', 'sparse_id_a', 'sparse_id_b', 'origin', by + '_b'])
    routing_df = routing_df.rename(columns={by + '_a': by})
    return routing_df


def Mapmatching(
    gps_track: pd.DataFrame,
    links: RoadLinks,
    n_neighbors: int = 10,
    distance_max: float = 1000,
    dijkstra_limit=None,
    nearest_method: str = 'radius',
    speed_limit: bool = False,
    turn_penalty: bool = False,
    MAX_SPEED: int = 500,
    SIGMA: float = 4.07,
    BETA: float = 3,
    POWER: float = 2,
    DIFF=True,
) -> pd.DataFrame:
    """
    gps_track: ordered list of geometry Point (in metre)
    links: RoadLinks object
    distance_max: max radius to search candidat road for each gps points
    dijkstra_limit: first dijkstra limit. if None. will use half the STD of the gps points coords.
    routing: True return the complete routing from the first to the last point on the road network (default = False)
    nearest_method: knn, radius or both.
    speed_limit: add a penalty if speed is larger dans maxspeed.
    turn_penalty: add a penalty depending on the road angle : must have 'bearing' in the road links

    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009

    Weight : 1/2 * 1/SIGMA**2 * (proj dist)**2 + 1/BETA * abs(dijkstra_dist - as_the_crow_flies_dist)
    """

    if dijkstra_limit is None:
        dijkstra_limit = add_geometry_coordinates(gps_track)[['x_geometry', 'y_geometry']].std().mean() / 2

    gps_dict = gps_track['geometry'].to_dict()
    gps_dict_arr = {key: item.coords[0] for key, item in gps_dict.items()}
    # GPS point distance to next point.
    gps_dist_dict = gps_track['geometry'].distance(gps_track.shift(-1)).to_dict()

    timestamp_dict = {}
    if speed_limit:
        convert_timestamp(gps_track)
        timestamp_dict = (gps_track['timestamp'].shift(-1) - gps_track['timestamp']).to_dict()
        # (dist/1000)/(time/1000/3600) # speed in kmh

    # ======================================================
    # Nearest roads and data preparation
    # ======================================================

    candidat_links = get_candidat_links(gps_track, links, method=nearest_method)
    candidat_links = add_distance_to_road(candidat_links, links.geom_dict_arr, gps_dict_arr, n_neighbors, distance_max)
    candidat_links = add_road_offset(candidat_links, links.geom_dict, gps_dict)
    dict_distance = candidat_links.set_index(['ix_one', 'index_nn'])['distance'].to_dict()
    candidat_links = candidat_links.drop(columns=['distance']).rename(columns={'index_nn': 'road'})
    if len(candidat_links) < 1:
        return pd.DataFrame()

    # add virtual nodes start and end.
    first = candidat_links.iloc[[0]].copy()
    first['ix_one'] -= 1
    last = candidat_links.iloc[[-1]].copy()
    last['ix_one'] += 1
    candidat_links = pd.concat([first, candidat_links, last], ignore_index=True)

    # dict of each linked point (ix_one). if pts 10 is NaN, point 9 will be linked to point 11
    ix_one_unique = candidat_links['ix_one'].unique()
    dict_point_link = dict(zip(ix_one_unique[:-1], ix_one_unique[1:]))

    # make a graph (connect each point to the next one)
    grouped = candidat_links.groupby('ix_one', sort=False).agg(list)
    grouped = grouped.merge(grouped.shift(-1), left_index=True, right_index=True, suffixes=['_a', '_b'])
    candidat_links = grouped.explode(['road_a', 'offset_a']).explode(['road_b', 'offset_b'])
    candidat_links = candidat_links.iloc[:-1]  # last node has no connection. remove
    candidat_links = candidat_links.reset_index()

    # ======================================================
    # DIJKSTRA sur road network
    # ======================================================

    candidat_links['node_a'] = candidat_links['road_a'].apply(lambda x: links.dict_node_a.get(x))
    candidat_links['node_b'] = candidat_links['road_b'].apply(lambda x: links.dict_node_b.get(x))

    candidat_links = get_routing_distance(candidat_links, links, dijkstra_limit)
    unfounded = candidat_links[np.isinf(candidat_links['routing_distance'])].copy()
    if len(unfounded) > 0:
        unfounded = get_routing_distance(unfounded, links, np.inf)
        candidat_links.loc[unfounded.index, 'routing_distance'] = unfounded['routing_distance']

    # ======================================================
    # Calcul probabilité
    # ======================================================

    road_b_length = candidat_links['road_b'].apply(lambda x: links.length_dict.get(x))
    candidat_links['road_distance'] = (
        candidat_links['routing_distance'] - candidat_links['offset_a'] - (road_b_length - candidat_links['offset_b'])
    )

    # apply gps distance computed earlier.
    candidat_links['distance_to_road'] = candidat_links.set_index(['ix_one', 'road_a']).index.map(dict_distance.get)
    candidat_links['points_distance'] = candidat_links['ix_one'].apply(lambda x: gps_dist_dict.get(x))

    # path prob
    candidat_links['path_prob'] = emission_logprob(candidat_links['distance_to_road'], SIGMA, POWER)
    candidat_links['path_prob'] += transition_logprob(
        candidat_links['road_distance'], candidat_links['points_distance'], BETA, DIFF
    )

    if turn_penalty and ('bearing_dict' in links.__dict__.keys()):
        start_angle = candidat_links['road_a'].map(links.bearing_dict.get)
        end_angle = candidat_links['road_b'].map(links.bearing_dict.get)
        candidat_links['angle'] = start_angle - end_angle
        candidat_links['path_prob'] += turning_penalty_logprob(candidat_links['angle'], BETA)

    if speed_limit:
        # (dist/1000)/(time/1000/3600) # speed in kmh
        candidat_links['gps_time'] = candidat_links['ix_one'].apply(lambda x: timestamp_dict.get(x, 0)) / 1000 / 3600
        candidat_links['speed'] = (candidat_links['road_distance'] / 1000) / (candidat_links['gps_time'])

        # correction, on ne veut pas filtrer les chemins qui sont le meme link.
        # si une route est en U par exemple, deux points peuvent se matche tres loins
        # en routing sur la meme route et la vitesse devient > max.
        same_road = candidat_links['road_a'] == candidat_links['road_b']
        candidat_links.loc[same_road, 'speed'] = 0
        # dont drop virtual nodes and observation at the same exact time (speed = inf)
        candidat_links.loc[candidat_links['gps_time'] == 0, 'speed'] = 0

        # add penality of 1 per km over the limit.
        # candidat_links= candidat_links[candidat_links['speed']<MAX_SPEED]
        candidat_links['path_prob'] += candidat_links['speed'].apply(lambda x: np.log10(np.max(x - MAX_SPEED, 1)))

    # tous les liens avec les noeuds virtuels (start finish) ont une prob constante (1 par defaut).
    start_mask = candidat_links['ix_one'] == -1
    end_mask = candidat_links['ix_one'] == candidat_links['ix_one'].max()

    candidat_links.loc[start_mask, 'path_prob'] = 1
    candidat_links.loc[end_mask, 'path_prob'] = 1

    # ======================================================
    # Dijkstra sur pseudo graph
    # ======================================================
    candidat_links['next'] = candidat_links['ix_one'].map(dict_point_link.get)
    candidat_links['a'] = list(zip(candidat_links['ix_one'], candidat_links['road_a'], candidat_links['offset_a']))
    candidat_links['b'] = list(zip(candidat_links['next'], candidat_links['road_b'], candidat_links['offset_b']))

    first_node = candidat_links.iloc[0]['a']
    last_node = candidat_links.iloc[-1]['b']
    pseudo_mat, pseudo_node_index = sparse_matrix(candidat_links[['a', 'b', 'path_prob']].values.tolist())
    pseudo_index_node = {v: k for k, v in pseudo_node_index.items()}

    origin = pseudo_node_index.get(first_node)
    dest = pseudo_node_index.get(last_node)

    _, pseudo_predecessors = dijkstra(
        csgraph=pseudo_mat, directed=True, indices=[origin], return_predecessors=True, limit=np.inf
    )

    path = get_path(pseudo_predecessors, origin, dest)
    path = [*map(pseudo_index_node.get, path)][1:-1]  # drop virtual nodes
    # path is a list of tuple (ix_one, road_b, offset_b)
    df = pd.DataFrame(path, columns=['index', 'road_id', 'offset']).set_index('index')
    df['road_id'] = df['road_id'].map(links.links_index_dict)
    return df
