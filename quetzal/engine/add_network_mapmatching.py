import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from shapely import get_coordinates
from sklearn.neighbors import NearestNeighbors
from quetzal.engine.pathfinder_utils import sparse_matrix
from syspy.spatial.spatial import add_geometry_coordinates
from quetzal.os.parallel_call import parallel_executor
from typing import Tuple
from numba import njit

@njit
def get_points_along_line(p1,p2,distance=20):
    direction = p2 - p1
    # Calculate the magnitude of the direction vector
    length = np.linalg.norm(direction)
    # ge number of points to interp
    points_count = int(length // distance) 
    resp=np.empty((points_count,2))
    # Normalize the direction vector to get the unit vector
    unit_vector = direction / length
    
    # Compute the point at distance t meters along the vector
    for i in range(points_count):
        pt = p1 + distance*(i + 1) * unit_vector
        resp[i]=pt

    return resp

def get_points_along_multiline(multiline,distance=10):
    resp=multiline
    num_pts = len(multiline)
    for i in range(num_pts-1):
        new_pts = get_points_along_line(multiline[i], multiline[i+1],distance)
        resp = np.concatenate([resp,new_pts])
    return resp

@njit
def _point_to_segment_distance(point, segment):
    """
    Calculate the distance between a point and a line segment.

    :param point: Tuple representing the point (x0, y0)
    :param segment: Tuple containing two points that define the segment ((x1, y1), (x2, y2))
    :return: The shortest distance from the point to the segment
    """
    (x0, y0) = point
    (x1, y1), (x2, y2) = segment
    resp=np.zeros(0)
    
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
    distance =  np.sqrt((x0 - nearest_point[0])**2 + (y0 - nearest_point[1])**2)
    return distance
    
@njit
def _point_to_multiLine_distance(point, line):
    num_pts = len(line)
    best=np.inf
    for i in range(num_pts-1):
        res = _point_to_segment_distance(point,[line[i], line[i+1]])
        if res < best:
            best = res
    return best

def point_to_line_distance(points:np.array, lines:np.array) -> np.array:
    '''
    gives the distance of a point to a line
    points should be an array of pts and not a shapely Point :[[x, y],[x, y]]
    lines should be an array of lines and not a shapely lines :[[[x1, y1], [x2, y2]], ...]

    This new version is up to 16X faster than shapely distance function

    transform geom with : 
    points -> geom.coords[0] 
    lines -> np.array(geom.coords)
    '''
    return [_point_to_multiLine_distance(p,l) for p, l in zip(points, lines)]

def project(A, B, normalized=False):
    return [a.project(b, normalized=normalized) for a, b in zip(A, B)]

def nearest(one, links, radius=False):
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
        indices = links.r_nbrs.radius_neighbors(y, radius = links.radius_search, return_distance=False)
    else:
        indices = links.nbrs.kneighbors(y,n_neighbors=links.n_neighbors_centroid,return_distance=False)
        
    indices = pd.DataFrame(indices)
    indices = pd.DataFrame(indices.stack(), columns=['index_nn']).reset_index().rename(
        columns={'level_0': 'ix_one', 'level_1': 'rank'}
    )
    if radius:
        indices = indices.explode('index_nn')
    
    indices['index_nn'] = indices['index_nn'].apply(lambda x: links.knn_dict.get(x))

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


def get_candidat_links(gps_track,links,method):
    
    if method == 'knn':
        candidat_links = nearest(gps_track, links,radius=False).drop(columns=['rank'])
    
    elif method == 'both':
        candidat_links = nearest(gps_track, links,radius=True).drop(columns=['rank'])
        candidat_links = candidat_links.dropna()
        temp_candidat = nearest(gps_track, links,radius=False).drop(columns=['rank'])
        candidat_links = pd.concat([candidat_links, temp_candidat]).sort_values('ix_one').reset_index(drop=True)
        
    elif method == 'radius':
        candidat_links = nearest(gps_track, links,radius=True).drop(columns=['rank'])
        unfound_points = candidat_links[candidat_links['index_nn'].isnull()]['ix_one'].values
        candidat_links = candidat_links.dropna()
    
        if len(unfound_points)>0:
            print(len(unfound_points), 'unfound with radius. use KNN for those')
            index_dict = {i:k for  i,k in enumerate(unfound_points)}
            temp_candidat = nearest(gps_track.loc[unfound_points], links,radius=False).drop(columns=['rank'])
            temp_candidat['ix_one'] = temp_candidat['ix_one'].apply(lambda x: index_dict.get(x))
            candidat_links = pd.concat([candidat_links,temp_candidat]).sort_values('ix_one').reset_index(drop=True)

    return candidat_links


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
        gps_track['timestamp'] = gps_track['timestamp'].apply(lambda x: pd.Timestamp(x)).apply(
            lambda x: int(x.timestamp() * 1000))
    else:  # it's float or int, must be ms tho.
        pass

class RoadLinks:
    '''
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
    '''
    def __init__(self, links, n_neighbors_centroid=10,radius_search=250,on_centroid=False):

        self.links = links
        assert self.links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        self.crs = links.crs
        self.n_neighbors_centroid = n_neighbors_centroid
        if len(links) < self.n_neighbors_centroid: self.n_neighbors_centroid = len(links)
        self.radius_search = radius_search
    
        try:
            self.links['length']
        except Exception:
            self.links['length'] = self.links.length

        if 'index' not in self.links.columns:
            self.links = self.links.reset_index()

        self.get_sparse_matrix()
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
        self.dict_link = self.links.sort_values('length', ascending=True).drop_duplicates(['a', 'b'], keep='first').set_index(['a', 'b'], drop=False)[ 'index'].to_dict()
        self.length_dict = self.links['length'].to_dict()
        self.geom_dict = dict(self.links['geometry'])
        self.geom_dict_arr = {key: get_coordinates(item) for key, item in self.geom_dict.items()}
        if 'bearing' in self.links.columns:
            self.bearing_dict = self.links['bearing'].to_dict()


    def fit_nearest_model(self):
        # Fit Nearest neighbors model
        # add points along each links at every self.radius_search
        # do the knn on those points. this will make sure that every links is found in the search radius,
        # even very long links with centroid realy far away.
        
        dist = self.radius_search #* 1.41 could divide by 2 or srt2. as we seach in diameter of double de radius
        interp_points_dict = {key:get_points_along_multiline(item,dist) for key,item in  self.geom_dict_arr.items()}

        indexes=[]; points=[]
        for key,item in interp_points_dict.items():
            indexes+=[key]*len(item)
            points.append(item)
        points = np.concatenate(points, axis=0)
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(points)
        self.r_nbrs = NearestNeighbors(radius=self.radius_search, algorithm='ball_tree').fit(points)
        # we added points to links. this give us: knn_index:link_index
        self.knn_dict = {i:j for i,j in enumerate(indexes)}

    def fit_nearest_centroid(self):
        # Fit Nearest neighbors model with links centroid
        links = add_geometry_coordinates(self.links, columns=['x_geometry', 'y_geometry'])
        x = links[['x_geometry', 'y_geometry']].values
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(x)
        self.r_nbrs = NearestNeighbors(radius=self.radius_search, algorithm='ball_tree').fit(x)
        # we added points to links. this give us: knn_index:link_index
        self.knn_dict = {i:i for i in range(len(x))}

def get_gps_tracks(links, nodes, by='trip_id', sequence='link_sequence'):
    '''
    format links to a format used by the Multi Mapmatching
    '''

    # Format links to a "gps track". keep node a,b of first links and node b of evey other ones.
    gps_tracks = links[['a', 'b', by, sequence]]
    gps_tracks = gps_tracks.sort_values([by, sequence])
    node_dict = nodes['geometry'].to_dict()
    # gps_tracks['geometry'] = gps_tracks['b'].apply(lambda x: node_dict.get(x))
    gps_tracks['node_seq'] = gps_tracks['b']
    # apply node a for the first link
    counter = gps_tracks.groupby(by).agg(len)['a'].values
    order = [i for j in range(len(counter)) for i in range(counter[j])]
    gps_tracks[sequence] = order

    # for trip with single links, duplicate them to have a mapmatching between a and b.
    single_points = gps_tracks[gps_tracks[sequence] == 0]
    single_points['node_seq'] = single_points['a']
    single_points[sequence] = -1
    single_points.index = 'node_a_' + single_points.index.map(str)

    gps_tracks = pd.concat([gps_tracks,single_points])

    # remove single links that a==b.
    gps_tracks = gps_tracks[gps_tracks['a'] != gps_tracks['b']]
    gps_tracks['geometry'] = gps_tracks['node_seq'].apply(lambda x: node_dict.get(x))

    gps_tracks = gps_tracks.sort_values([by, sequence])
    gps_tracks = gpd.GeoDataFrame(gps_tracks)
    gps_tracks = gps_tracks.drop(columns=['a', 'b', sequence])
    return gps_tracks


def Parallel_Mapmatching(gps_tracks: pd.DataFrame,
                        road_links: RoadLinks,
                        routing: bool = True,
                        n_neighbors: int = 10,
                        distance_max: float = 200,
                        by: str = 'trip_id',
                        num_cores: int = 1,
                        **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    trip_list = gps_tracks[by].unique()
    if num_cores>len(trip_list):
        num_cores = max(len(trip_list), 1)
    chunk_length =  round(len(trip_list)/ num_cores)
    # Split the list into four sub-lists
    chunks = [trip_list[j:j+chunk_length] for j in range(0, len(trip_list), chunk_length)]
    chunk_gps_tracks = [gps_tracks[gps_tracks[by].isin(trips)] for trips in chunks]


    kwargs = {'road_links':road_links,'routing':routing,'n_neighbors':n_neighbors,'distance_max':distance_max,'by':by,**kwargs}
    results = parallel_executor(Multi_Mapmatching,
                                num_workers=len(chunks),
                                parallel_kwargs={'gps_tracks': chunk_gps_tracks},
                                **kwargs)

    vals = pd.concat([res[0] for res in results])
    node_lists = pd.concat([res[1] for res in results])
    unmatched_trip=[]
    return vals, node_lists, unmatched_trip


def Multi_Mapmatching(gps_tracks: pd.DataFrame,
                        road_links: RoadLinks,
                        routing: bool = True,
                        n_neighbors: int = 10,
                        distance_max: float = 200,
                        by: str = 'trip_id',
                        **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    gps_track: use get_gps_tracks
    links: RoadLinks object
    distance_max: max radius to search candidat road for each gps point (default = 200m)
    routing: True return the complete routing from the first to the last point on the road network (default = False)
    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009
    """

    vals = gpd.GeoDataFrame()
    node_lists = gpd.GeoDataFrame()
    unmatched_trip = []
    trip_id_list = gps_tracks[by].unique()
    it=0
    for trip_id in trip_id_list:
        if it % max((len(trip_id_list)//5),5) == 0: # print 5 time
            print(f'{it} / {len(trip_id_list)}')
        it+=1
        gps_track = gps_tracks[gps_tracks[by] == trip_id].drop(columns=by)
        # format index. keep dict to reindex after the mapmatching
        gps_track = gps_track.reset_index()
        gps_track.index = gps_track.index - 1
        gps_index_dict = gps_track['index'].to_dict()
        gps_track.index = gps_track.index + 1
        gps_track = gps_track.drop(columns=['index'])

        if len(gps_track) < 2:  # cannot mapmatch less than 2 points.
            unmatched_trip.append(trip_id)
        else:
            val, node_list = Mapmatching(gps_track, road_links, 
                                                routing=routing,
                                                n_neighbors=n_neighbors,
                                                distance_max=distance_max,
                                                **kwargs)
            


            # add the by column to every data
            val[by] = trip_id
            node_list[by] = trip_id
            # apply input index
            val.index = val.index.map(gps_index_dict)
            node_list.index = node_list.index.map(gps_index_dict)
            vals = pd.concat([vals, val])
            node_lists = pd.concat([node_lists, node_list])

    # add matched points on road
    
    print(f'{len(trip_id_list)} / {len(trip_id_list)}')
    return vals, node_lists, unmatched_trip


def Mapmatching(gps_track:list,
                links:RoadLinks,
                n_neighbors:int=10, 
                distance_max:float=1000, 
                dijkstra_limit = None,
                routing:bool=False,
                nearest_method:str='radius',
                speed_limit:bool=False,
                turn_penalty:bool=False,
                plot:bool=False,
                MAX_SPEED:int=500,
                SIGMA:float=4.07,
                BETA:float=3, 
                POWER:float=2, 
                DIFF=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    

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
        dijkstra_limit = add_geometry_coordinates(gps_track)[['x_geometry','y_geometry']].std().mean() / 2

    gps_dict = gps_track['geometry'].to_dict()
    gps_dict_arr = {key: item.coords[0] for key, item in gps_dict.items()}
    # GPS point distance to next point.
    gps_dist_dict = gps_track['geometry'].distance(gps_track.shift(-1)).to_dict()
   


    if speed_limit:
        convert_timestamp(gps_track)
        timestamp_dict = (gps_track['timestamp'].shift(-1) - gps_track['timestamp']).to_dict()
        # (dist/1000)/(time/1000/3600) # speed in kmh

    # ======================================================
    # Nearest roads and data preparation
    # ======================================================

    
    candidat_links = get_candidat_links(gps_track, links, method=nearest_method)
   


    candidat_links['road_geom_arr'] = candidat_links['index_nn'].apply(lambda x: links.geom_dict_arr.get(x))
    candidat_links['gps_geom_arr'] = candidat_links['ix_one'].apply(lambda x: gps_dict_arr.get(x))

    # Add gps distance to road.
    candidat_links['distance'] = point_to_line_distance(candidat_links['gps_geom_arr'], candidat_links['road_geom_arr'])

    candidat_links.sort_values(['ix_one', 'distance'], inplace=True)
    len_list = candidat_links.groupby('ix_one')['index_nn'].agg(len).values
    ranks = [i  for length in len_list for i in range(length)]
    candidat_links['actual_rank'] = ranks
    candidat_links = candidat_links.loc[candidat_links['actual_rank'] < n_neighbors]
    candidat_links = candidat_links[candidat_links['distance'] < distance_max]
    candidat_links = candidat_links.reset_index().drop(columns=['index'])

    # Add offset
    candidat_links['road_geom'] = candidat_links['index_nn'].apply(lambda x: links.geom_dict.get(x))
    candidat_links['gps_geom'] = candidat_links['ix_one'].apply(lambda x: gps_dict.get(x))
    candidat_links['offset'] = project(candidat_links['road_geom'], candidat_links['gps_geom'], normalized=False)
    dict_distance = candidat_links.set_index(['ix_one', 'index_nn'])['distance'].to_dict()

    # make tuple with road index and offset.
    candidat_links['index_nn'] = list(zip(candidat_links['index_nn'], candidat_links['offset']))
    candidat_links = candidat_links.drop(columns=['road_geom', 'gps_geom', 'road_geom_arr', 'gps_geom_arr', 'offset', 'distance', 'actual_rank'])

    # add virtual nodes start and end.
    candidat_links.loc[len(candidat_links)] = [candidat_links['ix_one'].max() + 1, candidat_links['index_nn'].iloc[-1]]
    candidat_links.loc[-1] = [-1, candidat_links['index_nn'].iloc[0]]
    candidat_links.index = candidat_links.index + 1  # shifting index
    candidat_links = candidat_links.sort_index()  # sorting by index

    # dict of each linked point (ix_one). if pts 10 is NaN, point 9 will be linked to point 11
    dict_point_link = dict(
        zip(list(candidat_links['ix_one'].unique())[:-1], list(candidat_links['ix_one'].unique())[1:])
    )

    candidat_links = candidat_links.groupby('ix_one').agg(list)
    candidat_links = candidat_links.rename(columns={'index_nn': 'road_a'})
    candidat_links = candidat_links.reset_index()

    candidat_links['road_b'] = candidat_links['road_a'].shift(-1)  # .fillna(0)
    # remove last line (last node is virtual and linked to no one.)
    candidat_links = candidat_links.iloc[:-1]

    # df.explode(column='A').explode(column='B')
    candidat_links = candidat_links.explode(column='road_a').explode(column='road_b').reset_index(drop=True)

    # unpack tuple road_ID, offset
    candidat_links[['road_a', 'road_a_offset']] = pd.DataFrame(candidat_links['road_a'].tolist(),
                                                               index=candidat_links.index)
    candidat_links[['road_b', 'road_b_offset']] = pd.DataFrame(candidat_links['road_b'].tolist(),
                                                               index=candidat_links.index)

    # ======================================================
    # DIJKSTRA sur road network
    # ======================================================

    # lien de la route a vers b dans le pseudo graph
    # mais le dijkstra est entre le lien b(route a) vers le lien a(route b)
    candidat_links['node_b'] = candidat_links['road_a'].apply(lambda x: links.dict_node_b.get(x))
    candidat_links['node_a'] = candidat_links['road_b'].apply(lambda x: links.dict_node_a.get(x))

    # candidat_links = candidat_links.fillna(0)

    index_node = {v: k for k, v in links.node_index.items()}

    # liste des origines pour le dijkstra
    origin = list(candidat_links['node_b'].unique())
    origin_sparse = [links.node_index[x] for x in origin]

    # Dijktra on the road network from node = incices to every other nodes.
    # From b to a.
    dist_matrix = dijkstra(
        csgraph=links.mat,
        directed=True,
        indices=origin_sparse,
        return_predecessors=False,
        limit=dijkstra_limit
    )

    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.index = origin

    # Dijkstra Destinations list
    destination = list(candidat_links['node_a'].unique())
    destination_sparse = [links.node_index[x] for x in destination]

    # Filter. on garde seulement les destination d'intéret (les nodes a)
    dist_matrix = dist_matrix[destination_sparse]
    # Then rename (less columns then less time)
    dist_matrix = dist_matrix.rename(columns=index_node)

    # identifie les routes pas trouvées (limit sur Dijkstra de 2000)
    dist_matrix = dist_matrix.replace(np.inf, np.nan)

    # Applique la distance routing a candidat_link
    temp_dist_matrix = dist_matrix.stack().reset_index().rename(
        columns={'level_0': 'b', 'level_1': 'a', 0: 'dijkstra'}
    )
    candidat_links = candidat_links.merge(temp_dist_matrix, left_on=['node_b', 'node_a'], right_on=['b', 'a'],
                                          how='left').drop(columns=['b', 'a'])

    # si des pair origine detination n'ont pas été trouvé dans le routing limité
    # on refait un Dijktra sans limite avec ces origin (noeud b).
    unfound_origin_nodes = (candidat_links[np.isnan(candidat_links['dijkstra'])]['node_b'].unique())
    if len(unfound_origin_nodes) > 0:
        origin_sparse2 = [links.node_index[x] for x in unfound_origin_nodes]
        # Dijktra on the road network from node = incices to every other nodes.
        # from b to a.
        dist_matrix2 = dijkstra(
            csgraph=links.mat,
            directed=True,
            indices=origin_sparse2,
            return_predecessors=False,
            limit=np.inf
        )

        dist_matrix2 = pd.DataFrame(dist_matrix2)
        # dist_matrix2 = dist_matrix2.rename(columns=index_node)
        dist_matrix2.index = unfound_origin_nodes

        # Filter. on garde seulement les destination d'intéret (les nodes a)
        dist_matrix2 = dist_matrix2[destination_sparse]
        dist_matrix2 = dist_matrix2.rename(columns=index_node)

        # Applique les nouvelles valeurs a la matrice originale
        dist_matrix.loc[dist_matrix2.index] = dist_matrix2

        candidat_links = candidat_links.drop(columns='dijkstra')
        temp_dist_matrix = dist_matrix.stack().reset_index().rename(
            columns={'level_0': 'b', 'level_1': 'a', 0: 'dijkstra'}
        )
        candidat_links = candidat_links.merge(temp_dist_matrix, left_on=['node_b', 'node_a'], right_on=['b', 'a'],
                                              how='left').drop(columns=['b', 'a'])

    # ======================================================
    # Calcul probabilité
    # ======================================================


    candidat_links['length'] = candidat_links['road_a'].apply(lambda x: links.length_dict.get(x))

    candidat_links['dijkstra'] = candidat_links['dijkstra'] + candidat_links['length'] - candidat_links[
        'road_a_offset'] + candidat_links['road_b_offset']
    cond = candidat_links['road_a'] == candidat_links['road_b']
    candidat_links.loc[cond, 'dijkstra'] = candidat_links.loc[cond, 'road_b_offset'] - candidat_links.loc[
        cond, 'road_a_offset']
    candidat_links = candidat_links.drop(columns='length')

    # candidat_links['dijkstra'] = np.abs(candidat_links['dijkstra'])

    # applique la distance réelle entre la route et le point GPS.
    candidat_links['distance_to_road'] = candidat_links.set_index(['ix_one', 'road_a']).index.map(
        dict_distance.get)  # .fillna(5)

    # applique la distance entre les point gps a vers b
    candidat_links['gps_distance'] = candidat_links['ix_one'].apply(lambda x: gps_dist_dict.get(x))  # .fillna(3)

    # path prob
    candidat_links['path_prob'] = emission_logprob(candidat_links['distance_to_road'], SIGMA, POWER)
    candidat_links['path_prob'] += transition_logprob(candidat_links['dijkstra'], candidat_links['gps_distance'], BETA, DIFF)

    if (turn_penalty == True) and ('bearing_dict' in links.__dict__.keys()):
        candidat_links['angle'] = candidat_links['road_a'].apply(lambda x: links.bearing_dict.get(x)) - candidat_links[
            'road_b'].apply(lambda x: links.bearing_dict.get(x))
        candidat_links['path_prob'] += turning_penalty_logprob(candidat_links['angle'], BETA)
    

    if speed_limit:
        # (dist/1000)/(time/1000/3600) # speed in kmh
        candidat_links['gps_time'] = candidat_links['ix_one'].apply(lambda x: timestamp_dict.get(x,0)) / 1000 / 3600
        candidat_links['speed'] = (candidat_links['dijkstra'] / 1000) / (candidat_links['gps_time'])

        # correction, on ne veut pas filtrer les chemins qui sont le meme link.
        # si une route est en U par exemple, deux points peuvent se matche tres loins
        # en routing sur la meme route et la vitesse devient > max.
        candidat_links.loc[candidat_links['road_a']==candidat_links['road_b'],'speed'] = 0
        # dont drop virtual nodes and observation at the same exact time (speed = inf)
        candidat_links.loc[candidat_links['gps_time'] == 0, 'speed']=0

        # add penality of 1 per km over the limit.
        #candidat_links= candidat_links[candidat_links['speed']<MAX_SPEED]
        candidat_links['path_prob'] += candidat_links['speed'].apply(lambda x: np.log10(max(x - MAX_SPEED,1)))



    # tous les liens avec les noeuds virtuels (start finish) ont une prob constante (1 par defaut).
    ind = candidat_links['ix_one'] == -1
    candidat_links.loc[ind, 'path_prob'] = 1

    ind = candidat_links['ix_one'] == candidat_links['ix_one'].max()
    candidat_links.loc[ind, 'path_prob'] = 1

    # ======================================================
    # Dijkstra sur pseudo graph
    # ======================================================

    # candidat_links['a'] = candidat_links['ix_one'].astype(str)+'_'+candidat_links['road_a'].astype(str)  #+'_a'
    # candidat_links['b'] = candidat_links['ix_one'].apply(lambda x :dict_point_link.get(x)).astype(str)+'_'+candidat_links['road_b'].astype(str)  #+'_b'
    candidat_links['a'] = list(zip(candidat_links['ix_one'], candidat_links['road_a'], candidat_links['road_a_offset']))
    candidat_links['b'] = list(
        zip(candidat_links['ix_one'].apply(lambda x: dict_point_link.get(x)), candidat_links['road_b'],
            candidat_links['road_b_offset'])
    )
    first_node = candidat_links.iloc[0]['a']
    last_node = candidat_links.iloc[-1]['b']
    pseudo_mat, pseudo_node_index = sparse_matrix(candidat_links[['a', 'b', 'path_prob']].values.tolist())
    pseudo_index_node = {v: k for k, v in pseudo_node_index.items()}

    # Dijkstra on the road network from node = indices to every other nodes.
    # From b to a.
    pseudo_dist_matrix, pseudo_predecessors = dijkstra(
        csgraph=pseudo_mat,
        directed=True,
        indices=pseudo_node_index[first_node],
        return_predecessors=True,
        limit=np.inf
    )

    pseudo_dist_matrix = pd.DataFrame(pseudo_dist_matrix)

    # pseudo_dist_matrix = pseudo_dist_matrix.rename(columns=pseudo_index_node)
    pseudo_dist_matrix.index = pseudo_dist_matrix.index.map(pseudo_index_node)
    pseudo_dist_matrix

    pseudo_predecessors = pd.DataFrame(pseudo_predecessors)
    pseudo_predecessors.index = pseudo_predecessors.index.map(pseudo_index_node)
    pseudo_predecessors[0] = pseudo_predecessors[0].apply(lambda x: pseudo_index_node.get(x))

    path = []
    last_value = last_node
    for i in range(len(candidat_links['ix_one'].unique())):
        last_value = pseudo_predecessors.loc[last_value][0]
        path.append(last_value)
    temp_path = path.copy()
    temp_path.reverse()

    path = [x[1] for x in path]

    path.reverse()

    val = pd.DataFrame(temp_path, columns=['index', 'road_id', 'offset']).set_index('index')[1:]
    val['road_id_b'] = val['road_id'].shift(-1)
    val['offset_b'] = val['offset'].shift(-1)
    val = val[:-1]
    val = val.rename(columns={'road_id': 'road_id_a', 'offset':'offset_a'})
    dijkstra_dict = candidat_links.set_index(['ix_one', 'road_a', 'road_b'], drop=False)['dijkstra'].to_dict()
    val['length'] = val.set_index([val.index, 'road_id_a', 'road_id_b']).index.map(dijkstra_dict.get)

    if plot:
        f, ax = plt.subplots(figsize=(10, 10))
        gps_track.plot(ax=ax, marker='o', color='blue', markersize=20)
        links.links.loc[path].plot(ax=ax, color='red')
        plt.show()

    # ======================================================
    # Reconstruction du routing
    # ======================================================
    node_list = []
    if routing:
        # predecessors = predecessors.apply(lambda x : index_node.get(x))
        df_path = pd.DataFrame(path[1:], columns=['road_id'])
        df_path['from_a'] = df_path['road_id'].apply(lambda x: links.node_index.get(links.dict_node_a.get(x)))
        df_path['from_b'] = df_path['road_id'].apply(lambda x: links.node_index.get(links.dict_node_b.get(x)))
        df_path['to_a'] = df_path['from_a'].shift(-1)
        df_path['to_b'] = df_path['from_b'].shift(-1)

        routing_origins = df_path['from_b'].unique()
        _, routing_predecessors = dijkstra(
            csgraph=links.mat,
            directed=True,
            indices=routing_origins,
            return_predecessors=True,
            limit=np.inf
        )
        routing_predecessors = pd.DataFrame(routing_predecessors)
        routing_predecessors.index = routing_origins

        node_mat = []
        for ori, des, from_a, to_a in df_path[['from_b','to_b','from_a','to_a']].values[:-1]:
            node_list = [des] 
            # this make sure we us the correct path. between 2 matched point (there could be a faster path incoherent)
            if ori == to_a:
                node = to_a
            else:
                node = routing_predecessors.loc[ori, des]
            while node != -9999:  # Ajoute les noeds b jusqua ce qu'on arrive au prochain point gps
                node_list.append(node)
                node = routing_predecessors.loc[ori, node]
            node_list.append(from_a)
            node_list = [index_node[x] for x in node_list[::-1]]  # reverse and swap index
            node_mat.append(node_list)
        # transforme la liste de noeud en liste de route
        link_mat = []
        for node_list in node_mat:
            link_list = []
            if len(node_list)>=2: # if only 1 node. skip this
                for i in range(len(node_list) - 1):
                    link_list.append(links.dict_link[node_list[i], node_list[i + 1]])
            link_mat.append(link_list)
            try: # if first and last road are not equal: correct it
                if len(link_mat) >=2 :
                    prev = link_mat[-2]
                    current = link_mat[-1]
                    if prev[-1]!=current[0]:
                        prev.append(current[0])
            except Exception:
                pass
            
        # format en liste dans un dataframe
        node_mat = pd.Series(node_mat).to_frame('road_node_list')
        node_mat['road_link_list'] = link_mat

        if plot:
            from syspy.spatial.spatial import plot_lineStrings
            f, ax = plt.subplots(figsize=(10, 10))
            plot_lineStrings(links.links,ax=ax, linewidth=1)
            gps_track.plot(ax=ax, marker='o', color='red', markersize=20)
            ls = [x for xs in link_mat for x in xs]
            plot_lineStrings(links.links[links.links['index'].isin(ls)],ax=ax, color='orange', linewidth=2)
            plt.xlim([gps_track['geometry'].x.min() - 1000, gps_track['geometry'].x.max() + 1000])
            plt.ylim([gps_track['geometry'].y.min() - 100, gps_track['geometry'].y.max() + 1000])
            plt.show()
        return val, node_mat
    else:
        return val


def duplicate_nodes(original_links, original_nodes):
    nodes = original_links[['a','b','trip_id']]
    nodes_a = nodes[['a','trip_id']].set_index('a')
    nodes_b = nodes.groupby('trip_id')[['b']].agg('last').reset_index().set_index('b')
    nodes = pd.concat([nodes_a,nodes_b])
    nodes = nodes.reset_index()
    # get only the trips after the first one (first is not changing)
    nodes_dup_list = nodes.groupby('index')[['trip_id']].agg(list)
    nodes_dup_list['trip_id'] = nodes_dup_list['trip_id'].apply(lambda x: x[:-1])
    nodes_dup_list['len'] = nodes_dup_list['trip_id'].apply(len)
    nodes_dup_list = nodes_dup_list[nodes_dup_list['len']>0]

    if len(nodes_dup_list)>0: # skip if no duplicated nodes
        # explode and split tuple (uuid) in original trip_id  .
        nodes_dup_list = nodes_dup_list.explode('trip_id').reset_index()
        # new name!
        nodes_dup_list['new_index'] = nodes_dup_list['index'].astype(str) + '-' + nodes_dup_list['trip_id'].astype(str)
        # create dict (trip_id, index) : new_stop_name for changing the links
        # for b. remove 1 in  as we used link sequence for a. last node we did add 1 in link sequence.
        new_index_dict = nodes_dup_list.set_index(['trip_id','index'])['new_index'].to_dict()
        new_index_dict = nodes_dup_list.set_index(['trip_id','index'])['new_index'].to_dict()
        if len(nodes_dup_list[nodes_dup_list['index'].duplicated()]) > 0:
            print('there is at least a node with duplicated (index, trip_id), will not be split in different nodes')
            print(nodes_dup_list[nodes_dup_list['new_index'].duplicated()]['new_index'])

        #duplicate nodes and concat them to the existing nodes.
        new_nodes = nodes_dup_list[['index','new_index']].merge(original_nodes,left_on='index',right_on='index')
        new_nodes = new_nodes.drop(columns=['index']).rename(columns = {'new_index': 'index'})
        new_nodes = new_nodes.set_index('index')
        original_nodes = pd.concat([original_nodes, new_nodes])

        # change nodes stop_id with new ones in links

        original_links['new_a'] = original_links.set_index(['trip_id','a']).index.map(new_index_dict)
        original_links['a'] = original_links['new_a'].combine_first(original_links['a'])

        original_links['new_b'] = original_links.set_index(['trip_id','b']).index.map(new_index_dict)
        original_links['b'] = original_links['new_b'].combine_first(original_links['b'])

        original_links = original_links.drop(columns = ['new_a','new_b'])
    return original_links, original_nodes