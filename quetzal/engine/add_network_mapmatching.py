import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from quetzal.engine.pathfinder_utils import sparse_matrix
from syspy.spatial.spatial import add_geometry_coordinates
from multiprocessing import Process, Manager
from typing import Tuple


class RoadLinks:
    '''
    Link Object for mapmatching 

    parameters
    ----------
    links (gpd.GeoDataFrame): links in a metre projection (not 4326 or 3857)
    gps_track (gpd.GeoDataFrame): [['index','geometry']] ordered list of geometry Points. only needed if links is None (to import the links from osmnx)
    n_neighbors_centroid (int) : number of neighbor using the links centroid. first quick pass to find all good candidat road.

    returns
    ----------
    RoadLinks object for mapmatching
    '''
    def __init__(self, links, n_neighbors_centroid=100):

        self.links = links
        assert self.links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        self.crs = links.crs
        self.n_neighbors_centroid = n_neighbors_centroid
        
        try:
            self.links['length']
        except Exception:
            self.links['length'] = self.links.length

        if 'index' not in self.links.columns:
            self.links = self.links.reset_index()

        self.get_sparse_matrix()
        self.get_dict()
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

    def fit_nearest_model(self):
        # Fit Nearest neighbors model
        links = add_geometry_coordinates(self.links, columns=['x_geometry', 'y_geometry'])
        x = links[['x_geometry', 'y_geometry']].values

        if len(links) < self.n_neighbors_centroid: self.n_neighbors_centroid = len(links)

        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(x)



def get_gps_tracks(links, nodes, by='trip_id', sequence='link_sequence'):
    '''
    format links to a format used by the Multi Mapmatching
    '''

    # Format links to a "gps track". keep node a,b of first links and node b of evey other ones.
    gps_tracks = links[['a', 'b', by, sequence, 'route_id']]
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

    gps_tracks = gps_tracks.append(single_points)

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

    # multi threading!
    def process_wrapper(chunk, kwargs, result_list, index):
        result = Multi_Mapmatching(chunk, **kwargs)
        result_list[index] = result
    manager = Manager()
    result_list = manager.list([None] * len(chunks))
    processes = []
    pkwargs = {'road_links':road_links,'routing':routing,'n_neighbors':n_neighbors,'distance_max':distance_max,'by':by,**kwargs}
    for i, trips in enumerate(chunks):
        chunk_gps_tracks = gps_tracks[gps_tracks[by].isin(trips)]
        process = Process(target=process_wrapper, args=(chunk_gps_tracks, pkwargs, result_list, i))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    # Convert the manager list to a regular list for easier access
    result_list = np.array(result_list)

    vals = pd.concat(result_list[:,0])
    node_lists = pd.concat(result_list[:,1])
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
            vals = vals.append(val)
            node_lists = node_lists.append(node_list)

    # add matched points on road
    
    print(f'{len(trip_id_list)} / {len(trip_id_list)}')
    return vals, node_lists, unmatched_trip



def Mapmatching(gps_track:list, links:RoadLinks, n_neighbors:int=10, distance_max:float=1000,
                routing:bool=False, plot:bool=False, SIGMA:float=4.07, BETA:float=3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    

    """
    gps_track: ordered list of geometry Point (in metre)
    links: RoadLinks object
    distance_max: max radius to search candidat road for each gps point (default = 200m)
    routing: True return the complete routing from the first to the last point on the road network (default = False)
    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009
    """

    
    dijkstra_limit = 2000  # Limit on first dijkstra on road network.

    def nearest(one, nbrs, geometry=False):
        try:
            # Assert df_many.index.is_unique
            assert one.index.is_unique
        except AssertionError:
            msg = 'Index of one and many should not contain duplicates'
            print(msg)
            warnings.warn(msg)

        df_one = add_geometry_coordinates(one.copy())

        # x = df_many[['x_geometry','y_geometry']].values
        y = df_one[['x_geometry', 'y_geometry']].values

        distances, indices = nbrs.kneighbors(y)

        indices = pd.DataFrame(indices)
        distances = pd.DataFrame(distances)
        indices = pd.DataFrame(indices.stack(), columns=['index_nn']).reset_index().rename(
            columns={'level_0': 'ix_one', 'level_1': 'rank'}
        )
        return indices

    # Unused
    def routing_correction(length, x):
        if x['road_a'] == x['road_b']:
            return x['road_b_offset'] - x['road_a_offset']
        else:
            return x['dijkstra'] + length - x['road_a_offset'] + x['road_b_offset']

    def emission_logprob(distance, SIGMA=SIGMA):
        # c = 1 / (SIGMA * np.sqrt(2 * np.pi))
        # return c*np.exp(-0.5*(distance/SIGMA)**2)
        # return -np.log10(np.exp(-0.5*(distance/SIGMA)**2))
        return 0.5 * (distance / SIGMA) ** 2  # Drop constant with log. its the same for everyone.

    def transition_logprob(dijkstra_dist, gps_dist, BETA=BETA):
        c = 1 / BETA
        delta = abs(dijkstra_dist - gps_dist)
        # return c * np.exp(-c * delta)
        return c * delta

    # x=np.linspace(0,50,101)
    # y=transition_prob(x,25)
    # plt.plot(x,y)
    # plt.xlabel('metre')
    # plt.ylabel('probability')

    # path_prob = emission_prob(100)+transition_prob(x,3)
    # plt.plot(x,path_prob)

    # process map

    gps_dict = gps_track['geometry'].to_dict()
    # GPS point distance to next point.
    gps_dist_dict = gps_track['geometry'].distance(gps_track.shift(-1)).to_dict()


    # ======================================================
    # Nearest roads and data preparation
    # ======================================================
    candidat_links = nearest(gps_track, links.nbrs).drop(columns=['rank'])

    def project(A, B, normalized=False):
        return [a.project(b, normalized=normalized) for a, b in zip(A, B)]

    def distance(A, B):
        return [a.distance(b) for a, b in zip(A, B)]

    candidat_links['road_geom'] = candidat_links['index_nn'].apply(lambda x: links.geom_dict.get(x))
    candidat_links['gps_geom'] = candidat_links['ix_one'].apply(lambda x: gps_dict.get(x))

    # Add gps distance to road.
    candidat_links['distance'] = distance(candidat_links['gps_geom'], candidat_links['road_geom'])

    candidat_links.sort_values(['ix_one', 'distance'], inplace=True)
    ranks = list(range(links.n_neighbors_centroid)) * len(gps_track)
    candidat_links['actual_rank'] = ranks
    candidat_links = candidat_links.loc[candidat_links['actual_rank'] < n_neighbors]
    candidat_links = candidat_links[candidat_links['distance'] < distance_max]
    candidat_links = candidat_links.reset_index().drop(columns=['index'])

    # Add offset
    # candidat_links['normalized_offset'] = project(candidat_links['road_geom'],candidat_links['gps_geom'],normalized=True)
    candidat_links['offset'] = project(candidat_links['road_geom'], candidat_links['gps_geom'], normalized=False)
    dict_distance = candidat_links.set_index(['ix_one', 'index_nn'])['distance'].to_dict()

    # make tuple with road index and offset.
    candidat_links['index_nn'] = list(zip(candidat_links['index_nn'], candidat_links['offset']))
    candidat_links = candidat_links.drop(columns=['road_geom', 'gps_geom', 'offset', 'distance', 'actual_rank'])

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
    dist_matrix, predecessors = dijkstra(
        csgraph=links.mat,
        directed=True,
        indices=origin_sparse,
        return_predecessors=True,
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
    temp_dist_matrix = dist_matrix.stack(dropna=True).reset_index().rename(
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
        dist_matrix2, predecessors2 = dijkstra(
            csgraph=links.mat,
            directed=True,
            indices=origin_sparse2,
            return_predecessors=True,
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
        temp_dist_matrix = dist_matrix.stack(dropna=True).reset_index().rename(
            columns={'level_0': 'b', 'level_1': 'a', 0: 'dijkstra'}
        )
        candidat_links = candidat_links.merge(temp_dist_matrix, left_on=['node_b', 'node_a'], right_on=['b', 'a'],
                                              how='left').drop(columns=['b', 'a'])

    # ======================================================
    # Calcul probabilité
    # ======================================================

    # Apply offset difference to Dijkstra
    # candidat_links['dijkstra'] = candidat_links.apply(lambda x: routing_correction(links.loc[x['road_a']]['length'],x),axis=1)

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
    candidat_links['path_prob'] = emission_logprob(candidat_links['distance_to_road']) + transition_logprob(
        candidat_links['dijkstra'], candidat_links['gps_distance'])

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
        predecessors = pd.DataFrame(predecessors)
        predecessors.index = origin_sparse

        # Si on a fait deux dijkstra
        if len(unfound_origin_nodes) > 0:
            predecessors2 = pd.DataFrame(predecessors2)
            predecessors2.index = origin_sparse2

            predecessors.loc[predecessors2.index] = predecessors2

        # predecessors = predecessors.apply(lambda x : index_node.get(x))
        df_path = pd.DataFrame(path[1:], columns=['road_id'])
        df_path['sparse_node_b'] = df_path['road_id'].apply(lambda x: links.node_index.get(links.dict_node_b.get(x)))

        node_mat = []

        for i in range(len(df_path) - 1):
            node_list = []
            node_list.append(int(df_path.iloc[-(1 + i)]['sparse_node_b']))  # premier noed (noed b)
            node = predecessors.loc[df_path.iloc[-(1 + i + 1)]['sparse_node_b'], df_path.iloc[-(1 + i)]['sparse_node_b']]
            while node != -9999:  # Ajoute les noeds b jusqua ce qu'on arrive au prochain point gps
                node_list.append(node)
                node = predecessors.loc[df_path.iloc[-(1 + i + 1)]['sparse_node_b'], node]

            # if i==len(df_path)-2:
            node_list.append(int(links.node_index[links.links.loc[df_path.iloc[-(1 + i + 1)]['road_id']]['a']]))  # ajoute le noeud a.
            node_list = [index_node[x] for x in node_list[::-1]]  # reverse and swap index
            node_mat.append(node_list)
        # ajoute le noed a du premier point. puisque le Dijkstra a été calculé à partir des noeds b. le noed a du premier point
        # gps doit être ajouté manuellement.
        node_mat = node_mat[::-1]
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
        print(node_list)

        if plot:
            f, ax = plt.subplots(figsize=(10, 10))
            links.links.plot(ax=ax, linewidth=1)
            gps_track.plot(ax=ax, marker='o', color='red', markersize=20)
            links.links.loc[node_list['road_id']].plot(ax=ax, color='orange', linewidth=2)
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