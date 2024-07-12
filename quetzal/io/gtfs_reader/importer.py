# pylint: disable=no-member
import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from syspy.transitfeed import feed_links
from . import patterns
from .feed_gtfsk import Feed
from syspy.spatial import spatial
from quetzal.engine.pathfinder_utils import paths_from_edges
from pyproj import transform
from multiprocessing import Process, Manager



def get_epsg(lat: float, lon: float) -> int:
    '''
    return EPSG in meter for a given (lat,lon)
    lat is north south 
    lon is est west
    '''
    return int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))


def to_seconds(time_string):  # seconds
    return pd.to_timedelta(time_string).total_seconds()


def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()

    def geometry(row):
        return LineString(
            (point_dict[row[from_point]], point_dict[row[to_point]]))

    return df.apply(geometry, axis=1)


def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def cumulative_minus_first(group):
    return group.cumsum() - group.iloc[0]

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return [None, LineString(line)]
    if distance >= line.length:
        return [LineString(line), None]
    coords = list(line.coords)
    pd = 0
    for i, p in enumerate(coords):
        if i == 0:
            continue
        pd += euclidean_distance(p, coords[i - 1])
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]
        
def parallel_shape_geometry(self, from_point, to_point, max_candidates=10,emission_weight=0.5, log=False, num_cores=2):

    class SimpleFeed:
        # less memory intensive feed object with all we need for the shape_geometry() function
        def __init__(self, feed):
            self.links = feed.links
            self.nodes = feed.nodes
            self.shapes = feed.shapes
            self.stop_times = feed.stop_times

    trip_list = self.links['trip_id'].unique()

    if len(trip_list) < num_cores: # if 2 trips, and 4 cores, use 2 core (cannot split more)
        num_cores = max(len(trip_list), 1)

    chunk_length =  round(len(trip_list)/ num_cores)
    # Split the list into four sub-lists
    chunks = [trip_list[j:j+chunk_length] for j in range(0, len(trip_list), chunk_length)]
    # multi threading!
    def process_wrapper(chunk, kwargs, result_list, index):
        result = shape_geometry(chunk, **kwargs)
        result_list[index] = result
    manager = Manager()
    result_list = manager.list([None] * len(chunks))
    processes = []
    kwargs = {'from_point':from_point,'to_point':to_point,'max_candidates':max_candidates,'emission_weight':emission_weight,'log':log}
    for i, trips in enumerate(chunks):
        chunk_links = SimpleFeed(self.copy())
        chunk_links.links = chunk_links.links[chunk_links.links['trip_id'].isin(trips)]
        process = Process(target=process_wrapper, args=(chunk_links, kwargs, result_list, i))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    # Convert the manager list to a regular list for easier access
    return pd.concat(result_list)


def shape_geometry(self, from_point, to_point, max_candidates=10, log=False, emission_weight=0.5):
    to_concat = []

    point_dict = self.nodes.set_index('stop_id')['geometry'].to_dict()
    shape_dict = self.shapes.set_index('shape_id')['geometry'].to_dict()

    stop_ids = set(self.links[[from_point, to_point]].values.flatten())
    stops_pts = pd.DataFrame([point_dict.get(n) for n in stop_ids], index=list(stop_ids), columns=['geometry'])

    for trip in set(self.links['trip_id']):
        links = self.links[self.links['trip_id'] == trip].copy()
        links = links.drop_duplicates(subset=['link_sequence']).sort_values(by='link_sequence')
        s = shape_dict.get(links.iloc[0]['shape_id'])
        trip_stops = list(links[from_point]) + [links.iloc[-1][to_point]]
        stop_pts = stops_pts.loc[trip_stops]


        # Find segments candidates from shape for projection
        segments = pd.DataFrame(list(map(LineString, zip(s.coords[:-1], s.coords[1:]))), columns=['geometry'])
        n_candidates = min([len(segments), max_candidates])

        ng = spatial.nearest(stop_pts, segments, n_neighbors=n_candidates)
        ng = ng.set_index(['ix_one', 'rank'])['ix_many'].to_dict()

        # Distance matrix (stops * n_candidates)
        emission_dict={}
        distances_a = np.empty((len(trip_stops), n_candidates))
        for r in range(n_candidates):
            proj_pts = []
            for i,n in enumerate(trip_stops):
                seg = segments.loc[ng[(n, r)]]['geometry']
                emission_dict[f'{i}_{r}'] =  seg.distance(point_dict.get(n)) # get point to linestring dist
                proj_pts.append(seg.interpolate(seg.project(point_dict.get(n))))
            distances = [s.project(pts) for pts in proj_pts]
            distances_a[:, r] = distances

        # Differential distance matrix (transition cost between candidates)
        df = pd.DataFrame(distances_a)
        diff_df = pd.DataFrame({from_point: df.index[:-1], to_point: df.index[1:]})
        for i in range(n_candidates):
            for j in range(n_candidates):
                diff_col = df.iloc[:, j].shift(-1) - df.iloc[:, i]
                diff_df[(i, j)] = list(diff_col)[:-1]

        diff_df = diff_df.set_index([from_point, to_point])
        diff_df.columns = pd.MultiIndex.from_tuples(diff_df.columns, names=['from', 'to'])
        diff_df[diff_df <= 0.0] = 1e9

        # Compute diffrential between shape_dist_traveled and transition matrix
        shape_dist_traveled_diff = list(self.stop_times[self.stop_times['trip_id'] == trip].sort_values(by='stop_sequence')['shape_dist_traveled'].diff())[1:]
        for c in diff_df.columns:
            diff_df[c] = abs(shape_dist_traveled_diff - diff_df[c])

        diff_df = diff_df.stack().stack().reset_index()
        diff_df['from'] = diff_df[from_point].astype(str) + '_' + diff_df['from'].astype(str)
        diff_df['to'] = diff_df[to_point].astype(str) + '_' + diff_df['to'].astype(str)

        diff_df['emission'] = diff_df['from'].apply(lambda x: emission_dict.get(x))
        diff_df['prob'] = diff_df[0] + diff_df['emission']*emission_weight


        # Build edges + dijkstra
        candidates_e = diff_df[['from', 'to', 'prob']].values
        source_e = np.array([["source"] * n_candidates,
                            [str(0) + '_' + str(c) for c in range(n_candidates)],
                            np.ones(n_candidates)], dtype="O").T
        target_e = np.array([[str(len(trip_stops) - 1) + '_' + str(c) for c in range(n_candidates)],
                            ["target"] * n_candidates,
                            np.ones(n_candidates)], dtype="O").T

        edges = np.concatenate([candidates_e, source_e, target_e])
        path = paths_from_edges(edges=edges, od_set={('source', 'target')})
        best_distances = [int(p.split('_')[-1]) for p in path['path'][0][1:-1]]

        distances = list(np.take_along_axis(distances_a, np.array(best_distances)[:, np.newaxis], axis=1).flatten())

        # Cut shape
        cuts = []
        for d1, d2 in zip(distances[:-1], distances[1:]):
            first_cut = cut(s, d1)[1]
            cuts.append(cut(first_cut, d2 - d1)[0])

        for i, c in enumerate(cuts):
            if c is None:
                msg = f'''
                Failed to cut shape for trip {trip}. Replacing by A -> B Linestring.
                '''
                if log:
                    print(msg)
                cuts = linestring_geometry(links, point_dict, 'a', 'b').values
        if abs(1 - sum([c.length for c in cuts]) / s.length) > 0.1:
            msg = f'''
                Length of trip {trip} is more than 10% longer or shorter than shape.
                This may be caused by a bad cutting of the shape.
                Try increasing the maximum number of candidates (max_candidates)
                '''
            if log:
                print(msg)

        links['geometry'] = cuts
        
        to_concat.append(links)
    return gpd.GeoDataFrame(pd.concat(to_concat))


class GtfsImporter(Feed):

    """The file importer of quetzal contains a main class: GtfsImporter.
    
    It gives access to different method to handle GTFS data:

        - description (GtfsImporter.describe())
        - maps
        - filtering on specific area, dates, trips, stops (GtfsImporter.restrict())
        - stops / trips aggregation
        - frequency conversion
    """
    def __init__(self, epsg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsg = epsg

    from .directions import build_directions
    from .frequencies import compute_pattern_headways, convert_to_frequencies
    from .patterns import build_patterns
    from .services import group_services

    def clean(self):
        feed = super().clean()
        feed.stop_times = feed_links.clean_sequences(feed.stop_times)
        return feed

    def build_stop_clusters(self, **kwargs):
        self.stops = patterns.build_stop_clusters(self.stops, **kwargs)

    def build(self, date, time_range, cluster_distance_threshold=None, drop_unused=True):
        print('Restricting to date…')
        feed = self.restrict(dates=[date], drop_unused=drop_unused)
        print('Grouping services…')
        feed.group_services()
        # print('Cleaning…')
        # feed = feed.clean()
        if cluster_distance_threshold is not None:
            print('Clustering stops…')
            feed.build_stop_clusters(distance_threshold=cluster_distance_threshold)
            print('Building patterns…')
            feed.build_patterns(on='cluster_id')
        else:
            print('Building patterns…')
            feed.build_patterns()
        print('Converting to frequencies…')
        feed = feed.convert_to_frequencies(time_range=time_range, drop_unused=drop_unused)
        print('Building links and nodes…')
        feed.build_links_and_nodes()
        return feed

    def build_links_and_nodes(self, 
                              time_expanded=False, 
                              shape_dist_traveled=False, 
                              from_shape=False, 
                              stick_nodes_on_links=False,
                              log=True, 
                              num_cores=1,
                              **kwargs):
        """
        Build links and nodes from a GTFS.

        Parameters
        ----------
        time_expanded : bool, 
            (default = False)
        shape_dist_traveled : bool, 
            if True, add the shape_dist_travel to each links (their lenght). (default = False)
        from_shape : bool, 
            if True, create links with the GTFS shapes.
            else just conenct stops  with straight lines. (default = False)
        stick_nodes_on_links : bool, 
            Change nodes position to be on the links if true
            and duplicated nodes used on multiple links (default = False)
        log : bool
            (default = False)
        num_cores: int
            for the from_shape method. can parallel it as it is quite slow


      
        Builds
        links and nodes.
        ----------

        """        
        self.to_seconds()
        self.build_links(time_expanded=time_expanded, shape_dist_traveled=shape_dist_traveled, **kwargs)
        self.build_geometries(from_shape=from_shape, stick_nodes_on_links=stick_nodes_on_links, log=log, num_cores=num_cores, **kwargs)

    def to_seconds(self):
        # stop_times
        time_columns = ['arrival_time', 'departure_time']
        self.stop_times[time_columns] = self.stop_times[
            time_columns
        ].applymap(to_seconds)
        # frequencies
        if self.frequencies is not None:
            time_columns = ['start_time', 'end_time']
            self.frequencies[time_columns] = self.frequencies[
                time_columns
            ].applymap(to_seconds)

    def build_links(self, time_expanded=False, shape_dist_traveled=False, 
            keep_origin_columns=['departure_time'],
            keep_destination_columns=['arrival_time']):
        """
        Create links and add relevant information
        """
        if shape_dist_traveled and 'shape_dist_traveled' not in keep_origin_columns:
            keep_origin_columns += ['shape_dist_traveled']
            keep_destination_columns += ['shape_dist_traveled']
            
        self.stop_times = feed_links.clean_sequences(self.stop_times)
        self.links = feed_links.link_from_stop_times(
            self.stop_times,
            max_shortcut=1,
            stop_id='stop_id',
            keep_origin_columns=keep_origin_columns,
            keep_destination_columns=keep_destination_columns,
            stop_id_origin='origin',
            stop_id_destination='destination',
            out_sequence='link_sequence'
        ).reset_index()
        self.links['time'] = self.links['arrival_time'] - self.links['departure_time']
        self.links.rename(
            columns={
                'origin': 'a',
                'destination': 'b',
            },
            inplace=True
        )

        if shape_dist_traveled:
            self.links['shape_dist_traveled'] = self.links['shape_dist_traveled_destination'] - self.links['shape_dist_traveled_origin']
            self.links.drop(columns=['shape_dist_traveled_origin', 'shape_dist_traveled_destination'], inplace=True)

        if not time_expanded:
            self.links = self.links.merge(self.frequencies[['trip_id', 'headway_secs']], on='trip_id')
            self.links.rename(columns={'headway_secs': 'headway'}, inplace=True)
            # Filter on strictly positive headway (Headway = 0 : no trip)
            self.links = self.links.loc[self.links['headway'] > 0].reset_index(drop=True)
        links_trips = pd.merge(self.trips, self.routes, on='route_id')
        self.links = pd.merge(self.links, links_trips, on='trip_id')

    def build_geometries(self, 
                         use_utm=True, 
                         from_shape=False, 
                         simplify_dist=5, 
                         stick_nodes_on_links=False,
                         emission_weight=0.5,
                         log=True,
                         num_cores=1,
                           **kwargs):
        self.nodes = gk.stops.geometrize_stops_0(self.stops)
        if use_utm:
            epsg = get_epsg(self.stops.iloc[1]['stop_lat'], self.stops.iloc[1]['stop_lon'])
            if log: print('export geometries in epsg:', epsg)
            self.nodes = self.nodes.to_crs(epsg=epsg)
            if from_shape:
                self.shapes = gk.shapes.geometrize_shapes_0(self.shapes)
                self.shapes = self.shapes.to_crs(epsg=epsg)
        if len(self.links) > 0:
            if from_shape:
                if not use_utm:
                    raise Exception("If using shape as geometry, use_utm should be set to true")
                if num_cores==1:
                    self.links = shape_geometry(
                        self.copy(),
                        'a',
                        'b',
                        emission_weight=emission_weight,
                        log=log,
                    )
                else:
                    self.links = parallel_shape_geometry(
                    self.copy(),
                    'a',
                    'b',
                    emission_weight=emission_weight,
                    log=log,
                    num_cores=num_cores
                )
            else:
                self.links['geometry'] = linestring_geometry(
                    self.links,
                    self.nodes.set_index('stop_id')['geometry'].to_dict(),
                    'a',
                    'b'
                )
            self.links = gpd.GeoDataFrame(self.links)
            self.links.crs = self.nodes.crs
            # simplify Linestring geometry. (remove anchor nodes)
            if simplify_dist:
                self.links.geometry = self.links.simplify(simplify_dist)

            if stick_nodes_on_links:
                self.ori_nodes = self.nodes.copy()
                self.stick_nodes_on_links()


    def stick_nodes_on_links(self):
    
        """
        Replace Nodes geometry to match links geometry. This function will create new nodes
        if a node is used by mutiples trips (or links). Duplicated nodes stop_id will be replace with
        stop_id => "<stop_id> - <trip_id> - <link_sequence>",
        while the first occurence will keep the original stop_id.
        NOTE: need link_sequence correctly order (0,1,2,3,...)

            Parameters
            ----------
            Builds
            self.links : gpd.GeoDataFrame,
                links with new a,b nodes
            self.nodes : gpd.GeoDataFrame,
                nodes with geometry matching the links linetring geometry.
            ----------

            """        
        self.links = self.links.sort_values(by='trip_id').sort_values(by='link_sequence')

        #get all nodes in links.
        nodes = self.links[['a','b','trip_id','link_sequence']]
        nodes_a = nodes[['a','trip_id','link_sequence']].set_index('a')
        nodes_b = nodes.groupby('trip_id')[['b','link_sequence']].agg('last').reset_index().set_index('b')
        nodes_b['link_sequence'] = nodes_b['link_sequence']+1 # we will remove 1 in link sequence later for nodes b dict
        nodes = pd.concat([nodes_a,nodes_b])
        # all nodes are from a (first link) except last one from the last link.
        # later we will overwrite them with link sequence. for nodes b, it will be link_sequence -1
        # however, as the the last noode use the last link (node b). we add +1 to this link sequence
        nodes = nodes.reset_index()
        nodes = nodes.rename(columns={'index':'stop_id'})

        # create tuple trip_id,link_sequence for graoupby explode.
        nodes['uuid'] = list(zip(nodes['trip_id'],nodes['link_sequence']))
        #get all duplicated nodes and create new one (except the first encouter. stay the original node.)
        nodes_dup_list = nodes.groupby('stop_id')[['uuid']].agg(list)

        # get only the trips after the first one (first is not changing)
        nodes_dup_list['uuid'] = nodes_dup_list['uuid'].apply(lambda x: x[:-1])
        nodes_dup_list['len'] = nodes_dup_list['uuid'].apply(len)
        nodes_dup_list = nodes_dup_list[nodes_dup_list['len']>0]
        if len(nodes_dup_list)>0: # skip if no duplicated nodes
            # explode and split tuple (uuid) in original trip_id  link_sequence.
            nodes_dup_list = nodes_dup_list.explode('uuid').reset_index()
            nodes_dup_list['trip_id'], nodes_dup_list['link_sequence'] = zip(*nodes_dup_list['uuid'])
            # new name!
            nodes_dup_list['new_stop_id'] = nodes_dup_list['stop_id'].astype(str) + '-' + nodes_dup_list['trip_id'].astype(str)+ '-' + nodes_dup_list['link_sequence'].astype(str)
            # create dict (trip_id, stop_id) : new_stop_name for changing the links
            # for b. remove 1 in link_sequence as we used link sequence for a. last node we did add 1 in link sequence.
            nodes_dup_list['link_sequence_b'] = nodes_dup_list['link_sequence']-1
            new_stop_id_a_dict = nodes_dup_list.set_index(['trip_id','link_sequence','stop_id'])['new_stop_id'].to_dict()
            new_stop_id_b_dict = nodes_dup_list.set_index(['trip_id','link_sequence_b','stop_id'])['new_stop_id'].to_dict()
            if len(nodes_dup_list[nodes_dup_list['new_stop_id'].duplicated()]) > 0:
                print('there is at least a node with duplicated (stop_id, trip_id, link_sequence), will not be split in different nodes')
                print(nodes_dup_list[nodes_dup_list['new_stop_id'].duplicated()]['new_stop_id'])

            #duplicate nodes and concat them to the existing nodes.
            new_nodes = nodes_dup_list[['stop_id','new_stop_id']].merge(self.nodes,left_on='stop_id',right_on='stop_id')
            new_nodes = new_nodes.drop(columns=['stop_id']).rename(columns = {'new_stop_id': 'stop_id'})
            self.nodes = pd.concat([self.nodes, new_nodes],ignore_index=True)

            # change nodes stop_id with new ones in links

            self.links['new_a'] = self.links.set_index(['trip_id','link_sequence','a']).index.map(new_stop_id_a_dict)
            self.links['a'] = self.links['new_a'].combine_first(self.links['a'])

            self.links['new_b'] = self.links.set_index(['trip_id','link_sequence','b']).index.map(new_stop_id_b_dict)
            self.links['b'] = self.links['new_b'].combine_first(self.links['b'])

            self.links = self.links.drop(columns = ['new_a','new_b'])

        # apply new geometry (links geometry [0] or [-1] for last nodes.)
        nodes = self.links[['a','b','trip_id','geometry']]
        nodes_a = nodes[['a','trip_id','geometry']].set_index('a')
        nodes_a['geometry'] = nodes_a['geometry'].apply(lambda g: Point(g.coords[0]))
        nodes_b =nodes.groupby('trip_id')[['b','geometry']].agg('last').reset_index().set_index('b')
        nodes_b['geometry'] = nodes_b['geometry'].apply(lambda g: Point(g.coords[-1]))
        nodes = pd.concat([nodes_a,nodes_b])
        nodes = nodes.reset_index()
        nodes = nodes.rename(columns={'index':'stop_id'})
        geom_dict = nodes.set_index('stop_id')['geometry'].to_dict()


        self.nodes['new_geom'] = self.nodes['stop_id'].apply(lambda x: geom_dict.get(x))
        self.nodes['geometry'] = self.nodes['new_geom'].combine_first(self.nodes['geometry'])
        self.nodes = self.nodes.drop(columns = ['new_geom'])

    def append_dist_to_shapes(self):
        '''
        return self with self.shapes['shape_dist_traveled'] 
        '''
        #transform coords to meters
        epsg = get_epsg(self.stops.iloc[1]['stop_lat'], self.stops.iloc[1]['stop_lon'])
        # pyproj takes [y,x] and return [x,y].
        self.shapes['shape_pt_x'],self.shapes['shape_pt_y'] = transform(4326, epsg, self.shapes['shape_pt_lat'],self.shapes['shape_pt_lon'])
        
        #get distance between each points
        self.shapes = self.shapes.sort_values(['shape_id','shape_pt_sequence']).reset_index(drop=True)
        self.shapes['geom'] = self.shapes[['shape_pt_y','shape_pt_x']].apply(tuple,axis=1)
        self.shapes['previous_geom'] = self.shapes['geom'].shift(+1).fillna(method='bfill')

        self.shapes['dist'] = self.shapes[['previous_geom','geom']].apply(lambda x: euclidean_distance(x[0],x[1]), axis=1)
        # cumsum the dist minus the first one (should be 0 but its not due to the batch operation)
        self.shapes['shape_dist_traveled'] = self.shapes.groupby('shape_id')['dist'].apply(cumulative_minus_first).values

        self.shapes = self.shapes.drop(columns=['shape_pt_x','shape_pt_y','geom','previous_geom','dist'])
        if self.dist_units == 'km':
            self.shapes['shape_dist_traveled'] = self.shapes['shape_dist_traveled']/1000
        
    def append_dist_to_stop_times_fast(self):
        '''
        This function apply self.shapes['shape_dist_traveled'] to self.stop_times
        Use the departure time to interpolate the distance at each stop.
        This function is faster than gtfs_kit same method and use 3 times less memory.
        
        return self with self.shapes['shape_dist_traveled'] and 'time_traveled'
        '''
        
        self.stop_times['time'] = self.stop_times['departure_time'].apply(to_seconds)
        self.stop_times = self.stop_times.sort_values(['trip_id','stop_sequence']).reset_index(drop=True)
        self.stop_times['previous_time'] = self.stop_times['time'].shift(+1).fillna(method='bfill')

        self.stop_times['diff_time'] = self.stop_times[['previous_time','time']].apply(lambda x: x[1]-x[0], axis=1)
        # cumsum the dist minus the first one (should be 0 but its not due to the batch operation)
        self.stop_times['time_traveled'] = self.stop_times.groupby('trip_id')['diff_time'].apply(cumulative_minus_first).values


        df = self.stop_times.groupby('trip_id')['time_traveled'].agg(['last']).rename(columns={'last':'time_traveled'})

        shape_dict = self.trips.set_index('trip_id')['shape_id'].to_dict()
        df['shape_id']=df.index.map(shape_dict.get)

        shape_dist = self.shapes.sort_values(['shape_id','shape_dist_traveled']).groupby('shape_id')['shape_dist_traveled'].agg('last').to_dict()
        df['shape_dist_traveled'] = df['shape_id'].apply(lambda x: shape_dist.get(x))

        #for each trip. find a mapping time => distance
        conversion_dict = (df['shape_dist_traveled']/df['time_traveled']).to_dict()

        self.stop_times['dist_factor'] =self.stop_times['trip_id'].apply(lambda x: conversion_dict.get(x,0))
        self.stop_times['shape_dist_traveled'] = self.stop_times['time_traveled'] * self.stop_times['dist_factor']

        self.stop_times = self.stop_times.drop(columns=['time','previous_time','diff_time','dist_factor'])