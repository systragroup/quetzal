# pylint: disable=no-member
import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from syspy.transitfeed import feed_links
from syspy.spatial import spatial
from quetzal.engine.pathfinder_utils import paths_from_edges
from tqdm import tqdm
import math

from . import patterns
from .feed_gtfsk import Feed


def get_epsg(lat, lon):
    return int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))


def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def to_seconds(time_string):  # seconds
    return pd.to_timedelta(time_string).total_seconds()


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


def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()

    def geometry(row):
        return LineString(
            (point_dict[row[from_point]], point_dict[row[to_point]]))

    return df.apply(geometry, axis=1)


def shape_geometry(self, from_point, to_point, max_candidates=10, log=False):
    to_concat = []

    point_dict = self.nodes.set_index('stop_id')['geometry'].to_dict()
    shape_dict = self.shapes.set_index('shape_id')['geometry'].to_dict()

    stop_ids = set(self.links[[from_point, to_point]].values.flatten())
    stop_pts = pd.DataFrame([point_dict.get(n) for n in stop_ids], index=list(stop_ids), columns=['geometry'])

    for trip in tqdm(set(self.links['trip_id'])):
        links = self.links[self.links['trip_id'] == trip].copy()
        links = links.drop_duplicates(subset=['link_sequence']).sort_values(by='link_sequence')
        s = shape_dict.get(links.iloc[0]['shape_id'])
        trip_stops = list(links[from_point]) + [links.iloc[-1][to_point]]

        # Find segments candidates from shape for projection
        segments = pd.DataFrame(list(map(LineString, zip(s.coords[:-1], s.coords[1:]))), columns=['geometry'])
        n_candidates = min([len(segments), max_candidates])

        ng = spatial.nearest(stop_pts.loc[set(trip_stops)], segments, n_neighbors=n_candidates)
        ng = ng.set_index(['ix_one', 'rank'])['ix_many'].to_dict()

        # Distance matrix (stops * n_candidates)
        distances_a = np.empty((len(trip_stops), n_candidates))
        for r in range(n_candidates):
            proj_pts = []
            for n in trip_stops:
                seg = segments.loc[ng[(n, r)]]['geometry']
                proj_pts.append(seg.interpolate(seg.project(point_dict.get(n))))
            distances = [s.project(pts) for pts in proj_pts]
            distances_a[:, r] = distances
        # TODO (faster): create custom nearest to build distance matrix directly

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

        # Build edges + dijkstra
        candidates_e = diff_df[['from', 'to', 0]].values
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

    def build(self, date, time_range, cluster_distance_threshold=None):
        print('Restricting to date…')
        feed = self.restrict(dates=[date])
        print('Grouping services…')
        feed.group_services()
        print('Cleaning…')
        feed = feed.clean()
        if cluster_distance_threshold is not None:
            print('Clustering stops…')
            feed.build_stop_clusters(distance_threshold=cluster_distance_threshold)
            print('Building patterns…')
            feed.build_patterns(on='cluster_id')
        else:
            print('Building patterns…')
            feed.build_patterns()
        print('Converting to frequencies…')
        feed = feed.convert_to_frequencies(time_range=time_range)
        print('Building links and nodes…')
        feed.build_links_and_nodes()
        return feed

    def build_links_and_nodes(self, time_expanded=False, shape_dist_traveled=False, log=True, **kwargs):
        self.to_seconds()
        self.build_links(time_expanded=time_expanded, shape_dist_traveled=shape_dist_traveled)
        self.build_geometries(log=log, **kwargs)

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

    def build_links(self, time_expanded=False, shape_dist_traveled=False):
        """
        Create links and add relevant information
        """
        keep_origin_columns = ['departure_time', 'pickup_type']
        keep_destination_columns = ['arrival_time', 'drop_off_type']
        if shape_dist_traveled:
            keep_origin_columns += ['shape_dist_traveled']
            keep_destination_columns += ['shape_dist_traveled']

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

    def build_geometries(self, use_utm=True, from_shape=False, log=True, **kwargs):
        self.nodes = gk.stops.geometrize_stops_0(self.stops)
        if use_utm:
            epsg = get_epsg(self.stops.iloc[1]['stop_lat'], self.stops.iloc[1]['stop_lon'])
            if log:
                print('export geometries in epsg:', epsg)
            self.nodes = self.nodes.to_crs(epsg=epsg)
            if from_shape:
                self.shapes = gk.shapes.geometrize_shapes_0(self.shapes)
                self.shapes = self.shapes.to_crs(epsg=epsg)

        if from_shape:
            if not use_utm:
                raise Exception("If using shape as geometry, ust_utm should be set to true")
            self.links = shape_geometry(
                self.copy(),
                'a',
                'b',
                log=log,
                **kwargs
            )
        else:
            self.links['geometry'] = linestring_geometry(
                self,
                'a',
                'b'
            )
        self.links = gpd.GeoDataFrame(self.links)
        self.links.crs = self.nodes.crs
