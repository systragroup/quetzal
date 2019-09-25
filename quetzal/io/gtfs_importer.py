# -*- coding: utf-8 -*-

import os
import pandas as pd
from shapely.geometry import Point, LineString

from syspy.spatial import spatial, zoning
from syspy.transitfeed import feed_links, feed_stops
from syspy.syspy_utils import headway_utils

# seconds

def to_seconds(time_string):
    return pd.to_timedelta(time_string).total_seconds()

def point_geometry(row):
    return Point(row['stop_lon'], row['stop_lat'])

def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()
    
    def geometry(row):
        return LineString(
            (point_dict[row[from_point]], point_dict[row[to_point]]))
    return df.apply(geometry, axis=1)


class BaseGtfsImporter():

    """
    importer = BaseGtfsImporter(gtfs_path)

    importer.read()
    importer.build()

    sm = stepmodel.StepModel()

    sm.links = importer.links
    sm.nodes = importer.stops
    """

    def __init__(self, gtfs_path):
        self.gtfs_path = gtfs_path

    def read(self, encoding=None):

        self.stop_times = pd.read_csv(
            self.gtfs_path + 'stop_times.txt', 
            encoding=encoding, 
        )

        self.trips = pd.read_csv(
            self.gtfs_path + 'trips.txt', 
            encoding=encoding, 
            low_memory=False  # mixed types
        )

        self.routes = pd.read_csv(
            self.gtfs_path + 'routes.txt', 
            encoding=encoding
        )

        self.stops = pd.read_csv(
            self.gtfs_path + 'stops.txt',
            encoding=encoding
        )

        self.calendar_dates = pd.read_csv(
            self.gtfs_path + 'calendar_dates.txt',
            encoding=encoding
        )

        self.calendar = pd.read_csv(
            self.gtfs_path + 'calendar.txt',
            encoding=encoding
        )

        self.frequencies = pd.read_csv(
            self.gtfs_path + 'frequencies.txt',
            encoding=encoding
        )
        if 'shapes.txt' in os.listdir(self.gtfs_path):
            self.shapes = pd.read_csv(
                self.gtfs_path + 'shapes.txt',
                encoding=encoding
            )


    def pick_trips(self):
        # one trip by direction
        self.trips = pd.merge(self.trips, self.routes[['route_id']])
        
        self.trips = self.trips.groupby(
            ['route_id', 'direction_id'],
            as_index=False
            ).first()
        self.stop_times = pd.merge(self.stop_times, self.trips[['trip_id']])
        
        stop_id_set = set(self.stop_times['stop_id'])
        self.stops = self.stops.loc[self.stops['stop_id'].isin(stop_id_set)]

    def to_seconds(self):
        time_columns = ['arrival_time', 'departure_time']
        self.stop_times[time_columns] = self.stop_times[
            time_columns].applymap(to_seconds)

    def build_links(self):
        links = feed_links.link_from_stop_times(
            self.stop_times,
            max_shortcut=1,
            stop_id='stop_id',
            keep_origin_columns = ['departure_time'],
            keep_destination_columns = ['arrival_time'],
            stop_id_origin = 'origin',
            stop_id_destination = 'destination',
            out_sequence='link_sequence'
        ).reset_index()
        links['time'] = links['arrival_time'] - links['departure_time']
        links.rename(
            columns={
                'origin': 'a',
                'destination': 'b',
            },
            inplace=True
        )
        self.links = links

    def merge_tables(self):
        # merge
        self.trips = pd.merge(self.trips, self.routes, on='route_id')
        # [['trip_id', 'route_id', 'direction_id']]
        self.links = pd.merge(self.links, self.trips, on ='trip_id') 


    def build_geometries(self):
        self.stops['geometry'] = self.stops.apply(point_geometry, axis=1)
        self.links['geometry'] = linestring_geometry(
            self.links, 
            self.stops.set_index('stop_id')['geometry'].to_dict(), 
            'a', 
            'b'
        )

    def cast_columns_to_string(
        self, 
        columns=['trip_id', 'route_id', 'stop_id']
    ) :
        for key, attr in self.__dict__.items():
            try:
                cols = []
                for c in attr.columns :
                    if c in columns:
                        cols.append(c)
                        attr[c] = attr[c].astype(str)
                print(key, cols, 'converted to string')
            except AttributeError:  # 'str' object has no attribute 'columns'
                pass

    def build(self):
        self.pick_trips()
        self.to_seconds()
        self.build_links()
        self.merge_tables()
        self.build_geometries()

    def build_headways(self, timerange, service_ids=None):
        GTFSFrequencies = headway_utils.GTFS_frequencies_utils(self.frequencies, self.trips)
        timerange_sec = [
            headway_utils.hhmmss_to_seconds_since_midnight(x) for x in timerange
        ]
        self.trips['headway'] = self.trips.apply(
            lambda x: GTFSFrequencies.compute_average_headway(
                [x['trip_id']],
                timerange_sec,
                service_ids
            ),
            1
        )
        self.links = self.links.merge(
            self.trips[['trip_id', 'headway']],
        )

    def build_patterns_and_directions(
            self, group='route_id',
            direction_ratio_threshold=0.1, debug=False,
            stop_cluster_kwargs={}):
        """
        From a given links dataframe:
        1. Group by direction trips belonging to the same group
        2. Merge trips that have the same sorted list
        3. Create patterns dataframe
        """
        stops_parent_stations = feed_stops.stops_with_parent_station(self.stops, stop_cluster_kwargs=stop_cluster_kwargs)
        stop_to_parent_station = stops_parent_stations.set_index('stop_id')['parent_station'].to_dict()
        trips_direction = get_trips_direction(self.links, stop_to_parent_station, group=group, direction_ratio_threshold=direction_ratio_threshold)
        trips_direction['id'] = trips_direction['links_list'].apply(
            lambda x: str([[a[0], a[1]] for a in x if a[0]!=a[1]])
        )
        
        patterns = trips_direction.groupby([group, 'direction_id', 'id']).agg({'trip_id': list}).reset_index()
        
        patterns['pattern_id_n'] = patterns.groupby([group, 'direction_id']).cumcount()
        
        patterns['pattern_id'] = patterns.apply(
            lambda x: '{}_{}_{}'.format(x[group], x['direction_id'], x['pattern_id_n']),
            1
        )
        patterns.drop(['id','pattern_id_n'], 1, inplace=True)
        self.patterns = patterns
        if debug:
            self.trips_direction = trips_direction
            self.stops_parent_stations = stops_parent_stations


def get_trip_links_list(trip_id, links):
    """
    Given a trip_id and a links dataframe, return the list of links included in the trip.
    """
    arrays_result =  list(links[
        (links['trip_id'] == trip_id)&(links['a']!=links['b'])
    ][['a', 'b']].values)
    return [list(x) for x in arrays_result]


def same_direction_ratio(parent_links_list, links_list):
    """
    Given two lists of links, returns the shared directio ratio [-1,1]:
    if 1: all of the oriented links of links_list are included in parent_links_list
          and with the same orientation
    if -1: all of the oriented links of links_list are included in parent_links_list
           but in reversed orientation
    if 0: None of the links in links_list (or their reverse) are in parent_links_list
    """
    same_direction_score = 0
    for link in links_list:
        for link_0 in parent_links_list:
            if link == link_0:
                same_direction_score +=1
            elif link[::-1] == link_0:
                same_direction_score -= 1
    return same_direction_score / len(links_list)


def get_trips_direction(links, stop_to_parent_station=None, reference_trip_ids=None, group='route_id', direction_ratio_threshold=0.1):
    """
    Read a links dataframe and compute the direction for each trip within each group.
    Reference trip ids can be given to define trips that will be given the 0 direction.
    If not, the longest trip of each group (in terms of number of links) will be the reference.
    """
    # Create trip-direction df
    df = links.copy()[[group, 'trip_id']].drop_duplicates().reset_index(drop=True)
    df['direction_id'] = None
    # Aggregate links in list
    df['links_list'] = df['trip_id'].apply(lambda x: get_trip_links_list(x, links))
    if stop_to_parent_station is not None:
        df['links_list'] = df['links_list'].apply(lambda x: [[stop_to_parent_station[a] for a in l] for l in x])
        df['links_list'] = df['links_list'].apply(lambda a: [[x[0], x[1]] for x in a if x[0]!=x[1]])

    df['links_number'] = df['links_list'].apply(len)
    # Sort by descending length
    df = df.sort_values(
        [group, 'links_number'],
        ascending=False
    ).reset_index(drop=True)
    
    if reference_trip_ids is None:
        reference_trip_ids = df.groupby(group).first()['trip_id'].to_dict()
    # Set direction of reference trip to 0
    df['reference_trip'] = df[group].apply(lambda x: reference_trip_ids[x])
    # Compute shared direction ratio
    df['direction_0_ratio'] = df.apply(
        lambda x: same_direction_ratio(
            x['links_list'],
            df.loc[df['trip_id']==x['reference_trip'], 'links_list'].values[0]
        ),
        1
    )
    # Give 0 or 1 direction id if shared direction is not zero, 100 (=not found) otherwise
    df['direction_id'] = df['direction_0_ratio'].apply(
        lambda x: 0*(x>direction_ratio_threshold) + 1*(x<-direction_ratio_threshold) +\
                 100*(x>=-direction_ratio_threshold)*(x<=direction_ratio_threshold)
        )
    
    # While some direction id are 100 (not found): recursively call function
    count = 0
    while len(df[df['direction_id']==100])>0 and (count < 10):
        count += 1
        df_ = get_trips_direction(
            links.loc[links['trip_id'].isin(df[df['direction_id']==100]['trip_id'].values)],
            stop_to_parent_station=stop_to_parent_station,
            group=group,
            direction_ratio_threshold=direction_ratio_threshold
        )
        d=df[
            (df['direction_0_ratio']!=0)&(df['direction_id']<100)
        ].groupby(group)['direction_id'].max().to_dict()
        df_['direction_id'] = df_.apply(
            lambda x: 1 + x['direction_id'] + d[x[group]] if x['direction_id']!=100 else x,
            1
        )
        t = df_.set_index('trip_id')['direction_id'].to_dict()
        df['direction_id'] = df.apply(lambda x: t.get(x['trip_id'], x['direction_id']), 1)
    
    return df[[group, 'trip_id', 'direction_id', 'direction_0_ratio', 'links_list']]