# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString

from syspy.spatial import spatial, zoning
from syspy.transitfeed import feed_links, feed_stops
from . import frequency_utils
from .feed_gtfsk import Feed
import gtfs_kit as gk
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


class GtfsImporter(Feed):

    """
    importer = BaseGtfsImporter(gtfs_path)
    importer.read()
    importer.build()

    sm = stepmodel.StepModel()

    sm.links = importer.links
    sm.nodes = importer.stops
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clean(self):
        feed = super().clean()
        feed.stop_times = feed_links.clean_sequences(feed.stop_times)
        return feed

    def stop_times_to_seconds(self):
        time_columns = ['arrival_time', 'departure_time']
        self.stop_times[time_columns] = self.stop_times[
            time_columns
        ].applymap(to_seconds)

    def pick_trips(self):
        # Keep only one trip by direction
        self.trips = pd.merge(self.trips, self.routes[['route_id']])
        
        self.trips = self.trips.groupby(
            ['route_id', 'direction_id'],
            as_index=False
            ).first()
        self.stop_times = pd.merge(self.stop_times, self.trips[['trip_id']])
        stop_id_set = set(self.stop_times['stop_id'])
        self.stops = self.stops.loc[self.stops['stop_id'].isin(stop_id_set)]
    
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

    def build(self, service_ids=None, pick_trips=True, columns_to_cast_to_string=[]):
        self.cast_columns_to_string(columns=columns_to_cast_to_string)
        if service_ids is not None:
            self.filter_services(service_ids)
        if pick_trips:
            self.pick_trips()
        self.to_seconds()
        self.clean_sequences()
        self.build_links()
        self.merge_tables()
        self.build_geometries()


    def build_headways(self, timerange, service_ids=None):
        GTFSFrequencies = frequency_utils.GTFS_frequencies_utils(self.frequencies, self.trips)
        timerange_sec = [
            frequency_utils.hhmmss_to_seconds_since_midnight(x) for x in timerange
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
        2. Merge trips that have the same sorted list of links
        3. Create patterns dataframe
        """
        stops_parent_stations = feed_stops.stops_with_parent_station(self.stops, stop_cluster_kwargs=stop_cluster_kwargs)
        stop_to_parent_station = stops_parent_stations.set_index('stop_id')['parent_station'].to_dict()
        trips_direction = feed_links.get_trips_direction(self.links, stop_to_parent_station, group=group, direction_ratio_threshold=direction_ratio_threshold)
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

    def build_pattern_headways(self):
        """
        Requires patterns dataframe and links headways to be computed
        1. Compute the aggregated headway of each pattern
        2. Pick one trip per pattern
        3. Replace trip_id per pattern_id
        """
        # Filter on strictly positive headway (Headway = 0 : no trip)
        self.links = self.links.loc[self.links['headway']>0].reset_index(drop=True)
        self.patterns['trip_id'] = self.patterns['trip_id'].apply(
            lambda x: [t for t in x if t in self.links['trip_id'].unique()]
        )
        self.patterns = self.patterns[self.patterns['trip_id'].apply(len)>0]

        # Add trip headways to each pattern
        trip_id_to_headway = self.links.groupby('trip_id')['headway'].first().to_dict()
        self.patterns['headways'] = self.patterns['trip_id'].apply(lambda x: [trip_id_to_headway.get(a, np.inf) for a in x])
        self.patterns['headway'] = self.patterns['headways'].apply(lambda x : 1 / sum(1/np.array(x)))
        self.patterns = self.patterns[self.patterns['headway']<np.inf]     

        # Pick one trip per pattern
        self.patterns['trip_id_1'] = self.patterns['trip_id'].apply(lambda x: x[0])

        # Create new links
        new_links = self.links.copy()
        new_links['itsim_trip_id'] = new_links['trip_id']
        new_links = new_links[new_links['trip_id'].isin(self.patterns['trip_id_1'].values)]
        self.patterns = self.patterns.set_index('trip_id_1')
        new_links['direction_id'] = new_links['trip_id'].apply(lambda x: int(self.patterns.loc[x, 'direction_id']))
        new_links['headway'] = new_links['trip_id'].apply(lambda x: int(self.patterns.loc[x, 'headway']))
        new_links['trip_id'] = new_links['trip_id'].apply(lambda x: self.patterns.loc[x, 'pattern_id'])

        self.links = new_links




