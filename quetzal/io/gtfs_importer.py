# -*- coding: utf-8 -*-

import os
import pandas as pd
from shapely.geometry import Point, LineString

from syspy.spatial import spatial, zoning
from syspy.transitfeed import feed_links
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
        if 'frequencies.txt' in os.listdir(self.gtfs_path):
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

