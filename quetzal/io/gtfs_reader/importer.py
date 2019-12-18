# -*- coding: utf-8 -*-

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString

from syspy.spatial import spatial, zoning
from syspy.transitfeed import feed_links, feed_stops
from . import frequencies
from .feed_gtfsk import Feed
import gtfs_kit as gk
from . import patterns
# seconds

def to_seconds(time_string):
    return pd.to_timedelta(time_string).total_seconds()

def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()
    
    def geometry(row):
        return LineString(
            (point_dict[row[from_point]], point_dict[row[to_point]]))
    return df.apply(geometry, axis=1)


class GtfsImporter(Feed):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    from .directions import build_directions
    from .patterns import build_patterns
    from .services import group_services
    from .frequencies import convert_to_frequencies, compute_pattern_headways  

    def clean(self):
        feed = super().clean()
        feed.stop_times = feed_links.clean_sequences(feed.stop_times)
        return feed

    def build_stop_clusters(self, **kwargs):
        self.stops = patterns.build_stop_clusters(self.stops, **kwargs)
    
    def build_links_and_nodes(self, columns_to_cast_to_string=[]):
        self.clean()
        self.cast_columns_to_string(columns=columns_to_cast_to_string)
        self.to_seconds()
        self.build_links()
        self.build_geometries()

    def to_seconds(self):
        # stop_times
        time_columns = ['arrival_time', 'departure_time']
        self.stop_times[time_columns] = self.stop_times[
            time_columns
        ].applymap(to_seconds)
        # frequencies
        time_columns = ['start_time', 'end_time']
        self.frequencies[time_columns] = self.frequencies[
            time_columns
        ].applymap(to_seconds)

    def build_links(self):
        """
        Create links and add relevant information
        """
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
        self.links = links.merge(self.frequencies[['trip_id', 'headway_secs']], on='trip_id')
        self.links.rename(columns={'headway_secs': 'headway'}, inplace=True)
        # Filter on strictly positive headway (Headway = 0 : no trip)
        self.links = self.links.loc[self.links['headway']>0].reset_index(drop=True)
        links_trips = pd.merge(self.trips, self.routes, on='route_id')
        self.links = pd.merge(self.links, links_trips, on ='trip_id') 
        
    def build_geometries(self, use_utm=True):
        self.nodes = gk.stops.geometrize_stops_0(self.stops, use_utm=use_utm)
        self.links['geometry'] = linestring_geometry(
            self.links, 
            self.nodes.set_index('stop_id')['geometry'].to_dict(), 
            'a', 
            'b'
        )
        self.links = gpd.GeoDataFrame(self.links)

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
                # print(key, cols, 'converted to string')
            except AttributeError:  # 'str' object has no attribute 'columns'
                pass