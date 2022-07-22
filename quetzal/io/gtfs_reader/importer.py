# pylint: disable=no-member
import geopandas as gpd
import gtfs_kit as gk
import pandas as pd
from shapely.geometry import LineString
from syspy.transitfeed import feed_links

from . import patterns
from .feed_gtfsk import Feed


def get_epsg(lat, lon):
    return int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))



def to_seconds(time_string):  # seconds
    return pd.to_timedelta(time_string).total_seconds()


def linestring_geometry(dataframe, point_dict, from_point, to_point):
    df = dataframe.copy()

    def geometry(row):
        return LineString(
            (point_dict[row[from_point]], point_dict[row[to_point]]))

    return df.apply(geometry, axis=1)


class GtfsImporter(Feed):

    """This class allows to import GTFS and transform it into a Quetzal format, 
    that is to say a geodataframe of nodes (transit stops) and a geodataframe of links containing
    all the information about the public transport lines.
    
    It gives access to different method to handle GTFS data:

        - description (GtfsImporter.describe())
        - maps
        - filtering on specific area, dates, trips, stops (GtfsImporter.restrict())
        - stops / trips aggregation
        - frequency conversion

    Example
    --------

    full_feed  = BaseGtfsImporter(gtfs_path)

    You can either use the wrapper : 

    full_feed.build()

    Or process manually by applying the following functions.

    Can restrict to dates and hour with :

        * feed_r = full_feed.restrict(dates=['20200204']) # the dates must be within the feed start and end dates
        * feed_t = feed_r.restrict(time_range=['08:00:00', '09:00:00'])

    If converting timetable to frequency is needed :
    
        * feed.group_services()
        * feed.build_stop_clusters(distance_threshold=300)
        * feed.build_patterns(on='cluster_id')
        * feed_frequencies = feed.convert_to_frequencies(time_range=time_range)

    Last but not least :

        * feed_frequencies.build_links_and_nodes()
        * feed_frequencies.build_geometries(use_utm=False)

    And to save :

        * sm = stepmodel.StepModel()
        * sm.links = feed_frequencies.links
        * sm.nodes = feed_frequencies.nodes

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
        """Those stop clusters are used to regroup pattern of similar services.
        The aim is to simplify the number of stops and services in a public transport route.
        Works on the feed defined with BaseGtfsImporter(gtfs_path).

        Parameters
        ----------
        distance_threshold : int, optional, default 150
            Distance threshold for node to group
        col : str, optional, default 'cluster_id'
            Name of the column that will be created
        use_parent_station : bool, optional, default False
            Use parent station if defined
        
        Builds
        ----------
            feed.stop : 
                add cluster column

        """        
        self.stops = patterns.build_stop_clusters(self.stops, **kwargs)

    def build(self, date, time_range, cluster_distance_threshold=None):
        """Wrapper that allows to apply all required functions to build
        links and nodes from GTFS, including converting timetable to frequency.
        If keeping timetable is needed, use function build_links_and_nodes with parameter
        time_expanded = True.

        Parameters
        ----------
        date : string
            date of the gtfs feed to import. Example : '20200204'
        time_range : list
            Use if you want to restrict to a specific period of the day.
            Example : ['08:00:00', '09:00:00']
        cluster_distance_threshold : int, optional, default None
            Use to simplify the number of stops and services in a public transport route.
            Distance threshold for node to group.


        Builds
        ----------
        self.links
            Links of the public transport system and pt routes caracteristics, quetzal format
        self.nodes
            Public transport stations.
        """      

        print('Restricting to date…')
        feed = self.restrict(dates=[date])
        print('Restricting to time')
        feed = self.restrict(time_range=time_range)
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

    def build_links_and_nodes(self, time_expanded=False, log=True, **kwargs):
        """Transform the transit feed into a Quetzal format, 
        that is to say a geodataframe of nodes (transit stops) and a geodataframe of links containing
        all the information about the public transport lines.

        Parameters
        ----------
        time_expanded : bool, optional, default False
            Use true if you want to build a time expanded model, 
            using timetables for pathfinders
        log : bool, optional, default True
            print epsg
        use_utm : bool, optional, default True
            Compute projection from stops coordinates

        Builds
        ----------
        self.links
            Links of the public transport system and pt routes caracteristics, quetzal format
        self.nodes
            Public transport stations.
        """        
        self.to_seconds()
        self.build_links(time_expanded=time_expanded, **kwargs)
        self.build_geometries(log=log)

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
        keep_origin_columns=['departure_time', 'pickup_type']
        keep_destination_columns=['arrival_time', 'drop_off_type']
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
            self.links['shape_dist_traveled'] = self.links['shape_dist_traveled_destination'] -self.links['shape_dist_traveled_origin']
            self.links.drop(columns=['shape_dist_traveled_origin', 'shape_dist_traveled_destination'], inplace=True)

        if not time_expanded:
            self.links = self.links.merge(self.frequencies[['trip_id', 'headway_secs']], on='trip_id')
            self.links.rename(columns={'headway_secs': 'headway'}, inplace=True)
            # Filter on strictly positive headway (Headway = 0 : no trip)
            self.links = self.links.loc[self.links['headway'] > 0].reset_index(drop=True)
        links_trips = pd.merge(self.trips, self.routes, on='route_id')
        self.links = pd.merge(self.links, links_trips, on='trip_id')

    def build_geometries(self, use_utm=True, log=True):
        self.nodes = gk.stops.geometrize_stops_0(self.stops)
        if use_utm:
            epsg = get_epsg(self.stops.iloc[1]['stop_lat'], self.stops.iloc[1]['stop_lon'])
            if log: print('export geometries in epsg:', epsg)
            self.nodes = self.nodes.to_crs(epsg=epsg)

        self.links['geometry'] = linestring_geometry(
            self.links,
            self.nodes.set_index('stop_id')['geometry'].to_dict(),
            'a',
            'b'
        )
        self.links = gpd.GeoDataFrame(self.links)
        self.links.crs = self.nodes.crs
