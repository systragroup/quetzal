# pylint: disable=no-member

from collections import OrderedDict
from copy import deepcopy

import folium as fl
import folium.plugins as fp
import gtfs_kit as gk
import pandas as pd
from tqdm import tqdm

# See documentation of this library here: https://mrcagney.github.io/gtfs_kit_docs/index.html#document-index
# This file overwrites / customizes some methods of the library


class Feed(gk.feed.Feed):  # Overwrite Feed class

    from .filtering import (restrict, restrict_to_area, restrict_to_dates,
                            restrict_to_services, restrict_to_timerange,
                            restrict_to_trips)

    def __init__(self, dist_units: str, path=None):
        if path is not None:
            gtfs_dict = read_gtfs(path, dist_units)
            super().__init__(**gtfs_dict)
            self.path = path
        else:
            super().__init__(dist_units=dist_units)

    def describe(self, sample_date=None):
        return describe(self, sample_date)

    def get_trips(self, date=None, time=None):
        return get_trips(self, date=date, time=time)

    def copy(self):
        copy = deepcopy(self)
        return copy

    def map_stops(self, *args, embed=False, **kwargs):
        # The original method requires stop_code which is optional
        if embed:
            return embed_map(map_stops(self, *args, **kwargs))
        else:
            return map_stops(self, *args, **kwargs)

    def map_trips(self, *args, embed=False, **kwargs):
        if embed:
            return embed_map(gk.map_trips(self, *args, **kwargs))
        else:
            return gk.map_trips(self, *args, **kwargs)

    def map_routes(self, *args, embed=False, **kwargs):
        if embed:
            return embed_map(gk.map_routes(self, *args, **kwargs))
        else:
            return gk.map_routes(self, *args, **kwargs)

    def write_geojson(self, folder):
        from geojson import dump
        to_export = {
            'routes_to_geojson': 'routes',
            'stops_to_geojson': 'stops',
            'trips_to_geojson': 'trips'
        }
        for method, name in tqdm(to_export.items(), desc='routes - stops - trips'):
            gj = self.__getattribute__(method)()
            with open(folder + name + '.geojson', 'w') as f:
                dump(gj, f)

    def build_stops_timetable(self, stop_ids, dates):
        return build_stops_timetable(self, stop_ids, dates)

    def drop_unused(self):
        # drop stops without stop_times
        stop_set = list(self.stop_times.stop_id.unique())
        if 'parent_station' in self.stops.columns and not self.stops['parent_station'].isna().all():
            stop_set += list(self.stops.loc[self.stops.stop_id.isin(stop_set), 'parent_station'].values)
        self.stops = self.stops.loc[self.stops.stop_id.isin(stop_set)]

        # drops trips without stop_times
        trip_set = list(self.stop_times.trip_id.unique())
        self.trips = self.trips.loc[self.trips.trip_id.isin(trip_set)]

        # drops routes without trips
        route_set = list(self.trips.route_id.unique())
        self.routes = self.routes.set_index('route_id').loc[route_set].reset_index()

        # drops shapes without trips
        if self.shapes is not None:
            shape_set = list(self.trips.shape_id.unique())
            self.shapes = self.shapes.loc[self.shapes.shape_id.isin(shape_set)]

        # drop calendar without trips
        service_set = list(self.trips.service_id.unique())
        if self.calendar is not None:
            self.calendar = self.calendar.loc[self.calendar.service_id.isin(service_set)]
        if self.calendar_dates is not None:
            self.calendar_dates = self.calendar_dates.loc[self.calendar_dates.service_id.isin(service_set)]

        # drops agency without routes
        agency_set = list(self.routes.agency_id)
        self.agency = self.agency.loc[self.agency.agency_id.isin(agency_set)]


def read_gtfs(*args, **kwargs) -> dict:
    try:
        temp = gk.feed.read_gtfs(*args, **kwargs)
    except Exception:
        temp = gk.feed.read_feed(*args, **kwargs)
    init_temp = {key: value for key, value in temp.__dict__.items() if key in gk.constants.FEED_ATTRS_1}
    init_temp['calendar'] = temp.calendar
    init_temp['calendar_dates'] = temp.calendar_dates
    init_temp['trips'] = temp.trips
    init_temp['dist_units'] = temp.dist_units
    return init_temp


def get_trips(feed: "Feed", date=None, time=None):
    """
    Return ``Feed.trips``.
    If dates (YYYYMMDD date string or list of date string) are given then subset the result to trips
    that start on one of these dates.
    If a time or a time range (two HH:MM:SS strings, possibly with HH > 23) is given in addition to a date,
    then further subset the result to trips in service at that time or within that time.
    """

    # TODO: filtering on time
    # DATE FILTERING
    filtered = feed.trips.copy()
    if date is not None:
        if feed.calendar is not None and not feed.calendar.empty:
            weekday_str = gk.helpers.weekday_to_str(gk.helpers.datestr_to_date(date).weekday())
            services = set(
                feed.calendar[
                    (feed.calendar['start_date'] <= date)
                    & (feed.calendar['end_date'] >= date)
                    & (feed.calendar[weekday_str] == 1)
                ]['service_id'].values
            )
        else:
            services = set()
        if feed.calendar_dates is not None:
            to_add = set(
                feed.calendar_dates[
                    (feed.calendar_dates['date'] == date)
                    & (feed.calendar_dates['exception_type'] == 1)
                ]['service_id'].values
            )
            to_delete = set(
                feed.calendar_dates[
                    (feed.calendar_dates['date'] == date)
                    & (feed.calendar_dates['exception_type'] == 2)
                ]['service_id'].values
            )
            services = services.union(to_add).difference(to_delete)

        filtered = feed.trips[feed.trips['service_id'].isin(services)]

    return filtered


def describe(feed: "Feed", sample_date: str = None) -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of routes.
    Specialize some those indicators to the given YYYYMMDD sample date string,
    e.g. number of routes active on the date.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    """
    d = OrderedDict()
    dates = feed.get_dates()
    d["agencies"] = feed.agency["agency_name"].tolist()
    d["running_services"] = get_active_services(feed)
    d["timezone"] = feed.agency["agency_timezone"].iat[0]
    d["start_date"] = dates[0]
    d["end_date"] = dates[-1]
    d["num_routes"] = feed.routes.shape[0]
    d["num_trips"] = feed.trips.shape[0]
    d["num_stops"] = feed.stops.shape[0]
    if feed.shapes is not None:
        d["num_shapes"] = feed.shapes["shape_id"].nunique()
    else:
        d["num_shapes"] = 0

    if feed.frequencies is not None:
        d["num_frequencies"] = feed.frequencies.shape[0]
    else:
        d["num_frequencies"] = 0

    if sample_date is not None:
        d["sample_date"] = sample_date
        d["num_routes_active_on_sample_date"] = feed.get_routes(sample_date).shape[
            0
        ]
        trips = feed.get_trips(sample_date)
        d["num_trips_active_on_sample_date"] = trips.shape[0]
        d["num_stops_active_on_sample_date"] = feed.get_stops(sample_date).shape[0]

    f = pd.DataFrame(list(d.items()), columns=["indicator", "value"])
    return f


def get_active_services(feed: "Feed") -> list:
    cols = [
        'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday'
    ]
    if feed.calendar is not None:
        services = set(feed.calendar[feed.calendar[cols].T.sum() > 0]['service_id'])
    else:
        services = set()
    if feed.calendar_dates is not None:
        to_add = set(
            feed.calendar_dates[
                feed.calendar_dates['exception_type'] == 1
            ]['service_id'].values
        )
        services = services.union(to_add)
    return list(services)


STOP_STYLE = {
    "radius": 8,
    "fill": "true",
    "color": gk.constants.COLORS_SET2[1],
    "weight": 1,
    "fillOpacity": 0.75,
}


def embed_map(m):
    """
    Workaround taken from https://github.com/python-visualization/folium/issues/812
    for displaying Folium maps with lots of features in Chrome-based browsers.
    """
    import os

    from IPython.display import IFrame
    i = 1
    filename = 'index_1.html'
    while os.path.exists(filename):
        i += 1
        filename = 'index_{}.html'.format(i)
    m.save(filename)
    return IFrame(filename, width='100%', height='750px')


def map_stops(feed, stop_ids, stop_style=STOP_STYLE):
    """
    Return a Folium map showing the given stops of this Feed.
    If some of the given stop IDs are not found in the feed, then raise a ValueError.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create a feature group for the stops and add it to the map
    # group = fl.FeatureGroup(name="Stops")

    # Add stops to feature group
    stops = feed.stops.loc[lambda x: x.stop_id.isin(stop_ids)].fillna("n/a")

    # Add stops with clustering
    callback = f"""\
    function (row) {{
        var imarker;
        marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {stop_style}
        );
        marker.bindPopup(
            '<b>Stop name</b>: ' + row[2] + '<br>' +
            '<b>Stop ID</b>: ' + row[3]
        );
        return marker;
    }};
    """
    fp.FastMarkerCluster(
        data=stops[
            ["stop_lat", "stop_lon", "stop_name", "stop_id"]
        ].values.tolist(),
        callback=callback,
        disableClusteringAtZoom=14,
    ).add_to(my_map)

    # Fit map to stop bounds
    bounds = [
        (stops.stop_lat.min(), stops.stop_lon.min()),
        (stops.stop_lat.max(), stops.stop_lon.max()),
    ]
    my_map.fit_bounds(bounds, padding=[1, 1])

    return my_map


def build_stops_timetable(feed, stop_ids, dates):
    """
    Return a DataFrame containing the timetable for the given stop IDs
    and dates (YYYYMMDD date strings)

    Return a DataFrame whose columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times`` plus ``'date'``, and the stop IDs are restricted to the given
    stop IDs.
    The result is sorted by date then departure time.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    t = pd.merge(feed.trips, feed.stop_times)
    t = t[t["stop_id"].isin(stop_ids)].copy()
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to stops active on date
        ids = a.loc[a[date] == 1, "trip_id"]
        f = t[t["trip_id"].isin(ids)].copy()
        f["date"] = date
        frames.append(f)

    f = pd.concat(frames)
    return f.sort_values(["date", "departure_time"])
