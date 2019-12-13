from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import gtfs_kit as gk


def restrict(feed, dates=None, service_ids=None, time_range=None, trip_ids=None):

    feed = feed.copy()
    # Restrict to dates
    if dates is not None:
        feed = gk.restrict_to_dates(feed, dates)  #TODO: too slow, compute_trip_actitivy must be rewritten with .loc…
    # restrict to trip ids:
    if trip_ids is not None:
        feed = restrict_to_trips(feed, trip_ids=trip_ids)
    # restrict to services
    if service_ids is not None:
        feed = restrict_to_services(feed, service_ids=service_ids)
    # restrict to time_range
    if time_range is not None:
        feed = restrict_to_timerange(feed, time_range=time_range)

    return feed


def restrict_to_services(feed, service_ids: List[str]):
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. active on at least one of the given services
    (strings).
    Return the resulting feed, which will have empty non-agency tables
    if no trip is active on any of the given services.
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Get every trip that is active on at least one of the services
    feed.trips = feed.trips[feed.trips['service_id'].isin(service_ids)]
    trip_ids = feed.trips['trip_id'].unique()

    return restrict_to_trips(feed, trip_ids)


def restrict_to_timerange(feed, time_range):
    """
    Build a new feed by restricting this one to only the trips 
    that are active during this time range.
    The new feed can still possibly have trips running out of the time range,
    for instance if the old feed had trips running several times.
    Return the resulting feed, which will have empty non-agency tables
    if no trip is active on any of the given services.
    """
    # TODO: real filtering that would modify the stop_times / frequencies
    # tables based on this restriction

    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Get every trip that is active on the given time range
    frequency_trip_ids = []
    active_trips = []
    # Frequency case: trip is defined in feed.frequencies
    if feed.frequencies is not None:
        frequency_trip_ids = feed.frequencies['trip_id'].unique()
        active_trips += list(
            feed.frequencies[
                # case 1: start_time < end_time
                (
                    (feed.frequencies['start_time'] < feed.frequencies['end_time'])&
                    (feed.frequencies['start_time'] <= time_range[1])&
                    (feed.frequencies['end_time'] > time_range[0])
                ) |
                # case 2: start_time == end_time: one trip, that must start within time range
                (
                    (feed.frequencies['start_time'] == feed.frequencies['end_time'])&
                    (feed.frequencies['start_time'] <= time_range[1])&
                    (feed.frequencies['start_time'] >= time_range[0])
                )
            ]['trip_id'].unique()
        )

    # Non-frequency case: trip is not defined in feed.frequencies
    active_trips += list(
        feed.stop_times[
            (~(feed.stop_times['trip_id'].isin(frequency_trip_ids)))&
            (feed.stop_times['arrival_time'] <= time_range[1])&
            (feed.stop_times['departure_time'] >= time_range[0])
        ]['trip_id'].unique()
    )

    trip_ids = list(set(active_trips))
    
    return restrict_to_trips(feed, trip_ids)

def restrict_to_trips(feed, trip_ids):
    """
    Build a new feed by restricting this one to only the specified trips
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Slice trips
    feed.trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)]

    # Slice routes
    feed.routes = feed.routes.loc[
        lambda x: x.route_id.isin(feed.trips.route_id.unique())
    ]

    # Slice stop times
    feed.stop_times = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)]

    # Slice stops
    stop_ids = feed.stop_times.stop_id.unique()
    f = feed.stops.copy()
    cond = f.stop_id.isin(stop_ids)
    if "location_type" in f.columns:
        cond |= ~f.location_type.isin([0, np.nan])
    feed.stops = f[cond].copy()

    # Slice calendar
    service_ids = feed.trips['service_id'].unique()
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[
            lambda x: x.service_id.isin(service_ids)
        ]
        if feed.calendar.empty:
            feed.calendar = None

    # Get agency for trips
    if "agency_id" in feed.routes.columns:
        agency_ids = feed.routes.agency_id
        if len(agency_ids):
            feed.agency = feed.agency.loc[
                lambda x: x.agency_id.isin(agency_ids)
            ]

    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.loc[
            lambda x: x.service_id.isin(service_ids)
        ]

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[
            lambda x: x.trip_id.isin(trip_ids)
        ]

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes.loc[lambda x: x.shape_id.isin(shape_ids)]

    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers.loc[
            lambda x: x.from_stop_id.isin(stop_ids)
            & x.to_stop_id.isin(stop_ids)
        ]

    return feed