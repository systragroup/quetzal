import collections

import pandas as pd
from tqdm import tqdm
from math import floor

from .filtering import restrict_to_timerange

tqdm.pandas()


def convert_to_frequencies(feed, time_range, pattern_column='pattern_id', drop_unused=True):
    """
    Given:
        - a clean feed defined on one day / one service
        - a time range
    Returns a new feed, converted to frequencies
    """
    # Make sure only one service is defined
    n_services = len(feed.trips.service_id.unique())
    error_message = """
        Your GTFS still contains {} services. You must select one date
        or one service before converting to frequencies.
    """.format(n_services)
    assert n_services == 1, error_message

    # Make a restricted copy
    feed = restrict_to_timerange(feed, time_range, drop_unused=drop_unused)

    # Compute pattern headway
    pattern_headways = compute_pattern_headways(feed, time_range, pattern_column)

    # One trip per pattern
    feed = pattern_to_trip(feed, pattern_column=pattern_column)

    # Replace frequencies
    frequencies = pattern_headways.reset_index().rename(columns={pattern_column: 'trip_id'})
    frequencies['start_time'] = time_range[0]
    frequencies['end_time'] = time_range[1]
    feed.frequencies = frequencies
    # Clean
    feed = feed.restrict_to_trips(feed.trips.trip_id, drop_unused=drop_unused)
    # feed = feed.clean()
    return feed


def pattern_to_trip(feed, pattern_column='pattern_id'):
    # replace trip_id in trips by a single pattern_id
    # replace trip_id in stop_times with the new trip_id (pattern_id)

    # One trip per pattern
    feed.trips = feed.trips.groupby(pattern_column, as_index=False).first()
    to_replace_first = feed.trips.set_index('trip_id')[pattern_column].to_dict()
    feed.trips['trip_id'] = feed.trips['trip_id'].replace(to_replace_first)

    # Replace stop times
    feed.stop_times['trip_id'] = feed.stop_times['trip_id'].replace(to_replace_first)
    return feed


def compute_pattern_headways(feed, time_range, pattern_column='pattern_id'):
    """
    Args:
        feed
        time_range: ['08:00:00', '10:00:00'] time range in HH:MM.SS format
    returns:
        DataFrame(pattern_id, headway_secs)
    """
    if feed.frequencies is not None:
        temp_frequencies = feed.frequencies.copy()
    else:
        temp_frequencies = pd.DataFrame(columns=['trip_id', 'start_time', 'end_time', 'headway_secs'])

    temp_stop_times = feed.stop_times.copy()
    # remove all trips that are already defined in frequencies
    temp_stop_times = temp_stop_times[~temp_stop_times['trip_id'].isin(temp_frequencies.trip_id)]

    temp_stop_times = (
        temp_stop_times.sort_values(['trip_id', 'stop_sequence'])
        .groupby('trip_id')
        .first()[['arrival_time', 'departure_time']]
        .rename(columns={'arrival_time': 'start_time', 'departure_time': 'end_time'})
        .reset_index()
    )
    temp_stop_times['headway_secs'] = 0
    temp_frequencies = pd.concat([temp_frequencies, temp_stop_times])

    time_range_sec = [hhmmss_to_seconds_since_midnight(x) for x in time_range]

    freq_conv = GTFS_frequencies_utils(temp_frequencies, feed.trips.copy())
    pattern_headways = feed.trips.groupby(pattern_column)[['trip_id']].agg(list)
    pattern_headways['headway_secs'] = pattern_headways['trip_id'].apply(
        lambda x: freq_conv.compute_average_headway(x, time_range_sec)
    )

    return pattern_headways[['headway_secs']]


class GTFS_frequencies_utils:
    """
    Example:
    >>> frequencies = pd.read_csv(gtfs_path + 'frequencies.txt')
    >>> trips = pd.read_csv(gtfs_path + 'trips.txt')
    >>> GTFS_frequencies_utils = frequencies.GTFS_frequencies_utils(frequencies, trips)
    >>> GTFS_frequencies_utils.compute_average_headway(['trip_1', 'trip_2'], [36000, 39600], ['JOB'])
    --> return 300

    """

    def __init__(self, frequencies, trips):
        self.frequencies = frequencies
        self.frequencies['start_time_sec'] = self.frequencies['start_time'].apply(hhmmss_to_seconds_since_midnight)
        self.frequencies['end_time_sec'] = self.frequencies['end_time'].apply(hhmmss_to_seconds_since_midnight)

        self.trips = trips

    def compute_average_headway(self, trip_ids, time_range, service_ids=None):
        """
        Compute the average headway over different time ranges for a specified
        list of trip ids and one or several services.

        :param trip_ids: (list, int) list of trip_ids
        :param time_range: (list, gtfs_format) [start_time, end_time]
        :param service_ids: (int or list of int, default:None) the service(s).
            If several services are given, they are supposed to be all running
            during the same day.
        :return: (int) average_headway

        The function does the following:
        - Extract the selected trips
        - Get the list of time intervals contained in the time_range
        - Compute the average headway with respect to time for this interval.
        Note that the given time_ranges intervals are inclusive.
        """
        if service_ids:
            if isinstance(service_ids, int):
                service_ids = [service_ids]
        elif service_ids is None:
            service_ids = list(self.trips['service_id'].unique())

        # Get headways aggregate for the whole period, discretize with list of times
        aggregated_headways = self.aggregate_headways_with_time(trip_ids, time_range, service_ids)
        # print(aggregated_headways)
        return average_headway_over_time(aggregated_headways, time_range[0], time_range[1])

    def aggregate_headways_with_time(self, trip_ids, time_range, service_ids=None):
        """
        Compute the total headway over a given time range with specified times, for a specified
        service and list of trips.

        :param trip_ids:(list - int) the list of trip_id whose headway is computed.
        :param time_range: (list, gtfs) time range
        :param service_ids:(int or list of int) the id(s) of the service(s) for
            which we want to compute the headway. If several services are given,
            they are supposed to be all running during the same day.
        :return: a dict with the start_time of each interval and its associated
            average headway.

        The function does the following:
        - For the specified trips and services, find all timetables which intersect
        the time range
        - Separate this time range into intervals according to the timetables that
        were found.
        - For each interval, compute the average headway.
        """
        if not trip_ids or not service_ids:
            return {time_range[0]: 0}

        if service_ids:
            if isinstance(service_ids, int):
                service_ids = [service_ids]
        elif service_ids is None:
            service_ids = list(self.trips['service_id'].unique())

        # Select the intersecting timetables
        intersecting_timetables = self.frequencies[
            (self.frequencies['trip_id'].isin(trip_ids))
            & (self.frequencies['start_time_sec'] < time_range[1])
            & (self.frequencies['end_time_sec'] > time_range[0])
            & (self.frequencies['headway_secs'] > 0)
        ]
        # print('intersecting_timetables', intersecting_timetables)

        intersecting_null_timetables = self.frequencies[
            (self.frequencies['trip_id'].isin(trip_ids))
            & (self.frequencies['start_time_sec'] < time_range[1])
            & (self.frequencies['end_time_sec'] >= time_range[0])
            & (self.frequencies['headway_secs'] == 0)
        ]
        # print('intersecting_null_timetables', intersecting_null_timetables)

        # Time range separation
        time_list = (
            list(intersecting_timetables['start_time_sec'].values)
            + list(intersecting_timetables['end_time_sec'].values)
            + list(intersecting_null_timetables['start_time_sec'].values)
            + list(intersecting_null_timetables['end_time_sec'].values)
        )
        time_list = list(set(time_list))
        time_list = [a for a in time_list if a > time_range[0] and a < time_range[1]]
        intervals = sorted([time_range[0]] + time_list + [time_range[1]])

        def append_headway_if_included(row, array, interval):
            if row['start_time_sec'] < interval[1] and row['end_time_sec'] > interval[0]:
                array.append(row['headway_secs'])

        def append_null_headway_if_included(row, array, interval):
            if row['start_time_sec'] < interval[1] and row['end_time_sec'] >= interval[0]:
                array.append(interval[1] - interval[0])

        result = {}
        # Average headway computation
        for i in range(len(intervals) - 1):
            headways_to_include = []
            # Taking into account not null or 0 headways
            intersecting_timetables.apply(
                lambda x: append_headway_if_included(x, headways_to_include, [intervals[i], intervals[i + 1]]), 1
            )
            # Taking into account NULL or 0 headways
            # In this case, it means one vehicle starts a trip at start_time
            intersecting_null_timetables.apply(
                lambda x: append_null_headway_if_included(x, headways_to_include, [intervals[i], intervals[i + 1]]), 1
            )
            result[intervals[i]] = compute_avg_headway(headways_to_include)
        return result


def hhmmss_to_seconds_since_midnight(time_int):
    """
    Convert  HH:MM:SS into seconds since midnight.
    For example "01:02:03" returns 3723. The leading zero of the hours may be
    omitted. HH may be more than 23 if the time is on the following day.
    :param time_int: HH:MM:SS string. HH may be more than 23 if the time is on the following day.
    :return: int number of seconds since midnight
    """
    time_int = int(''.join(time_int.split(':')))
    hour = time_int // 10000
    minute = (time_int - hour * 10000) // 100
    second = time_int % 100
    return hour * 3600 + minute * 60 + second


def seconds_since_midnight_to_hhmmss(time_int):
    """
    Convert  seconds since midnight into HH:MM:SS.
    For example 3723 returns "01:02:03". Numbers higher than 86400 (24:00:00) will return
    HH over 24 to represent the following day.
    :param time_int: int.
    :return: HH:MM:SS string
    """
    time = time_int / 60
    return '{:02d}:{:02d}:{:02d}'.format(floor(time / 60), int(time % 60), round(time % 1 * 60))


def compute_avg_headway(headways):
    """
    Compute the average headways from the sum of all trips with the
    specified headways.
    :param headways: (list of int) list of headways
    """
    if not headways:
        return 0
    frequency = sum([1 / headway for headway in headways])
    return 1 / frequency


def average_headway_over_time(aggregated_headways, start_time, end_time):
    """
    Given a continuous headway dicts (from aggregate_headways_with_time) and
    a start_time and end_time included in its time range, returns the
    average headway over the given duration.

    :param aggregated_headways:(dict of start times and headways
        {start time: headway}
        Example: {0: 0, 21600: 1800.0, 36000: 0, 64800: 1800.0, 79200: 0}
    :param start_time: (int) start_time to compute avg headway, in gtfs format.
        It must be greater than the min time in aggregated_headways
    :param end_time: (int) end_time to compute avg headway, in gtfs format.
        If it is greater than the max time in aggregated headways, the function
        will assume the last headways is valid until the given end_time.

    :returns: (int) the average headway

    Examples :
    average_headway_over_time({21600: 1800.0, 36000: 0}, 21600, 28800)
        --> returns 1800
    average_headway_over_time({21600: 1800.0, 36000: 0}, 21600, 72000)
        --> assumes the 0 headway is valid in the range 36000-72000 and returns
            therefore ((36000-21600)*1800 + (72000-36000)*0)/(72000-21600)
    """
    # Create reduced time intervals
    new_intervals = [time for time in aggregated_headways.keys() if time > start_time and time < end_time]
    new_intervals = new_intervals + [start_time, end_time]
    new_intervals = sorted(set(new_intervals))
    # Create a reduced headway aggregate over the given time range
    reduced_aggregated_headways = {}
    for time in new_intervals:
        if time in aggregated_headways.keys():
            # if time is an already existing key we take its headway value
            reduced_aggregated_headways[time] = aggregated_headways[time]
        else:
            # Otherwise we take the headway of the interval it is in
            keys = sorted(list(aggregated_headways.keys()))
            headway_found = False
            for i in range(len(keys) - 1):
                if time > keys[i] and time < keys[i + 1]:
                    reduced_aggregated_headways[time] = aggregated_headways[keys[i]]
                    headway_found = True
                    break
            if not headway_found:
                reduced_aggregated_headways[time] = aggregated_headways[keys[-1]]

    # dict sort
    reduced_aggregated_headways = collections.OrderedDict(sorted(reduced_aggregated_headways.items()))
    # Now we compute the average value
    nb_trips = 0
    keys = list(reduced_aggregated_headways.keys())
    for i in range(len(reduced_aggregated_headways) - 1):
        if reduced_aggregated_headways[keys[i]]:
            nb_trips += (keys[i + 1] - keys[i]) / reduced_aggregated_headways[keys[i]]
    if nb_trips > 0:
        averaged_headway = (end_time - start_time) / nb_trips
    else:
        averaged_headway = 0
    return round(averaged_headway)
