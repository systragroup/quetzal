def group_services(feed, distinct_by_pattern=False):
    service_groups = get_groupable_services(feed, distinct_by_pattern)
    to_replace = {}
    for master, children in service_groups.items():
        for child in children:
            to_replace.update({child: master})
    # Replace in feed
    feed.trips['service_id'] = feed.trips['service_id'].replace(to_replace)
    if feed.calendar is not None:
        feed.calendar['service_id'] = feed.calendar['service_id'].replace(to_replace)
        feed.calendar.drop_duplicates(inplace=True)
    if feed.calendar_dates is not None:
        feed.calendar_dates['service_id'] = feed.calendar_dates['service_id'].replace(to_replace)
        feed.calendar_dates.drop_duplicates(inplace=True)


def get_groupable_services(feed, distinct_by_pattern=False):
    """Groups the services in a feed in order to minimize the number of services

    Args:
        feed_id (int): a feed id
        distinct_by_pattern (boolean): if True, the function groups services
                                       only if they are used by only one
                                       pattern
    """
    services = None
    services_groups = {}
    if distinct_by_pattern:
        services_trips = feed.trips.groupby('service_id').agg({'trip_id': list})['trip_id'].to_dict()
        # Services().get_trips(feed_id=feed_id, as_dict=True)
        patterns_trips = feed.trips.groupby('pattern_id').agg({'trip_id': list})['trip_id'].to_dict()
        # Patterns().get_trips(feed_id=feed_id, as_dict=True)

        patterns_services = {}
        services_patterns = {}
        for key, values in patterns_trips.items():
            services = []
            for s, t in services_trips.items():
                if len(set(values) & set(t)) > 0:
                    services.append(s)
                    services_patterns.setdefault(s, set()).add(key)
            patterns_services[key] = services
        services = set(services_trips.keys())
        # we create a list of the services that are groupable
        # (they belong to one pattern only)
        groupables = set()
        for p, s_list in patterns_services.items():
            for s in s_list:
                if len(services_patterns[s]) == 1:
                    groupables.add(s)
        for p, s in patterns_services.items():
            groupable = set(s) & groupables
            if len(groupable) > 0:
                services_groups.update(group_services_from_list(feed, groupable))

    else:
        services = feed.trips.service_id.unique()
        services_groups = group_services_from_list(feed, services)
    return services_groups


def group_services_from_list(feed, serviceslist):
    """Groups identical services given a service_id list
    Args:
        feed
        serviceslist (list(int)): a list of services ids
    Returns:
        a dict of groups: {master_id: [child_id_1, child_id_2, ...]}
    """
    # grouping by calendar
    if feed.calendar is not None:
        calendar_groups = calendar_service_groups(feed.calendar)
    else:
        calendar_groups = {serviceslist[0]: serviceslist}

    # grouping by calendar_exceptions
    calendar_dates_groups = {}
    if feed.calendar_dates is not None:
        calendar_dates_groups = calendar_dates_service_groups(feed.calendar_dates)
    else:
        calendar_dates_groups = {serviceslist[0]: serviceslist}

    services_groups = intersection_of_groups_of_sets(
        calendar_groups, calendar_dates_groups)

    return services_groups


def intersection_of_groups_of_sets(dict_set_a, dict_set_b):
    """Computes the multiple instersections of two groups of sets, and keeps
    the exclusive values.
    """
    groups = {}
    included = set()
    excluded = set()
    for a_elements in dict_set_a.values():
        for b_elements in dict_set_b.values():
            intersect = list(set(a_elements) & set(b_elements))
            included = included.union(intersect)
            excluded = excluded.union(
                set(a_elements) ^ set(b_elements)) - included
            if len(intersect) > 0:
                groups[intersect[0]] = intersect
    for elements in dict_set_a.values():
        intersect = list(set(elements) & excluded)
        if len(intersect) > 1:
            groups[intersect[0]] = intersect
            included.union(intersect)
            excluded = excluded - included
    for elements in dict_set_b.values():
        intersect = list(set(elements) & excluded)
        if len(intersect) > 1:
            groups[intersect[0]] = intersect
            included.union(intersect)
            excluded = excluded - included
    for element in list(excluded):
        groups[element] = [element]
    return groups


def calendar_service_groups(calendar):
    """
    Return the dict of groupable service: {master : [children]}
    """

    calendar = calendar.copy()

    calendar['calendar'] = calendar[
        [x for x in calendar.columns if x != 'service_id']
    ].apply(lambda x: '-'.join(x.map(str)), 1)
    calendar['dumb'] = 1
    calendar_groups = calendar[['service_id', 'calendar', 'dumb']].set_index(
        ['service_id', 'calendar']
    ).unstack().fillna(-1)['dumb']
    calendar_groups = calendar_groups.reset_index().groupby(
        list(calendar_groups.columns)
    ).agg({'service_id': list})
    calendar_groups['master'] = calendar_groups['service_id'].apply(lambda x: x[0])
    calendar_groups = calendar_groups.set_index('master')['service_id'].to_dict()
    return calendar_groups


def calendar_dates_service_groups(calendar_dates):
    """
    Return the dict of groupable service: {master : [children]}
    """
    calendar_dates = calendar_dates.copy()
    calendar_dates_groups = calendar_dates.set_index(
        ['service_id', 'date']
    ).unstack().fillna(0)['exception_type']
    calendar_dates_groups = calendar_dates_groups.reset_index().groupby(
        list(calendar_dates_groups.columns)
    ).agg({'service_id': list})
    calendar_dates_groups['master'] = calendar_dates_groups['service_id'].apply(lambda x: x[0])
    calendar_dates_groups = calendar_dates_groups.set_index('master')['service_id'].to_dict()
    return calendar_dates_groups
