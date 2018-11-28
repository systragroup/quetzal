__author__ = 'qchasserieau'

from tqdm import tqdm
import pandas as pd


def print_if_debug(to_print, debug):
    if debug:
        print(to_print)


def build_connection_dicts(
    link_array,
    origin_column,
    destination_column,
    start=0,
    first_only=True,
    connection_time=False,
    footpath_list=frozenset(),
    max_wait=24*3600,
    preprocess=True,
    alpha_wait=1,
    alpha_in_vehicle=1,
    alpha_footpath=1,
    beta_footpath=0,
    beta_transfer=0,
    transfer_stops=None,
    debug=False
):

    """
    :param link_array: link DataFrame with at least: 'departure_time', 'arrival_time',
        origin_column and destination_column
    :param origin_column: the name of the column to use as origin for the link
    :param destination_column: the name of the column to use as destination for the link
    :param start: additional time to use
    :param first_only: if True, only the first departure for each destination is kept
    :param connection_time: dict containing minimum time
        for a transfer at each stop {stop : minimum_transfer_time}
    :param max_wait: maximum waiting time for a transfer
    :return:
    """

    _links = link_array.copy()

    if not transfer_stops:
        transfer_stops = set(_links[origin_column]).union(
            set(_links[destination_column]))

    # next_label holds the next link of the trip for each link
    _links['transfer'] = 0
    _links['cost'] = _links['duration'] * alpha_in_vehicle
    _links['from_stop'] = False
    l = _links.sort_values(
        ['trip_id', 'link_sequence']).to_dict(orient='record')
    next_link = {
        l[i]['index']: l[i+1] for i in range(len(l) - 1)
        if l[i]['trip_id'] == l[i + 1]['trip_id']
    }

    # origins, destinations and stop_set will be used as keys
    # when building the dictionaries of connections
    origins = set(_links[origin_column])
    destinations = set(_links[destination_column])
    stop_set = origins.union(destinations)

    foot_paths = {stop: [] for stop in stop_set}
    for foot_path in footpath_list:
        try:
            foot_paths[foot_path[0]].append((foot_path[1], foot_path[2]))
            foot_paths[foot_path[1]].append((foot_path[0], foot_path[2]))
        except KeyError:  # the footpaths does not link two stops of the stop_set
            pass

    _links['transfer'] = 1
    _links.drop(
        ['link_sequence', 'reachable'],
        axis=1,
        inplace=True,
        errors='ignore'
    )
    sorted_links = _links.set_index(
        [origin_column, 'departure_time'],
        drop=False
    ).sort().copy()

    indexed_links = _links.set_index('index').to_dict(orient='index')

    def arrivals(stop):
        return set(_links[_links[destination_column] == stop]['arrival_time'])

    neighbors_time_stop = {
        stop: {time: [] for time in arrivals(stop)} for stop in stop_set
    }
    connection_time = connection_time if connection_time else {s: 0 for s in stop_set}

    def cost(row, arrival_time):
        row_cost = row['duration'] * alpha_in_vehicle
        row_cost += row['transfer'] * beta_transfer
        row_cost += (row['departure_time'] - arrival_time) * alpha_wait
        return row_cost

    def direct_neighbors(
        stop,
        arrival_time,
        footpath_cost=0,
        from_stop=False,
        max_wait=max_wait
    ):
        try:
            print_if_debug(
                ('direct_neighbors', stop, arrival_time, max_wait),
                debug
            )
            assert stop in transfer_stops, 'transfer not allowed at this stop'
            connection_slice = sorted_links.loc[stop].loc[
                arrival_time + connection_time[stop]: arrival_time + max_wait
            ].copy()
            connection_slice['cost'] = cost(
                connection_slice, arrival_time) + footpath_cost
            connection_slice['from_stop'] = from_stop
            if first_only:
                connection_slice = connection_slice.groupby(
                    destination_column,
                    as_index=False
                ).first()
            return connection_slice.to_dict(orient='records')
        except (KeyError, AssertionError):
            # stop may not belong to the index of sorted_links
            # transfers may not be allowed at this stop
            print_if_debug(
                ('error in direct_neighbors', stop, arrival_time),
                debug)
            return []

    def all_neighbors(stop, time, max_wait=max_wait):
        including_foot_paths_connections = direct_neighbors(
            stop,
            time,
            footpath_cost=0,
            from_stop=False,
            max_wait=max_wait
        )
        for foot_path in foot_paths[stop]:
            footpath_cost = foot_path[1] * alpha_footpath + beta_footpath
            including_foot_paths_connections += direct_neighbors(
                stop=foot_path[0],
                arrival_time=time+foot_path[1],
                footpath_cost=footpath_cost,
                from_stop=stop,
                max_wait=max_wait
            )
        return including_foot_paths_connections

    if preprocess:
        #  when possible_connection is to be called many times with the same
        #  arguments, we have better of storing its results in a dictionary

        for _stop in tqdm(stop_set, desc='preprocessing'):
            arrival_times = [start] + list(arrivals(_stop))
            neighbors_time_stop[_stop] = {
                arrival: all_neighbors(_stop, arrival)
                for arrival in arrival_times
            }

        def all_neighbors(stop, time, max_wait=max_wait):
            try:
                return neighbors_time_stop[stop][time]
            except:
                print(stop, time, max_wait)
                return direct_neighbors(stop, time, max_wait=max_wait)

    # _ix functions and dicts use the link index
    def all_neighbors_ix(
        index,
        destination_column='destination',
        max_wait=max_wait
    ):
        link = indexed_links[index]
        possible = all_neighbors(
            link[destination_column],
            link['arrival_time'],
            max_wait=max_wait
        )

        # if the link is not the last of the trip:
        # we replace the link that lead to the destination
        # of the next link of the trip nextlink by nextlink itself
        # doing so, we may miss the opportunity
        # to board on a faster trip to our next destination...
        if index in next_link.keys():
            possible = [
                c for c in possible
                if c[destination_column] != next_link[index][destination_column]
            ]
            possible += [next_link[index]]
        return possible

    neighbors_ix = {}
    if preprocess:
        indexed = list(indexed_links.keys())

        for _index in tqdm(indexed, desc='mapping functions to dicts'):
            neighbors_ix[_index] = all_neighbors_ix(
                _index,
                destination_column=destination_column
            )

        # if we chose to preprocess the data, we call the function on all links
        # and store the values in a dict
        def all_neighbors_ix(index):
            return neighbors_ix[index]

    to_return = {
        'all_neighbors': all_neighbors,
        'all_neighbors_ix': all_neighbors_ix,
        'next_link': next_link,
        'neighbors_ix': neighbors_ix,
        'neighbors_time_stop': neighbors_time_stop,
        'indexed_links': indexed_links
    }

    return to_return


def distinct(l, lists):
    differences = [len(set(r)-set(l)) for r in lists]
    try:
        return max(differences)
    except:
        return 1


def single_source_labels(
    source,
    all_neighbors,
    all_neighbors_ix,
    source_index=None,
    infinity=999999,
    stop_set=False,
    start_from=1,
    departure_time=0,
    max_sequence=False,
    spread=1,
    absolute=1,
    unique_route_sets=True,
    origin_column='origin',
    destination_column='destination',
    debug=False,
    max_wait=24*3600
):

    """
    :param source: id of the source (it belongs to the keys of connections)
    :param connections: connection dict {origin:{arrival_time:
        [possible_connections(origin, arrival_time)}}
    :param possible_connections: function object that return a list of
        connections for a given node at a given time
    :param infinity: number to use as the initial distance
        of the nodes from the source
    :param stop_set: should contain all the ids of the nodes
    :param start_from: first label id
    :param departure_time: time from which the graph search is started
    :param max_sequence: maximum length of the path
    :param spread: at every stop, the label that arrive to a destination d
        is added to the stack if its cost is not
    bigger than spread * the minimal cost to d so far
    :param unique_route_sets: if True; labels with redundant route_sets
        are added to the stack only if they beat the
    best cost to their destination. (when spread=1, this )
    :param origin_column:
    :param destination_column:
    :return:
    """
    max_sequence = max_sequence if max_sequence else infinity

    root = {
        'label_id': start_from,
        'parent': 0,
        'sequence': 0,
        destination_column: source,
        'arrival_time': departure_time,
        origin_column: None,
        'trip_id': None,
        'visited': [source],
        'route': frozenset([]),
        'cumulative': 0,
        'cost': 0,
        'from_stop': False
    }

    if source_index:
        root['index'] = source_index

    pile = [root]

    label_id = iter(range(start_from, 1000000))
    store = []
    earliest = {stop: infinity for stop in stop_set}
    best = {stop: infinity for stop in stop_set}
    routes = {stop: frozenset({}) for stop in stop_set}

    def next_labels(label, label_id):
        print_if_debug('in', debug)

        """ retourne les labels accessibles depuis un label """

        time = label['arrival_time']
        stop = label[destination_column]
        route = label['route']
        cumulative = label['cumulative']

        # à corriger pour utiliser un spread en temps et non
        # un spread en cumulative cost
        if (time-departure_time) >= (earliest[stop]-departure_time)*spread:
            pass
            # print('tim')
            # return []

        if cumulative >= best[stop] * spread and cumulative - best[stop] > absolute:
            print_if_debug('dominated', debug)
            return []

        if cumulative < best[stop]:  # time < earliest[stop]:
            best[stop] = cumulative
            earliest[stop] = time
            routes[label[destination_column]] = routes[
                label[destination_column]].union({route})

        elif not distinct(
            route,
            routes[label[destination_column]]
        ) and unique_route_sets:
            print_if_debug('not_distinct', debug)
            return []

        label['label_id'] = label_id
        store.append(label)
        sequence = label['sequence'] + 1

        if sequence > max_sequence:  # nombre de correspondances
            print_if_debug('over max_sequence', debug)
            return []

        visited = label['visited']

        labels = []
        try:
            proto_labels = list(all_neighbors_ix(
                label['index'],
                destination_column=destination_column,
                max_wait=max_wait
            ))
        except:
            print_if_debug('not in neighbors ', debug)
            proto_labels = all_neighbors(stop, time, max_wait)

        for proto_label in proto_labels:
            # proto_label est le dictionnaire stocké dans connections,
            #  il faut ajouter des champs

            if proto_label[destination_column] not in visited:

                from_stop = proto_label['from_stop']

                to_append = {'parent': label['label_id'],
                             'sequence': sequence,
                             'visited': visited + [proto_label[destination_column]] + ([from_stop] if from_stop else []),
                             'route': frozenset(route.union({proto_label['route_id']}))}

                to_append.update(proto_label)
                to_append['cumulative'] = cumulative + to_append['cost']
                labels.append(to_append)

        print_if_debug(('number of labels:', len(labels)), debug)
        return labels

    while len(pile):
        print_if_debug(len(pile), debug)
        # on remplace le dernier élément de la pile par tous ses enfants
        pile = next_labels(pile.pop(), next(label_id)) + pile

    return store


def single_source_label_dataframe(
    source,
    source_column='source',
    **label_from_stop_kwargs
):

    df = pd.DataFrame(single_source_labels(source, **label_from_stop_kwargs))
    df[source_column] = source

    return df


def multiple_source_label_dataframe(origins, **label_from_stop_kwargs):

    to_concat = []

    for stop in tqdm(origins):
        to_concat.append(
            single_source_label_dataframe(stop, **label_from_stop_kwargs))

    return pd.concat(to_concat)
