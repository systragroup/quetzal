import itertools
import itertools
import numpy as np


def _get_consecutive_counts(fare_id_list):
    fare_id_list = list(fare_id_list)
    # if their is no fare for a route, a nan is used
    # it is necessary in order to break the sequence in the loop
    if len(fare_id_list) == 0:
        return []
    current = fare_id_list[0]
    consecutive = []
    count = 1
    for fare_id in fare_id_list[1:] + [np.nan]:
        if fare_id != current:
            consecutive.append((current, count))
            count = 1
        else:
            count += 1
        current = fare_id
    return [(fare_id, count) for fare_id, count in consecutive if fare_id is not np.nan]

def get_counts(fare_id_list,  consecutive=False):
    if consecutive :
        return _get_consecutive_counts(fare_id_list)
    fare_counts = []
    for fare_id in list(np.unique(fare_id_list)):
        if fare_id != 'nan':
            fare_counts.append((fare_id, fare_id_list.count(fare_id)))
    return fare_counts

def get_fare(count, allowed_transfers, price):
    if np.isnan(allowed_transfers):
        return price
    else:
        return max(np.ceil(count / (allowed_transfers + 1)), 1) * price


def get_price_breakdown(counts, transfers, price):
    breakdown = {}
    for fare_id, count in counts:
        add = 0
        try:
            add = get_fare(count, transfers[fare_id], price[fare_id])
            breakdown[fare_id] += add
        except KeyError:
            breakdown[fare_id] = add
    return breakdown


def get_fare_options(arod_list, route_fares_dict):
    return tuple(route_fares_dict.get(arod[1],frozenset({'nan'})) for arod in arod_list)


def get_fare_id_combinations(fare_options, irrelevant_consecutive_fares=None):
    if irrelevant_consecutive_fares == None:
        return list(itertools.product(*fare_options))
    else :
        flat = [('root',)]
        for iterable in fare_options:
            inner_flat = []
            for combination in flat :
                for i in iterable - irrelevant_consecutive_fares.get(combination[-1], set()):
                    inner_flat.append(combination + (i, ))
            flat = inner_flat
        flat = [f[1:] for f in flat]
        return flat

def get_breakdown_options(fare_options, transfers, price, irrelevant_consecutive_fares=None, consecutive=False):
    fare_id_combinations = get_fare_id_combinations(fare_options, irrelevant_consecutive_fares=irrelevant_consecutive_fares)
    counts = [
        get_counts(fare_id_list, consecutive=consecutive)
        for fare_id_list in fare_id_combinations
    ]
    return [get_price_breakdown(count, transfers, price) for count in counts]


def get_cheapest_breakdown(breakdown_options):
    cheapest = np.nan
    lowest_price = np.inf
    for breakdown in breakdown_options:
        price = sum(breakdown.values())
        if price < lowest_price:
            cheapest = breakdown
            lowest_price = price
    return cheapest

def get_tap_free_transfers(tap_free_networks):
    tap_free = {}
    for network in tap_free_networks:
        for route in network:
            try : 
                tap_free[route] += network
            except KeyError:
                tap_free[route] = network
    return tap_free

def merge_tuple(leg_tuple, route_dict, tap_free):
    if len(leg_tuple) < 2:
        return leg_tuple
    head = np.nan
    tail = np.nan
    route = np.nan
    node = np.nan
    merged_tuple = []

    for leg in leg_tuple :
        next_route = route_dict[leg[1]]
        next_node = leg[0]
        
        tap = not (next_node == node and next_route in tap_free.get(route, set()))
        if tap :
            merged_tuple.append(head + tail)
            head = leg[:2]
        
        tail = leg[2:]
        node = leg[2] # alighting_node
        route = next_route
    merged_tuple.append(head + tail)

    merged_tuple = tuple(t for t in merged_tuple[1:])
    return merged_tuple