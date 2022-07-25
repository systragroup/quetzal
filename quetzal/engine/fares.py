import itertools
import itertools
import numpy as np


def get_consecutive_counts(fare_id_list):
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


def get_fare(count, allowed_transfers, price):
    if np.isnan(allowed_transfers):
        return price
    else:
        return max(np.ceil(count / (allowed_transfers + 1)), 1) * price


def get_price_breakdown(consecutive_counts, transfers, price):
    breakdown = {}
    for fare_id, count in consecutive_counts:
        add = 0
        try:
            add = get_fare(count, transfers[fare_id], price[fare_id])
            breakdown[fare_id] += add
        except KeyError:
            breakdown[fare_id] = add
    return breakdown


def get_fare_options(arod_list, route_fares_dict):
    return tuple(route_fares_dict[arod[1]] for arod in arod_list)


def get_fare_id_combinations(fare_options, forbidden=None):
    if forbidden == None:
        return list(itertools.product(*fare_options))
    else :
        flat = [('root',)]
        for iterable in fare_options:
            inner_flat = []
            for combination in flat :
                for i in iterable - forbidden.get(combination[-1], set()):
                    inner_flat.append(combination + (i, ))
            flat = inner_flat
        flat = [f[1:] for f in flat]
        return flat

def get_breakdown_options(fare_options, transfers, price, forbidden=None):
    fare_id_combinations = get_fare_id_combinations(fare_options, forbidden=forbidden)
    counts = [
        get_consecutive_counts(fare_id_list)
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