__author__ = 'qchasserieau'

from IPython.html.widgets import FloatProgress
from IPython.display import display
from syspy.syspy_utils import syscolors


def csa(_links, start, infinity=999999, connection_time=False, origins=False):

    links = _links.copy()
    origin_set = set(links['origin']).intersection(set(origins)) if origins else set(links['origin'])
    stop_set = set(links['origin']).union(set(links['destination']))

    progress = FloatProgress(min=0, max=len(origin_set), width=975,
                             height=10, color=syscolors.rainbow_shades[1], margin=5)
    progress.value = 1
    display(progress)

    links['reachable'] = False
    csa_connections = links.to_dict(orient='records')

    earliest_arrival_time_dict = {}
    earliest_arrival_link_dict = {}
    reachable_connections_dict = {}
    reachable_trips_dict = {}

    connection_time = connection_time if connection_time else {s:0 for s in stop_set}

    print(len(origin_set))
    for origin in list(origin_set):

        progress.value += 1

        reachable_connections = {l: 0 for l in list(links['index'])}
        reachable_trips = {t: 0 for t in list(links['trip_id'])}

        earliest_arrival_time = {s: infinity for s in stop_set}
        earliest_arrival_time[origin] = start
        earliest_arrival_link = {}

        def is_reachable(label):
            r = reachable_trips[label['trip_id']] or \
                earliest_arrival_time[label['origin']] + connection_time[label['destination']] \
                    <= label['departure_time']

            return r

        def scan(label):
            reachable = is_reachable(label)
            reachable_trips[label['trip_id']], reachable_connections[label['index']] = reachable, reachable

            if reachable:
                if earliest_arrival_time[label['destination']] > label['arrival_time']:
                    earliest_arrival_time[label['destination']] = label['arrival_time']
                    earliest_arrival_link[label['destination']] = label['index']

        for connection in csa_connections:
            scan(connection)

        earliest_arrival_time_dict[origin] = earliest_arrival_time
        earliest_arrival_link_dict[origin] = earliest_arrival_link
        reachable_connections_dict[origin] = reachable_connections
        reachable_trips_dict[origin] = reachable_trips

    return{'earliest_arrival_time_dict': earliest_arrival_time_dict,
           'earliest_arrival_link_dict': earliest_arrival_link_dict,
           'reachable_connections_dict': reachable_connections_dict,
           'reachable_trips_dict': reachable_trips}
