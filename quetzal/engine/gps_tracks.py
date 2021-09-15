import networkx as nx
from scipy import interpolate
from shapely import geometry
from syspy.spatial import spatial


def get_path(points, road_links, road_nodes, buffer=50, penalty_factor=2, n_neighbors=10):
    # create copies for the recursion
    base_road_links, base_road_nodes = road_links.copy(), road_nodes.copy()

    # geographic filter
    linestring = geometry.LineString(list(points['geometry']))
    polygon = linestring.buffer(buffer).simplify(buffer)
    road_links = road_links.loc[road_links['geometry'].within(polygon)]
    road_node_set = set(road_links['a']).union(road_links['b'])
    road_nodes = road_nodes.loc[road_node_set]

    # access to origin and destination
    od = points.iloc[[1, -1]]
    od.index = ['o', 'd']
    connectors = spatial.nearest(od, road_nodes, geometry=False, n_neighbors=n_neighbors)
    connectors = connectors.loc[connectors['distance'] < buffer]
    connectors['weight'] = connectors['distance'] * penalty_factor
    connectors.index = connectors['ix_one']
    try:
        # graph
        g = nx.DiGraph()
        g.add_weighted_edges_from(road_links[['a', 'b', 'length']].values)
        # access from the first point
        g.add_weighted_edges_from(connectors.loc[['o'], ['ix_one', 'ix_many', 'weight']].values)
        # egress to the last point
        g.add_weighted_edges_from(connectors.loc[['d'], ['ix_many', 'ix_one', 'weight']].values)
        path = nx.dijkstra_path(g, 'o', 'd')

    except (nx.NetworkXNoPath, KeyError):
        path = get_path(
            points=points,
            road_links=base_road_links,
            road_nodes=base_road_nodes,
            buffer=buffer * 1.2,
            penalty_factor=penalty_factor,
        )
    return path


def get_times(points, road_links, road_nodes, smoothing_span=None, *args, **kwargs):
    path = get_path(points, road_links, road_nodes, *args, **kwargs)
    path = [i for i in path if i in road_nodes.index]
    checkpoints = road_nodes.loc[path].dropna(subset=['geometry'])
    linestring = geometry.LineString(list(checkpoints['geometry']))
    # interpolate
    points['s'] = points['geometry'].apply(lambda g: linestring.project(g))
    checkpoints['s'] = checkpoints['geometry'].apply(lambda g: linestring.project(g))

    checkpoints = checkpoints.loc[checkpoints['s'] <= points['s'].max()]
    checkpoints = checkpoints.loc[checkpoints['s'] >= points['s'].min()]

    points['t'] = (points['time'] - points['time'].iloc[0]).apply(lambda t: t.total_seconds())
    f = interpolate.interp1d(points['s'], points['t'], kind='linear')
    if smoothing_span is None:
        smooth = f
    else:
        # we smooth the interpolation function with
        # a linear interpolation by steps
        steps = list(
            range(
                int(points['s'].min()) + 1,
                int(points['s'].max()),
                smoothing_span
            )
        )
        steps = [points['s'].min()] + steps + [points['s'].max()]
        smooth = interpolate.interp1d(steps, f(steps))

    checkpoints['s'] = checkpoints['geometry'].apply(lambda g: linestring.project(g))
    checkpoints['t'] = smooth(checkpoints['s'])

    # index
    checkpoints['index'] = checkpoints.index
    b = checkpoints.iloc[1:].reset_index(drop=True)
    a = checkpoints.iloc[:-1].reset_index(drop=True)
    delta = b[['t']] - a[['t']]
    delta['dt'] = delta['t']
    delta['t'] = a['t']
    delta['ds'] = b['s'] - a['s']
    delta['s'] = a['s']
    delta['b'] = b['index']
    delta['a'] = a['index']
    delta['speed'] = delta['ds'] / delta['dt']
    return delta
