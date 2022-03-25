import networkx as nx
import numpy as np
import pandas as pd
import syspy.assignment.raw as assignment_raw
from syspy.distribution import distribution
from syspy.routing.frequency import graph as frequency_graph
from syspy.skims import skims
from syspy.spatial import spatial
from tqdm import tqdm


def od_volume_from_zones(zones, deterrence_matrix=None, power=2,
                         coordinates_unit='degree', intrazonal=False):
    """
    Use this function to build an Origin -> Destination demand matrix.
        * a deterrence matrix can be provided, if not:
            * The euclidean distance matrix will be calculated automatically.
            * The deterrence matrix will be calculated from the distance matrix
                (elevated to a given power)
        * A doubly constrained distribution will be performed in the end,
            it is based on zones['emission'] and zones['attraction']

    examples:
    ::
        volumes = engine.od_volume_from_zones(zones, power=2)
        volumes = engine.od_volume_from_zones(zones, friction_matrix)

    :param zones: zoning GeoDataFrame, indexed by the zones numeric
        contains a geometry column named 'geometry', productions and attractions
    :param deterrence_matrix: Default None. The deterrence matrix used to compute
        the distribution. If not provided, a deterrence matrix will automatically
        be computed from the euclidean distance elevated to a given power.
    :param power: the friction curve used in the gravity model is equal to
        the euclidean distance to the power of this exponent
    :param intrazonal: (bool) if True, computes intrazonal volumes using a
        zone characteristic distance. Default False.

    :return: volumes, a DataFrame with the following columns
        ['origin', 'destination', 'volume']
    """
    if deterrence_matrix is None:
        euclidean = skims.euclidean(zones, coordinates_unit=coordinates_unit, intrazonal=intrazonal)
        distance = euclidean.set_index(
            ['origin', 'destination'])['euclidean_distance'].unstack('destination')

        # If intrazonal volumes are not wanted, the distance is set to infinity
        if not intrazonal:
            # friction the impedance matrix
            distance.replace(0, 1e9, inplace=True)
        deterrence_matrix = np.power(distance, -power)

        deterrence_matrix.replace(float('inf'), 1e9, inplace=True)
    # gravitaire doublement contraint
    volume_array = distribution.CalcDoublyConstrained(
        zones['emission'].values,
        zones['attraction'].values,
        deterrence_matrix.values
    )

    volumes = pd.DataFrame(
        volume_array,
        index=list(zones.index),
        columns=list(zones.index)
    ).stack()

    volumes = volumes.reset_index()
    volumes.columns = ['origin', 'destination', 'volume']

    return volumes


def ntlegs_from_centroids_and_nodes(
    centroids,
    nodes,
    short_leg_speed=3,
    long_leg_speed=15,
    threshold=500,
    n_neighbors=5,
    coordinates_unit='degree'
):

    """
    From a given zoning an a given set of nodes, links the nearest nodes to each
    centroid of the zoning.

    example:
    ::
        centroids = zones.copy()
        centroids['geometry'] = centroids['geometry'].apply(lambda g: g.centroid)
        ntlegs = engine.ntlegs_from_centroids_and_nodes(
            centroids,
            nodes,
            ntleg_speed=3,
            n_neighbors=6
        )

    :param centroids: a point GeoDataFrame indexed by a numeric index;
    :param nodes: a point GeoDataFrame indexed by a numeric index;
    :param ntleg_speed: speed, as a crow flies, along the ntlegs km/h;
    :param n_neighbors: number of ntleg of each centroid;
    :param coordinates_unit: degree or meter

    :return: ntlegs, a line GeoDataFrame containing the ids of the
        the centroid and the node, the distance and the proximity rank;
    :rtype : line GeoDataFrame
    """
    ntlegs = spatial.nearest(
        centroids,
        nodes,
        geometry=True,
        n_neighbors=n_neighbors
    ).rename(
        columns={'ix_one': 'centroid', 'ix_many': 'node'}
    )[['centroid', 'node', 'rank', 'distance', 'geometry']]

    access = ntlegs.rename(columns={'centroid': 'a', 'node': 'b'})
    eggress = ntlegs.rename(columns={'centroid': 'b', 'node': 'a'})
    access['direction'] = 'access'
    eggress['direction'] = 'eggress'

    ntlegs = pd.concat([access, eggress], ignore_index=True)

    if coordinates_unit == 'degree':
        ntlegs['distance'] = skims.distance_from_geometry(ntlegs['geometry'])
    elif coordinates_unit == 'meter':
        ntlegs['distance'] = ntlegs['geometry'].apply(lambda x: x.length)
    else:
        raise('Invalid coordinates_unit.')

    ntlegs['speed_factor'] = np.power(ntlegs['distance'] / threshold, 0.5)
    ntlegs['short_leg_speed'] = short_leg_speed
    ntlegs['long_leg_speed'] = long_leg_speed
    ntlegs['speed'] = ntlegs['short_leg_speed'] * ntlegs['speed_factor']
    ntlegs['speed'] = np.minimum(ntlegs['speed'], ntlegs['long_leg_speed'])
    ntlegs['speed'] = np.maximum(ntlegs['speed'], ntlegs['short_leg_speed'])

    ntlegs['time'] = ntlegs['distance'] / ntlegs['speed'] / 1000 * 3600

    return ntlegs


def graph_links(links):
    """
    Decorates a link dataframe so it can be used as a basis for building a graph;
    It is meant to be used on a LineDraft export link DataFrame;
    :param links: LineDraft link DataFrame;
    :return: Basically the same as links but with different names and trivial columns;
    """
    links = links.copy()

    # links['arrival_time'] = 1
    links['duration'] = links['time']
    if 'cost' not in links.columns:
        links['cost'] = links['duration'] + links['headway'] / 2

    links['origin'] = links['a']
    links['destination'] = links['b']

    # if link_sequence already exists, we do not want two columns
    links.rename(columns={'sequence': 'link_sequence'}, inplace=True)
    links = links.loc[:, ~links.columns.duplicated()]

    return links


def multimodal_graph(
    links,
    ntlegs,
    pole_set,
    footpaths=None,
    boarding_cost=300,
    ntlegs_penalty=1e9,
):

    """
    This is a graph builder and a pathfinder wrapper
        * Builds a public transport frequency graph from links;
        * Adds access and eggress to the graph (accegg) using the ntlegs;
        * Search for shortest paths in the time graph;
        * Returns an Origin->Destination stack matrix with path and duration;

    example:
    ::
        links = engine.graph_links(links) # links is a LineDraft export
        path_finder_stack = engine.path_and_duration_from_links_and_ntlegs(
            links,
            ntlegs
        )

    :param links: link DataFrame, built with graph_links;
    :param ntlegs: ntlegs DataFrame built;
    :param boarding_cost: an artificial cost to add the boarding edges
        of the frequency graph (seconds);
    :param ntlegs_penalty: (default 1e9) high time penalty in seconds to ensure
        ntlegs are used only once for access and once for eggress;
    :return: Origin->Destination stack matrix with path and duration;
    """
    print(boarding_cost)
    pole_set = pole_set.intersection(set(ntlegs['a']))
    links = links.copy()
    ntlegs = ntlegs.copy()

    links['index'] = links.index  # to be consistent with frequency_graph

    nx_graph, _ = frequency_graph.graphs_from_links(
        links,
        include_edges=[],
        include_igraph=False,
        boarding_cost=boarding_cost
    )
    ntlegs.loc[ntlegs['direction'] == 'access', 'time'] += ntlegs_penalty
    nx_graph.add_weighted_edges_from(ntlegs[['a', 'b', 'time']].values.tolist())

    if footpaths is not None:
        nx_graph.add_weighted_edges_from(
            footpaths[['a', 'b', 'time']].values.tolist()
        )

    return nx_graph


def path_and_duration_from_links_and_ntlegs(
    links,
    ntlegs,
    pole_set,
    footpaths=None,
    boarding_cost=300,
    ntlegs_penalty=1e9
):

    """
    This is a graph builder and a pathfinder wrapper
        * Builds a public transport frequency graph from links;
        * Adds access and eggress to the graph (accegg) using the ntlegs;
        * Search for shortest paths in the time graph;
        * Returns an Origin->Destination stack matrix with path and duration;

    example:
    ::
        links = engine.graph_links(links) # links is a LineDraft export
        path_finder_stack = engine.path_and_duration_from_links_and_ntlegs(
            links,
            ntlegs
        )

    :param links: link DataFrame, built with graph_links;
    :param ntlegs: ntlegs DataFrame built;
    :param boarding_cost: an artificial cost to add the boarding edges
        of the frequency graph (seconds);
    :param ntlegs_penalty: (default 1e9) high time penalty in seconds to ensure
        ntlegs are used only once for access and once for eggress;
    :return: Origin->Destination stack matrix with path and duration;
    """
    pole_set = pole_set.intersection(set(ntlegs['a']))
    links = links.copy()
    ntlegs = ntlegs.copy()

    links['index'] = links.index  # to be consistent with frequency_graph

    nx_graph, _ = frequency_graph.graphs_from_links(
        links,
        include_edges=[],
        include_igraph=False,
        boarding_cost=boarding_cost
    )
    ntlegs['time'] += ntlegs_penalty
    nx_graph.add_weighted_edges_from(ntlegs[['a', 'b', 'time']].values.tolist())

    if footpaths is not None:
        nx_graph.add_weighted_edges_from(
            footpaths[['a', 'b', 'time']].values.tolist()
        )

    # return nx_graph
    allpaths = {}
    alllengths = {}
    iterator = tqdm(list(pole_set))

    for pole in iterator:
        iterator.desc = str(pole) + ' '
        olengths, opaths = nx.single_source_dijkstra(nx_graph, pole)
        opaths = {target: p for target, p in opaths.items() if target in pole_set}
        olengths = {target: p for target, p in olengths.items() if target in pole_set}
        alllengths[pole], allpaths[pole] = olengths, opaths

    duration_stack = assignment_raw.nested_dict_to_stack_matrix(
        alllengths, pole_set, name='gtime')
    # Remove access and egress ntlegs penalty
    duration_stack['gtime'] -= 2 * ntlegs_penalty
    duration_stack['gtime'] = np.clip(duration_stack['gtime'], 0, None)

    path_stack = assignment_raw.nested_dict_to_stack_matrix(
        allpaths, pole_set, name='path')

    los = pd.merge(duration_stack, path_stack, on=['origin', 'destination'])
    los['path'] = los['path'].apply(tuple)

    return los, nx_graph


def modal_split_from_volumes_and_los(
    volumes,
    los,
    time_scale=1 / 3600,
    alpha_car=2,
    beta_car=0
):
    """
    This is a modal split wrapper essentially based on duration and modal penalties.
    Based on all modes demand and levels of services, it returns the volume by mode.

    example:
        los = pd.merge(
            car_skims,
            path_finder_stack,
            on=['origin', 'destination'],
            suffixes=['_car', '_pt']
        )
        shared = engine.modal_split_from_volumes_and_los(
            volumes,
            los,
            time_scale=1/1800
            alpha_cal=2,
            beta_car=600
        )

    :param volumes: all mode, origin->destination demand matrix;
    :param los: levels of service. An od stack matrix with:
        'duration_pt', 'duration_car'
    :param time_scale: time scale of the logistic regression that compares
        'duration_pt' to alpha_car * 'duration_car' + beta_car
    :param alpha_car: multiplicative penalty on 'duration_car' for the calculation
        of 'utility_car'
    :param beta_car: additive penalty on 'duration_car' for the calculation
        of 'utility_car'
    :return:
    """
    los = los.copy()
    mu = time_scale

    def share_pt(row):
        return np.exp(-time_scale * row['duration_pt']) / (
            np.exp(-time_scale * row['duration_pt']) + np.exp(
                -time_scale * (alpha_car * row['duration_car'] + beta_car)))

    los['utility_pt'] = -los['duration_pt']
    los['utility_car'] = -los['duration_car'] * alpha_car - beta_car
    los['delta_utility'] = los['utility_car'] - los['utility_pt']
    los['share_pt'] = 1 / (
        1 + np.exp(mu * (los['delta_utility']))
    )
    los['share_car'] = 1 - los['share_pt']

    shared = pd.merge(volumes, los, on=['origin', 'destination'])
    shared['volume_pt'] = shared['volume'] * shared['share_pt']
    shared['volume_car'] = shared['volume'] * shared['share_car']
    return shared


def emission_share_pt(shared, origin):
    try:
        df = shared.loc[shared['origin'] == origin]
        return np.average(df['share_pt'], weights=df['volume'])
    except Exception:
        return np.nan


def attraction_share_pt(shared, destination):
    try:
        df = shared.loc[shared['destination'] == destination]
        return np.average(df['share_pt'], weights=df['volume'])
    except Exception:
        return np.nan


def aggregate_shares(shared, zones):

    """
    Aggregates modal shares by zone. Weights the share by the demand;
    :param shared: straight output from the modal split;
    :param zones: zoning GeoDataFrame
    :return: aggregated_shares, the zoning augmented with the modal shares as columns;
    """
    aggregated_shares = pd.DataFrame(
        [
            (emission_share_pt(shared, i), attraction_share_pt(shared, i))
            for i in zones.index
        ],
        columns=['pt_share_emission', 'pt_share_attraction']
    )

    aggregated_shares['geometry'] = zones['geometry']

    try:
        aggregated_shares[
            ['emission', 'attraction']] = zones[
            ['emission', 'attraction']]
    except Exception:
        pass

    return aggregated_shares


def loaded_links_and_nodes(
    links,
    nodes,
    volumes=None,
    path_finder_stack=None,
    volume_column='volume',
    path_column='path',
    pivot_column=None,
    path_pivot_column=None,
    boardings=False,
    alightings=False,
    transfers=False,
    link_checkpoints=set(),
    node_checkpoints=set(),
    checkpoints_how='all',
    **kwargs
):
    """
    The assignment function. The last step of modelling the demand.
        * the shortest path are known in the public transport graph;
        * the demand by mode is known;
        * we want to assign the demand to the edges and nodes of the shortest path for every OD

    example:
    ::
        links, nodes = engine.loaded_links_and_nodes(
        links, nodes, shared, path_finder_stack, 'volume_pt')

    :param links: links DataFrame to be loaded
    :param nodes: nodes DataFrame to be loaded
    :param volumes: demand stack matrix by mode
    :param path_finder_stack: paths stack matrix (pathfinder output)
    :param od_stack: path / demand merged stack matrix
    :param volume_column: name of the column from 'volumes' to use in
        order to load the links and nodes
    :param pivot_column: name of the column from od_stack to multiply with
        volumes. Default None
    :return: (links, nodes) a tuple containing the loaded DataFrames
    """
    # TODO: factoriser tout le boarding / alighting / transfer etc
    checkpoints = node_checkpoints.union(link_checkpoints)

    links = links.copy()
    nodes = nodes.copy()
    volumes = volumes.copy()

    if pivot_column:
        volumes[volume_column] = volumes[volume_column] * volumes[pivot_column]

    analysis_col = []
    if boardings:
        analysis_col += ['boardings', 'boarding_links']
    if alightings:
        analysis_col += ['alightings', 'alighting_links']
    if transfers:
        analysis_col += ['transfers']

    # use it in order to add probability to paths
    path_finder_stack['pivot'] = 1
    if path_pivot_column:
        path_finder_stack['pivot'] = path_finder_stack[path_pivot_column]

    # we don't want name collision
    merged = pd.merge(
        volumes[['origin', 'destination', volume_column]],
        path_finder_stack[['origin', 'destination', 'pivot', path_column, ] + analysis_col],
        on=['origin', 'destination'])

    merged[volume_column] = merged[volume_column] * merged['pivot']
    volume_array = merged[volume_column].values
    paths = merged[path_column].values

    def assigned_node_links(paths):
        assigned = assignment_raw.assign(
            volume_array,
            paths,
            checkpoints=checkpoints,
            checkpoints_how=checkpoints_how
        )

        try:
            link_index = [s for s in list(assigned.index) if s in links.index]
            assigned_links = assigned.loc[link_index]['volume']
        except KeyError:  # None of [...] are in the [index]
            assigned_links = None

        node_index = [s for s in list(assigned.index) if s in nodes.index]
        assigned_nodes = assigned.loc[node_index]['volume']

        assigned_nodes.index.name = 'id'
        return assigned_nodes, assigned_links

    assigned_nodes, assigned_links = assigned_node_links(paths)

    links[volume_column] = assigned_links.fillna(0)
    nodes[volume_column] = assigned_nodes.fillna(0)

    if boardings:
        paths_boardings = merged['boardings'].values
        paths_boarding_links = merged['boarding_links'].values
        boarding_nodes, _ = assigned_node_links(paths_boardings)
        _, boarding_links = assigned_node_links(paths_boarding_links)
        nodes['boardings'], links['boardings'] = boarding_nodes, boarding_links

    if alightings:
        paths_alightings = merged['alightings'].values
        paths_alighting_links = merged['alighting_links'].values
        alighting_nodes, _alighting_links = assigned_node_links(paths_alightings)
        _, alighting_links = assigned_node_links(paths_alighting_links)
        nodes['alightings'], links['alightings'] = alighting_nodes, alighting_links

    if transfers:
        paths_transfers = merged['transfers'].values
        transfer_nodes, transfer_links = assigned_node_links(paths_transfers)
        nodes['transfers'], links['transfers'] = transfer_nodes, transfer_links

    return links, nodes
