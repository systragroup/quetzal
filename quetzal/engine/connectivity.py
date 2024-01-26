import geopandas as gpd
import pandas as pd
import shapely
from syspy.skims import skims
from syspy.spatial import spatial



def node_clustering(links, nodes, n_clusters=None, prefixe='', group_id=None, **kwargs):
    disaggregated_nodes = nodes.copy()
    if group_id is None:
        assert n_clusters is not None, 'n_clusters must be defined if group_id is None'
        clusters, cluster_series = spatial.zone_clusters(
            nodes,
            n_clusters=n_clusters,
            **kwargs
        )
    else:
        clusters = gpd.GeoDataFrame(nodes).dissolve(group_id)['geometry'].apply(lambda x: x.convex_hull)
        clusters = pd.DataFrame(clusters)
        cluster_series = nodes[group_id]

    cluster_dict = cluster_series.to_dict()
    centroids = clusters.copy()
    centroids['geometry'] = centroids['geometry'].apply(lambda g: g.centroid)

    try:
        links = links.copy()

        links['disaggregated_a'] = links['a']
        links['disaggregated_b'] = links['b']

        links['a'] = links['a'].apply(lambda x: prefixe + str(cluster_dict[x]))
        links['b'] = links['b'].apply(lambda x: prefixe + str(cluster_dict[x]))
    except AttributeError:
        links = None

    clusters['count'] = cluster_series.value_counts()
    disaggregated_nodes['cluster'] = cluster_series

    parenthood = pd.merge(
        disaggregated_nodes,
        centroids,
        left_on='cluster',
        right_index=True,
        suffixes=['_node', '_centroid']
    )

    parenthood['geometry'] = parenthood.apply(parenthood_geometry, axis=1)
    centroids.index = prefixe + pd.Series(centroids.index).astype(str)

    return links, centroids,  clusters, parenthood


def parenthood_geometry(row):
    g = shapely.geometry.LineString(
        [row['geometry_node'], row['geometry_centroid']]
    )
    return g


def geo_join_method(geo):
    return geo.convex_hull.buffer(1e-4)


def voronoi_graph_and_tesselation(nodes, max_length=None, coordinates_unit='degree'):

    v_tesselation, v_graph = spatial.voronoi_diagram_dataframes(nodes['geometry'])

    # Compute length
    if coordinates_unit == 'degree':  # Default behaviour, assuming lat-lon coordinates
        v_graph['length'] = skims.distance_from_geometry(v_graph['geometry'])
    elif coordinates_unit == 'meter':  # metric
        v_graph['length'] = v_graph['geometry'].apply(lambda x: x.length)
    else:
        raise('Invalid coordinates_unit.')

    if max_length:
        v_graph = v_graph.loc[v_graph['length'] <= max_length]
    return v_graph, v_tesselation


def build_footpaths(nodes, speed=3, max_length=None, clusters_distance=None, coordinates_unit='degree'):
    if clusters_distance :
        nodes, links = agg_nodes_and_links(nodes, clusters_distance, coordinates_unit=coordinates_unit)
        
        # not a bool for the geodataframe to be serializabe
        links['voronoi'] = 0

    graph, tesselation = voronoi_graph_and_tesselation(
        nodes,
        max_length,
        coordinates_unit=coordinates_unit
    )
    footpaths = pd.concat(
        [
            graph,
            graph.rename(columns={'a': 'b', 'b': 'a'})
        ]
    )
    footpaths['voronoi'] = 1
    try:
        footpaths = footpaths.append(links)
        if max_length:
            footpaths = footpaths.loc[footpaths['length'] <= max_length]
    except NameError:
        pass

    footpaths.reset_index(drop=True, inplace=True)
    footpaths.index = 'footpath_' + pd.Series(footpaths.index).astype(str)
    footpaths['time'] = footpaths['length'] / speed / 1000 * 3600
    return footpaths


def agg_nodes_and_links(nodes, clusters_distance, coordinates_unit='degree'):
    label = spatial.agglomerative_clustering(nodes, distance_threshold = clusters_distance)
    nodes['cluster'] = label

    index_name = nodes.index.name
    index_name = index_name if index_name else 'index'

    node_lists = nodes.reset_index().groupby('cluster')[index_name].agg(list)
    node_geo_dict = nodes['geometry'].to_dict()
    first = nodes.drop_duplicates('cluster')


    def link_geometry(a, b):
        return shapely.geometry.LineString([node_geo_dict[a], node_geo_dict[b]])
    
    # create links from the agg_node (first one) and all other one (in both direction).
    values = []
    for node_list in node_lists:
        agg_node = node_list[0]
        for node in node_list[1:]:
            values.append([agg_node, node, link_geometry(agg_node, node)])
            values.append([node, agg_node, link_geometry(node, agg_node)])

    links = pd.DataFrame(values, columns=['a', 'b', 'geometry'])
    if coordinates_unit == 'degree':
        links['length'] = skims.distance_from_geometry(links['geometry'])
    else:
        links['length'] = gpd.GeoDataFrame(links).length
    return first, links


def adaptive_clustering(nodes, zones, mean_distance_threshold=None, distance_col=None):
    """
    Compute cluster_id for each node, based on given zoning and agglomerative_clustering.
    For each zone, distance_threshold is computed as follow:
    - take value of distance_col if parameter is given
    - otherwise:
        - consider twice the characteristic distance of each zone (area**0.5)
        - scale in average to mean_distance_threshold if given
    """
    # define distance_threshold
    zone_df = gpd.GeoDataFrame(zones.copy())

    if distance_col is not None:
        zone_df['distance_threshold'] = zone_df[distance_col]
    else:
        zone_df['distance_threshold'] = zone_df['geometry'].area ** 0.5
        
    if mean_distance_threshold is not None:
        zone_df['distance_threshold'] *= mean_distance_threshold / zone_df['distance_threshold'].mean()
    
    print('Mean distance threshold is {}'.format(int(zone_df['distance_threshold'].mean())))

    #  take twice max value for outer zones
    d_max = zone_df['distance_threshold'].max() * 2

    def group_clusters(g, zone_df=zone_df):
        z_id = g['zone_id'].values[0]

        if len(g) > 1:
            # a quarter of the characteristic distance of the zone
            if z_id != 'outer':
                d = zone_df.loc[z_id, 'distance_threshold']
            else:
                d = d_max

            cluster_ids = spatial.agglomerative_clustering(g, d)

        else:
            cluster_ids = [0]

        g['adaptive_cluster_id'] = ['{}_{}'.format(z_id, x) for x in cluster_ids]

        return g

    # find zone
    nodes['zone_id'] = 'outer'
    for z_id, z in zones.iterrows():
        nodes.loc[nodes.within(z.geometry), 'zone_id'] = z_id

    # perform clustering
    nodes = nodes.groupby('zone_id').apply(group_clusters)

    return nodes
