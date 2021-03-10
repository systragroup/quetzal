# -*- coding: utf-8 -*-

from syspy.spatial import spatial
from syspy.skims import skims
import pandas as pd
import geopandas as gpd
import shapely

def node_clustering(links, nodes, n_clusters, prefixe='', group_id=None,**kwargs):
    
    disaggregated_nodes = nodes.copy()
    if group_id is None:
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
    if coordinates_unit=='degree':  # Default behaviour, assuming lat-lon coordinates
        v_graph['length'] = skims.distance_from_geometry(v_graph['geometry'])
    elif coordinates_unit=='meter':  # metric
        v_graph['length'] = v_graph['geometry'].apply(lambda x: x.length)
    else:
        raise('Invalid coordinates_unit.')

    if max_length:
        v_graph = v_graph.loc[v_graph['length'] <= max_length]
        
    return v_graph, v_tesselation

def build_footpaths(nodes, speed=3, max_length=None, n_clusters=None, coordinates_unit='degree'):

    if n_clusters and n_clusters < len(nodes):
        centroids, links = centroid_and_links(nodes, n_clusters, coordinates_unit=coordinates_unit)
        nodes=nodes.loc[centroids]
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


def centroid_and_links(nodes, n_clusters, coordinates_unit='degree'):
    
        clusters, cluster_series = spatial.zone_clusters(
            nodes, 
            n_clusters=n_clusters, 
            geo_union_method=lambda lg: shapely.geometry.MultiPoint(list(lg)),
            geo_join_method=geo_join_method
        )

        index_name = cluster_series.index.name
        index_name = index_name if index_name else 'index'
        grouped = cluster_series.reset_index().groupby('cluster')
        first = list(grouped[index_name].first())
        node_lists = list(grouped[index_name].agg(lambda s: list(s)))

        node_geo_dict = nodes['geometry'].to_dict()

        def link_geometry(a, b):
            return shapely.geometry.LineString([node_geo_dict[a], node_geo_dict[b]])

        values = []
        for node_list in node_lists:
            for a in node_list:
                for b in node_list:
                    if a != b:
                        values.append([a, b, link_geometry(a, b)])

        links = pd.DataFrame(values, columns=['a', 'b', 'geometry'])
        if coordinates_unit=='degree':
            links['length'] = skims.distance_from_geometry(links['geometry'])
        else:
            links['length'] = gpd.GeoDataFrame(links).length

        return first, links