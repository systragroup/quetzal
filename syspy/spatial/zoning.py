import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely
from shapely.ops import cascaded_union
from syspy.spatial import polygons, spatial
from syspy.syspy_utils import neighbors, pandas_utils, syscolors
from tqdm import tqdm


def compute_coverage_layer(layer, buffer, extensive_cols=[]):
    """
    From a given GeoDataFrame layer and a shapely 2D geometry buffer, computes the coverage layer,
    i.e. the GeoDataFrame of layer's entities included in the geometry buffer.
    Inputs:
        - layer: a GeoDataFrame object
        - buffer: a shapely Polygon or MultiPolygon
        - extensives_cols: a subset of columns whose value are extensives and have to be recomputed
            for the new layer (for instance the population of the zone)
    Outputs:
        a GeoDataFrame with the same columns as the input layer, but different geometry and extensive_cols
    """
    # Create
    layer_in_buffer = layer.copy()
    layer_in_buffer['geometry_intersect'] = layer_in_buffer.intersection(buffer)

    # Explode the multipolygons in polygons
    layer_in_buffer['geometries'] = layer_in_buffer['geometry_intersect'].apply(
        lambda x: x.geoms if x.type == 'MultiPolygon' else [x]
    )
    layer_in_buffer_exploded = pandas_utils.df_explode(layer_in_buffer, 'geometries')

    # Compute intersection area
    layer_in_buffer_exploded['area_intersected'] = gpd.GeoSeries(layer_in_buffer_exploded['geometries']).area

    # Drop row with null areas
    layer_in_buffer_exploded.drop(
        layer_in_buffer_exploded[layer_in_buffer_exploded['area_intersected'] == 0].index,
        inplace=True
    )

    # Recompute extensive columns values
    for col in extensive_cols:
        layer_in_buffer_exploded[col] = layer_in_buffer_exploded.apply(
            lambda x: x[col] * x['geometries'].area / x['geometry'].area, 1
        )

    layer_in_buffer_exploded.drop(['geometry', 'geometry_intersect', 'area_intersected'], 1, inplace=True)
    layer_in_buffer_exploded.rename(columns={'geometries': 'geometry'}, inplace=True)
    layer_in_buffer_exploded = gpd.GeoDataFrame(layer_in_buffer_exploded)
    return layer_in_buffer_exploded


def merge_zonings(background, foreground, min_area_factor=0.01, min_area=None):
    back = background.copy()
    front = foreground.copy()

    stencil = shapely.geometry.MultiPolygon(
        list(front['geometry'])
    ).buffer(1e-9)

    back['geometry'] = back['geometry'].apply(lambda g: g.difference(stencil))
    back['geometry'] = polygons.biggest_polygons(list(back['geometry']))

    back['area'] = [g.area for g in back['geometry']]
    min_area = min_area if min_area else back['area'].mean() * min_area_factor

    back = back.loc[back['area'] > min_area]

    back['id'] = back.index
    front['id'] = front.index
    back['zoning'] = 'back'
    front['zoning'] = 'front'

    columns = ['zoning', 'id', 'geometry']

    concatenated = pd.concat(
        [back[columns], front[columns]]
    )

    df = concatenated

    zones = list(df['geometry'])
    clean_zones = polygons.clean_zoning(
        zones,
        buffer=1e-4,
        fill_buffer=2e-3,
        fill_gaps=False,
        unite_gaps=True
    )
    df['geometry'] = clean_zones
    return df.reset_index(drop=True)


def pool_and_geometries(pool, geometries):
    done = []

    while len(pool):
        # start another snail
        done.append(pool[0])
        current = geometries[pool[0]]
        pool = [p for p in pool if p not in done]

        for i in range(len(pool)):
            for p in pool:
                if geometries[p].intersects(current):
                    done.append(p)
                    current = geometries[p]
                    pool = [p for p in pool if p not in done]
                    break
    return done


def snail_number(zones, center, distance_to='zone'):
    if distance_to == 'zone':
        distance_series = zones['geometry'].apply(lambda g: center.distance(g))
    elif distance_to == 'centroid':
        distance_series = zones['geometry'].apply(lambda g: center.distance(g.centroid))
    distance_series.name = 'cluster_distance'
    distance_series.sort_values(inplace=True)
    geometries = zones['geometry'].to_dict()

    pool = list(distance_series.index)

    done = pool_and_geometries(pool, geometries)

    snail = pd.Series(done)
    snail.index.name = 'cluster_snail'
    snail.name = 'cluster'

    indexed = snail.reset_index().set_index('cluster')['cluster_snail']
    return indexed.loc[zones.index]  # we use zones.index to sort the result


def cluster_snail_number(zones, n_clusters=20, centre=None, buffer=10):
    """
    zones: GeoSeries
    """

    df = pd.DataFrame(zones['geometry']).copy()
    df['index'] = zones.index

    if centre is None:
        union = cascaded_union(df.geometry).buffer(buffer)
        centre = union.centroid

    # Snail clusterize
    clusters, cluster_series = spatial.zone_clusters(df, n_clusters=n_clusters)
    df['cluster'] = cluster_series
    snail = snail_number(clusters, centre)
    clusters['snail'] = snail
    df = df.merge(snail.reset_index(), on='cluster')
    df.drop('cluster', 1, inplace=True)

    # snail numbering within cluster
    to_concat = []
    for cluster in set(df['cluster_snail']):
        temp_df = df.loc[df['cluster_snail'] == cluster]
        temp_centre = cascaded_union(temp_df.geometry).centroid
        temp_snail = snail_number(temp_df, temp_centre)
        temp_df['snail'] = temp_snail
        to_concat.append(temp_df)

    concat = pd.concat(to_concat)
    concat = concat.sort_values(['cluster_snail', 'snail']).reset_index(drop=True)
    concat = concat.reset_index().rename(
        columns={
            'level_0': 'id',
            'cluster_snail': 'cluster',
            'index': 'original_index'
        }
    )
    ids = concat.set_index('original_index')['id']
    clusters = concat.set_index('original_index')['cluster']
    return ids, clusters


def greedy_color(zoning, colors=syscolors.rainbow_shades, buffer=1e-6):
    zoning = zoning.copy()
    zoning['geometry'] = zoning['geometry'].apply(lambda g: g.buffer(buffer))

    # TODO change the edge construction to make it independant from neighbors
    n = neighbors.neighborhood_dataframe(zoning)
    edges = n[['origin', 'destination']].values

    g = nx.Graph()
    g.add_edges_from(edges)
    d = nx.coloring.greedy_color(
        g,
        strategy=nx.coloring.strategy_largest_first
    )

    color_list = list(colors)

    def index_to_color(index):
        return color_list[index]
    return pd.Series(d).apply(index_to_color)


########################################################################


def intersection_area(geoa, geob):
    if geoa.intersects(geob):
        intersection = geoa.intersection(geob)
        return intersection.area
    else:
        return 0


def intersection_area_matrix(x_geometries, y_geometries):
    array = []
    for g in tqdm(x_geometries, desc=str(len(y_geometries))):
        array.append(
            [
                intersection_area(y_geometry, g)
                for y_geometry in y_geometries
            ]
        )
    return array


def intersection_area_dataframe(front, back):
    front.index.name = 'front_index'
    back.index.name = 'back_index'
    ia_matrix = intersection_area_matrix(
        list(front['geometry']),
        list(back['geometry'])
    )

    df = pd.DataFrame(ia_matrix)
    df.index = front.index
    df.columns = back.index
    return df


def front_distribution(front_zone, intersection_dataframe):
    """
    share of the front zone in intersection with every back zone
    """
    df = intersection_dataframe
    intersection_series = df.loc[front_zone]
    area = intersection_series.sum()
    return intersection_series / area


def back_distribution(front_zone, intersection_dataframe):
    df = intersection_dataframe
    """
    share of of every back zone in intersection with the front zone
    """
    intersection_series = df.loc[front_zone]
    area_series = df.sum()
    return intersection_series / area_series


def share_intensive_columns(front_zone, back, intersection_dataframe, columns):
    shares = front_distribution(front_zone, intersection_dataframe)
    shared_series = back[columns].apply(lambda s: s * shares)
    return shared_series.sum()


def share_extensive_columns(front_zone, back, intersection_dataframe, columns):
    shares = back_distribution(front_zone, intersection_dataframe)
    shared_series = back[columns].apply(lambda s: s * shares)
    return shared_series.sum()


def concatenate_back_columns_to_front(front, back, intensive, extensive):
    df = intersection_area_dataframe(front, back)
    apply_series = pd.Series(front.index, index=front.index)

    intensive_dataframe = apply_series.apply(
        lambda z: share_extensive_columns(z, back, df, intensive)
    )
    extensive_dataframe = apply_series.apply(
        lambda z: share_extensive_columns(z, back, df, extensive)
    )
    return pd.concat(
        [front, intensive_dataframe, extensive_dataframe],
        axis=1
    )


def normalize_columns(df):
    column_sums = df.sum()
    normalized = df / column_sums
    return normalized


def share_od_extensive_columns(
    od_dataframe,
    intersection_dataframe,
    extensive_columns
):
    normalized = normalize_columns(intersection_dataframe)
    # series (front, back) -> normalized_intersection
    stack = normalized.stack()

    origin_stack = stack.loc[stack > 0].copy()
    destination_stack = stack.loc[stack > 0].copy()
    origin_stack.index.names = ['front_index_origin', 'back_index_origin']
    dest_index_names = ['front_index_destination', 'back_index_destination']
    destination_stack.index.names = dest_index_names

    # dense matrix of OD shares (origin_share * destination_share)
    share_matrix = origin_stack.apply(lambda v: v * destination_stack)
    share_matrix = share_matrix.sort_index(axis=0).sort_index(axis=1)

    # we stack the two columns index
    share_stack = share_matrix.stack(dest_index_names)
    share_stack.name = 'shares'
    share_stack = share_stack.reset_index()

    pool = od_dataframe.rename(
        columns={
            'origin': 'back_index_origin',
            'destination': 'back_index_destination'
        }
    )

    # we expen the od_dataframe by mergint it on the shares
    merged = pd.merge(
        pool,
        share_stack,
        on=['back_index_origin', 'back_index_destination']
    )
    print(len(merged))

    # we reduce merged by grouping it by front indexes,
    # multiplying each row by its' share
    shared = merged.copy()
    shared[extensive_columns] = shared[extensive_columns].apply(
        lambda c: c * shared['shares'])

    grouped = shared.groupby(
        ['front_index_origin', 'front_index_destination'],
    )
    extensive_sums = grouped[extensive_columns].sum()
    extensive_sums.index.names = ['origin', 'destination']

    return extensive_sums.reset_index()
