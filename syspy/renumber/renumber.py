__author__ = 'qchasserieau'
import shapely
from syspy.spatial import spatial
import pandas as pd


def _join_geometry(link_row, one, many):
    return shapely.geometry.LineString(
        [one['geometry'].loc[link_row['ix_one']],many['geometry'].loc[link_row['ix_many']]])


def add_geometry_coordinates(df, columns=['x_geometry','y_geometry']):
    df[columns[0]] = df['geometry'].apply(lambda g: g.coords[0][0])
    df[columns[1]] = df['geometry'].apply(lambda g: g.coords[0][1])
    return df


def renumber(
    zones,
    volume,
    n_clusters=10,
    cluster_column=None,
    volume_columns=['volume']
):
    clusters, cluster_series = spatial.zone_clusters(
        zones,
        n_clusters=n_clusters,
        cluster_column=cluster_column
    )
    grouped = renumber_volume(
        volume,
        cluster_series,
        volume_columns=volume_columns
    )
    return clusters, grouped, cluster_series


def renumber_quetzal(
    zones,
    volume,
    od_stack,
    n_clusters=10,
    cluster_column=None,
    volume_columns=['volume'],
    volume_od_columns=['volume_pt'],
    distance_columns=['euclidean_distance']
):
    clusters, cluster_series = spatial.zone_clusters(
        zones,
        n_clusters=n_clusters,
        cluster_column=cluster_column
    )
    grouped = renumber_volume(
        volume,
        cluster_series,
        volume_columns=volume_columns
    )
    od_stack_grouped = renumber_od_stack(
        od_stack,
        cluster_series,
        volume_od_columns,
        distance_columns)
    return clusters, grouped, cluster_series, od_stack_grouped


def renumber_volume(volume, cluster_series, volume_columns):
    proto = pd.merge(volume, pd.DataFrame(cluster_series), left_on='origin', right_index=True)
    proto = pd.merge(proto, pd.DataFrame(cluster_series), left_on='destination',
                     right_index=True, suffixes=['_origin', '_destination'])
    grouped = proto.groupby(['cluster_origin', 'cluster_destination'])[volume_columns].sum()
    grouped.index.names = ['origin', 'destination']
    grouped.reset_index(inplace=True)

    return grouped


def renumber_od_stack(od_stack, cluster_series, volume_columns, distance_columns):
    proto = pd.merge(od_stack, pd.DataFrame(cluster_series), left_on='origin', right_index=True)
    proto = pd.merge(proto, pd.DataFrame(cluster_series), left_on='destination',
                     right_index=True, suffixes=['_origin', '_destination'])
    f = {v: 'sum' for v in volume_columns}
    f.update({d: 'mean' for d in distance_columns})
    grouped = proto.groupby(['cluster_origin', 'cluster_destination']).agg(f)
    grouped.index.names = ['origin', 'destination']
    grouped = pd.DataFrame(grouped)
    grouped.reset_index(inplace=True)

    return grouped
