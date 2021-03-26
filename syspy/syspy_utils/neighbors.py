"""

**This Neighborhood class builds a standalone affectation model on top of a zoning and an origin-destination volume matrix**

No transport network is required.
The model runs on a Neighborhood graph that is automatically built from the zoning :

* the vertices of this graph are the centroids of the zones;
* the edges link together the centroids of the zones that touche each other;

The Dijkstra algorithm is used to find the shortest path in the neighborhood graph,
its results may be found in the following nested dictionaries :

* dijkstra_paths;
* dijkstra_path_length;

both dictionaries use the 'dict[origin][destination]' convention

* volume is the origin-destination volume matrix.
* transit is the volume affected to each edge by Dijkstra

pos is a dictionary that contains the coordinates of the centroids
"""

__author__ = 'qchasserieau'

import itertools
import json
import os
import shutil
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from IPython.display import display
from ipywidgets import FloatProgress
from sklearn.neighbors import NearestNeighbors
from syspy.io.geojson_utils import set_geojson_crs
from syspy.io.pandasshp import pandasshp
from syspy.skims import skims
from syspy.spatial import spatial
from syspy.syspy_utils import data_visualization, syscolors
from tqdm import tqdm

r_path = os.path.dirname(os.path.realpath(__file__))
gis_resources = r_path + r'/gis_resources/'

# gis_resources = r'G:\PLT\L-Lignes Produit\0. Dev\python\modules\pandasshp\gis_resources/'
line_style = gis_resources + r'styles/line.qml'
line_offset_style = gis_resources + r'styles/line_with_offset.qml'
epsg4326_string = pandasshp.read_prj(gis_resources + r'projections/epsg4326.prj')

warnings.simplefilter("ignore")


class Neighborhood:
    """
    Assigned volume on the official zoning of Monterrey (1307 zones called AGEB):
    ::

        from pandasshp import pandasshp
        from syspy_utils.neighbors import Neighborhood

        zones = pandasshp.read_shp(sig_path + 'zonificacion/zonificacion_epsg4326.shp').set_index('n')
        volume = pd.read_csv(data_path + 'volumen.csv')

        links = {(142,141),(142,148),(141,142),(148,142),(140,139),(139,136)}
        add = links.union({(link[1], link[0]) for link in links})             # list of directional links to add

        neighborhood = Neighborhood(zones, volume, volume_columns=['volume_pt', 'volume_car'], additional_links=add)
        neighborhood_few.to_shp('volume_pt', sig_path + 'affected_volume.shp', affected=True, projection_string=wgs84)


    .. figure:: ./pictures/loaded_neighborhood_ageb.png
        :width: 75%
        :align: center
        :alt: affected desire matrix of Monterrey
        :figclass: align-center

        affected desire matrix of Monterrey

    Assigned volume in Monterrey, modelled at an agregated level of 150 zones, runs much faster than the example above:
    ::
        # pass n_clusters as an argument in order to aggregate the zones | additional links are then ignored
        neighborhood = Neighborhood(zones, volume, volume_columns=['volume_pt', 'volume_car'], n_clusters=150)
        neighborhood_few.to_shp('volume_pt', sig_path + 'macro_affected_volume.shp', affected=True, projection_string=wgs84)


    .. figure:: ./pictures/neighborhood.png
        :width: 75%
        :align: center
        :alt: affected desire matrix of Monterrey (aggregated)
        :figclass: align-center

        affected desire matrix of Monterrey (aggregated)


    Pairwise OD volume in Monterrey, represented at an aggregated level of 25 zones :
    ::
        # pass od_geometry as an argument in order to build every OD geometry (required for raw desire matrix)
        neighborhood = Neighborhood(zones, volume, volume_columns=['volume_pt', 'volume_car'], n_clusters=25, od_geometry=True)
        neighborhood_few.to_shp('volume_pt', sig_path + 'macro_volume.shp', affected=False, projection_string=wgs84)

    .. figure:: ./pictures/macro_desire_ageb.png
        :width: 75%
        :align: center
        :alt: raw desire matrix of Monterrey (aggregated in 25 clusters of zones)
        :figclass: align-center

        raw desire matrix of Monterrey (aggregated in 25 clusters of zones)
    """
    def __init__(
        self,
        zones_shp,
        volume,
        volume_columns,
        additional_links=frozenset({}),
        drop_internal=False,
        od_geometry=False,
        build_sparse=False,
        n_clusters=False,
        compute_centroids=True,
        display_progress=True,
        buffer=0.001
    ):
        progress = FloatProgress(
            min=0, max=8, width=975, height=10,
            color=syscolors.rainbow_shades[1], margin=5
        )
        progress.value = 0
        if display_progress:
            display(progress)

        self.z = n_clusters if n_clusters else len(zones_shp) + 1

        if self.z > 1000 and not n_clusters:
            _input = '%i zones, that is a lot! the assignment process may crash if z > 1000 \n' % self.z
            _input += 'enter a number –z– then press enter if you want to run the assignment on –z– clusters of zones\n'
            _input += 'OR press enter to continue with %i zones: ' % self.z
            rep = input(_input)
            n_clusters = int(rep) if bool(rep) else False
            self.z = n_clusters

        if self.z > 250 and od_geometry:
            _input = '%i zones and od_geometry=True : %i OD pairs ~ %i min(s)\n' % (self.z, self.z**2, self.z**2 / 100000)
            _input += 'do you want to build every OD geometry (y/n): '
            rep = input(_input)
            od_geometry = rep == 'y'

        # Create a dense volume dataframe and add the input volumes
        vol = skims.euclidean(zones_shp)[['origin', 'destination']]
        vol = pd.merge(vol, volume[['origin', 'destination'] + volume_columns], on=['origin', 'destination'], how='left')
        vol[volume_columns].fillna(1e-9, inplace=True)
        volume = vol
        if n_clusters:
            processed_zones, processed_volume, self.cluster_series = renumber(zones_shp, volume, n_clusters, volume_columns)
        else:
            processed_zones, processed_volume, self.cluster_series = zones_shp, volume, False

        progress.value += 1
        self.od_geometry = od_geometry
        self.build_sparse = build_sparse
        if compute_centroids:
            self.zones = add_centroid(processed_zones.copy())
        else:
            self.zones = processed_zones
        self.z = len(self.zones) + 1
        self.zones['geometry'] = self.zones['geometry'].apply(lambda g: g.buffer(buffer))

        progress.value += 1
        self.edges = neighborhood_dataframe(self.zones, additional_links)  #: the edges link together the centroids of the zones that touche each other
        self.pos = self.zones[['latitude', 'longitude']]  #: the {centroid: [latitude, longitude]} dictionary
        iterate = [self.pos.index] * 2
        self.od = pd.DataFrame(index=pd.MultiIndex.from_product(iterate, names=['origin', 'destination'])).reset_index()  #: the od column matrix
        self.od = pd.merge(self.od, self.pos, left_on='origin', right_index=True)
        self.od = pd.merge(self.od, self.pos, left_on='destination', right_index=True,
                           suffixes=['_origin', '_destination'])
        progress.value += 1

        if od_geometry:
            print('building edge geometries')
            self.od['geometry'] = self.od[['origin', 'destination']].apply(
                lambda r: shapely.geometry.LineString(
                    [self.zones.loc[r['origin'], 'centroid_geometry'],
                     self.zones.loc[r['destination'], 'centroid_geometry']]
                ), axis=1)
        else:
            self.od = pd.merge(self.od, self.edges, on=['origin', 'destination'], how='left')
        progress.value += 1

        self.graph = nx.Graph()  # networkx.Graph
        self.edge_list = list(self.edges.set_index(['origin', 'destination']).index)
        self.graph.add_weighted_edges_from(self.edges[['origin', 'destination', 'distance']].values)
        progress.value += 1

        # nx2 syntax
        self.dijkstra_paths = dict(nx.all_pairs_dijkstra_path(self.graph))
        self.dijkstra_paths_length = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        progress.value += 1

        self._link_index = pd.DataFrame(index=self.edge_list).reset_index().reset_index().set_index('index')
        self._link_index.columns = ['index']
        self._link_index.index.names = ['link']
        self._link_index_dict = dict(self._link_index['index'])
        self.volume_columns = volume_columns
        progress.value += 1

        self.update_volume(processed_volume, drop_internal)
        progress.value += 1

    def update_volume(self, volume, volume_columns=None, drop_internal=True):
        self.volume = volume.copy()
        if volume_columns:
            self.volume_columns = volume_columns
        if drop_internal:
            self.volume.loc[self.volume['origin'] == self.volume['destination'], self.volume_columns] = 0

        try:
            self.volume['link'] = self.links
            self.volume['link_list'] = self.link_lists
        except Exception:
            self.volume['link'] = self.volume.apply(lambda r: (r['origin'], r['destination']), axis=1)
            self.volume['link_list'] = self.volume.apply(lambda r: link_list_from_path(
                self.try_path_from_od(r['origin'], r['destination'])), axis=1)
            self.links = self.volume['link']
            self.link_lists = self.volume['link_list']

        column_indices = [self.columns(link) for link in list(self.link_lists)]
        row_indices = [[i] * len(column_indices[i]) for i in range(len(self.link_lists))]
        volume_list = {column: list(volume[column]) for column in self.volume_columns}

        def deep_list(column):
            return [[volume_list[column][i]] * len(column_indices[i]) for i in range(len(self.link_lists))]

        flat_row_indices = list(itertools.chain.from_iterable(row_indices))
        flat_column_indices = list(itertools.chain.from_iterable(column_indices))

        volumes = {column: deep_list(column) for column in self.volume_columns}
        flat_volumes = {column: list(itertools.chain.from_iterable(volumes[column])) for column in self.volume_columns}

        del row_indices
        del column_indices

        df_dict = flat_volumes
        df_dict['od'] = flat_row_indices
        df_dict['link_index'] = flat_column_indices

        def grouped(df_dict, dict_range, volume_columns):
            start = dict_range[0]
            end = dict_range[1]
            df_dict_chunk = {key: value[start:end] for key, value in df_dict.items()}
            return pd.DataFrame(df_dict_chunk).groupby(['link_index'], as_index=False)[volume_columns].sum()

        if self.build_sparse:
            sparse = pd.DataFrame(df_dict)
            sparse['origin'] = np.floor((self.z + sparse['od']) / self.z)
            sparse['destination'] = sparse['od'] + 1 % self.z
            sparse = pd.merge(sparse, pd.DataFrame(self._link_index.index), left_on='link_index', right_index=True)

            to_merge = sparse.groupby('link')[self.volume_columns].sum()
        else:
            flat_length = len(flat_volumes[self.volume_columns[0]])
            chunk_length = flat_length // 100
            g = [(i * chunk_length, min((i + 1) * chunk_length, flat_length)) for i in range(101)]
            grouped_df_list = [grouped(df_dict, dict_range, self.volume_columns) for dict_range in g]
            concatenated = pd.concat(grouped_df_list).groupby('link_index', as_index=False).sum()
            concatenated = pd.merge(concatenated, pd.DataFrame(self._link_index.index), left_on='link_index', right_index=True)
            to_merge = concatenated.set_index('link')[self.volume_columns]

        to_merge.columns = map(lambda s: s + '_transit', to_merge.columns)

        self.volume = pd.merge(self.volume, to_merge, left_on='link', right_index=True, how='left').fillna(0)
        self.volume = pd.merge(self.volume, self.od, on=['origin', 'destination'])

    def try_path_from_od(self, origin, destination):
        try:
            return self.dijkstra_paths[origin][destination]
        except Exception:
            return []

    def columns(self, links):
        return [self._link_index_dict[k] for k in links]

    def export(
        self,
        volume_column,
        file=None,
        affected=False,
        outer_average_width=15,
        max_value=None,
        projection_string=None,
        style_file=line_style,
        epsg=None,
        color=None
    ):
        if projection_string is None and epsg is None:
            print('No projection defined --> considered as EPSG:4326')
            projection_string = epsg4326_string
        if affected:
            volume_column += '_transit'
        else:
            if not self.od_geometry:
                print('can only map affected volumes with arg : od_geometry=False')
        to_shape = self.volume.sort_values(by=volume_column).copy()
        to_shape = to_shape[to_shape[volume_column] > 0]
        to_shape['label'] = to_shape[volume_column]
        if color is not None:
            to_shape['color'] = color
        else:
            to_shape['color'] = data_visualization.color_series(to_shape[volume_column], max_value=max_value)
        to_shape['width'] = data_visualization.width_series(
            to_shape[volume_column],
            outer_average_width=outer_average_width,
            max_value=max_value)
        if not file:
            return to_shape
        else:
            extension = file.split('.')[-1]
            if extension == 'shp':
                pandasshp.write_shp(file, to_shape, projection_string=projection_string, style_file=style_file, epsg=epsg)
            elif extension == 'geojson':
                gpd.GeoDataFrame(to_shape).to_file(file, driver='GeoJSON')
                if epsg:
                    with open(file, 'r') as infile:
                        data = json.load(infile)
                        infile.close()
                    with open(file, 'w') as outfile:
                        data['crs'] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::{}".format(epsg)}}
                        json.dump(data, outfile)


def link_vector_from_link_list(link_list, index_dict):
    values = [0] * len(index_dict)
    indexes = [index_dict[k] for k in link_list]
    for i in indexes:
        values[i] = 1
    return values


def neighborhood_dataframe(zones, additional_links=frozenset({})):
    geometries = dict(zones['geometry'])
    neighbors = []

    for id_origin, geometry_origin in tqdm(
        geometries.items(),
        'neighborhood_dataframe'
    ):
        for id_destination, geometry_destination in geometries.items():
            if is_neighbor(geometry_origin, geometry_destination) or (id_origin, id_destination) in additional_links:
                try:
                    origin_centroid = shapely.geometry.Point(
                        [zones.loc[id_origin]['longitude'], zones.loc[id_origin]['latitude']]
                    )
                    destination_centroid = shapely.geometry.Point(
                        [zones.loc[id_destination]['longitude'], zones.loc[id_destination]['latitude']]
                    )
                except Exception as e:
                    print(str(e), ' --> computing centroid')
                    origin_centroid = geometry_origin.centroid
                    destination_centroid = geometry_destination.centroid
                link_geometry = shapely.geometry.LineString([origin_centroid, destination_centroid])
                neighbors.append({
                    'origin': id_origin,
                    'destination': id_destination,
                    'geometry': link_geometry
                })

    df_edges = pd.DataFrame(neighbors)
    df_edges['distance'] = df_edges['geometry'].apply(lambda g: g.length)
    return df_edges


def add_centroid(zones_shp):
    inner_zones = zones_shp.copy()
    inner_zones['centroid_geometry'] = inner_zones['geometry'].apply(lambda g: g.centroid)
    inner_zones['centroid_coordinates'] = inner_zones['geometry'].apply(lambda g: g.centroid.coords[0])
    inner_zones['latitude'] = inner_zones['geometry'].apply(lambda g: g.centroid.y)
    inner_zones['longitude'] = inner_zones['geometry'].apply(lambda g: g.centroid.x)
    return inner_zones


def link_list_from_path(path):
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def is_neighbor(geometry_a, geometry_b, buffer=1e-9):
    # returns True if the two polygons touche, handles invalid geometries.
    try:
        return geometry_a.overlaps(geometry_b) or geometry_a.touches(geometry_b)
    except Exception:
        return is_neighbor(geometry_a.buffer(1e-9), geometry_b.buffer(1e-9))


def nearest(one, many, geometry=False, n_neighbors=1):

    df_many = add_geometry_coordinates(many.copy(), columns=['x_geometry', 'y_geometry'])
    df_one = add_geometry_coordinates(one.copy(), columns=['x_geometry', 'y_geometry'])

    x = df_many[['x_geometry', 'y_geometry']].values
    y = df_one[['x_geometry', 'y_geometry']].values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(y)

    index_one = pd.DataFrame(df_one.index.values, columns=['ix_one'])
    index_many = pd.DataFrame(df_many.index.values, columns=['ix_many'])

    to_concat = []
    for i in range(n_neighbors):
        links = pd.merge(index_one, pd.DataFrame(
            indices[:, i], columns=['index_nn']), left_index=True, right_index=True
        )
        links = pd.merge(links, index_many, left_on='index_nn', right_index=True)
        links = pd.merge(
            links,
            pd.DataFrame(distances[:, i], columns=['distance']),
            left_index=True,
            right_index=True
        )
        links['rank'] = i
        to_concat.append(links)

    links = pd.concat(to_concat)

    if geometry:
        links['geometry'] = links.apply(lambda r: _join_geometry(r, one, many), axis=1)

    return links


def nearest_deprecated(one, many, geometry=False):
    print('deprecated')

    df_many = add_geometry_coordinates(many.copy(), columns=['x_geometry', 'y_geometry'])
    df_one = add_geometry_coordinates(one.copy(), columns=['x_geometry', 'y_geometry'])

    x = df_many[['x_geometry', 'y_geometry']].values
    y = df_one[['x_geometry', 'y_geometry']].values

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(y)

    index_one = pd.DataFrame(df_one.index.values, columns=['ix_one'])
    index_many = pd.DataFrame(df_many.index.values, columns=['ix_many'])

    links = pd.merge(index_one, pd.DataFrame(
        indices, columns=['index_nn']), left_index=True, right_index=True
    )
    links = pd.merge(links, index_many, left_on='index_nn', right_index=True)
    links = pd.merge(links, pd.DataFrame(
        distances, columns=['distance']), left_index=True, right_index=True
    )
    if geometry:
        links['geometry'] = links.apply(lambda r: _join_geometry(r, one, many), axis=1)
    return links


def _join_geometry(link_row, one, many):
    return shapely.geometry.LineString(
        [one['geometry'].loc[link_row['ix_one']], many['geometry'].loc[link_row['ix_many']]])


def add_geometry_coordinates(df, columns=['x_geometry', 'y_geometry']):
    df[columns[0]] = df['geometry'].apply(lambda g: g.coords[0][0])
    df[columns[1]] = df['geometry'].apply(lambda g: g.coords[0][1])
    return df


def renumber(zones, volume, n_clusters=10, volume_columns=['volume'], cluster_column=None):
    clusters, cluster_series = spatial.zone_clusters(zones, n_clusters, cluster_column)
    grouped = renumber_volume(volume, cluster_series, volume_columns=volume_columns)
    return clusters, grouped, cluster_series


def renumber_volume(volume, cluster_series, volume_columns):
    proto = pd.merge(volume, pd.DataFrame(cluster_series), left_on='origin', right_index=True)
    proto = pd.merge(proto, pd.DataFrame(cluster_series), left_on='destination',
                     right_index=True, suffixes=['_origin', '_destination'])
    grouped = proto.groupby(['cluster_origin', 'cluster_destination'])[volume_columns].sum()
    grouped.index.names = ['origin', 'destination']
    grouped.reset_index(inplace=True)
    return grouped



def volumes_classified_neighborhood(
    zones, volume_matrix, folder, by, volume_col='vol',
    colors_df=None, max_value=None, style_file=line_offset_style
):
    """
    This function applies the neighboorhood algorithm several times, in order to keep track of the
    information of the volume category for all loaded links.
    For each volume category:
        - the neighboorhood algorithm assigns these volumes on the graph
        - The links are given the color of the volume category and exported as shpfile
    """
    # Check number of macro zones
    if len(volume_matrix[by].unique()) > 12:
        print("{} categories, that's is going to be difficult to plot. Please give a color palette to try.")
        return(None)

    if colors_df is None:
        # Color from Colorbrewer
        colors = [
            '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
            '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
        ]
        colors_df = pd.Series({a: b for a, b in zip(list(volume_matrix[by].unique()), colors)})

    # Compute assignment for each macro zone
    affected_all = []
    for i_cat, color in colors_df.items():
        print(i_cat)
        vol_matrix_i = volume_matrix.loc[volume_matrix[by]==i_cat]
        neighborhood_i = Neighborhood(zones, vol_matrix_i, volume_columns=[volume_col])
        affected_all.append(neighborhood_i.volume)

    # Create df with all volumes
    loaded_links = affected_all[0][['link', 'geometry']].dropna()
    
    i = 0
    for df in affected_all:
        i_cat = colors_df.index[i]
        loaded_links = loaded_links.merge(
            df[['link', volume_col + '_transit']].rename(
                columns={volume_col + '_transit': volume_col + '_transit_{}'.format(i_cat)}
                ),
            on='link'
        )
        i += 1

    # Compute width dataframe, for each macro zone of origin
    width_df = loaded_links.set_index('link')

    # Compute offset dataframe, for each macro zone of origin
    offset_df = width_df.drop('geometry', 1).cumsum(1) - 0.5 * width_df.drop('geometry', 1)

#     Compute width and offset for a nice plotting, and export.
    for i_cat, color in tqdm(colors_df.items()):
        col = volume_col + '_transit_{}'.format(i_cat)
        offset = offset_df[[col]].rename(columns={col: 'offset_vol'})
        width = width_df[[col, 'geometry']].rename(columns={col: 'width_vol'})
        df = width.merge(offset, left_index=True, right_index=True)
        a = get_shp_with_offset(
            df, 'width_vol', outer_average_width=15,
            max_value=max_value, color=color, offset_col='offset_vol'
        )
        to_export = gpd.GeoDataFrame(a).reset_index()
        to_export['a'] = to_export['link'].apply(lambda x: x[0])
        to_export['b'] = to_export['link'].apply(lambda x: x[1])
        to_export.drop('link', 1, inplace=True)
        
        if len(to_export):
            filename = folder + '/desire_lines_{}.geojson'.format(i_cat)
            to_export.to_file(filename, driver='GeoJSON')
            if zones.crs:
                epsg = zones.crs.to_epsg(min_confidence=20)
                if epsg is not None:
                    set_geojson_crs(filename, "urn:ogc:def:crs:EPSG::{}".format(epsg))
    
            if style_file:
                shutil.copyfile(style_file, folder + '/desire_lines_{}.qml'.format(i_cat))            


def zones_classified_neighborhood(
    zones, volumes, folder, macro_col, by='origin', volume_col='vol', **kwargs):
    """
    This function applies the neighboorhood algorithm several times, in order to keep track of the
    information of the macro zone of origin/destination, for all loaded links.
    For each macro zone:
        - the volumes having their origin/destination within this macrozone are identified.
        - the neighboorhood algorithm assigns these volumes on the graph
        - The links are given the color of the macro zone and exported as shpfile
    """
    # Check number of macro zones
    if len(zones[macro_col].unique()) > 12:
        print("{} macro zones, that's is going to be difficult to plot. Please give a color palette to try.")
        return(None)
    
    # sort volumes
    volume_matrix = volumes[['origin', 'destination', volume_col]].copy()
    volume_matrix['category'] = 0
    for i_macro in zones[macro_col].unique():
        loc = volume_matrix[by].isin(zones[zones[macro_col]==i_macro].index.values)
        volume_matrix.loc[loc, 'category'] = i_macro
    
    volumes_classified_neighborhood(zones, volume_matrix, folder, 'category', volume_col, **kwargs)


def get_shp_with_offset(
    df,
    volume_column,
    outer_average_width=15,
    max_value=None,
    color=None,
    offset_col=None
):
    to_shape = df.sort_values(by=volume_column).copy()
    to_shape = to_shape[to_shape[volume_column] > 0]
    to_shape['label'] = to_shape[volume_column]
    if color is not None:
        to_shape['color'] = color
    else:
        to_shape['color'] = data_visualization.color_series(to_shape[volume_column], max_value=max_value)
    to_shape['width'] = data_visualization.width_series(
        to_shape[volume_column],
        outer_average_width=outer_average_width,
        max_value=max_value)
    if offset_col is not None:
        to_shape['offset'] = data_visualization.width_series(
            to_shape[offset_col],
            outer_average_width=outer_average_width,
            max_value=max_value)
    return to_shape
