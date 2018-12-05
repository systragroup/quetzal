# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry, ops
import os

from syspy.io.pandasshp import pandasshp
from syspy.syspy_utils import data_visualization as visual
from syspy.spatial.geometries import line_list_to_polyline

from syspy.paths import gis_resources

# gis_resources = r'G:\PLT\L-Lignes Produit\0. Dev\python\modules\pandasshp\gis_resources/'

epsg4326 = gis_resources + r'projections/epsg4326.prj'
epsg4326_string = pandasshp.read_prj(gis_resources + 'projections/epsg4326.prj')
# todo use path to local ressources

bordered_line_style = gis_resources + 'styles/bordered_line.qml'
line_style = gis_resources + 'styles/line.qml'
point_style = gis_resources + 'styles/point.qml'
polygon_style = gis_resources + 'styles/polygon.qml'

RdYlGn = [
    '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
    '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837']



def shares_to_shp(
    aggregated_shares,
    zones,
    gis_path,
    epsg=None,
    projection_file=None
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    as_attraction = aggregated_shares.copy()
    as_attraction['color'] = visual.color_series(
        as_attraction['pt_share_attraction'],
        colors=RdYlGn,
        method='linear',
        reversed_colors=False)
    as_emission = aggregated_shares.copy()
    as_emission['color'] = visual.color_series(
        as_emission['pt_share_emission'],
        colors=RdYlGn,
        method='linear',
        reversed_colors=False)

    pandasshp.write_shp(
        gis_path + 'share_attraction.shp',
        as_attraction,
        epsg=epsg,
        projection_file=projection_file,
        style_file=polygon_style)

    pandasshp.write_shp(
        gis_path + 'share_emission.shp',
        as_emission,
        epsg=epsg,
        projection_file=projection_file,
        style_file=polygon_style)


def assigned_nodes_to_shp(
    nodes,
    gis_path,
    load_column='load',
    max_value=None,
    outer_average_width=10,
    method='surface',
    projection_file=None,
    epsg=None,
    color=True,
    nodes_name='loaded_nodes.shp'
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    nodes = nodes.copy()

    # Add width
    nodes['width'] = visual.width_series(
        nodes[load_column],
        max_value=max_value,
        outer_average_width=outer_average_width,
        method=method,
    )

    # Add color
    if color:
        nodes['color'] = visual.color_series(
            nodes[load_column].fillna(0),
            max_value=max_value,
        )

    # Export
    pandasshp.write_shp(
        gis_path + nodes_name,
        nodes,
        style_file=point_style,
        epsg=epsg,
        projection_file=projection_file
    )


def assigned_links_nodes_to_shp(
    links,
    nodes,
    gis_path,
    epsg=None,
    projection_file=None,
    link_name='loaded_links.shp',
    node_name='loaded_nodes.shp'
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    links = links.copy()
    nodes = nodes.copy()

    try:
        links['color'] = links['line_color']
    except KeyError:
        links['color'] = links['line_color'] = 'gray'

    links['width'] = 3
    nodes['color'] = 'gray'
    nodes['width'] = 3

    pandasshp.write_shp(
        gis_path + 'nodes_linedraft.shp',
        nodes,
        epsg=epsg,
        projection_file=projection_file,
        style_file=point_style)

    pandasshp.write_shp(
        gis_path + 'links_linedraft.shp',
        links,
        epsg=epsg,
        projection_file=projection_file,
        style_file=bordered_line_style
    )

    links['width'] = visual.width_series(links['load'], outer_average_width=20)
    links['color'] = links['line_color']
    pandasshp.write_shp(
        gis_path + 'loaded_links_linedraft_color.shp',
        links,
        epsg=epsg,
        style_file=line_style,
        projection_file=projection_file
    )

    links['color'] = visual.color_series(links['load'].fillna(0))
    pandasshp.write_shp(
        gis_path + link_name, links,
        style_file=line_style, epsg=epsg, projection_file=projection_file)

    nodes['width'] = visual.width_series(nodes['load'], outer_average_width=10)
    nodes['color'] = visual.color_series(nodes['load'].fillna(0))
    pandasshp.write_shp(
        gis_path + node_name,
        nodes,
        style_file=point_style,
        epsg=epsg,
        projection_file=projection_file
    )


def loaded_links_to_shp(
    loaded_links,
    gis_path,
    load_column='load',
    max_value=None,
    outer_average_width=20,
    color=True,
    epsg=None,
    projection_file=None,
    name='loaded_links.shp',
    create_legend=False,
    **legend_kwargs
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    loaded_links = loaded_links.copy()

    # Add colors
    if color:
        loaded_links['color'] = visual.color_series(
            loaded_links[load_column].fillna(0),
            max_value=max_value
        )

    # Add width
    loaded_links['width'] = visual.width_series(
        loaded_links[load_column].fillna(0),
        max_value=max_value,
        outer_average_width=outer_average_width
    )

    # Export
    pandasshp.write_shp(
        gis_path + name,
        loaded_links,
        style_file=line_style,
        epsg=epsg,
        projection_file=projection_file
    )

    # Legend
    if create_legend==True:
        create_load_legend(
            **legend_kwargs,
            legend_file_path=gis_path + name.split('.shp')[0] + '_legend.shp',
            outer_average_width=outer_average_width,
            epsg=epsg,
            projection_file=projection_file,
            max_value=max_value,
            legend_type='LineString'
            )



def ntlegs_centroids_to_shp(
    ntlegs,
    centroids,
    gis_path,
    epsg=None,
    projection_file=None,
    weighted=True
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    ntlegs['width'] = 1
    ntlegs['color'] = 'gray'
    pandasshp.write_shp(
        gis_path + 'ntlegs.shp',
        ntlegs,
        epsg=epsg,
        projection_file=projection_file,
        style_file=line_style)

    try:
        centroids['color'] = visual.color_series(
            centroids['emission_rate'].fillna(0))
    except:
        pass
    pandasshp.write_shp(
        gis_path + 'centroids.shp',
        centroids,
        epsg=epsg,
        projection_file=projection_file
    )

    if weighted:
        centroids['width'] = visual.width_series(
            np.sqrt(centroids['weight']), outer_average_width=15)
        pandasshp.write_shp(
            gis_path + 'weighted_centroids.shp',
            centroids,
            style_file=point_style,
            epsg=epsg,
            projection_file=projection_file
        )

def build_lines(links, line_columns='all', group_id='trip_id', sum_columns=[], mean_columns=[]):

    if line_columns is 'all':
        # all the columns that are shared by all the links of the group are listed
        variability = links.groupby(group_id).agg(lambda s: s.nunique()).max()
        line_columns = list(variability.loc[variability == 1].index)

    lines = links.groupby(group_id)[line_columns].first()

    if len(sum_columns):
        lines[sum_columns] = links.groupby(group_id)[sum_columns].sum()
    if len(mean_columns):
        lines[mean_columns] = links.groupby(group_id)[mean_columns].mean()

    links = links.loc[~links['geometry'].apply(lambda g: g.is_empty)]
    lines['geometry'] = links.groupby(group_id)['geometry'].agg(
        ops.linemerge)

    lines = lines.dropna(subset =['geometry'])
    iloc = lines['geometry'].apply(lambda g: g.geom_type) == 'MultiLineString'
    loc = iloc, 'geometry'

    lines.loc[loc] = lines.loc[loc].apply(line_list_to_polyline)
    lines = lines.loc[~lines['geometry'].apply(lambda g: g.is_empty)]
    
    return lines.reset_index()

def lines_to_shp(
    links,
    gis_path,
    group_id='trip_id',
    color_id=None,
    colors=['green'],
    width=1,
    epsg=None,
    projection_file=None
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    if False:
        to_concat = []
        for line in set(links[group_id]):
            sample = links[links[group_id] == line]
            color = sample[color_id].iloc[0] if color_id else None
            l = list(sample['geometry'])

            to_concat.append((line, color, ops.linemerge(l)))
        

        df = pd.DataFrame(to_concat, columns=[group_id,'color' ,'geometry'])


    df = build_lines(links, [color_id] if color_id else None)
    
    if not color_id:
        df['color'] = pd.Series(colors * 10000)
    
    df['width'] = width
    pandasshp.write_shp(
        gis_path + 'lines.shp',
        df,
        epsg=epsg,
        projection_file=projection_file,
        style_file=bordered_line_style
    )


def save_boardings_by_length_by_line(lines, path):
    plt.clf()
    fig = (lines['boardings'] / lines['length'] * 1000).plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Boardings by km of line')
    plt.xlabel('line')
    plt.ylabel('passenger by km of line')
    fig.savefig(path + 'boardings_by_length_by_line.png', bbox_inches='tight')


def save_boardings_by_line(lines, path):
    plt.clf()
    fig = lines['boardings'].plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Boardings')
    plt.xlabel('line')
    plt.ylabel('boardings')
    fig.savefig(path + 'boardings_by_line.png', bbox_inches='tight')


def save_passenger_km_by_line(lines, path):
    plt.clf()
    fig = lines['passenger_km'].plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Passenger * km by line')
    plt.xlabel('line')
    plt.ylabel('passenger * km')
    fig.savefig(path + 'passenger_km_by_line.png', bbox_inches='tight')


def save_line_length(lines, path):
    plt.clf()
    fig = (lines['length'] / 1000).plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Line length')
    plt.xlabel('line')
    plt.ylabel('length (km)')
    fig.savefig(path + 'line_length.png', bbox_inches='tight')


def save_line_travel_time(lines, path):
    plt.clf()
    fig = (lines['time'] / 60).plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Travel time')
    plt.xlabel('line')
    plt.ylabel('travel time (min)')
    fig.savefig(path + 'line_travel_time.png', bbox_inches='tight')


def save_line_transfer(lines, path):
    plt.clf()
    fig = (lines['transfer'] - 1).plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Average number of transfers')
    plt.xlabel('line')
    plt.ylabel('number of transfers')
    fig.savefig(path + 'line_transfer.png', bbox_inches='tight')


def save_line_headway(lines, path):
    plt.clf()
    fig = (lines['headway']).plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Headway')
    plt.xlabel('line')
    plt.ylabel('headway (seconds)')
    fig.savefig(path + 'line_headway.png', bbox_inches='tight')


def save_line_load(lines, path):
    plt.clf()
    fig = lines['max_load'].plot(
        kind='bar', colors=list(lines['color'])).get_figure()
    plt.title('Passengers on the most loaded link during the modeling period')
    plt.xlabel('line')
    plt.ylabel('passengers')
    fig.savefig(path + 'line_max_load.png', bbox_inches='tight')


def save_line_plots(lines, path):

    mean_lines = lines.groupby(['name']).mean()
    mean_lines['color'] = lines.groupby('name')['color'].first()
    sum_lines = lines.groupby(['name']).mean()
    sum_lines['color'] = lines.groupby('name')['color'].first()

    save_boardings_by_length_by_line(lines, path)
    save_passenger_km_by_line(lines, path)
    save_boardings_by_line(lines, path)

    save_line_length(mean_lines, path)
    save_line_travel_time(mean_lines, path)
    save_line_transfer(mean_lines, path)
    save_line_headway(mean_lines, path)

    save_line_load(sum_lines, path)


def aggregation_summary(micro_zones, macro_zones, cluster_series, size=1):

    macro_centroids_dict = macro_zones['geometry'].apply(
        lambda g: g.centroid).to_dict()
    micro_centroids_dict = micro_zones['geometry'].apply(
        lambda g: g.centroid).to_dict()

    centroid_link_list = [
        geometry.linestring.LineString(
            [micro_centroids_dict[key], macro_centroids_dict[value]]
        ) for key, value in cluster_series.to_dict().items()
    ]

    centroid_links = pd.DataFrame(
        pd.Series(centroid_link_list), columns=['geometry'])
    macro_centroids = pd.DataFrame(
        pd.Series(macro_centroids_dict), columns=['geometry'])
    micro_centroids = pd.DataFrame(
        pd.Series(micro_centroids_dict), columns=['geometry'])

    centroid_links['width'], centroid_links['color'] = size, 'gray'
    macro_centroids['width'], macro_centroids['color'] = 1.5*size, 'gray'
    micro_centroids['width'], micro_centroids['color'] = size, 'gray'

    return micro_centroids, macro_centroids, centroid_links


def three_level_aggregation_summary_to_shp(
    micro_zones,
    meso_zones,
    macro_zones,
    micro_meso_cluster_series,
    meso_macro_cluster_series,
    gis_path,
    epsg=None,
    projection_file=None,
):
    if epsg is None and projection_file is None:
        print('No projection defined --> considered as EPSG:4326')
        projection_file = epsg4326

    micro_centroids, meso_centroids, centroid_links = aggregation_summary(
        micro_zones, meso_zones, micro_meso_cluster_series)

    pandasshp.write_shp(
        gis_path + 'micro_centroids.shp',
        micro_centroids,
        epsg=epsg,
        projection_file=projection_file,
        style_file=point_style)
    pandasshp.write_shp(
        gis_path + 'meso_centroids.shp',
        meso_centroids,
        epsg=epsg,
        projection_file=projection_file,
        style_file=point_style)
    pandasshp.write_shp(
        gis_path + 'centroid_links.shp',
        centroid_links,
        epsg=epsg,
        projection_file=projection_file,
        style_file=line_style)

    meso_centroids, macro_centroids, macro_centroid_links = aggregation_summary(
        meso_zones, macro_zones, meso_macro_cluster_series, size=1.5)
    pandasshp.write_shp(
        gis_path + 'meso_centroids.shp',
        meso_centroids,
        epsg=epsg,
        projection_file=projection_file,
        style_file=point_style)
    pandasshp.write_shp(
        gis_path + 'macro_centroids.shp',
        macro_centroids,
        epsg=epsg,
        projection_file=projection_file,
        style_file=point_style)
    pandasshp.write_shp(
        gis_path + 'macro_centroid_links.shp',
        macro_centroid_links,
        epsg=epsg,
        projection_file=projection_file,
        style_file=line_style)

    pandasshp.write_shp(
        gis_path + 'micro_zones.shp',
        micro_zones,
        epsg=epsg,
        projection_file=projection_file,
        style_file=polygon_style)
    pandasshp.write_shp(
        gis_path + 'meso_zones.shp',
        meso_zones,
        epsg=epsg,
        projection_file=projection_file,
        style_file=polygon_style)
    pandasshp.write_shp(
        gis_path + 'macro_zones.shp',
        macro_zones,
        epsg=epsg,
        projection_file=projection_file,
        style_file=polygon_style)


def create_load_legend(legend_file_path, coordinates, legend_type, values, max_value=None,  style_file=line_style,
                       categories=None, outer_average_width=20, colors=True, delta=[1000, 1500], method='linear',
                       epsg=None, projection_file=None):
    """
    Create a georeferenced legend in shp format for loaded links or points.
    Args:
        legend_file_path: name and path of the legend shp file
        style_file: path to the style file to use
        coordinates: coordinates of the georeferenced legend
        type: Point or LineString
        values: list of values to include in the legend
        max_value: max value to calibrate the legend
        categories: categories associated with the values
        outer_average_width: width
        delta: legend spacing (one value for Point legend, two values for Linestring)        
    """
    x = coordinates[0]
    y = coordinates[1]

    if legend_type == 'Point':
        delta_y = delta[0] if isinstance(delta, list) else delta
        legend_geometries = [geometry.Point([(x, y)])]
        for i in range(len(values) - 1):
            legend_geometries.append(
                geometry.Point([(x, y + (i + 1) * delta_y)]))
    elif legend_type == 'LineString':
        delta_x = delta[0]
        delta_y = delta[1]
        legend_geometries = [geometry.LineString(
            [(x, y), (x + delta_x, y)])]
        for i in range(len(values) - 1):
            legend_geometries.append(
                geometry.LineString(
                    [
                        (x + (i +1) * delta_x, y ),
                        (x + (i + 2) * delta_x, y )
                    ]
                )
            )
    else:
        raise Exception('Not implemented')

    # Create legend df
    legend_scale = pd.DataFrame({
        'value': values,
        'category': categories if categories else values,
        'geometry': legend_geometries

    }
    )
    if colors:
        legend_scale['color'] = visual.color_series(
            legend_scale['value'], max_value=max_value)

    legend_scale['width'] = visual.width_series(
        legend_scale['value'],
        outer_average_width=outer_average_width,
        max_value=max_value,
        method=method
    )
    legend_scale['label'] = legend_scale['value']

    # Write shp legend file
    pandasshp.write_shp(
        legend_file_path,
        legend_scale,
        style_file=style_file,
        epsg=epsg,
        projection_file=projection_file
        )
