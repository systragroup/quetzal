# -*- coding: utf-8 -*-

"""

This modules provides tools for creating folium map object from pandas.DataFrames and save them as .png

example::

    from syspy_utils import sysfolium
    from selenium import webdriver

    rox = webdriver.Firefox()                                                       # launches the driver

    my_place = sysfolium.location(my_volume_dataframe)                              # finds the location of the map

    f = sysfolium.folium_map_from_edges(
        my_volume_dataframe,
        zoom=12,
        popup_columns=['origin', 'destination','volume_transit'],
        min_width=1,
        location=my_place)                                                          # creates the folium map object

    f.save(r'Q:/temp.html')                                                         # saves the map in a html file
    sysfolium.html_to_png(r'Q:/temp.html', r'Q:/my_plot.png', driver=rox, delay=1)  # saves the .html as .png

    rox.quit()                                                                      # kills the driver

"""


__author__ = 'qchasserieau'

import folium
import pandas as pd
import numpy as np

import branca.colormap as cm
from selenium import webdriver
import time

from syspy.syspy_utils.syscolors import rainbow_shades

def folium_map_from_edges(edges, zoom=5, popup_columns=[], min_width=1, location=None):

    center_coordinates = location if location else [
        (edges['latitude_origin'].max() + edges['latitude_origin'].min() + 2*edges['latitude_origin'].mean()) / 4,
        (edges['longitude_origin'].max() + edges['longitude_origin'].min() + 2*edges['longitude_origin'].mean()) / 4
    ]

    # construction de la carte
    fmap = folium.Map(location=center_coordinates, zoom_start=zoom,
                        tiles='OpenStreetMap', width=width, height=height)

    # ajout des arcs sur la carte
    for key, row in edges[edges['width'] > min_width].iterrows():  # <0.5 is not even visible
        point_o = [row['latitude_origin'], row['longitude_origin']]
        point_d = [row['latitude_destination'],row['longitude_destination']]
        fmap.line([point_o, point_d],
                  popup=str(row[popup_columns]),
                  line_weight=row['width'],
                  line_color=row['line_color'],
                  line_opacity=1)



    return fmap

def add_zone_centroid_to_fmap(fmap, zones, id_field):
    # ajout des nœuds sur la carte
    for key, row in zones.reset_index().iterrows():
        fmap.circle_marker(location=(row['latitude'],row['longitude']),
                               radius=20,
                               popup=str(row[id_field]),
                               line_color= rainbow_shades[1],
                               fill_color= rainbow_shades[1])
    return fmap


def edges_from_graph_and_dict(g, pos, width_dict={}, color_dict={}, outer_average_width=3):

    print('building edge dataframe')
    # construction du tableau des arcs
    edges = pd.DataFrame(g.edges(), columns=['origin', 'destination'])
    edges['latitude_origin'] = edges['origin'].apply(lambda origin: pos[origin]['latitude'])
    edges['longitude_origin'] = edges['origin'].apply(lambda origin: pos[origin]['longitude'])
    edges['latitude_destination'] = edges['destination'].apply(lambda destination : pos[destination]['latitude'])
    edges['longitude_destination'] = edges['destination'].apply(lambda destination : pos[destination]['longitude'])

    edges['width'] = edges.apply(lambda r : width_of_od(r['origin'], r['destination'], width_dict, outer_average_width), axis=1)
    edges['line_color'] = edges.apply(lambda r : color_of_od(r['origin'], r['destination'], color_dict), axis=1)

    return edges


def folium_map_from_graph(g, pos, width_dict={}, color_dict={}, outer_average_width=3, zoom=5):
    return folium_map_from_edges(edges_from_graph_and_dict(g, pos, width_dict, color_dict, outer_average_width), zoom)


folium_width, folium_height = 1300, 1300
width, height = folium_width, folium_height

def width_of_od(origin, destination, width_dict, outer_average_width):
    inner_max_width = np.max(list(width_dict.values()))
    return width_dict[(origin, destination)]/inner_max_width*outer_average_width

def color_of_od(origin, destination, color_dict):
    linear = cm.LinearColormap([rainbow_shades[1],rainbow_shades[0]], vmin=0, vmax=1)

    inner_max_color = np.max(list(color_dict.values()))
    color = min(color_dict[(origin, destination)], inner_max_color)
    return linear(color/inner_max_color)


def html_to_png(html, png, driver=None, delay=5):
    """

    :param html: the path to the html to save
    :param png:  the path to the png file
    :param driver: the driver that will fetch the html and save it. If no driver is provided,
        one will automatically be instantiated, used, and taken down afterwards.
        If you use this function many times, you should instantiate an external driver and pass it as an argument.
    :type driver: selenium.webdriver.firefox.webdriver.WebDriver
    :param delay: the time for your browser to wait before taking a screenshot of the map (rendered html)
    :returns: None

    if you need a single map::

        f = sysfolium.folium_map_from_edges(
            my_volume_dataframe,
            zoom=12,
            popup_columns=['origin', 'destination','volume_transit'],
            min_width=1)                                                          # creates the folium map object

        f.save(r'Q:/temp.html')                                                   # saves the map in a html file
        sysfolium.html_to_png(r'Q:/temp.html', r'Q:/my_plot.png', delay=1)        # saves the .html as .png

    if you need several maps of the very same location to be saved quickly::

        # rox if a tiny furry driver that you do not want to create and kill at every map you save
        rox = webdriver.Firefox()

        # finds the location of the map to plot, passing it as an argument to sysfolium.folium_map_from_edges ensures that all the maps focus on the very same place.
        my_place = sysfolium.location(my_volume_dataframe)

        f = sysfolium.folium_map_from_edges(
            my_volume_dataframe,
            zoom=12,
            popup_columns=['origin', 'destination','volume_transit'],
            min_width=1,
            location=my_place)                                                          # creates the folium map object

        f.save(r'Q:/temp.html')                                                         # saves the map in a html file
        sysfolium.html_to_png(r'Q:/temp.html', r'Q:/my_plot.png', driver=rox, delay=1)  # saves the .html as .png

        rox.quit()                                                                      # kills the driver

    """


    url='file://{mapfile}'.format(mapfile=html)

    if driver:

        # if a driver is provided
        driver.get(url)
        time.sleep(delay)
        driver.save_screenshot(png)

    else:

        driver = webdriver.Firefox()
        driver.get(url)

        # Give the map tiles some time to load
        time.sleep(delay)

        driver.save_screenshot(png)
        driver.quit()

def location(edges):
        return [
        (edges['latitude_origin'].max() + edges['latitude_origin'].min() + 2*edges['latitude_origin'].mean()) / 4,
        (edges['longitude_origin'].max() + edges['longitude_origin'].min() + 2*edges['longitude_origin'].mean()) / 4
    ]


"""Takes an IPython.core.display.HTML instance and save it to the path both as a png and a html file,  """
