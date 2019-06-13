# -*- coding: utf-8 -*-

"""
This module provides tools for geometry processing.
"""

__author__ = 'qchasserieau'

from tqdm import tqdm
import shapely
import json
from math import pi
import numpy as np
import pandas as pd
import geopandas as gpd

def reversed_polyline(polyline):
    coords = list(polyline.coords)
    return shapely.geometry.LineString(reversed(coords))



def linestring_geometry(row):
    return shapely.geometry.LineString(
        [
            [row['x_origin'], row['y_origin']],
            [row['x_destination'], row['y_destination']]
        ]
    )


def point_geometry(row):
    return shapely.geometry.Point(row['stop_lon'], row['stop_lat'])


def linestring_from_indexed_point_geometries(indexed, points):
    try:
        geometries = indexed.loc[points]
        coordinates = []
        for geometry in list(geometries):
            coordinates += list(geometry.coords)
        return shapely.geometry.linestring.LineString(coordinates)
    except ValueError:
        return None

def line_list_to_polyline(geometries):
    coord_sequence = []
    last = False
    for geometry in geometries:
        coords = list(geometry.coords)
        coord_sequence += coords[1:] if last == coords[0] else coords
        last = coords[-1]
    try:
        return shapely.geometry.linestring.LineString(coord_sequence)
    except ValueError:
        return None

def polyline_to_line_list(geometry, tolerance=0):
    sequence = geometry.simplify(tolerance).coords if tolerance else geometry.coords
    couples = [(sequence[i], sequence[i+1]) for i in range(len(sequence) - 1)]
    return [shapely.geometry.linestring.LineString(couple) for couple in couples]


def string_to_geometry(string_series):
    iterator = tqdm(list(string_series), 'string_to_geometry')
    return [shapely.geometry.shape(json.loads(x)) for x in iterator]


def geometry_to_string(geometry_series):
        iterator = tqdm(list(geometry_series), 'geometry_to_string')
        return [json.dumps(shapely.geometry.mapping(x)) for x in iterator]


def coexist(
    line_a,
    line_b,
    rate=0.25,
    buffer=1e-4,
    check_collinearity=True
):
    buffer_a = line_a.buffer(buffer)
    buffer_b = line_b.buffer(buffer)
    min_area = min(buffer_a.area, buffer_b.area)
    inter = buffer_a.intersection(buffer_b)
    intersect = (inter.area / min_area) > rate

    clause = True
    if check_collinearity:
        clause = collinear(line_a, line_b)

    return intersect * clause


def angle(geometry):
    xa = geometry.coords[0][0]
    ya = geometry.coords[0][1]
    xb = geometry.coords[-1][0]
    yb = geometry.coords[-1][1]
    
    delta_x = xb - xa
    delta_y = yb - ya
    
    if xb != xa:
        tan = delta_y / delta_x 
        a = np.arctan(tan)
    else:
        a = 0
    
    if delta_x < 0:
        a += pi
        
    return (a  + 2*pi) % (2*pi)

def b_crosses_a_to_the_left(a, b):
    ab_angle = angle(b) - angle(a) 
    ab_angle =  (ab_angle + 2*pi) % (2*pi)
    return ab_angle < pi


def delta_angle(g_a, g_b):
    delta = angle(g_a) - angle(g_b)
    return (delta + 2*pi) % (2*pi)


def collinear(g_a, g_b, tol=pi/4):
    return np.absolute(delta_angle(g_a, g_b) - pi) >= (pi - tol)


def dissociate_collinear_lines(lines, coexist_kwargs={}):
    conflicts = [
        [
            coexist(line_a, line_b, **coexist_kwargs)
            for line_a in lines
        ]
        for line_b in tqdm(lines)
    ]

    df = pd.DataFrame(conflicts)
    uniques = {i: None for i in range(len(conflicts))}
    sorted_lines = list(df.sum().sort_values(ascending=True).index)
    possibilities = {i for i in range(len(lines))}

    for line in sorted_lines:
        taken = {
            uniques[other_line]
            for other_line in sorted_lines
            if conflicts[line][other_line] and
            other_line != line
        }
        uniques[line] = min(possibilities - taken)

    return uniques


def line_rows(row, tolerance):
    """
    Splits the geometry of a row and returns the list of chunks as a series
    The initial geometry is a polyline. It is simplified then cut at its checkpoints.
    """
    line_list = polyline_to_line_list(row['geometry'], tolerance)
    df = pd.DataFrame([row]*(len(line_list))).reset_index(drop=True)
    df['geometry'] = pd.Series(line_list)
    return df


def simplify(dataframe, tolerance=False):
    """
    from a dataframe of polylines,
    returns a longer dataframe of straight lines
    """
    to_concat = []
    for name, row in dataframe.iterrows():
        to_concat.append(line_rows(row, tolerance))
    return pd.concat(to_concat)


def cut_ab_at_c(geometry, intersection):
    """
    Geometry is a line. intersection is a point.
    returns two lines : origin->intersection and intersection->destination

    """

    coords = list(geometry.coords)
    a = coords[0]
    b = coords[-1]
    c = list(intersection.coords)[0]

    if c in {a, b}:
        return [geometry]
    else:
        return shapely.geometry.LineString([a, c]), shapely.geometry.LineString([c, b])


def add_centroid_to_polyline(polyline, polygon):
    """
    polyline is actualy two points geometry. Returns a three points geometry
    if the line intersects the polygon. The centroid of the polygon is added 
    to the line (in the midle)
    """
    lines = polyline_to_line_list(polyline)
    to_concatenate = []
    centroid = polygon.centroid
    for line in lines:
        to_concatenate += cut_ab_at_c(line, centroid) if polygon.intersects(line) else [line]

    chained = line_list_to_polyline(to_concatenate)

    return chained


def add_centroids_to_polyline(geometry, intersections, buffer=1e-9):
    """
    Recursive:
    geometry is a line. Every point in itersections is added to it recursively.
    In the end, a polyline is returned. All the points that were in intersections can 
    be found in the coordinates of the polyline.
    """

    if not len(intersections):
        return [geometry]

    coords = list(geometry.coords)
    remaining_intersections = intersections - set(geometry.coords)

    coord_intersections = set(intersections).intersection(coords)
    sequence_dict = {coords[i]: i for i in range(len(coords))}
    cuts = sorted([0] + [sequence_dict[coord] for coord in coord_intersections] + [len(coords)-1])
    coord_lists = [coords[cuts[i]: cuts[i+1] + 1] for i in range(len(cuts)-1)]
    polylines = [shapely.geometry.LineString(coord_list) for coord_list in coord_lists if len(coord_list) > 1]

    if len(remaining_intersections) == 0:
        return polylines

    else:
        polygons = [shapely.geometry.point.Point(i).buffer(buffer) for i in remaining_intersections]
        centroids = [polygon.centroid for polygon in polygons]

        centroid_coords = {list(centroid.coords)[0] for centroid in centroids if len(centroid.coords)}

        while len(polygons):
            polygon = polygons.pop()

            polylines = [add_centroid_to_polyline(polyline, polygon) for polyline in polylines]

    # recursive
    return add_centroids_to_polyline(
            line_list_to_polyline(polylines), 
            coord_intersections.union(centroid_coords), 
            buffer
        )


def intersects_in_between(geometry_a, geometry_b):
    """
    Returns True if :
    geometry_a and geometry_b form a T intersection or a cross intersection
    """

    # they dont even intersect, it is not an intersection
    if not geometry_a.intersects(geometry_b):
        return False

    boundaries_a = [list(geometry_a.coords)[0], list(geometry_a.coords)[-1]]
    boundaries_b = [list(geometry_b.coords)[0], list(geometry_b.coords)[-1]]

    # the two geometries share an endpoint.
    if set(boundaries_a).intersection(set(boundaries_b)):
        return False

    return True

def connected_geometries(sorted_edges):
    try:
        edges = sorted_edges.copy()
        a = list(edges.index)
        #a = [i for i in a if i in edges.index]
        b = [edges.loc[a[i], 'b'] == edges.loc[a[i+1], 'a'] for i in range(len(a) - 1)]

        s = [0] + [i+1 for i in range(len(b) - 1) if not b[i]] + [len(a)]
        slices = [(s[i], s[i+1]) for i in range(len(s) - 1)]

        geometry_list = []
        for s in slices:
            line_series = edges.loc[a].iloc[s[0]: s[1]]['geometry']
            geometry = line_list_to_polyline(list(line_series))
            geometry_list.append(geometry)
    except ValueError as e: # Can only compare identically-labeled Series objects
        pass
        return []
    return geometry_list

def geometries_with_side(
    tuple_indexed_geometry_lists, 
    width=1
    ):
    tigl = tuple_indexed_geometry_lists
    
    # explode geometry lists
    tuples = []
    for index_tuple,  geometry_list in tigl.items():
        for geometry in geometry_list:
            tuples.append([index_tuple , geometry])
            
    tuple_geometries = pd.DataFrame(tuples, columns=['index_tuple', 'geometry'])

    # explode line tuples
    tuples = []
    for index_tuple in tigl.keys():
        for item in index_tuple:
            tuples.append([item, index_tuple])

    df = pd.DataFrame(tuples, columns=['id', 'index_tuple'])
    df = pd.merge(tuple_geometries, df, on='index_tuple').dropna()
    
    df['geometry_string'] = df['geometry'].astype(str)
    l = df.copy()
    groupby_columns = ['geometry_string']
    l.sort_values(groupby_columns + ['id'], inplace=True)

    l['index'] = 0
    sides = []
    for n in list(l.groupby(groupby_columns)['index'].count()):
        sides += list(range(n))

    l['side'] = sides
    l['width'] = width
    l['offset'] =  l['width'] *( l['side'] + 0.5)

    return gpd.GeoDataFrame(l[['geometry', 'width', 'side', 'offset', 'id']])