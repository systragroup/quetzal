# -*- coding: utf-8 -*-

import shapely
import pandas as pd
from tqdm import tqdm


def remove_overlaps(geometries):
    """
    Run through the list of geometries. Each geometry is cropped by
    the union of previously tackled geometries.
    """
    crop = shapely.geometry.polygon.Polygon()
    cropped = []
    for g in tqdm(geometries, desc='remove_overlaps'):
        cropped.append(g.difference(crop))
        crop = crop.union(g)
    return cropped


def fill_gaps_with_buffer(geometries, fill_buffer=1e-6):
    polygons = []
    for i in tqdm(range(len(geometries)), desc='fill_gaps'):
        g = geometries[i].buffer(fill_buffer)
        for j in range(len(geometries)):
            if j != i:
                g = g.difference(geometries[j])

        polygons.append(g)

    return polygons


def interstitial_polygons(geometries, buffer=1e-9, hull_buffer=1):
    multi = shapely.geometry.MultiPolygon(geometries).buffer(buffer)
    convex_hull = multi.convex_hull
    convex_hull_buffer = convex_hull.buffer(hull_buffer)
    voids = convex_hull_buffer.difference(multi)
    voids = [v for v in voids if convex_hull.contains(v)]
    return voids


def border_length(geoa, geob, buffer=1e-9):

    if geoa.intersects(geob):
        intersection = geoa.buffer(buffer).intersection(geob.buffer(buffer))
        return intersection.area / buffer
    else:
        return 0


def border_length_matrix(x_geometries, y_geometries, buffer=1e-9):
    x_geometries = [g.buffer(buffer) for g in x_geometries]
    y_geometries = [g.buffer(buffer) for g in y_geometries]
    array = []
    for g in tqdm(x_geometries, desc=str(len(y_geometries))):
        array.append(
            [
                border_length(y_geometries[i], g, buffer=buffer)
                for i in range(len(y_geometries))
            ]
        )
    return array


def gap_nearest_polygon_index(gap_idex, length_matrix):
    index = length_matrix[gap_idex].sort_values().index
    polygon_index = list(index)[-1]
    return polygon_index


def unite_gaps_to_polygons(gaps, polygons, buffer=1e-4):

    geometries = [p for p in polygons]

    array = border_length_matrix(polygons, gaps,  buffer=buffer)
    df = pd.DataFrame(array)

    for gap_index in range(len(gaps)):
        polygon_index = gap_nearest_polygon_index(gap_index, df)
        geometries[polygon_index] = geometries[polygon_index].union(gaps[gap_index])

    return geometries


def buffer_if_not_polygon(g, buffer):
    if type(g) == shapely.geometry.polygon.Polygon:
        return g
    else:
        return g.buffer(buffer)


def biggest_polygon(multipolygon):
    m = 0
    r = shapely.geometry.Polygon()
    for p in list(multipolygon):
        if p.area > m:
            m = p.area
            r = p
    return r


def biggest_polygons(multipolygons):
    polygons = [
        p if type(p) == shapely.geometry.Polygon else biggest_polygon(p)
        for p in multipolygons
    ]
    return polygons


def clean_zoning(
    zones,
    coordinates='degree', # or meter
    buffer=None,
    fill_buffer=None,
    hull_buffer=None,
    mini_buffer=None,
    fill_gaps=True,
    unite_gaps=True,
    **kwargs
):
    # Default values if not given
    if coordinates=='degree':
        if buffer is None:
            buffer = 1e-4
        if fill_buffer is None:
            fill_buffer = 1e-4
        if hull_buffer is None:
            hull_buffer = 1
        if mini_buffer is None:
            mini_buffer = 1e-9
    elif coordinates=='meter':
        if buffer is None:
            buffer = 10
        if fill_buffer is None:
            fill_buffer = 10
        if hull_buffer is None:
            hull_buffer = 10000
        if mini_buffer is None:
            mini_buffer = 1
    # clean geom
    polygons = [g.simplify(buffer/10) for g in zones]
    polygons = [g.buffer(buffer, **kwargs) for g in polygons]
    polygons = [g.simplify(buffer) for g in polygons]

    if fill_gaps:
        polygons = fill_gaps_with_buffer(polygons, fill_buffer=fill_buffer)
        polygons = [g.buffer(buffer, **kwargs) for g in polygons]

    polygons = biggest_polygons(polygons)

    if unite_gaps:
        voids = interstitial_polygons(polygons, buffer=mini_buffer, hull_buffer=hull_buffer)
        polygons = unite_gaps_to_polygons(voids, polygons, buffer=mini_buffer)
        polygons = [buffer_if_not_polygon(g, buffer) for g in polygons]

    polygons = remove_overlaps(polygons)
    polygons = biggest_polygons(polygons)

    return polygons
