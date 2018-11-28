import shapely
import pandas as pd
from tqdm import tqdm


def remove_overlaps(geometries):
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


def interstitial_polygons(geometries, buffer=1e-9):
    multi = shapely.geometry.MultiPolygon(geometries).buffer(buffer)
    convex_hull = multi.convex_hull
    convex_hull_buffer = convex_hull.buffer(1)
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
                border_length(y_geometries[i], g)
                for i in range(len(y_geometries))
            ]
        )
    return array


def gap_nearest_polygon_index(gap_idex, length_matrix):
    index = length_matrix[gap_idex].sort_values().index
    polygon_index = list(index)[-1]
    return polygon_index


def unite_gaps_to_polygons(gaps, polygons):

    geometries = [p for p in polygons]

    array = border_length_matrix(polygons, gaps)
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
    buffer=1e-4,
    fill_buffer=1e-4,
    fill_gaps=True,
    unite_gaps=True,
    **kwargs
):

    polygons = [g.buffer(buffer, **kwargs) for g in zones]
    polygons = [g.simplify(buffer) for g in polygons]

    if fill_gaps:
        polygons = fill_gaps_with_buffer(polygons, fill_buffer=fill_buffer)
        polygons = [g.buffer(buffer, **kwargs) for g in polygons]

    polygons = biggest_polygons(polygons)

    if unite_gaps:

        voids = interstitial_polygons(polygons)
        polygons = unite_gaps_to_polygons(voids, polygons)
        polygons = [buffer_if_not_polygon(g, buffer) for g in polygons]

    polygons = remove_overlaps(polygons)
    polygons = biggest_polygons(polygons)

    return polygons
