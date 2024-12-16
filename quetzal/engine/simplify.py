import h3
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point


def polygon_to_h3_zones_gdf(polygon_4326, resolution):
    poly = h3.LatLngPoly([(y, x) for x, y in polygon_4326.exterior.coords[1:]])
    cells = h3.h3shape_to_cells(poly, res=resolution)
    zones = gpd.GeoDataFrame({"geometry" : [Point(h3.cell_to_latlng(c)[::-1]) for c in cells]})
    return zones.set_crs(epsg=4326)


def simplify_road_network(sm, resolution=8, to_keep_indices=[], force=False):
    rl_cols = sm.road_links.columns
    ## BUILD H3 ZONING
    # get latlong perimeter
    print("build H3 zoning")
    perimeter_df = gpd.GeoDataFrame({"geometry": [sm.road_links.unary_union.convex_hull]})
    perimeter_df = perimeter_df.set_crs(epsg=sm.epsg).to_crs(epsg=4326)
    perimeter = perimeter_df.geometry.values[0]
    # build zoning
    zones = polygon_to_h3_zones_gdf(perimeter, resolution)

    # check n zones
    if not force:
        assert(len(zones)<1500), f"Warning: the algorithm will use {len(zones)} which is high and might kill your computer! Rerun with force=True if you are sure to proceed"
    
    # prepare model and build dumb volumes
    print("prepare model")
    sm.zones = zones.set_crs(epsg=4326).to_crs(epsg=sm.epsg)
    volumes = pd.DataFrame(
        index=pd.MultiIndex.from_product([zones.index, zones.index]),
        data=np.ones(len(zones.index)**2),
        columns=["volume"]
    )
    sm.volumes = volumes.reset_index().rename(columns={"level_0": "origin", "level_1": "destination"})
    sm.preparation_ntlegs(zone_to_road=True, road_to_transit=True)

    # run pathfinder and assignment
    print("run pathfinder")
    sm.step_road_pathfinder(method='aon')
    sm.los = sm.car_los
    sm.car_los[("volume", "probability")] = 1
    sm.segments=["volume"]
    print("run assigmnent")
    sm.step_assignment(road=True)

    # Filter road_links
    # to_keep_indices = 
    print("filter")
    keep_loc = sm.road_links.index.isin(to_keep_indices)
    rl = sm.road_links.loc[(keep_loc)|(sm.road_links[("volume", "car")]>0)]

    # run integrity checks
    print("integrity checks")
    clean = sm.copy()
    clean.road_links = rl
    node_indices = list(set(rl["a"]).union(set(rl["b"])))
    clean.road_nodes = sm.road_nodes.loc[node_indices]
    clean.integrity_fix_road_network(recursive_depth=10)

    return clean.road_nodes, clean.road_links[rl_cols]
        