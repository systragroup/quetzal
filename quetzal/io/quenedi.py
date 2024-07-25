import numpy as np
import pandas as pd
import geopandas as gpd
from quetzal.io.gtfs_reader.frequencies import hhmmss_to_seconds_since_midnight, seconds_since_midnight_to_hhmmss
from syspy.spatial.geometries import reverse_geometry
from pathlib import Path
import tempfile
import json
import shutil
import os


def quenedi_to_quetzal_schedule(links, pattern_col='pattern_id'):
    """Convert scheduled links as defined in QUENEDI in links for Quetzal

    Parameters
    ----------
    links : geodataframe
        Links GeoDataFrame containing lists of departures and arrivals as defined in Quenedi
    pattern_col : str, optional, default 'pattern_id'
        Name of the new columns for the pattern index (named trip_id in Quenedi)

    Returns
    -------
    links : geodataframe
        Links GeoDataFrame were all links represent a departure as used in Quetzal
    """
    # Explode Schedule
    df = pd.DataFrame(links)
    df = df.sort_values(by=['trip_id', 'link_sequence']).explode(['departures', 'arrivals'])
    suffixes = df.groupby(level=0).cumcount()
    df.index = df.index.astype(str) + '_' + suffixes.astype(str)
    suffixes.index = df.index
    df[pattern_col] = df['trip_id'].copy()
    df['trip_id'] = df['trip_id'] + '_' + suffixes.astype(str)

    # Convert Departure/Arrival Time
    df['departure_time'] = df['departures'].apply(hhmmss_to_seconds_since_midnight)
    df['arrival_time'] = df['arrivals'].apply(hhmmss_to_seconds_since_midnight)
    df.drop(columns=['departures', 'arrivals', 'time', 'speed'], inplace=True, errors='ignore')

    return gpd.GeoDataFrame(df, crs=links.crs)


def quetzal_to_quenedi_schedule(links, group='route_id'):
    """Convert scheduled links as defined in Quetzal for Quenedi

    Parameters
    ----------
    links : geodataframe
        Links GeoDataFrame were all links represent a departure as used in Quetzal
    group : str, optional, default 'route_id'
        Column representing a group of similar route to be aggregated in pattern_id

    Returns
    -------
    links : geodataframe
        Links GeoDataFrame containing lists of departures and arrivals as defined in Quenedi
    """
    crs = links.crs
    # Build Footprints (list of stops representing the path of the trip)
    grouped = links.sort_values(['trip_id', 'link_sequence']).groupby('trip_id').agg(
        {'a': lambda x: list(x),
         'b': lambda x: list(x)})
    trip_nodes = grouped['a'] + grouped['b'].apply(lambda x: [x[-1]])
    footprints = trip_nodes.map(str)
    footprints.name = 'footprint'

    # Define a pattern_id for each trip
    patterns = pd.concat([
        links.groupby('trip_id')['route_id'].first(),
        footprints],
        axis=1)
    pattern_n = patterns.drop_duplicates().set_index(['footprint', group]).groupby(
        group,
        as_index=False
    ).cumcount()
    pattern_n.name = 'pattern_num'
    patterns = patterns.reset_index().merge(
        pattern_n,
        on=['footprint', group]
    )
    patterns['pattern_id'] = patterns[['route_id', 'pattern_num']].apply(
        lambda x: '_'.join(x.map(str)), 1
    )
    links = links.merge(patterns[['trip_id', 'pattern_id']], on='trip_id')

    # Convert time format
    links['departure_time'] = links['departure_time'].apply(seconds_since_midnight_to_hhmmss)
    links['arrival_time'] = links['arrival_time'].apply(seconds_since_midnight_to_hhmmss)

    # Aggregate by pattern_id
    agg = {c: 'first' for c in links.columns}
    agg['departure_time'] = lambda x: list(x)
    agg['arrival_time'] = lambda x: list(x)
    links = links.sort_values(['trip_id', 'link_sequence']).groupby(['pattern_id', 'link_sequence'], as_index=False).agg(agg)

    links.rename(columns={'departure_time': 'departures', 'arrival_time': 'arrivals'}, inplace=True)

    return gpd.GeoDataFrame(links, crs=crs)


def compute_time_and_speed(links, dwell_from='departure'):
    """Compute time and speed from a Quetzal format scheduled trip with departure_time and arrival_time

    Parameters
    ----------
    links : geodataframe
        Links GeoDataFrame containing departure_time and arrival_time
    dwell_from : {'departure', 'arrival'}, optional, default 'departure'
        Wherea to include dwell time from
        - departure: departure station
        - arrival: arrival station

    Returns
    -------
    links : geodataframe
        Links GeoDataFrame with 'time' and 'speed' columns
    """

    links['time'] = np.nan
    if dwell_from == 'departure':
        links['time'] = links.groupby('trip_id', group_keys=False)['arrival_time'].apply(lambda x: x.diff(1))
    elif dwell_from == 'arrival':
        links['time'] = -links.groupby('trip_id', group_keys=False)['departure_time'].apply(lambda x: x.diff(-1))
    else:
        raise Exception('Unknown value for dwell_time')

    m = links['time'].isna()
    links.loc[m, 'time'] = links.loc[m, 'arrival_time'] - links.loc[m, 'departure_time']

    links['speed'] = links['length'] * 3.6 / links['time']

    return links


def split_quenedi_rlinks(road_links, oneway='0'):
    """
    split road_links into two directions.
    attributes with _r suffix are applied to the reverse links.
    """
    if 'oneway' not in road_links.columns:
        print('no column oneway. do not split')
        return
    links_r = road_links[road_links['oneway']==oneway].copy()
    if len(links_r) == 0:
        print('all oneway, nothing to split')
        return
    # apply _r features to the normal non r features
    r_cols = [col for col in links_r.columns if col.endswith('_r')]
    cols = [col[:-2] for col in r_cols]
    for col, r_col in zip(cols, r_cols):
        links_r[col] = links_r[r_col]
    # reindex with _r 
    links_r.index = links_r.index.astype(str) + '_r'
    # reverse links (a=>b, b=>a)
    links_r = links_r.rename(columns={'a': 'b', 'b': 'a'})
    links_r['geometry'] = links_r['geometry'].apply(lambda g: reverse_geometry(g))
    road_links = pd.concat([road_links, links_r])
    return road_links


def merge_quenedi_rlinks(road_links,new_cols=[]):
    if 'oneway' not in road_links.columns:
        print('no column oneway. do not merge')
        return
    #get reversed links
    index_r = [idx for idx in road_links.index if idx.endswith('_r')]
    if len(index_r) == 0:
        print('all oneway, nothing to merge')
        return
    links_r = road_links.loc[index_r].copy()
    # create new reversed column with new columns
    for col in new_cols:
        links_r[col + '_r'] = links_r[col]
    # reindex with initial non _r index to merge
    links_r.index = links_r.index.map(lambda x: x[:-2])
    new_cols_r = [col+ '_r' for col in new_cols]
    links_r = links_r[new_cols_r]
    # drop added _r links, merge new columns to inital two way links.
    road_links = road_links.drop(index_r, axis=0)
    # drop column if they exist before merge. dont want duplicates
    for col in new_cols_r:
        if col in road_links.columns:
            road_links = road_links.drop(columns=col)
    
    road_links = pd.merge(road_links, links_r, left_index=True, right_index=True, how='left')
    return road_links


def _to_geojson(gdf,tmp_path,new_dir,name,to_4326=True, engine='pyogrio'):
    if to_4326:
        gdf = gdf.to_crs(4326)
    p = tmp_path / os.path.join(new_dir, name + '.geojson')
    gdf.to_file(str(p),driver='GeoJSON',engine=engine)

def to_zip(sm, path='test.zip', to_export=['pt','road'],inputs=[],outputs=[],to_4326=False, engine='pyogrio'):
    """
    Export model to zip file (readable in quenedi)
    sm: Quetzal stepmodel 
    path: str. path to the zip fil
    to_export: list of str ['pt','road']. only ['pt'] to export only links and nodes.
    inputs: list of str. names of the input sm attributes to export
    outputs: list of str. names of the output sm attributes to export
    to_4326: bool. if True, perform to_crs(4326) on GeoDataFrames
    engine: str. 'pyogrio' or 'fiona'.
    """
    if not path.endswith('.zip'):
        path = path + '.zip'
    
    path = Path(path)
    # Write to temporary directory before zipping
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    if 'links' in sm.__dict__.keys() and 'pt' in to_export:
        new_dir = tmp_path / 'inputs/pt'
        os.makedirs(new_dir)
        for name in ['links','nodes']:
            gdf = getattr(sm, name)
            _to_geojson(gdf,tmp_path,new_dir,name,to_4326,engine)
        
    if 'road_links' in sm.__dict__.keys() and 'road' in to_export:
        new_dir = tmp_path / 'inputs/road'
        os.makedirs(new_dir)
        for name in ['road_links','road_nodes']:
            gdf = getattr(sm, name)
            _to_geojson(gdf,tmp_path,new_dir,name,to_4326,engine)

    for name in inputs:
        if name not in sm.__dict__.keys():
            continue
        new_dir = tmp_path / 'inputs/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        data = getattr(sm, name)
        if type(data) == gpd.GeoDataFrame:
            _to_geojson(data,tmp_path,new_dir,name,to_4326,engine)
        elif type(data) == pd.DataFrame:
            p = tmp_path / os.path.join(new_dir, name + '.csv')
            print( name + '.csv')
            data.to_csv(str(p))
            
    for name in outputs:
        if name not in sm.__dict__.keys():
            continue
        new_dir = tmp_path / 'outputs/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        data = getattr(sm, name)
        if type(data) == gpd.GeoDataFrame:
            _to_geojson(data,tmp_path,new_dir,name,to_4326,engine)
        elif type(data) == pd.DataFrame:
            p = tmp_path / os.path.join(new_dir, name + '.csv')
            print( name + '.csv')
            data.to_csv(str(p))

    basename = str(path.parent / path.stem)
    shutil.make_archive(basename, format="zip", root_dir=tmp_dir.name)
    tmp_dir.cleanup()


def read_geojson(filename,**kwargs):
    '''
    read geojson with gpd.read_file but add unreadable columns susch as List.
    Set index if index in column else set index name as 'index'
    '''
    gdf = gpd.read_file(filename,**kwargs)

    json_prop = json_data['features'][0]['properties'].keys()
    missing_columns = set(json_prop).difference(gdf.columns)
    missing_columns.discard('index')# dont want to add index
    with open(filename, 'r') as j:
        json_data = json.loads(j.read())
    for col in missing_columns:
        d = {f['properties']['index']: f['properties'][col] for f in json_data['features']}
        gdf[col] = gdf.index.map(d)

    if 'index' in gdf.columns:
        gdf.set_index('index', inplace=True)
    else:
        gdf.index.name = 'index'

    return gdf