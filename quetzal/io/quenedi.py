import numpy as np
import pandas as pd
import geopandas as gpd
from quetzal.io.gtfs_reader.frequencies import hhmmss_to_seconds_since_midnight, seconds_since_midnight_to_hhmmss
from pathlib import Path
import tempfile
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


def to_zip(sm, path='test.zip', to_export=['pt','road'], engine='pyogrio'):
    """
    Export model to zip file (readable in quenedi)
    sm: Quetzal stepmodel 
    path: str. path to the zip fil
    to_export: list of str ['pt','road']. only ['pt'] to export only links and nodes.
    engine: str. 'pyogrio' or 'fiona'.
    """
    if not path.endswith('.zip'):
        path = path + '.zip'
    
    path = Path(path)
    # Write to temporary directory before zipping
    tmp_dir = tempfile.TemporaryDirectory()
    new_path = Path(tmp_dir.name)
    if 'links' in sm.__dict__.keys() and 'pt' in to_export:
        new_dir = new_path / 'inputs/pt'
        os.makedirs(new_dir)
        for name in ['links','nodes']:
            gdf = getattr(sm, name)
            p = new_path / os.path.join(new_dir, name + '.geojson')
            gdf.to_file(str(p),driver='GeoJSON',engine=engine)

    if 'road_links' in sm.__dict__.keys() and 'road' in to_export:
        new_dir = new_path / 'inputs/road'
        os.makedirs(new_dir)
        for name in ['road_links','road_nodes']:
            gdf = getattr(sm, name)
            p = new_path / os.path.join(new_dir, name + '.geojson')
            gdf.to_file(str(p),driver='GeoJSON',engine=engine)

    basename = str(path.parent / path.stem)
    shutil.make_archive(basename, format="zip", root_dir=tmp_dir.name)
    tmp_dir.cleanup()