import geopandas as gpd
import pandas as pd
import gtfs_kit as gtk
import numpy as np
from quetzal.io.gtfs_reader import importer
from quetzal.io.gtfs_reader.frequencies import hhmmss_to_seconds_since_midnight 
from quetzal.model import stepmodel
from s3_utils import DataBase
from pydantic import BaseModel
from typing import  Optional

import warnings
warnings.filterwarnings("ignore")

# docker build -f api/GTFS_importer/Dockerfile -t gtfs_importer:latest .

# docker run -p 9000:8080 --env-file 'api/GTFS_importer/test.env' gtfs_importer 

# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"test","files":["https://storage.googleapis.com/storage/v1/b/mdb-latest/o/ca-british-columbia-translink-vancouver-gtfs-1222.zip?alt=media"]}'
# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"345e4cb4-4875-449a-bb32-666fb6e4dcb9","files":["stm.zip","stl.zip"]}'



DAY_DICT = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}


class Model(BaseModel):
    callID: Optional[str] = 'test'
    files: Optional[list] = []
    start_time: Optional[str] = '6:00:00'
    end_time: Optional[str] = '12:00:00'
    day: Optional[str] = 'tuesday'
    dates: Optional[list] = []

db = DataBase()


    
def handler(event, context):
    args = Model.parse_obj(event)
    print('start')
    print(args)
    uuid = args.callID
    start_time = args.start_time
    end_time = args.end_time
    day = args.day
    dates = args.dates

    selected_day = DAY_DICT[day]
    time_range = [start_time,end_time]

    #if paths are not url (its from S3.)
    # need to download locally in lambda as gtfs_kit doesnt support s3 buffers.
    if any([f[:4] != 'http' for f in args.files]):
        db.download_s3_folder(uuid, '/tmp')
        files = ['/tmp/' + f for f in args.files]
    else:
        files = args.files
    feeds=[]
    for file in files:
        print('Importing {f}'.format(f=file))
        feeds.append(importer.GtfsImporter(path=file, dist_units='m'))


    for i in range(len(feeds)):
        print('cleaning ', files[i])
        if 'agency_id' not in feeds[i].routes:
            print(f'add agency_id to routes in {files[i]}')
            feeds[i].routes['agency_id'] = feeds[i].agency['agency_id'].values[0]
        
        if 'pickup_type' not in feeds[i].stop_times:
            print(f'pickup_type missing in stop_times. set to 0 in {files[i]}')
            feeds[i].stop_times['pickup_type'] = 0
        
        if 'drop_off_type' not in feeds[i].stop_times:
            print(f'drop_off_type missing in stop_times. set to 0 in {files[i]}')
            feeds[i].stop_times['drop_off_type'] = 0
            
        if 'parent_station' not in feeds[i].stops:
            print(f'parent_station missing in stops. set to NaN in {files[i]}')
            feeds[i].stops['parent_station'] = np.nan
        feeds[i].stop_times['pickup_type'].fillna(0, inplace=True)
        feeds[i].stop_times['drop_off_type'].fillna(0, inplace=True)

        feeds[i].stop_times['arrival_time'] = feeds[i].stop_times['departure_time']

    # if dates is not provided as inputs.
    # get it from first dates of each GTFS
    if len(dates)==0:
        for feed in feeds:
            min_date = feed.calendar['start_date'].unique().min()
            max_date = feed.calendar['end_date'].unique().max()
            # get date range 
            s = pd.date_range(min_date, max_date, freq='D').to_series()
            # get dayofweek selected and take first one
            s = s[s.dt.dayofweek==selected_day][0]
            # format  ex: ['20231011'] and append
            dates.append(f'{s.year}{str(s.month).zfill(2)}{str(s.day).zfill(2)}')


    feeds_t = []
    print('restrict feed')
    for i, feed in enumerate(feeds):
        feed_t = feed.restrict(dates=[dates[i]], time_range=time_range)
        if len(feed_t.trips) > 0:
            feeds_t.append(feed_t)
    #del feeds
    print('add shape_dist_traveled to shapes')
    for feed in feeds_t:
        if 'shape_dist_traveled' not in feed.shapes.columns:
            feed.append_dist_to_shapes()

    print('add shape_dist_traveled to stop_times')
    for feed in feeds_t:
        if 'shape_dist_traveled' not in feed.stop_times.columns:
            feed.append_dist_to_stop_times_fast()
        else:
            nan_sequence=feed.stop_times[feed.stop_times['shape_dist_traveled'].isnull()]['stop_sequence'].unique()
            # if there but all nan are at seq=1. just fill wwith 0.
            if all(seq==1 for seq in nan_sequence):
                feed.stop_times['shape_dist_traveled'] = feed.stop_times['shape_dist_traveled'].fillna(0)
            else:
                feed.append_dist_to_stop_times_fast()
    print('convert to meter if necessary')
    for feed in feeds_t:
        if feed.stop_times['shape_dist_traveled'].max() < 100:
                print(f'convert to meters')
                feed.dist_units = 'km'
                feed = gtk.convert_dist(feed, new_dist_units='m')

    feeds_frequencies = []
    for i in range(len(feeds_t)):
        print('Building links and nodes ', files[i])
        feed_s = feeds_t[i].copy()
        feed_s.group_services()

        feed_s.build_stop_clusters(distance_threshold=50)
        feed_s.build_patterns(on='cluster_id')

        feed_frequencies = feed_s.convert_to_frequencies(time_range=time_range)
        shapes = feed_frequencies.shapes is not None
        feed_frequencies.build_links_and_nodes(log=False, 
                                            shape_dist_traveled=True, 
                                            from_shape=shapes, 
                                            stick_nodes_on_links=shapes,
                                            keep_origin_columns=['departure_time','pickup_type'],
                                            keep_destination_columns=['arrival_time','drop_off_type'],
                                            num_cores=4)
        feeds_frequencies.append(feed_frequencies)
    #del feeds_t
    mapping = {0:'tram', 1:'subway', 2:'rail', 3:'bus',4:'ferry',5:'cable_car',6:'gondola',7:'funicular', 700:'bus', 1501:'taxi'}
    retire = ['taxi']
    for feed_frequencies in feeds_frequencies:
        feed_frequencies.links['route_type'] = feed_frequencies.links['route_type'].apply(
            lambda t: mapping.get(t, np.nan)
        )
        
        assert not any(feed_frequencies.links['route_type'].isna())
        feed_frequencies.links = feed_frequencies.links[~feed_frequencies.links['route_type'].isin(retire)]

    for feed_frequencies in feeds_frequencies:
        feed_frequencies.links.loc[feed_frequencies.links['time'] == 0,'time'] = 1.0



    columns=['trip_id','route_id','agency_id','direction_id','a','b', 'shape_dist_traveled',
                                        'link_sequence','time','headway','pickup_type', 'drop_off_type',
                                        'route_short_name','route_type','route_color','geometry']


    sm = stepmodel.StepModel(epsg=4326, coordinates_unit='meter')

    links_concat = []; nodes_concat = []
    for feed_frequencies in feeds_frequencies:
        links_concat.append(feed_frequencies.links)
        nodes_concat.append(feed_frequencies.nodes)

    # nothing to export. export empty geojson
    if len(links_concat) == 0:
        links = gpd.GeoDataFrame(columns=['feature'], geometry='feature',crs=4326)
        nodes = gpd.GeoDataFrame(columns=['feature'], geometry='feature',crs=4326)
        links.to_file(f's3://{db.BUCKET}/{uuid}/links.geojson', driver='GeoJSON')
        nodes.to_file(f's3://{db.BUCKET}/{uuid}/nodes.geojson', driver='GeoJSON')
        return
    
    sm.links = pd.concat(links_concat)
    for col in columns:
        if col not in sm.links.columns:
            sm.links[col] = np.nan
            
    sm.links = sm.links[columns]
    sm.nodes = pd.concat(nodes_concat)[['stop_id','stop_name','stop_code','geometry']]

    sm.nodes = sm.nodes.reset_index(drop=True).sort_index()
    sm.links = sm.links.reset_index(drop=True).sort_index()


    sm.nodes.loc[sm.nodes['stop_code'].isna(),'stop_code'] = sm.nodes.loc[sm.nodes['stop_code'].isna(),'stop_id'] 
    sm.nodes.drop_duplicates(subset=['stop_id'], inplace=True)

    sm.links['trip_id'] = sm.links['agency_id'] +'_' +sm.links['trip_id']
    sm.links['route_id'] = sm.links['agency_id'] +'_' +sm.links['route_id']

    sm.links = sm.links.sort_values(['route_type','trip_id']).reset_index(drop=True)

    dnodes = ('node_' +sm.nodes.reset_index().set_index('stop_id')['index'].astype(str)).to_dict()
    sm.nodes.index = 'node_' +sm.nodes.index.astype(str)

    sm.links.index = 'link_' +sm.links.index.astype(str)

    sm.links['a'] = sm.links['a'].apply(lambda a: dnodes.get(a))
    sm.links['b'] = sm.links['b'].apply(lambda a: dnodes.get(a))

    sm.links.drop_duplicates(subset=['trip_id','link_sequence'], inplace=True)

    # Tag route with only one trip
    time_slot = np.diff([hhmmss_to_seconds_since_midnight(time) for time in time_range])[0]
    sm.links.loc[(time_slot/sm.links['headway']) < 2.0, 'headway'] = np.nan

    sm.links = sm.links.to_crs(4326)
    sm.nodes = sm.nodes.to_crs(4326)
    print('Saving on S3'), 
    sm.links.to_file(f's3://{db.BUCKET}/{uuid}/links.geojson', driver='GeoJSON')
    sm.nodes.to_file(f's3://{db.BUCKET}/{uuid}/nodes.geojson', driver='GeoJSON')
    print('done')

