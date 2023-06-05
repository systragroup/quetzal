import os
import io
import requests
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import boto3
from sklearn.cluster import KMeans
from shapely.geometry import Point
from quetzal.engine.road_model import *
from s3_utils import DataBase
from io import BytesIO
from pydantic import BaseModel
from typing import Union, Optional
# docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .


# docker run -p 9000:8080 --env-file 'api/ML_MatrixRoadCaster/test.env' ml_matrixroadcaster 

# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'


# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"test"}'



class Model(BaseModel):
    callID: Optional[str] = 'test'
    num_zones: Optional[int] = 100
    train_size: Optional[int] = 10
    date_time: Optional[str] = '2022-12-13T08:00:21+01:00'
    ff_time_col: Optional[str] = 'time'
    max_speed: Optional[float] = 100
    num_cores: Optional[int] = 1
    hereApiKey: str = '' 
   

db = DataBase()

def create_zones_from_nodes(nodes,num_zones=100):
    nodes['x'] = nodes['geometry'].apply(lambda p:p.x)
    nodes['y'] = nodes['geometry'].apply(lambda p:p.y)
    cluster = KMeans(n_clusters=num_zones,random_state=0,n_init='auto')
    cluster.fit(nodes[['x','y']].values)
    geom = [Point(val) for val in cluster.cluster_centers_]
    zones = gpd.GeoDataFrame(range(len(geom)),geometry=geom,crs=4326).drop(columns=0)
    zones.index = 'zone_' + zones.index.astype(str)
    return zones


    
def handler(event, context):
    args = Model.parse_obj(event)
    print('start')
    print(args)
    uuid = args.callID
    num_zones = args.num_zones
    train_size = args.train_size
    date_time = args.date_time
    ff_time_col = args.ff_time_col
    max_speed = args.max_speed
    num_cores = args.num_cores
    hereApiKey= args.hereApiKey
    
    print('read files')
    links = db.read_geojson(uuid,'road_links.geojson')
    links.set_index('index',inplace=True)
    nodes = db.read_geojson(uuid,'road_nodes.geojson')
    nodes.set_index('index',inplace=True)

    print('create zones')
    zones = create_zones_from_nodes(nodes,num_zones=num_zones)

    print('init road_model')
    self = RoadModel(links,nodes,zones,ff_time_col=ff_time_col)
    print('split rlinks to oneways')
    self.split_quenedi_rlinks()

    print('find nearest nodes')
    self.zones_nearest_node()
    print('create OD mat')
    self.create_od_mat()
    print(len(self.od_time),'OD')
    print(len(self.zones_centroid), 'zones')

    train_od = self.get_training_set(train_size=train_size,seed=42)


    #read Here matrix
    try:
        mat = db.read_csv(uuid,'here_OD.csv')
        mat = mat.set_index('origin')
        mat.columns.name='destination'
    except:
        mat = self.call_api_on_training_set(train_od,
                                            apiKey=hereApiKey,
                                            api='here',
                                            mode='car',
                                            time=date_time,
                                            verify=True)
        db.save_csv(uuid, 'here_OD.csv', mat)

    # apply OD mat
    self.apply_api_matrix(mat,api_time_col='here_time')

    # train and predict
    print('train and predict')
    self.train_knn_model(weight='distance', n_neighbors=5)
    self.predict_zones()

    print('apply OD time on road links')
    err = self.apply_od_time_on_road_links(gap_limit=0.5,max_num_it=15, num_cores=num_cores, max_speed=max_speed,log_error=True)

    #plots

    self.merge_quenedi_rlinks()
    print('Saving on S3'), 
    db.save_geojson(uuid, 'road_links2.geojson', self.road_links)
    db.save_geojson(uuid, 'road_nodes2.geojson', self.road_nodes)
    print('done')
    