import os
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import boto3
from sklearn.cluster import KMeans
from shapely.geometry import Point
from quetzal.engine.road_model import *

from io import BytesIO
# docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .


# docker build -t ml_matrixroadcaster:latest .
# docker run -p 9000:8080  ml_matrixroadcaster
# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

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
    print('start')
    num_zones = 100
    train_size = 10
    date_time = '2022-12-13T08:00:21+01:00'
    ff_time_col = 'time'
    max_speed = 100
    num_cores = 1
    hereApiKey= 'QMopayRokz2AnlodBD6oYt-gPxUjZe4DN-XBhv4Wnx0' 

    print('read files')
    links = gpd.read_file('road_links.geojson')
    links.set_index('index',inplace=True)
    nodes = gpd.read_file('road_nodes.geojson')
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
        mat = pd.read_csv( 'here_OD.csv')
        mat = mat.set_index('origin')
        mat.columns.name='destination'
    except:
        mat = self.call_api_on_training_set(train_od,
                                            apiKey=hereApiKey,
                                            api='here',
                                            mode='car',
                                            time=date_time,
                                            verify=True)
        mat.to_csv( 'here_OD.csv')

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
    print('done!')