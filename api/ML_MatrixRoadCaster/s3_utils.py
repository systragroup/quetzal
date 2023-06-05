import os
from io import BytesIO, StringIO
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import boto3

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.nan):
            return None
        return super(NpEncoder, self).default(obj)


class DataBase:
    def __init__(self):
        print('init db')
        self.BUCKET = os.environ['BUCKET_NAME']
        self.s3_resource = boto3.resource('s3')
        self.s3 = boto3.client('s3')    

    def read_geojson(self, uuid, name):
        path =  uuid  + '/' + name
        result = self.s3.get_object(Bucket=self.BUCKET, Key=path) 
        dict = json.load(result["Body"])
        return gpd.read_file(json.dumps(dict))
    
    def save_geojson(self, uuid, name, payload):
        '''
            parameters
            ----------
            payload: geopdataframe file to s3
            name: name of the file (with .geojson at the end)
            returns
            ----------
        '''
        json_data = json.loads(payload.to_json())
        uploadByteStream = bytes(json.dumps(json_data, cls=NpEncoder).encode('UTF-8'))

        filename =  uuid + '/' + name
        self.s3_resource.Object(self.BUCKET,filename).put(Body=uploadByteStream)

    
    def read_csv(self, uuid, name):
        path = uuid  + '/' + name
        obj = self.s3.get_object(Bucket=self.BUCKET, Key=path)
        return pd.read_csv(BytesIO(obj['Body'].read()))
    
    def save_csv(self, uuid, name, payload):
        '''
            parameters
            ----------
            payload: pandas df to send to s3 as csv.
            name: name of the file (with .csv at the end)
            returns
            ----------
        '''
        csv_buffer = StringIO()
        payload.to_csv(csv_buffer)

        filename =  uuid + '/' + name
        self.s3_resource.Object(self.BUCKET, filename).put(Body=csv_buffer.getvalue())
