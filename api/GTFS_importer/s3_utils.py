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

    def save_image(self, uuid, name, img_buffer):
        '''
            parameters
            ----------
            payload: pandas df to send to s3 as csv.
            name: name of the file (with .csv at the end)
            returns
            ----------
        '''
        img_buffer.seek(0)
        bucket = self.s3_resource.Bucket(self.BUCKET)
        filename =  uuid + '/' + name
        bucket.put_object(Body=img_buffer, ContentType='image/png', Key=filename)

    
    def download_s3_folder(self, s3_folder, local_dir='/tmp'):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        bucket = self.s3_resource.Bucket(self.BUCKET)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)

