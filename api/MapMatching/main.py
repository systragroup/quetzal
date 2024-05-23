import sys
import os
import boto3
from pydantic import BaseModel
from typing import  Optional
import json
import pandas as pd
import geopandas as gpd
import numba as nb
import os

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath('quetzal'))

BUCKET_NAME = os.environ.get('BUCKET_NAME')
ON_LAMBDA = bool(os.environ.get('AWS_EXECUTION_ENV'))


class Model(BaseModel):
    step: Optional[str]='preparation'
    callID: Optional[str] = 'test'
    exclusions: Optional[list]= [] 
    exec_id: Optional[int] = 0
    
    
    
def handler(event, context):
    args = Model.parse_obj(event)
    print('start')
    print(args)
    uuid = args.callID
    step = args.step
    exec_id = args.exec_id
    exclusions = args.exclusions
    num_cores = nb.config.NUMBA_NUM_THREADS
    if step == 'preparation':
        num_machine = preparation(uuid,exclusions,num_cores)
        event['num_machine'] = num_machine
    elif step == 'mapmatching':
        mapmatching(uuid,exec_id,num_cores)
    else:
        merge(uuid)

    return event



def preparation(uuid,exclusions,num_cores):
    from quetzal.engine.add_network_mapmatching import duplicate_nodes

    basepath = f's3://{BUCKET_NAME}/{uuid}/' if ON_LAMBDA else '../test/'

    links = gpd.read_file(os.path.join(basepath,'links.geojson'),engine='pyogrio')
    links.set_index('index',inplace=True)
    nodes = gpd.read_file(os.path.join(basepath,'nodes.geojson'),engine='pyogrio')
    nodes.set_index('index',inplace=True)


    # if already mapmatched. remove road_links_list (will be redone here)
    if 'road_link_list' in  links.columns:
        print('remove road_links_list')
        links = links.drop(columns = ['road_link_list'])

    links, nodes = duplicate_nodes(links,nodes)

    excluded_links = links[links['route_type'].isin(exclusions)]

    links = links[~links['route_type'].isin(exclusions)]

    #Split 

    trip_list = links['trip_id'].unique()
    num_trips = len(trip_list)

    tot_num_iteration = num_trips//num_cores
    def get_num_machine(num_it,target_it=20,choices=[12,8,4,1]):
        # return the number of machine (in choices) requiresd to have target_it per machine).
        num_machine = num_it//target_it
        best_diff=100
        best_val=12
        for v in choices: # choice of output.
            diff = abs(num_machine-v)
            if diff < best_diff:
                best_diff = diff
                best_val=v
        return best_val

    num_machine =  get_num_machine(tot_num_iteration, target_it=20, choices=[12,8,4,1])

    print('num it per machine',tot_num_iteration//num_machine)

    chunk_length =  round(len(trip_list) / num_machine)
    # Split the list into four sub-lists
    chunks = [trip_list[j:j+chunk_length] for j in range(0, len(trip_list), chunk_length)]
    sum([len(c) for c in chunks]) == len(trip_list)

    for i,trips in enumerate(chunks):
        print(i)
        tlinks = links[links['trip_id'].isin(trips)]
        nodes_set = set(tlinks['a'].unique()).union(set(tlinks['b'].unique()))
        tnodes = nodes[nodes.reset_index()['index'].isin(nodes_set).values]
        tlinks.to_file(os.path.join(basepath, 'parallel', f'links_{i}.geojson'),driver='GeoJSON')
        tnodes.to_file(os.path.join(basepath, 'parallel', f'nodes_{i}.geojson'),driver='GeoJSON')

    if len(excluded_links)>0:
        nodes_set = set(excluded_links['a'].unique()).union(set(excluded_links['b'].unique()))
        tnodes = nodes[nodes.reset_index()['index'].isin(nodes_set).values]
        excluded_links.to_file(os.path.join(basepath, 'parallel', f'links_excluded.geojson'),driver='GeoJSON')
        excluded_links.to_file(os.path.join(basepath, 'parallel', f'nodes_excluded.geojson'),driver='GeoJSON')
    return num_machine

from shapely.ops import transform


def _reverse_geom(geom):
    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]
    return transform(_reverse, geom) 

def split_quenedi_rlinks(road_links, oneway='0'):
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
    links_r['geometry'] = links_r['geometry'].apply(lambda g: _reverse_geom(g))
    road_links = pd.concat([road_links, links_r])
    return road_links


def mapmatching(uuid,exec_id,num_cores):
    from shapely.geometry import LineString
    from quetzal.model import stepmodel
    from quetzal.io.gtfs_reader.importer import get_epsg

    basepath = f's3://{BUCKET_NAME}/{uuid}/' if ON_LAMBDA else '../test'

    links = gpd.read_file(os.path.join(basepath, 'parallel', f'links_{exec_id}.geojson'),engine='pyogrio')
    if 'index' in links.columns:
        links.set_index('index',inplace=True)
    nodes = gpd.read_file(os.path.join(basepath, 'parallel', f'nodes_{exec_id}.geojson'),engine='pyogrio')
    if 'index' in nodes.columns:
        nodes.set_index('index',inplace=True)

    road_links = gpd.read_file(os.path.join(basepath,'road_links.geojson'), engine='pyogrio')
    road_links.set_index('index',inplace=True)
    road_nodes = gpd.read_file(os.path.join(basepath,'road_nodes.geojson'), engine='pyogrio')
    road_nodes.set_index('index',inplace=True)

    print('split rlinks to oneways')
    road_links = split_quenedi_rlinks(road_links)


    # if already mapmatched. remove road_links_list (will be redone here)
    if 'road_link_list' in  links.columns:
        print('remove road_links_list')
        links = links.drop(columns = ['road_link_list'])

    sm = stepmodel.StepModel(epsg=4326)
    sm.links = links
    sm.nodes = nodes
    sm.road_links = road_links
    sm.road_nodes = road_nodes

    centroid = [*LineString(sm.nodes.centroid.values).centroid.coords][0]
    crs = get_epsg(centroid[1],centroid[0])

    sm = sm.change_epsg(crs,coordinates_unit='meter')

    sm.preparation_map_matching(sequence='link_sequence',
                            by='trip_id',
                            routing=True,
                            n_neighbors_centroid=100,
                            n_neighbors=25,
                            distance_max=3000,
                            overwrite_geom=True,
                            overwrite_nodes=True,
                            num_cores=num_cores)
    
    sm.nodes = sm.nodes.to_crs(4326)
    sm.links = sm.links.to_crs(4326)

    sm.links = sm.links.drop(columns=['road_a','road_b','offset_b','offset_a','road_node_list'])
    sm.links['road_link_list'] = sm.links['road_link_list'].fillna('[]')
    sm.links['road_link_list'] = sm.links['road_link_list'].astype(str)

    sm.links.to_file(os.path.join(basepath, 'parallel', f'links_{exec_id}.geojson'), driver='GeoJSON')
    sm.nodes.to_file(os.path.join(basepath, 'parallel', f'nodes_{exec_id}.geojson'), driver='GeoJSON')

def merge(uuid):
    '''
    merge all linka and nodes in uuid/parallel/ folder on s3.
    '''
    basepath = f's3://{BUCKET_NAME}/{uuid}/' if ON_LAMBDA else '../test'
    s3path = f's3://{BUCKET_NAME}/'
    s3 = boto3.client('s3')
    prefix = f'{uuid}/parallel/'
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME,Prefix=prefix)
    links_concat = []; nodes_concat = []
    for obj in resp['Contents']:
        key = obj['Key']
        name = key.replace(prefix, '')
        if name.startswith('links') and name.endswith('.geojson'):
            links_concat.append( gpd.read_file(os.path.join(s3path, key), engine='fiona') )
        elif name.startswith('nodes') and name.endswith('.geojson'):
            nodes_concat.append( gpd.read_file(os.path.join(s3path, key), engine='fiona') )

    links = pd.concat(links_concat)
    nodes = pd.concat(nodes_concat)

    links['road_link_list'] = links['road_link_list'].fillna('[]')
    links['road_link_list'] = links['road_link_list'].astype(str)
    links.set_index('index').to_file(os.path.join(basepath,f'links_final.geojson'), driver='GeoJSON', engine='pyogrio')
    nodes.set_index('index').to_file(os.path.join(basepath,f'nodes_final.geojson'), driver='GeoJSON', engine='pyogrio')
