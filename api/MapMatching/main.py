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
from s3_utils import DataBase

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath('quetzal'))

ON_LAMBDA = bool(os.environ.get('AWS_EXECUTION_ENV'))
db = DataBase()


class Model(BaseModel):
    step: Optional[str]='preparation'
    callID: Optional[str] = 'test'
    exclusions: Optional[list]= [] 
    SIGMA: Optional[float] = 4.07
    BETA: Optional[float] = 3
    POWER: Optional[float] = 2
    DIFF: Optional[bool] = True
    ptMetrics: Optional[bool] = True
    exec_id: Optional[int] = 0
    
    
    
def handler(event, context):
    args = Model.parse_obj(event)
    print('start')
    print(args)

    uuid = args.callID
    step = args.step
    exec_id = args.exec_id
    exclusions = args.exclusions
    kwargs = {'SIGMA':args.SIGMA,
              'BETA':args.BETA,
              'POWER':args.POWER,
              'DIFF':args.DIFF
              }
    add_pt_metrics = args.ptMetrics

    num_cores = nb.config.NUMBA_NUM_THREADS
    if step == 'preparation':
        num_machine = preparation(uuid,exclusions,num_cores)
        event['num_machine'] = num_machine
    elif step == 'mapmatching':
        mapmatching(uuid,exec_id,num_cores,**kwargs)
    else:
        merge(uuid,add_pt_metrics)

    return event



def preparation(uuid,exclusions,num_cores):
    from quetzal.engine.add_network_mapmatching import duplicate_nodes
    basepath = f's3://{db.BUCKET}/{uuid}/' if ON_LAMBDA else '../test/'

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
    def get_num_machine(num_it,target_it=20,choices=[12,8,4,2,1]):
        # return the number of machine (in choices) requiresd to have target_it per machine).
        num_machine = num_it /target_it
        best_diff=100
        best_val=12
        for v in choices: # choice of output.
            diff = abs(num_machine-v)
            if diff < best_diff:
                best_diff = diff
                best_val=v
        return best_val

    num_machine =  get_num_machine(tot_num_iteration, target_it=20, choices=[12,8,4,2,1])

    print('num it per machine',tot_num_iteration/num_machine)

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
        tnodes.to_file(os.path.join(basepath, 'parallel', f'nodes_excluded.geojson'),driver='GeoJSON')
    return num_machine



def mapmatching(uuid,exec_id,num_cores,**kwargs):
    from shapely.geometry import LineString
    from quetzal.model import stepmodel
    from quetzal.io.gtfs_reader.importer import get_epsg
    from quetzal.io.quenedi import split_quenedi_rlinks

    basepath = f's3://{db.BUCKET}/{uuid}/' if ON_LAMBDA else '../test'

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
                            n_neighbors_centroid=1000,
                            n_neighbors=20,
                            distance_max=3000,
                            overwrite_geom=True,
                            overwrite_nodes=True,
                            num_cores=num_cores,
                            **kwargs)
    
    # ovewrite Length with the real length and time
    sm.links['length'] = sm.links['geometry'].apply(lambda g: g.length)
    if 'speed' in sm.links.columns:
        sm.links['time'] = sm.links['length']/sm.links['speed'] * 3.6

    sm.nodes = sm.nodes.to_crs(4326)
    sm.links = sm.links.to_crs(4326)

    sm.links = sm.links.drop(columns=['road_a','road_b','offset_b','offset_a','road_node_list'])
    sm.links['road_link_list'] = sm.links['road_link_list'].fillna('[]')
    sm.links['road_link_list'] = sm.links['road_link_list'].astype(str)

    sm.links.to_file(os.path.join(basepath, 'parallel', f'links_{exec_id}.geojson'), driver='GeoJSON')
    sm.nodes.to_file(os.path.join(basepath, 'parallel', f'nodes_{exec_id}.geojson'), driver='GeoJSON')

def merge(uuid, add_pt_metrics=False):
    '''
    merge all linka and nodes in uuid/parallel/ folder on s3.
    '''
    from syspy.spatial.utils import get_acf_distance

    basepath = f's3://{db.BUCKET}/{uuid}/' if ON_LAMBDA else '../test'
    s3path = f's3://{db.BUCKET}/'
    s3 = boto3.client('s3')
    prefix = f'{uuid}/parallel/'
    resp = s3.list_objects_v2(Bucket=db.BUCKET,Prefix=prefix)
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

    if add_pt_metrics:
        road_links = gpd.read_file(os.path.join(basepath,'road_links.geojson'), engine='pyogrio')
        road_links.set_index('index',inplace=True)
        road_links = add_metrics_to_rlinks(links, road_links)
        road_links.to_file(os.path.join(basepath,f'road_links.geojson'), driver='GeoJSON', engine='pyogrio')



    links['road_link_list'] = links['road_link_list'].fillna('[]')
    links['road_link_list'] = links['road_link_list'].astype(str)
    links.set_index('index').to_file(os.path.join(basepath,f'links_final.geojson'), driver='GeoJSON', engine='pyogrio')
    nodes.set_index('index').to_file(os.path.join(basepath,f'nodes_final.geojson'), driver='GeoJSON', engine='pyogrio')


    df = links[['index','trip_id']].copy()
    df['acf_distance'] = links['geometry'].apply(lambda x: get_acf_distance([x.coords[0],x.coords[-1]],True))
    df['routing_distance'] = links['length']
    df['routing - acf'] = df['routing_distance']-df['acf_distance']
    if 'shape_dist_traveled' in links.columns:
        df['shape_dist_traveled'] = links['shape_dist_traveled']
        df['routing - sdt'] = df['routing_distance']-df['shape_dist_traveled']

    df2 = df.groupby('trip_id').agg(sum)
    df2['routing - acf'] = df2['routing_distance']-df2['acf_distance']
    if 'shape_dist_traveled' in df2.columns:
        df2['routing - sdt'] = df2['routing_distance']-df2['shape_dist_traveled']

    db.save_csv(uuid, 'links_distances.csv', df)
    db.save_csv(uuid, 'trips_distances.csv', df2)

def add_metrics_to_rlinks(links, rlinks):
    # add metrics to road links

    from quetzal.io.quenedi import split_quenedi_rlinks, merge_quenedi_rlinks

    rlinks = split_quenedi_rlinks(rlinks)

    df = links[['trip_id','route_id','headway','road_link_list']].explode('road_link_list')
    df['frequency'] = 1/(df['headway']/3600)

    agg_dict = {'trip_id':'nunique','route_id':'nunique','headway':lambda x: 1 / sum(1/x),'frequency':'sum'}
    df = df.groupby('road_link_list').agg(agg_dict)

    rename_dict = {'trip_id':'trip_id_count',
                'route_id':'route_id_count',
                'headway':'combine_headway (secs)',
                'frequency':'combine_frequency (veh/h)'}
    df = df.rename(columns=rename_dict)
    new_columns = df.columns
    
    rlinks = rlinks.merge(df,left_index=True,right_index=True,how='left')
    #cols_to_fill = [col for col in df.columns if col != 'combine_headway']
    #rlinks[cols_to_fill] = rlinks[cols_to_fill].fillna(0)

    rlinks = merge_quenedi_rlinks(rlinks,new_cols=new_columns)
    return rlinks