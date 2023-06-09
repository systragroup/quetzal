import networkx as nx
import geopandas as gpd
from sklearn.cluster import KMeans
from shapely.geometry import Point
from quetzal.engine.road_model import *
from s3_utils import DataBase
from io import BytesIO
from pydantic import BaseModel
from typing import  Optional
import matplotlib.pyplot as plt
# docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .

# docker run -p 9000:8080 --env-file 'api/ML_MatrixRoadCaster/test.env' ml_matrixroadcaster 

# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"test"}'



class Model(BaseModel):
    callID: Optional[str] = 'test'
    num_zones: Optional[int] = 100
    train_size: Optional[int] = 100
    date_time: Optional[str] = '2022-12-13T08:00:21-04:00'
    ff_time_col: Optional[str] = 'time'
    max_speed: Optional[float] = 100
    num_cores: Optional[int] = 1
    gap_limit: Optional[float] = 0.5
    max_num_it: Optional[int] = 30
    num_random_od: Optional[int] = 1
    create_zone: Optional[bool] = True
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

# from OSM_api
def main_strongly_connected_component(links, nodes=None, split_direction=False):
    graph = nx.DiGraph()
    graph.add_edges_from(links[['a', 'b']].values.tolist())
    if 'oneway' in links.columns and split_direction :
        graph.add_edges_from(
            links.loc[~links['oneway'].astype(bool)][['b', 'a']].values.tolist()
        )

    main_scc = None
    size = 0
    for scc in nx.strongly_connected_components(graph):
        if len(scc) > size :
            size = len(scc)
            main_scc = scc

    l = links.loc[links['a'].isin(main_scc) & links['b'].isin(main_scc)]
    if nodes is not None:
        n = nodes.loc[list(main_scc)]
        return l, n
    return l 


    
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
    gap_limit = args.gap_limit
    max_num_it = args.max_num_it
    num_random_od = args.num_random_od
    create_zone = args.create_zone
    hereApiKey = args.hereApiKey
    
    print('read files')
    #links = db.read_geojson(uuid,'road_links.geojson')

    links = gpd.read_file(f's3://{db.BUCKET}/{uuid}/road_links.geojson', driver='GeoJSON')
    links.set_index('index',inplace=True)
    nodes = gpd.read_file(f's3://{db.BUCKET}/{uuid}/road_nodes.geojson', driver='GeoJSON')
    nodes.set_index('index',inplace=True)

    if create_zone:
        print('create zones')
        zones = create_zones_from_nodes(nodes,num_zones=num_zones)
    else: 
        zones = gpd.read_file(f's3://{db.BUCKET}/{uuid}/zones.geojson', driver='GeoJSON')
        zones.set_index('index',inplace=True)

    print('init road_model')
    self = RoadModel(links,nodes,zones,ff_time_col=ff_time_col)
    print('split rlinks to oneways')
    self.split_quenedi_rlinks()

    print('remove cul-de-sac')
    self.road_links, self.road_nodes = main_strongly_connected_component(self.road_links, self.road_nodes, split_direction=False)

    #remove NaN time. use 20kmh
    null_idx = self.road_links[self.ff_time_col].isnull()
    print(len(self.road_links[null_idx]),'links with null time. replace with 20kmh time')
    self.road_links.loc[null_idx,self.ff_time_col] = self.road_links.loc[null_idx,'length'] * 3.6 / 20

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
                                            verify=True,
                                            saving=False)
        db.save_csv(uuid, 'here_OD.csv', mat)

    # apply OD mat
    self.apply_api_matrix(mat,api_time_col='here_time')

    # train and predict
    print('train and predict')
    self.train_knn_model(weight='distance', n_neighbors=5)
    self.predict_zones()

    print('apply OD time on road links')
    
    err = self.apply_od_time_on_road_links(gap_limit=gap_limit,max_num_it=max_num_it, num_cores=num_cores, max_speed=max_speed,log_error=True)
    print('merge links back to two ways')
    self.merge_quenedi_rlinks()

    print('creating and saving plots')
    plot_zones(self, uuid)
    plot_error(err, uuid)
    
    img_data = BytesIO()
    plot_correlation(self.od_time[self.api_time_col]/60, 
                    self.od_time['routing_time']/60, 
                    alpha=0.5,
                    xlabel='OD time (mins)', 
                    ylabel='Routing time (mins)',
                    title = 'Road network calibration (yellow = 95th percentile)')
    plt.savefig(img_data, format='png')
    db.save_image(uuid,'3_HERE_road_calibration.png', img_data)

    for i in range(num_random_od):
        f1,f2 = plot_random_od(self,seed=i)
        img_data = BytesIO()
        f1.savefig(img_data, format='png')
        db.save_image(uuid,'4_HERE_OD_prediction_{idx}.png'.format(idx=i+1), img_data)
        img_data = BytesIO()
        f2.savefig(img_data, format='png')
        db.save_image(uuid,'4_HERE_speed_prediction_{idx}.png'.format(idx=i+1), img_data)


    print('Saving on S3'), 
    self.road_links.to_file(f's3://{db.BUCKET}/{uuid}/road_links.geojson', driver='GeoJSON')
    self.road_nodes.to_file(f's3://{db.BUCKET}/{uuid}/road_nodes.geojson', driver='GeoJSON')
    self.zones.to_file(f's3://{db.BUCKET}/{uuid}/zones.geojson', driver='GeoJSON')
    print('done')

def plot_error(err, uuid):
    img_data = BytesIO()
    _, ax = plt.subplots(figsize=(10, 6))
    plt.plot([x[0] for x in err[1:]],[x[1] for x in err[1:]], linewidth=3)
    plt.grid(True, 'major', linestyle='-', axis='both')
    ax.set_axisbelow(True)
    plt.xlabel('iteration')
    plt.title('Road calibration error per iteration')
    plt.ylabel('mean error on OD VS routing (mins)')
    plt.savefig(img_data, format='png')
    db.save_image(uuid, '2_HERE_iteration_error.png', img_data)

def plot_zones(self,uuid):
    img_data = BytesIO()
    _, ax = plt.subplots(figsize=(10,10))
    self.road_links.plot(ax=ax,alpha=0.2,zorder=1)
    self.zones_centroid.plot(ax=ax,color='orange',zorder=2)
    plt.title('Zones Centroids')
    plt.savefig(img_data, format='png')
    db.save_image(uuid, '1_HERE_zones_centroids.png', img_data)