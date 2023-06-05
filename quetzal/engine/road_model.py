import pandas as pd
from numba import jit
import numba as nb
import numpy as np
from tqdm import tqdm
from time import sleep
import copy

from sklearn.model_selection import train_test_split
import ray

from sklearn.neighbors import KNeighborsRegressor
from syspy.spatial.spatial import nearest
from quetzal.engine.pathfinder_utils import get_path, parallel_dijkstra, simple_routing, build_index, sparse_matrix
from quetzal.engine.msa_utils import get_zone_index
from syspy.clients.api_proxy import multi_get_distance_matrix

class RoadModel:
    def __init__(self, road_links, road_nodes, zones, ff_time_col='time_ff'):
        """RoadModel class is used to calibrate a road network with the actual time from each OD.
    1) self.zones_nearest_node() find the nearest road_nodes to each zones
    2) self.create_od_mat() create a OD mat with each zones and get each routing time on road_links['time_ff']
    3.1) self.get_training_set() : get a random sample in the OD matrix to get actual value from an API
    3.2) self.call_api_on_training_set() : call the API for each training OD and save it.
    4) self.apply_api_matrix() : apply the now known OD time to the OD mat and flag the others one as interpolated.
    5) self.train_knn_model() : train a KNN model on the known OD. [o_lon,o_lat,d_lon,d_lat] => time
    6) self.predict_zones() : predict the travel time for every unknowned OD
    7) self.apply_od_time_on_road_links :  ajust speed on road_links to fit the each OD time when we route.


    Requires
    ----------
    road_links : geoDataFrame
    road_nodes : geoDataFrame
    zones : geoDataFrame
    
    Parameters
    ----------

    ff_time_col : str, optional, default 'time_ff'
        column to use in road_links as a basic time per links (in seconds)
    
    Builds
    ----------
    self.ff_time_col
    self.road_links 
    self.road_nodes
    self.zones 
    self.units
        """

        self.ff_time_col = ff_time_col
        self.road_links = road_links
        self.road_nodes = road_nodes
        self.zones = zones
        assert self.zones.crs == self.road_links.crs == self.road_nodes.crs, "CRS doesn't match on zones, nodes and links"
        if road_links.crs.is_geographic:
            self.units = 'degrees'
        else:
            self.units = 'meters'
        # set index
        if 'index' in road_links.columns:
            road_links.set_index('index', inplace=True)
        if 'index' in road_nodes.columns:
            road_nodes.set_index('index', inplace=True)
        if 'index' in zones.columns:
            zones.set_index('index', inplace=True)
        rcol = self.road_links.columns.values
        # columns verifications
        for col in ['a', 'b', 'length', ff_time_col]:
            if col not in rcol:
                raise Exception('columns {col} not found in road_links with columns {road_col}'.format(col=col, road_col=rcol))

    def copy(self):
        return copy.deepcopy(self)
    
    def split_quenedi_rlinks(self,oneway='0'):
        links_r = self.road_links[self.road_links['oneway']==oneway].copy()
        # apply _r features to the normal non r features
        r_cols = [col for col in links_r.columns if col.endswith('_r')]
        cols = [col[:-2] for col in r_cols]
        for col,r_col in zip(cols,r_cols):
            links_r[col] = links_r[r_col]
        # reindex with _r
        links_r.index = links_r.index.astype(str)+ '_r'
        # reverse links (a=>b, b=>a)
        links_r = links_r.rename(columns={'a':'b','b':'a'})
        self.road_links = pd.concat([self.road_links,links_r])
    
    def merge_quenedi_rlinks(self):
        #get reversed links
        index_r = [idx for idx in self.road_links.index if idx.endswith('_r')]
        links_r = self.road_links.loc[index_r].copy()
        # create new reversed column with here speed and time
        links_r[self.api_time_col+'_r'] = links_r[self.api_time_col]
        links_r[self.api_speed_col+'_r'] = links_r[self.api_speed_col]
        #reindex with initial non _r index to merge
        links_r.index = links_r.index.map(lambda x:x[:-2])
        links_r = links_r[[self.api_time_col+'_r',self.api_speed_col+'_r']]
        # drop added _r links, merge new here columns to inital two way links.
        self.road_links = self.road_links.drop(index_r, axis=0)
        self.road_links =  pd.merge(self.road_links,links_r,left_index=True,right_index=True,how='left')
                
    def zones_nearest_node(self):
        # getting zones centroids
        centroid = self.zones.copy()
        centroid['geometry'] = centroid.centroid
        # finding nearest node
        neigh = nearest(centroid, self.road_nodes, n_neighbors=1).rename(columns={'ix_one': 'zone_index', 'ix_many': 'node_index'})
        self.zone_node_dict = neigh.set_index('zone_index')['node_index'].to_dict()
        centroid['node_index'] = centroid.index.map(self.zone_node_dict.get)
        print('max_distance found: ', neigh['distance'].max(), self.units)
        # check for duplicated nodes. if there is. drop the duplicated zones.
        if len(centroid.drop_duplicates('node_index')) != len(centroid):
            print('there is zones associates to the same road_node')
            # duplicated = centroid[centroid['node_index'].duplicated()]['node_index'].values
            print('dropping zones: ')
            print(centroid[centroid['node_index'].duplicated()].index.values)
            centroid = centroid.drop_duplicates('node_index')
        self.zones_centroid = centroid
        
    def create_od_mat(self, od_set=None):
        '''
        od_set=None. use all Origin and destination in self.zones_centroid. provide with r_node as o and d.
        '''
        od_time = simple_routing(origin=self.zones_centroid['node_index'].values,
                                 destination=self.zones_centroid['node_index'].values,
                                 links=self.road_links,
                                 weight_col=self.ff_time_col)

        od_time = od_time.stack().reset_index().rename(columns={'level_0': 'origin', 'level_1': 'destination', 0: self.ff_time_col})
        # remove Origin  ==  destination
        od_time = od_time[od_time['origin'] != od_time['destination']]
        if od_set: # need to be rnodes 
            od_time = od_time.set_index(['origin', 'destination']).loc[list(od_set)].reset_index()
        
        inf_od = od_time[~np.isfinite(od_time[self.ff_time_col])]
        assert len(inf_od) == 0, 'fail: there is infinitely long path, fix your links or your zones \n {val}'.format(val=inf_od[['origin', 'destination']])        
            
        x_dict = self.road_nodes['geometry'].apply(lambda p: p.x).to_dict()
        y_dict = self.road_nodes['geometry'].apply(lambda p: p.y).to_dict()
        od_time['o_lon'] = od_time['origin'].apply(lambda x: x_dict.get(x))
        od_time['o_lat'] = od_time['origin'].apply(lambda x: y_dict.get(x))
        od_time['d_lon'] = od_time['destination'].apply(lambda x: x_dict.get(x))
        od_time['d_lat'] = od_time['destination'].apply(lambda x: y_dict.get(x))

        od_time = od_time[['origin', 'destination', self.ff_time_col, 'o_lon', 'o_lat', 'd_lon', 'd_lat']]
        self.od_time = od_time
        
    def get_training_set(self, train_size=10000, seed=42):
        '''
        train_size: number of OD to call (10 000  = 100 ori x 100 des)
        '''
        X = self.od_time[['o_lon', 'o_lat', 'd_lon', 'd_lat']]
        y = self.od_time[[self.ff_time_col]]
        size = 1 - (train_size) / len(self.zones_centroid)**2
        if size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state=seed)

            print(len(X_test), ' OD to interpolate')  
            print(len(X_train), ' OD for training (API call)')
            print(round(100 * len(X_train) / len(X_test), 1), '% training')

            train_od = self.od_time.loc[X_train.index][['origin', 'destination']]
            print('number of unique origin and destination in training set')
            print(len(train_od['origin'].unique()), len(train_od['destination'].unique()))
        else:
            train_od = self.od_time[['origin', 'destination']].copy()
            print('100% of OD for training (API call)')

        print('max destination for an origin:', train_od.groupby('origin').agg(len).max()['destination'])

        train_od = train_od.groupby('origin').agg(list)
        return train_od

    def call_api_on_training_set(self,
                                 train_od,
                                 apiKey='',
                                 api='here',
                                 mode='car',
                                 time=None,
                                 verify=False,
                                 saving=True):
        mat = pd.DataFrame()
        for row in tqdm(list(train_od.iterrows())):
            origin_nodes = row[0]
            destination_nodes = row[1]['destination']
            origins = self.zones_centroid[self.zones_centroid['node_index'] == origin_nodes]
            destinations = self.zones_centroid[self.zones_centroid['node_index'].isin(destination_nodes)]
            try:
                res = multi_get_distance_matrix(origins=origins,
                                                destinations=destinations,
                                                batch_size=(1, 100),
                                                apiKey=apiKey,
                                                mode=mode,
                                                api=api,
                                                time=time,
                                                verify=verify)
                sleep(1)
            except:
                sleep(5)
                res = multi_get_distance_matrix(origins=origins,
                                                destinations=destinations,
                                                batch_size=(1, 100),
                                                apiKey=apiKey,
                                                api=api,
                                                mode=mode,
                                                time=time,
                                                verify=verify)

            mat = pd.concat([mat, res])
            sleep(0.2)
        if saving:
            print('saving mat')
            mat.to_csv('Here_OD.csv')
        return mat

    def apply_api_matrix(self, mat, api_time_col='time'):
        '''
        take a matrix of time row (zone origin) column (zone destination).
        apply each time to self.od_time (api_time_col), get residual (free_flow-api_time)
        and add a interpolated=True to every other row of self.od_time.
        the zone are convert to nodes index (like self.od_time) with self.zone_node_dict
        '''
        self.api_time_col = api_time_col
        if api_time_col in self.od_time.columns:
            print('"{col}" columns will be overwrite in self.od_time'.format(col=api_time_col))
        mat = mat.stack().reset_index().rename(columns={0: 'time'})
        # unfound OD in the here api call. remove them with nan
        print(len(mat[mat['time'] == -9999]), 'unfound OD in the api call, they will be interpolated')
        mat.loc[mat['time'] == -9999, 'time'] = np.nan

        mat['origin'] = mat['origin'].apply(lambda x: self.zone_node_dict.get(x))
        mat['destination'] = mat['destination'].apply(lambda x: self.zone_node_dict.get(x))

        train_time_dict = mat.set_index(['origin', 'destination'])['time'].to_dict()

        self.od_time[api_time_col] = self.od_time.set_index(['origin', 'destination']).index.map(train_time_dict.get)

        self.od_time['interpolated'] = True
        self.od_time.loc[~np.isnan(self.od_time[api_time_col]), 'interpolated'] = False
        
        self.od_time['residual'] = self.od_time[api_time_col] - self.od_time[self.ff_time_col]

        print('od time applied on', len(self.od_time[~self.od_time['interpolated']]))
        print('od to interpolate', len(self.od_time[self.od_time['interpolated']]))

    def train_knn_model(self, weight='distance', n_neighbors=5):
        '''
        weight: distance, uniform or a function. (distance = 1/x)
        '''
        train_df = self.od_time[~self.od_time['interpolated']]

        neigh = KNeighborsRegressor(weights=weight, algorithm='ball_tree', n_neighbors=n_neighbors,)
        neigh.fit(train_df[['o_lon', 'o_lat', 'd_lon', 'd_lat']].values, train_df['residual'].values)
        self.knn_model = neigh

    def predict_zones(self):
        interp_df = self.od_time[self.od_time['interpolated']]
        pred_res = self.knn_model.predict(interp_df[['o_lon', 'o_lat', 'd_lon', 'd_lat']].values)
        self.od_time.loc[interp_df.index, 'residual'] = pred_res
        self.od_time[self.api_time_col] = self.od_time[self.ff_time_col] + self.od_time['residual']

    def apply_od_time_on_road_links(self, gap_limit=1, max_num_it=10, num_cores=4, max_speed=100,api_speed_col='here_speed', log_error=True):
        '''
        gap_limit: stop iterative process at this error value (minutes)
        max_num_it: maximum number of iteration
        num_cores: for parallel ijkstra
        max_speed: limit on road speed, cannot go faster  (kmh)

        return
        -------
        self.road_links with api_time_col and api_speed_col
        '''
        self.api_speed_col = api_speed_col
        df = self.road_links[['a', 'b', 'length', self.ff_time_col]].copy()

        # Build Sparse Matrix indexes for routing down the line
        edges = df[['a', 'b', self.ff_time_col]].values  # to build the index once and for all
        index = build_index(edges)
        reversed_index = {v: k for k, v in index.items()}
        # apply sparse index on zones
        od_time, zones = get_zone_index(df, self.od_time, index)

        # apply sparse index on links
        df['sparse_a'] = df['a'].apply(lambda x: index.get(x))
        df['sparse_b'] = df['b'].apply(lambda x: index.get(x))
        time_sparse_keys = list(zip(df['sparse_a'], df['sparse_b']))

        odt = od_time[['o', 'd', self.api_time_col]].values

        # algo
        nb.set_num_threads(num_cores)
        # init New_time to freeflow time
        df['new_time'] = df[self.ff_time_col]
        errors = []
        for i in range(max_num_it):

            edges = df[['a', 'b', 'new_time']].values
            sparse, _ = sparse_matrix(edges, index=index)
            time_matrix, predecessors = parallel_dijkstra(sparse,
                                                          directed=True,
                                                          indices=zones,
                                                          return_predecessors=True,
                                                          num_core=num_cores,
                                                          keep_running=True)

            # this give OD_time/time_matrix on each links. then X links time for the ratio links_time/tot_time
            ab_times = _assign_time(odt, predecessors, time_matrix, time_sparse_keys, reversed_index)
            df['time_correction'] = df.set_index(['a', 'b']).index.map(ab_times.get)
            cond = (np.isfinite(df['time_correction'])) & (df['time_correction'] > 0)
            df['time_correction'] = df['time_correction'] * df['new_time']
            df.loc[cond, 'new_time'] = df.loc[cond, 'time_correction']
            df['new_speed'] = df['length'] / df['new_time'] * 3.6
            df.loc[df['new_speed'] > max_speed, 'new_time'] = 3.6 * df.loc[df['new_speed'] > max_speed, 'length'] / max_speed

            od_time['routing_time'] = _get_od_time(od_time[['o', 'd']].values, time_matrix)
            errors.append((i, (round(np.mean(abs(od_time[self.api_time_col] - od_time['routing_time']) / 60), 2))))

            if log_error:
                print(i, errors[-1][1])
            if errors[-1][1]<=gap_limit:
                break
        if num_cores > 1:
            ray.shutdown()
        # last time for consistency
        edges = df[['a', 'b', 'new_time']].values  # to build the index once and for all
        sparse, _ = sparse_matrix(edges, index=index)
        time_matrix, predecessors = parallel_dijkstra(sparse,
                                                      directed=True,
                                                      indices=zones,
                                                      return_predecessors=True,
                                                      num_core=num_cores,
                                                      keep_running=True)
        od_time['routing_time'] = _get_od_time(od_time[['o', 'd']].values, time_matrix)
        errors.append((i + 1, (round(np.mean(abs(od_time[self.api_time_col] - od_time['routing_time']) / 60), 2))))
        print(i + 1, errors[-1][1])
       
        print(round(100 * len(df[df[self.ff_time_col] != df['new_time']]) / len(df), 1), '% of links used')
        
        self.road_links = pd.concat([self.road_links, df[['new_time']]], axis=1).rename(columns={'new_time': self.api_time_col})
        #self.road_links.loc[self.road_links[self.ff_time_col] == self.road_links[self.api_time_col], self.api_time_col] = np.nan
        self.road_links[self.api_speed_col] = self.road_links['length'] / self.road_links[self.api_time_col] * 3.6
        self.od_time['routing_time'] = od_time['routing_time']
        
        return errors


@jit(nopython=True, locals={'predecessors': nb.int32[:, ::1], 'time_matrix': nb.float64[:, ::1]}, parallel=True)
def _fast_assign_time(odt, predecessors, time_matrix, times, counter):
    # this function use parallelization (or not).nb.set_num_threads(num_cores)
    # volumes is a numba dict with all the key initialized
    for i in nb.prange(len(odt)):
        origin = nb.int32(odt[i, 0])
        destination = nb.int32(odt[i, 1])
        t = odt[i, 2]
        path = get_path(predecessors, origin, destination)
        path = list(zip(path[:-1], path[1:]))
        tot_time = time_matrix[origin, destination]
        new_t = t / tot_time
        for key in path:
            count = counter[key]
            times[key] = (times[key] * count + new_t) / (count + 1)
            counter[key] = count + 1
    return times


def _assign_time(odt, predecessors, time_matrix, time_sparse_keys, reversed_index):
    # create dict to create an average of time on each link. 
    # the average in weighted by the links_time/total_time
    numba_times = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, 2),
        value_type=nb.types.float64)

    numba_counter = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, 2), 
        value_type=nb.types.int64)

    for key in time_sparse_keys:
        numba_times[key] = 0
        numba_counter[key] = 0

    # assign volumes from each od
    times = dict(_fast_assign_time(odt, predecessors, time_matrix, numba_times, numba_counter))
    
    ab_times = {(reversed_index[k[0]], reversed_index[k[1]]): t
                for k, t in times.items()}
    return ab_times


def _get_od_time(od, time_matrix):
    times = []
    for origin, destination in od:
        times.append(time_matrix[origin, destination])
    return times


# vizualisation
def plot_correlation(x, y, alpha=0.1, xlabel='actual time (mins)', ylabel='time interpolation (mins)', title='comparaison temps OD (jaune = 5% des points)'):
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from sklearn.metrics import r2_score
    
    errors = abs(x - y)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'mins.')

    x = x.values
    y = y.values
    regr = linear_model.LinearRegression(fit_intercept=False)

    regr.fit(x[:, np.newaxis], y)
    # y_pred = regr.predict(x[:, np.newaxis])
    r2 = r2_score(y, x)
    slope = regr.coef_[0]
    intercept = 0

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.scatter(x, y,
                s=10,
                c=errors,
                norm=matplotlib.colors.Normalize(vmin=0, vmax=errors.quantile(0.95), clip=True),
                alpha=alpha)

    maxvalue = max(max(x), max(y))
    plt.plot([0, maxvalue], [0, maxvalue], '--r', alpha=0.5)
    mean = np.mean(x - y)
    median = np.median(x - y)
    sigma = np.std(x - y)

    textstr = '\n'.join((
        r'$\mathrm{slope}=%.2f$' % (slope, ),
        r'$\mathrm{intercept}=%.2f$' % (intercept, ),
        r'$\mathrm{R^2}=%.2f$' % (r2, ),
        r'$\mathrm{mean}=%.2f$mins' % (mean, ),
        r'$\mathrm{median}=%.2f$mins' % (median, ),
        r'$\sigma=%.2f$mins' % (sigma, )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    plt.plot(x, x * slope + intercept, 'r', alpha=0.5)
    # plt.plot(x_worst,x_worst*slope_worst+intercept,'r',alpha=0.3)

    plt.grid(True, 'major', linestyle='-', axis='both')
    ax.set_axisbelow(True)
    plt.xlim([0, max(x)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
