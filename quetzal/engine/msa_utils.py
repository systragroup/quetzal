from numba import jit
import pandas as pd
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path, get_path, sparse_matrix, parallel_dijkstra
import numba as nb





def jam_time(links:pd.DataFrame, vdf,flow:str='flow',time_col:str='time') -> pd.Series:
    # vdf is a {function_name: function } to apply on links 
    # 3 types of functions are accepted:
    # 1) str that will be evaluated with pandas.eval
    # 2) python function of the form func(df, flow_col, time_col).
    # 3) Jit function with single input matrix ['alpha','beta','limit',flow_col ,time_col,'penalty','capacity','base_flow']]
    keys = set(links['vdf'])
    missing_vdf = keys - set(vdf.keys())

    assert len(missing_vdf) == 0, 'you should provide methods for the following vdf keys' + str(missing_vdf)
    for key in keys:
        if type(vdf[key])==str:
            str_expression = vdf[key].replace('flow',flow).replace('time',time_col)
            links.loc[links['vdf']==key,'result'] = links.loc[links['vdf']==key].eval(str_expression)
        elif type(vdf[key]).__name__=='function': #normal python function.
            links.loc[links['vdf']==key,'result'] = vdf[key](links.loc[links['vdf']==key], flow, time_col) 
        else: # numba function.
            links.loc[links['vdf']==key,'result'] = vdf[key](links.loc[links['vdf']==key,['alpha','beta','limit',flow,time_col,'penalty','capacity','base_flow']].values) 
        
    return links['result']


def z_prime(links, vdf, phi, **kwargs):
    # min sum(on links) integral from 0 to formerflow + φΔ of Time(f) df
    # approx constant + jam_time(FormerFlow) x φΔ + 1/2 (jam_time(Formerflow + φΔ) - jam_time(FormerFlow) ) x φΔ
    # Δ = links['aux_flow'] - links['former_flow']
    delta = (links['auxiliary_flow'] - links['flow']).values
    links['new_flow'] = delta*phi+links['flow']
    #z = (jam_time(links,vdf={'default_bpr': default_bpr_phi},flow='delta',phi=phi) - links['jam_time']) / (links['delta']*phi + links['former_flow'])
    t_f = jam_time(links,vdf=vdf,flow='flow',**kwargs)
    t_del = jam_time(links,vdf=vdf,flow='new_flow',**kwargs)
    links['t_f'] = t_f
    links['t_del'] = t_del
    t_f = links['t_f'].values
    t_del = links['t_del'].values
    z = t_f*delta*phi + (t_del-t_f)*delta*phi*0.5

    return np.ma.masked_invalid(z).sum()    


def find_phi(links,vdf, phi=0, step=0.5, num_it=10, **kwargs):
    a = z_prime(links, vdf, phi, **kwargs)
    for i in range(num_it):
        b = z_prime(links, vdf, phi + step,**kwargs)
        if b<a:
            phi+=step
            step=step/2
            a=b
        else: 
            step=-step/2
        if a+step<0:
            step=-step
    return phi


def get_zone_index(df,v,index):
    # INDEX 
    seta = set(df['a'])
    setb = set(df['b'])
    v = v.loc[v['origin'].isin(seta) & v['destination'].isin(setb)]

    sources = set(v['origin']).union(v['destination'])

    pole_list = sorted(list(sources))  # fix order
    source_list = [zone for zone in pole_list if zone in sources]

    zones = [index[zone] for zone in source_list]
    zone_index = dict(zip(pole_list, range(len(pole_list))))

    v['o'] = v['origin'].apply(zone_index.get)
    v['d'] = v['destination'].apply(index.get)
    
    return v, zones


'''
def assign_volume(odv,predecessors,reversed_index):
    volumes={}
    for origin, destination, volume in odv: 
        path = get_edge_path(get_node_path(predecessors, origin, destination))
        for key in path:
            try:
                volumes[key] += volume
            except KeyError:
                volumes[key] = volume
    #return volumes            
    # ab_volume is the volume assigned to each link indexed by the (a, b) tuple
    ab_volumes = {
        (reversed_index[k[0]], reversed_index[k[1]]) : v 
        for k, v in volumes.items()
    }
    return ab_volumes
'''

@jit(nopython=True,locals={'predecessors':nb.int32[:,::1]},parallel=True) #parallel=True
def fast_assign_volume(odv,predecessors,volumes):
    # this function use parallelization (or not).nb.set_num_threads(num_cores)
    # volumes is a numba dict with all the key initialized
    for i in nb.prange(len(odv)): #nb.prange(len(odv)):
        origin = odv[i,0]
        destination = odv[i,1]
        v=odv[i,2]
        path = get_node_path(predecessors, origin, destination)
        path = list(zip(path[:-1], path[1:]))
        for key in path:
            volumes[key]+=v
    return volumes

def assign_volume(odv,predecessors,volumes_sparse_keys,reversed_index):
        # create a numba dict.
    numba_volumes = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.int64, 2), 
        value_type=nb.types.float64
        )
    for ind in volumes_sparse_keys:
        numba_volumes[ind]=0
    #assign volumes from each od
    volumes = dict(fast_assign_volume(odv,predecessors,numba_volumes))

    ab_volumes = {
        (reversed_index[k[0]], reversed_index[k[1]]) : v 
        for k, v in volumes.items()
        }
    return ab_volumes



def get_car_los(v,df,index,reversed_index,zones,ntleg_penalty,num_cores=1):
    car_los = v[['origin','destination','o','d']]
    edges = df['jam_time'].reset_index().values # build the edges again, useless
    sparse, _ = sparse_matrix(edges, index=index)
    dist_matrix, predecessors = parallel_dijkstra(sparse, directed=True, indices=zones, return_predecessors=True, num_core=num_cores)
    odlist = list(zip(car_los['o'].values, car_los['d'].values))
    time_dict = {(o,d):dist_matrix[o,d]-ntleg_penalty for o,d in odlist} # time for each od
    car_los['time'] = car_los.set_index(['o','d']).index.map(time_dict)

    path_dict = {}
    for origin, destination in odlist:
        path = get_path(predecessors, origin, destination)
        path = [*map(reversed_index.get, path)]
        path_dict[(origin,destination)] = path

    car_los.loc[car_los['origin'] == car_los['destination'], 'time'] = 0.0

    car_los['path'] = car_los.set_index(['o','d']).index.map(path_dict)
    car_los['gtime'] = car_los['time']

    car_los = car_los.drop(columns=['o','d'])

    return car_los
    
def find_beta(df,phi_1):
    # The Stiff is Moving - Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment from Mitradjieva maria

    b = [0,0,0]
    dk_1 = df['s_k-1'] - df['flow']
    dk_2 = phi_1 * df['s_k-1'] + (1 - phi_1) * df['s_k-2'] - df['flow']
    dk = df['auxiliary_flow'] - df['flow']
    # put a try here except mu=0 if we have a division by 0...
    mu = - sum( dk_2 * df['derivative'] * dk) / sum( dk_2 * df['derivative'] * ( df['s_k-2'] - df['s_k-1']  ) )
    mu = max(0,mu) # beta_k >=0
    # same try here.
    nu = - sum( dk_1 * df['derivative'] * dk ) / sum( dk_1 * df['derivative'] * dk_1 )  + ( mu * phi_1 / ( 1 - phi_1 ) )
    nu = max(0,nu)
    b[0] = 1 / ( 1+ mu + nu )
    b[1] = nu * b[0]
    b[2] = mu * b[0]
    return b


def get_derivative(df, vdf, flow_col='flow', h=0.001,**kwargs):
    df['x1'] = df[flow_col] + h
    df['x2'] = df[flow_col] - h
    df['x1'] = jam_time(df,vdf,flow='x1',**kwargs)
    df['x2'] = jam_time(df,vdf,flow='x2',**kwargs)
    return (df['x1'] - df['x2'])/(2*h)