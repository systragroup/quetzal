from numba import jit
import pandas as pd
import numpy as np
from quetzal.engine.pathfinder_utils import get_node_path, get_edge_path, get_path, sparse_matrix
from scipy.sparse.csgraph import dijkstra

def default_bpr(links,flow='flow'):
    alpha = 0.15
    limit=10
    beta = 4
    V = links[flow]
    Q = links['capacity']
    t0 = links['time']
    res = t0 * (1 + alpha*np.power(V/Q, beta))
    #res.loc[res>limit*t0]=limit*t0
    speed = links['length']/res *(60*60/1000)
    res.loc[speed<limit]=links['length']/limit *(60*60/1000)
    return res

def smock(links,flow='flow'):
    V = links[flow]
    Qs = links['capacity']
    t0 = links['time']
    return t0 * np.exp(V/Qs)

def free_flow(links,flow='flow'):
    t0 = links['time']
    return t0


def jam_time(links, vdf={'default_bpr': default_bpr},flow='flow'):
    # vdf is a {function_name: function } to apply on links 
    keys = set(links['vdf'])
    missing_vdf = keys - set(vdf.keys())
    assert len(missing_vdf) == 0, 'you should provide methods for the following vdf keys' + missing_vdf
    return pd.concat([vdf[key](links[links['vdf']==key],flow) for key in keys])


def z_prime(links,vdf, phi):
    # min sum(on links) integral from 0 to formerflow + φΔ of Time(f) df
    # approx constant + jam_time(FormerFlow) x φΔ + 1/2 (jam_time(Formerflow + φΔ) - jam_time(FormerFlow) ) x φΔ
    # Δ = links['aux_flow'] - links['former_flow']
    delta = (links['auxiliary_flow'] - links['flow']).values
    links['new_flow'] = delta*phi+links['flow']
    #z = (jam_time(links,vdf={'default_bpr': default_bpr_phi},flow='delta',phi=phi) - links['jam_time']) / (links['delta']*phi + links['former_flow'])
    t_f = jam_time(links,vdf=vdf,flow='flow')
    t_del = jam_time(links,vdf=vdf,flow='new_flow')
    links['t_f'] = t_f
    links['t_del'] = t_del
    t_f = links['t_f'].values
    t_del = links['t_del'].values
    z = t_f*delta*phi + (t_del-t_f)*delta*phi*0.5

    return np.ma.masked_invalid(z).sum()    

def find_phi(links,vdf, phi=0, step=0.5, num_it=10):
    a = z_prime(links,vdf,phi)
    for i in range(num_it):
        b = z_prime(links,vdf,phi+step)
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


def get_car_los(v,df,index,reversed_index,zones,ntleg_penalty):
    car_los = v[['origin','destination','o','d']]
    edges = df['jam_time'].reset_index().values # build the edges again, useless
    sparse, _ = sparse_matrix(edges, index=index)
    dist_matrix, predecessors = dijkstra(sparse, directed=True, indices=zones, return_predecessors=True)

    odlist = list(zip(car_los['o'].values, car_los['d'].values))
    time_dict = {(o,d):dist_matrix[o,d]-ntleg_penalty for o,d in odlist} # time for each od
    car_los['time'] = car_los.set_index(['o','d']).index.map(time_dict)

    path_dict = {}
    for origin, destination in odlist:
        path = get_path(predecessors, origin, destination)
        path = [*map(reversed_index.get, path)]
        path_dict[(origin,destination)] = path

    car_los['path'] = car_los.set_index(['o','d']).index.map(path_dict)
    car_los['gtime'] = car_los['time']

    car_los = car_los.drop(columns=['o','d'])

    return car_los



#old version
'''
def z_prime(row, phi):
    delta = row['aux_flow'] - row['former_flow']
    return (delta * row['time'] * (row['former_flow'] + phi * delta)).sum()


def find_phi(links, inf=0, sup=1, tolerance=1e-6):
    if z_prime(links, inf) > 0:
        print('fin: ', inf)
        return inf
    if z_prime(links, sup) < 0:
        return sup
    m = (inf + sup) / 2

    if (sup - inf) < tolerance:
        return m

    z_prime_m = z_prime(links, m)
    if z_prime_m == 0:
        return m
    elif z_prime_m < 0:
        inf = m
    elif z_prime_m > 0:
        sup = m
    return find_phi(links, inf, sup, tolerance)
'''