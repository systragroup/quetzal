import networkx as nx
import numpy as np
import pandas as pd
from quetzal.engine.pathfinder import sparse_los_from_nx_graph
from syspy.assignment import raw as raw_assignment
from tqdm import tqdm



def default_bpr(links,flow='flow'):
    alpha = 0.15
    limit=10
    beta = 4
    V = links[flow]
    Q = links['capacity']
    t0 = links['time']
    res = t0 * (1 + alpha*np.power(V/Q, beta))
    res.loc[res>limit*t0]=limit*t0
    return res





def smock(links,flow='flow'):
    V = links[flow]
    Qs = links['capacity']
    t0 = links['time']
    return t0 * np.exp(V/Qs)


def jam_time(links, vdf={'default_bpr': default_bpr, 'smock': smock},flow='flow'):
    # vdf is a {function_name: function } to apply on links 
    keys = set(links['vdf'])
    missing_vdf = keys - set(vdf.keys())
    assert len(missing_vdf) == 0, 'you should provide methods for the following vdf keys' + missing_vdf
    return pd.concat([vdf[key](links,flow) for key in keys])

def z_prime(links, phi):
    # min sum(on links) integral from 0 to formerflow + φΔ of Time(f) df
    # approx constant + jam_time(FormerFlow) x φΔ + 1/2 (jam_time(Formerflow + φΔ) - jam_time(FormerFlow) ) x φΔ
    # Δ = links['aux_flow'] - links['former_flow']
    delta = links['aux_flow'] - links['former_flow']
    links['new_flow'] = delta*phi+links['former_flow']
    #z = (jam_time(links,vdf={'default_bpr': default_bpr_phi},flow='delta',phi=phi) - links['jam_time']) / (links['delta']*phi + links['former_flow'])
    t_f = jam_time(links,vdf={'default_bpr': default_bpr},flow='former_flow')
    t_del = jam_time(links,vdf={'default_bpr': default_bpr},flow='new_flow')
    z = t_f*delta*phi + (t_del-t_f)*delta*phi*0.5

    return np.ma.masked_invalid(z).sum()    

def find_phi(links, inf=0, sup=1, tolerance=1e-5):
    #solution inf plus petite que sup, on diminue la limit sup
    a = z_prime(links,inf)
    b = z_prime(links,sup)
    m = (inf + sup) / 2
    c = z_prime(links,m)
    limits = [(a,inf),(b,sup),(c,m)]
    limits.sort()
    # loop infini. les deux bornes sont dans des minimims locales. prend le min des deux.
    if inf == limits[0][1] and sup == limits[1][1]:
        print('loop')
        return (inf+sup)/2
    inf=limits[0][1]
    sup=limits[1][1]   
    if abs(sup-inf)<=tolerance:
        return inf
    return find_phi(links, inf, sup, tolerance)

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

class RoadPathFinder:
    def __init__(self, model):
        self.zones = model.zones.copy()
        self.road_links = model.road_links.copy()
        self.zone_to_road = model.zone_to_road.copy()
        try:
            self.volumes = model.volumes.copy()
        except AttributeError:
            pass

    def aon_road_pathfinder(
        self,
        time='time',
        ntleg_penalty=1e9,
        cutoff=np.inf,
        access_time='time',
        **kwargs,
    ):
        road_links = self.road_links
        road_links['index'] = road_links.index
        indexed = road_links.set_index(['a', 'b']).sort_index()
        ab_indexed_dict = indexed['index'].to_dict()

        road_graph = nx.DiGraph()
        road_graph.add_weighted_edges_from(
            self.road_links[['a', 'b', time]].values.tolist()
        )
        zone_to_road = self.zone_to_road.copy()
        zone_to_road['time'] = zone_to_road[access_time]
        zone_to_road = zone_to_road[['a', 'b', 'direction', 'time']]
        zone_to_road.loc[zone_to_road['direction'] == 'access', 'time'] += ntleg_penalty
        road_graph.add_weighted_edges_from(
            zone_to_road[['a', 'b', 'time']].values.tolist()
        )

        def node_path_to_link_path(road_node_list, ab_indexed_dict):
            tuples = list(zip(road_node_list[:-1], road_node_list[1:]))
            road_link_list = [ab_indexed_dict[t] for t in tuples]
            return road_link_list

        def path_to_ntlegs(path):
            try:
                return [(path[0], path[1]), (path[-2], path[-1])]
            except IndexError:
                return []

        los = sparse_los_from_nx_graph(
            road_graph,
            pole_set=set(self.zones.index),
            cutoff=cutoff + ntleg_penalty,
            **kwargs
        )
        los['node_path'] = los['path'].apply(lambda p: p[1:-1])
        los['link_path'] = los['node_path'].apply(
            lambda p: node_path_to_link_path(p, ab_indexed_dict)
        )
        los['ntlegs'] = los['path'].apply(path_to_ntlegs)
        los.loc[los['origin'] != los['destination'], 'gtime'] -= ntleg_penalty
        self.car_los = los.rename(columns={'gtime': 'time'})

    def frank_wolfe_step(
        self,
        iteration=0,
        log=False,
        speedup=True,
        volume_column='volume_car',
        vdf={'default_bpr': default_bpr, 'smock': smock},
        **kwargs,
    ):
        links = self.road_links  # not a copy

        # a
        links['eq_jam_time'] = links['jam_time'].copy()
        links['jam_time'] = jam_time(links, vdf=vdf)

        # b
        self.aon_road_pathfinder(time='jam_time', **kwargs)
        merged = pd.merge(
            self.volumes,
            self.car_los,
            on=['origin', 'destination']
        )
        auxiliary_flows = raw_assignment.assign(
            merged[volume_column],
            merged['link_path']
        )

        auxiliary_flows.columns = ['flow']
        links['aux_flow'] = auxiliary_flows['flow']
        links['aux_flow'].fillna(0, inplace=True)
        links['former_flow'] = links['flow'].copy()
        # c
        phi = 2 / (iteration + 2)
        if iteration > 0 and speedup:
            phi = find_phi(links)
        if phi == 0:
            return True
        if log:
            print('step: %i ' % iteration, 'moved = %.1f %%' % (phi * 100))

        self.car_los['iteration'] = iteration
        self.car_los['phi'] = phi

        links['flow'] = (1 - phi) * links['flow'] + phi * links['aux_flow']
        links['flow'].fillna(0, inplace=True)
        return False  # fin de l'algorithme

    def process_car_los(self, car_los_list):
        df = pd.concat(car_los_list).sort_values('iteration')
        phi_series = df.groupby('iteration')['phi'].first()
        phi_series = phi_series.loc[phi_series > 0]

        # will not work if road_links.index have mixed types
        groupby = df.groupby(df['link_path'].apply(lambda l: tuple(l)))
        iterations = groupby['iteration'].apply(lambda s: tuple(s))
        los = groupby.first()
        los['iterations'] = iterations

        def path_weight(iterations):
            w = 0
            for i in phi_series.index:
                phi = phi_series[i]
                if i in iterations:
                    w = w + (1 - w) * phi
                else:
                    w = w * (1 - phi)
            return w

        combinations = {
            i: path_weight(i)
            for i in set(los['iterations'].apply(lambda l: tuple(l)))
        }

        # weight
        los['weight'] = los['iterations'].apply(lambda l: combinations[l])

        # ntleg_time
        time_dict = self.zone_to_road.set_index(['a', 'b'])['time'].to_dict()
        los['ntleg_time'] = los['ntlegs'].apply(lambda p: sum([time_dict[l] for l in p]))

        # equilibrium_jam_time
        time_dict = self.road_links['eq_jam_time'].to_dict()
        los['link_eq_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['eq_time'] = los['link_eq_time'] + los['ntleg_time']

        # actual_time
        time_dict = self.road_links['jam_time'].to_dict()
        los['link_actual_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['actual_time'] = los['link_actual_time'] + los['ntleg_time']

        # free_time
        time_dict = self.road_links['time'].to_dict()
        los['link_free_time'] = los['link_path'].apply(lambda p: sum([time_dict[l] for l in p]))
        los['free_time'] = los['link_free_time'] + los['ntleg_time']
        return los.reset_index(drop=True)

    def get_relgap(self, car_los):
        los = car_los.copy()
        los = pd.merge(los, self.volumes, on=['origin', 'destination'])
        min_time = los.groupby(['origin', 'destination'], as_index=False)['actual_time'].min()
        los = pd.merge(los, min_time, on=['origin', 'destination'], suffixes=['', '_minimum'])
        los['delta'] = los['actual_time'] - los['actual_time_minimum']
        gap = (los['delta'] * los['weight'] * los['volume_car']).sum()
        total_time = (los['actual_time_minimum'] * los['weight'] * los['volume_car']).sum()
        return gap / total_time

    def frank_wolfe(
        self,
        all_or_nothing=False,
        reset_jam_time=True,
        maxiters=20,
        tolerance=0.01,
        log=False,
        vdf={'default_bpr': default_bpr, 'smock': smock},
        *args,
        **kwargs
    ):
        if all_or_nothing:
            self.aon_road_pathfinder(*args, **kwargs)
            return
        if 'vdf' in self.road_links.columns:
            self.road_links['vdf'] = 'default_bpr'
        assert 'capacity' in self.road_links.columns

        if reset_jam_time:
            self.road_links['flow'] = 0
            self.road_links['jam_time'] = self.road_links['time']

        car_los_list = []
        for i in range(maxiters):
            done = self.frank_wolfe_step(iteration=i, log=log, vdf=vdf, *args, **kwargs)
            c = self.car_los
            car_los_list.append(c)

            los = self.process_car_los(car_los_list)
            relgap = self.get_relgap(los)
            if log:
                print('relgap = %.1f %%' % (relgap * 100))
            if i > 0:
                if done or relgap < tolerance:
                    break
        self.car_los = los.reset_index(drop=True)
