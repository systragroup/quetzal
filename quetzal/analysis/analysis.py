# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from quetzal.analysis import on_demand

from tqdm import tqdm

def tp_summary(links, shared):

    links = links.copy()
    links['index'] = links.index
    line_link_dict = links.groupby('trip_id')['index'].agg(
        lambda s: set(s)).to_dict()
    line_list = list(line_link_dict.keys())
    link_set = set(links['index'])

    df = shared.copy()

    for key, value in line_link_dict.items():
        df[key] = df['path'].apply(
            lambda p: bool(len(value.intersection(p)))) * df['volume_pt']

    # walk
    df['all_walk'] = df['path'].apply(
        lambda p: len(link_set.intersection(p)) == 0)

    df['transfer'] = df.loc[:, line_list].astype(bool).T.sum()
    df['exclusivity'] = 1 / df['transfer']
    # we do not want to reach infinite exclusivity rates where no line is used
    df.loc[df['transfer'] == 0, 'exclusivity'] = 0

    def average_value(line, column):
        try:
            return np.average(df[column], weights=df[line])
        except ZeroDivisionError:  # Weights sum to zero, can't be normalized
            return 0

    #  transfer
    transfers = pd.Series(
        [average_value(line, 'transfer') for line in line_list],
        index=line_list
    )
    exclusivities = pd.Series(
        [average_value(line, 'exclusivity') for line in line_list],
        index=line_list
    )
    right = pd.DataFrame({'transfer': transfers, 'exclusivity': exclusivities})

    # boardings
    lines = pd.DataFrame(
        df[line_list].sum().sort_index(),
        columns=['boardings']
    )

    # passenger_km
    links['passenger_km'] = links['load'] * links['length'] / 1000

    grouped = links.groupby(['trip_id'])

    sums = grouped[['passenger_km', 'length', 'time']].sum()
    firsts = grouped[['line_color']].first()
    means = links.groupby('trip_id')[['headway', 'load']].mean()
    maxs = links.groupby('trip_id')[['load']].max()

    lines = pd.concat(
        [
            lines,
            sums,
            firsts,
            means.rename(columns={'load': 'mean_load'}),
            maxs.rename(columns={'load': 'max_load'})
        ],
        axis=1
    )

    lines = pd.concat([lines, right], axis=1)

    lines[['name']] = pd.DataFrame(
        [s.split('_')[0] for s in lines.index],
        index=lines.index
    )
    lines['color'] = lines['line_color']
    return lines

def analysis_path(path, vertex_type):

    boarding_links = []
    alighting_links = []
    boardings = []
    alightings = []
    node_path = []
    link_path = []
    footpaths = []
    ntlegs = []
    
    for i in range(1, len(path)-1):
    
        from_node = path[i-1]
        node = path[i]
        to_node = path[i+1]

        from_vtype = vertex_type.get(from_node)
        vtype = vertex_type.get(node)
        to_vtype = vertex_type.get(to_node)

        if vtype == 'link':
            link_path.append(node)
            if to_vtype != 'link':
                alighting_links.append(node)
            if from_vtype != 'link':
                boarding_links.append(node)

        elif vtype == 'node': 
            node_path.append(node)
            if from_vtype == 'link':
                alightings.append(node)
            elif from_vtype == 'centroid':
                ntlegs.append((from_node, node))
            elif from_vtype == 'node':
                footpaths.append((from_node, node))
            if to_vtype == 'link':
                boardings.append(node)
            elif to_vtype == 'centroid':
                ntlegs.append((node, to_node))
                
    transfers = [n for n in boardings if n in alightings]
                
    to_return = {
        'boardings': boardings,
        'alightings': alightings,
        'node_path': node_path,
        'link_path': link_path,
        'footpaths': footpaths,
        'ntlegs': ntlegs,
        'transfers': transfers,
        'boarding_links': boarding_links,
        'alighting_links': alighting_links
    }
                
    return to_return

def path_analysis_od_matrix(
    od_matrix, 
    links, 
    nodes, 
    centroids, 
    agg={'link_path': ['time', 'length']},
):
    vertex_sets = {
        'node': set(nodes.index),
        'link': set(links.index),
        'centroid': set(centroids.index)
    }
    vertex_type = {}
    for vtype, vset in vertex_sets.items():
        for v in vset:
            vertex_type[v] = vtype
    link_dict = links.to_dict()


    analysis_path_list = [
        analysis_path(p, vertex_type) 
        for p in tqdm(list(od_matrix['path']), desc='path_analysis')
    ]

    analysis_path_dataframe =  pd.DataFrame(
        analysis_path_list, 
        index=od_matrix.index
    )

    df = pd.concat([od_matrix, analysis_path_dataframe], axis=1)
    df['all_walk'] = df['link_path'].apply(lambda p: len(p) == 0)

    for key, extensive_columns in agg.items():
        for column in extensive_columns:
            column_dict = link_dict[column]
            df[column + '_' + key] = df[key].apply(
                lambda p: sum([column_dict[i] for i in p])
            )

    return df

def volume_analysis_od_matrix(od_matrix):

    df = od_matrix.copy()

    df['volume_car'] = df['volume'] - df['volume_pt']
    df['volume_walk'] = df['volume_pt'] * df['all_walk'].astype(int)
    df['volume_pt'] = df['volume_pt'] * (1-df['all_walk'].astype(int))

    df['volume_duration_car'] = df['volume_car'] * df['duration_car']
    df['volume_duration_pt'] = df['volume_pt'] * df['duration_pt']

    df['volume_distance_car'] = df['volume_car'] * df['distance_car']
    df['volume_distance_pt'] = df['volume_pt'] * df['distance_pt']

    df['pivot'] = np.ones(len(df))

    return df

def analysis_od_matrix(od_matrix, links, nodes, centroids):
    paod_matrix = path_analysis_od_matrix(od_matrix, links, nodes, centroids)
    return volume_analysis_od_matrix(paod_matrix)


def analysis_tp_summary(lines, period_duration=1):
    lines = lines.copy()
    lines['vehicles'] = 1 / lines['headway'] * period_duration
    lines['exclusive_passenger'] = lines['boardings'] * lines['exclusivity']
    lines['vehicles_distance'] = lines['vehicles'] * lines['length'] / 1000

    return lines


def economic_series(od_stack, lines, period_length=1):

    df = pd.concat([

        od_stack[
            [
                'volume',
                'volume_pt',
                'volume_car',
                'volume_walk',
                'volume_duration_car',
                'volume_duration_pt',
                'volume_distance_car',
                'volume_distance_pt'
            ]
        ].sum(),
        lines[
            [
                'boardings',
                'passenger_km',
                'vehicles',
                'exclusive_passenger',
                'vehicles_distance'

            ]
        ].sum()
    ])

    return df
    

# function definition
checkpoint_demand = on_demand.checkpoint_demand
