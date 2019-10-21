import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def rank_paths(paths, by='utility'):

    assert paths[by].isnull().sum() == 0

    columns = ['origin', 'destination', 'route_type']
    sorted_paths = paths.sort_values(by=columns + [by], ascending=True)

    flat = []
    for i in sorted_paths.groupby(columns)[by].count():
        flat += list(range(i))

    sorted_paths['rank'] = flat
    return sorted_paths

def nest_probabilities(utilities, phi=1):
    exponential_df = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = exponential_df.T.sum()
    probability_df = exponential_df.apply(lambda s: s / exponential_s)
    return probability_df

def nest_utility(utilities, phi=1):
    exponential_df = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = exponential_df.T.sum()
    emu = np.log(exponential_s)
    composite_utility = phi * emu
    return composite_utility

def plot_nests(nests):
    g = nx.DiGraph(nests)
    root = [n for n in g.nodes if g.in_degree(n) == 0][0]
    lengths = nx.single_source_shortest_path_length(g, root)
    pos = {}
    levels = [0] * (max(lengths.values()) + 1)
    for key, x in lengths.items():
        pos[key] = [levels[x],  - x]
        levels[x] += 1
    
    plot = plt.axes()
    nx.draw(
        g, 
        pos=pos, 
        node_color='white', 
        alpha=1, 
        node_size=1000,
        arrows=False,
        edge_color='green',
        font_size=15,
        font_weight='normal',
        labels={k: k for k in g.nodes},
        axes=plot
    )
    return plot

def nested_logit_from_paths(paths, mode_nests=None, phi=None, verbose=False):
    """
    mode_nests = {
        'modes': ['pt', 'car', 'walk'],
        'pt':['massive','bus'],
        'massive':['subway', 'tram', 'rail']
    }

    phi = {
        'modes': 1,
        'pt': 0.75,
        'car': 0.75,
        'walk': 0.75,
        'massive': 0.5,
        'subway': 0.25,
        'tram': 0.25,
        'rail': 0.25,
        'bus':0.5
    }
    """
    
    if mode_nests is None:
        mode_nests = {'root': list(set(paths['route_type']))}

    g = nx.DiGraph(mode_nests)
    root = [n for n in g.nodes if g.in_degree(n) == 0][0]
    lengths = nx.single_source_shortest_path_length(g, root)
    
    if phi is None:
        phi = {mode:1 for mode in g.nodes}

    # fill phi_dict if is not complete (make it collapse)
    def recursive_phi(mode):
        try: 
            return phi[mode]
        except KeyError:
            parent = list(nx.neighbors(g.reverse(), mode))[0]
            phi[mode] = recursive_phi(parent)
            return phi[mode]
    phi = {n: recursive_phi(n) for n in g.nodes}
    
    ascending_modes = []
    depth = max(g.out_degree(n) for n in g.nodes)
    for degree in range(depth, -1, -1):
        leaf_modes = [n for n in g.nodes if lengths[n] == degree]
        ascending_modes += leaf_modes
    descending_modes = list(reversed(ascending_modes))
        
    # rank_utilities
    paths = rank_paths(paths)
    stack = paths.set_index(
        ['route_type','origin', 'destination', 'rank']
        )['utility'].sort_index()
    rank_utilities= stack.unstack(['route_type',  'rank']).T.sort_index().T
    rank_utilities.fillna(-np.inf, inplace=True)
    mode_utilities = pd.DataFrame(index=rank_utilities.index)
    mode_utilities.columns.name = 'route_type'

    # initialize all utilities at -inf
    for mode in ascending_modes:
        if mode not in mode_utilities.columns:
            mode_utilities[mode] = -np.inf

    # aggregate rank_utilities
    for mode in list(rank_utilities.columns.levels[0]):
        if verbose:
            print('path utilities', mode, phi[mode], '->', mode)
        mode_utilities[mode] = nest_utility(
            rank_utilities[mode], 
            phi[mode]
        )

    # propagate utilities to higher modes (bottom -> up)
    for mode in ascending_modes :
        children = list(nx.neighbors(g, mode))
        if len(children):
            if verbose:
                print('mode utilities', children, phi[mode], '->', mode)
            mode_utilities[mode] = nest_utility(
                mode_utilities[children], 
                phi=phi[mode]
            )
    mode_utilities = mode_utilities[descending_modes]

    # initialize probabilities
    mode_probabilities = pd.DataFrame(index=rank_utilities.index)
    mode_probabilities.columns.name = 'route_type'

    # propagate probabilities
    mode_probabilities[descending_modes[0]] = 1 # root mode
    for mode in descending_modes:
        children = list(nx.neighbors(g, mode))
        if len(children):
            if verbose:
                print('mode probabilities', mode, phi[mode],'->', children)
            partial = nest_probabilities(
                utilities=mode_utilities[children], 
                phi=phi[mode]
            )
            mode_probabilities[children] =  partial.multiply(
                mode_probabilities[mode], 
                axis='index'
            )
            
    mode_probabilities = mode_probabilities[descending_modes].fillna(0)

    # rank_probabilities
    rank_probabilities = rank_utilities.copy()
    for mode in list(rank_utilities.columns.levels[0]):
        if verbose:
            print('path probabilities', mode, phi[mode], '->', mode)
        rank_probabilities[mode] = nest_probabilities(
            rank_utilities[mode], 
            phi[mode]
        )
          
    # merge assignment probablities on paths
    rank_probabilities_s = rank_probabilities.stack().stack()
    rank_probabilities_s.name = 'assignment_share'

    mode_probabilities_s = mode_probabilities.stack()
    mode_probabilities_s.name = 'modal_split_share'

    merged = pd.merge(
        rank_probabilities_s.reset_index(),
        mode_probabilities_s.reset_index(),
        on=['origin', 'destination', 'route_type']
    )
    stack = rank_probabilities_s.reset_index()
    merged['probability'] = merged['assignment_share'] * merged['modal_split_share']
    
    merge_columns = ['origin', 'destination', 'route_type', 'rank']

    paths['index'] = paths.index
    paths = pd.merge(
        paths.drop('probability', axis=1, errors='ignore'), 
        merged[merge_columns + ['probability']], 
        on=merge_columns, 
        suffixes=['_old',  ''],
        how='left'
    ).set_index('index')
    return paths,  mode_utilities.reset_index(), mode_probabilities.reset_index()