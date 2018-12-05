# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import json

from syspy.surveys.array_example import base


def get_array(array_class, verbose=False):
    selected  = base.copy()
    for level, factors in array_class.items():
        selected = selected .loc[
            selected [level] >= factors
        ].copy()
        
    first = selected.iloc[0]
    exponents = first['exponents']
    df = pd.DataFrame(json.loads(first['array']))
    df.columns = get_index(exponents)
    
    array_classes = selected.loc[
        selected['runs'] == first['runs'],
        'exponents'
    ]
    if verbose:
        print('required array class:')
        print(array_class)
        print('available array classes (using first one): ')
        print(array_classes)
    return df[get_index(array_class)]

def build_array(array_class, *args, **kwargs):
    try: 
        return get_array(array_class, *args, **kwargs)
    except ValueError: 
        i = 2
        m = max(array_class.keys())
        pop = array_class.pop(m)
        l = []
        for key in [int(m/i), i]:
            base = 0
            if key in array_class.keys(): 
                base = array_class[key]
            array_class[key] = base + 1 
            l.append('%ie%i' % (key, base))
            
        oa = get_array(array_class)
        oa['%ie%i'% (m, 0)] = oa[l[0]] + oa[l[1]] * int(m/i)
        return oa.drop(l, axis=1)

def get_index(exponents):
    index = []
    for level, factors in  exponents.items():
        for i in range( factors):
            c = str(level) + 'e' + str(i)
            index.append(c)
    return sorted(index)

def get_selection(array_class):
    selected  = base.copy()
    for level, factors in array_class.items():
        selected = selected .loc[
            selected [level] >= factors
        ].copy()
    return selected


def get_array_class(factors):
    array_class = {}
    join = {}
    
    for factor, levels in factors.items():
        key = len(levels)
        if key in array_class:
            array_class[key] += 1
        else:
            array_class[key] = 1
        
        factor_index = array_class[key] - 1
        join[str(key) + 'e' + str(factor_index)] = factor
    return array_class, join


def orthogonal_array(factors, *args, **kwargs):

    array_class, match = get_array_class(factors)

    df = build_array(array_class, *args, **kwargs)

    df.columns = [match[c] for c in df.columns]

    for factor, factor_list in factors.items():
        df[factor] = df[factor].apply(lambda v: factor_list[v])

    columns = list(set(df.columns))
    identity = pd.DataFrame(
        np.identity(len(columns)),
        columns=columns,
        index=columns
    )
    delta_corr = df.corr() - identity
    assert delta_corr.max().max() < 1e-9
    return df