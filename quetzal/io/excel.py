import json
import os

import pandas as pd
from tqdm import tqdm


def read_var(file='parameters.xlsx', scenario='base', period=None, return_ancestry=False):
    parameter_frame = pd.read_excel(file, sheet_name='parameters').dropna(axis=1, how='all')
    try:
        types = parameter_frame.set_index(
            ['category', 'parameter']
        )['type'].dropna().to_dict()
    except KeyError:
        types = dict()
    
    if period is not None:
        mask  = ((parameter_frame['period'].isna()) | 
                (parameter_frame['period'].str.casefold() == period.casefold()))
        parameter_frame = parameter_frame[mask]
        parameter_frame.sort_values('period', inplace=True)
        parameter_frame.drop_duplicates(subset=['category','parameter'], inplace=True)
        parameter_frame.sort_index(inplace=True)
    parameter_frame.drop(['description', 'desc', 'unit', 'type', 'period'], axis=1, errors='ignore', inplace=True)
    parameter_frame.drop_duplicates(['category', 'parameter'], inplace=True)
    parameter_frame.set_index(['category', 'parameter'], inplace=True)
    if return_ancestry:
        ancestry = get_ancestry(parameter_frame, scenario=scenario)
    for c in parameter_frame.columns:
        parent = parameter_frame[c][('general', 'parent')]
        parameter_frame[c] = parameter_frame[c].fillna(parameter_frame[parent])
    var = parameter_frame[scenario]
    for k, v in types.items():
        try:
            if v == 'float':
                var.loc[k] = float(var.loc[k])
            elif v == 'int':
                var.loc[k] = int(var.loc[k])
            elif v == 'bool':
                var.loc[k] = bool(var.loc[k])
            elif v == 'str':
                var.loc[k] = str(var.loc[k])
            elif v == 'json':
                var.loc[k] = json.loads(var.loc[k])
        except KeyError:
            pass
    if return_ancestry:
        return var, ancestry
    return var


def merge_files(
    parameters_filepath=r'inputs/parameters.xlsx',
    scenario_filepath=r'model/{scenario}/stacks.xlsx',
    merged_filepath=r'outputs/stacks.xlsx'
):
    parameters = pd.read_excel(parameters_filepath)
    scenarios = [c for c in parameters.columns if c not in {'category', 'parameter'}]

    base = scenarios[0]
    base_dict = pd.read_excel(scenario_filepath.format(scenario=base), sheet_name=None)
    pool = {key: [] for key in base_dict.keys()}

    notfound = []
    for scenario in tqdm(scenarios, desc='reading'):
        try:
            df_dict = pd.read_excel(scenario_filepath.format(scenario=scenario), sheet_name=None)
            for key, value in df_dict.items():
                value['scenario'] = scenario
                col = [c for c in value.columns if 'scenario' not in c]
                col.insert(-1, 'scenario')
                value = value[col]
                pool[key].append(value)
        except FileNotFoundError:
            notfound.append(scenario)

    stacks = {k: pd.concat(v) for k, v in pool.items()}
    with pd.ExcelWriter(merged_filepath) as writer:  # doctest: +SKIP
        for name, stack in tqdm(stacks.items(), desc='writing'):
            stack.to_excel(writer, sheet_name=name, index=False)

def get_ancestry(parameter_frame, scenario='base'):
    child = scenario
    ancestry = [child]
    while True:
        parent = parameter_frame.loc[('general','parent'), child]
        if parent == child: break
        ancestry.append(parent)
        child = parent
    return ancestry

def get_filepath(filepath, ancestry=['base'], log=True):
    for scen in ancestry:
        relpath = filepath.format(s=scen)
        if os.path.exists(relpath):
            if log: 
                print(f"specified file found: {relpath}")
            return relpath
        if log:
            print(f"{relpath} does not exist")
    if log:
        print("specified file or input path does not exist")
    return None
