import json
import pandas as pd
from tqdm import tqdm


def read_var(file='parameters.xlsx', scenario='base'):
    parameter_frame = pd.read_excel(file)
    try:
        types = parameter_frame.set_index(
            ['category', 'parameter']
        )['type'].dropna().to_dict()
    except KeyError:
        types = dict()
    parameter_frame.drop(['description', 'unit', 'type'], axis=1, errors='ignore', inplace=True)
    parameter_frame.set_index(['category', 'parameter'], inplace=True)
    parameter_frame.dropna(how='all', inplace=True)
    for c in parameter_frame.columns:
        parent = parameter_frame[c][('general', 'parent')]
        parameter_frame[c] = parameter_frame[c].fillna(parameter_frame[parent])
    var = parameter_frame[scenario]
    for k, v in types.items():
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
