import pandas as pd
def read_var(file='parameters.xlsx', scenario='base'):
    parameter_frame = pd.read_excel(file)
    parameter_frame.drop('description', axis=1, errors='ignore', inplace=True)
    parameter_frame.set_index(['category','parameter'], inplace=True)
    for c in parameter_frame.columns:
        parent = parameter_frame[c][('general', 'parent')]
        parameter_frame[c] = parameter_frame[c].fillna(parameter_frame[parent])
    return parameter_frame[scenario]