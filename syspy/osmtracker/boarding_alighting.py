import pandas as pd
import numpy as np


def add_load_difference(df):
    boarding_cols = [c for c in df.columns if 'boardings_' in c]
    df['boardings'] = df[boarding_cols].T.sum()
    df.iloc[-1, df.columns.get_loc('number_passengers')] = 0
    df['load'] = df['boardings'].cumsum() - df['alightings'].cumsum()
    df['load_difference'] = df['number_passengers'] - df['load']
    df['load_difference'].fillna(0, inplace=True)

    return df


def _correct_ba_first(boarding_list, alightings, load_difference):
    new_boarding_list = []
    for b in boarding_list:
        b += load_difference / len(boarding_list)
        new_boarding_list.append(b)
    return new_boarding_list, alightings, load_difference


def _correct_ba_last(boarding_list, alightings, load_difference):
    alightings += load_difference
    return boarding_list, alightings, load_difference


def correct_ba(boarding_list, alightings, load_difference, first=False, last=False):

    # TODO: handle case where boarding or alighting data are missing

    if first:
        return _correct_ba_first(boarding_list, alightings, load_difference)
    elif last:
        return _correct_ba_last(boarding_list, alightings, load_difference)

    total_boardings = sum(boarding_list)
    new_boarding_list = []
    if load_difference > 0:  # pas assez de b ou trop de a
        alightings_to_remove = min(load_difference / 2, alightings / 2)
        boardings_to_add = load_difference - alightings_to_remove

        alightings -= alightings_to_remove
        for b in boarding_list:
            b += boardings_to_add / len(boarding_list)
            new_boarding_list.append(b)

    elif load_difference < 0:  # trop de b ou pas assez de a
        boardings_to_remove = min(abs(load_difference) / 2, total_boardings / 2)
        alightings_to_add = abs(load_difference) - boardings_to_remove

        alightings += alightings_to_add
        for b in boarding_list:
            b -= boardings_to_remove * b / total_boardings
            new_boarding_list.append(b)
    else:
        for b in boarding_list:
            new_boarding_list.append(b)

    return new_boarding_list, alightings, load_difference


def correct_ba_row(row, first, last):

    boarding_list = []
    boarding_cols = []
    for col in row.index:
        if 'boardings_' in col:
            boarding_list.append(row[col])
            boarding_cols.append(col)

    new_boarding_list, alightings, load_difference = correct_ba(
        boarding_list, row['alightings'], row['load_difference'], first, last)

    result = {}

    for c, b in zip(boarding_cols, new_boarding_list):
        result.update({c: b})
    result.update({'alightings': alightings})

    return pd.Series(result), load_difference


def correct_boarding_alighting_data(input_dict):
    df = pd.DataFrame(input_dict).T
    df = df.sort_values('stop_time').reset_index().rename(
        columns={'index': 'stop_id'}
    )
    df = df.replace(-1, np.nan)
    df = add_load_difference(df)

    new_df = pd.DataFrame()
    load_diff = 0
    for i, row in df.iterrows():
        if i == 0:
            first = True
        else:
            first = False
        if i == len(df) - 1:
            last = True
        else:
            last = False

        row['load_difference'] -= load_diff
        new_row, load_d = correct_ba_row(row, first, last)
        load_diff += load_d
        new_row.name = i
        new_df = new_df.append(new_row)

    new_df = new_df.merge(
        df[['stop_id', 'stop_time', 'number_passengers']],
        left_index=True, right_index=True
    )
    # fillna values for number passengers
    new_df = add_load_difference(new_df)
    new_df['number_passengers'] = new_df['load']
    new_df = new_df.drop(['load', 'load_difference', 'boardings'], 1)

    return new_df.set_index('stop_id').T.to_dict()
