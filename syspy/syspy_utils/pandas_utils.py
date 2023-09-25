import copy
import pandas as pd


def df_explode(df, column_to_explode):
    """
    Take a column with iterable elements, and flatten the iterable to one element
    per observation in the output table.
    Slow and therefore not adapted to huge df.

    :param df: A dataframe to explod
    :type df: pandas.DataFrame
    :param column_to_explode:
    :type column_to_explode: str
    :return: An exploded data frame
    :rtype: pandas.DataFrame
    """
    # Create a list of new observations
    new_observations = list()

    # Iterate through existing observations
    for row in df.to_dict(orient='records'):
        # Take out the exploding iterable
        explode_values = row[column_to_explode]
        del row[column_to_explode]
        # Create a new observation for every entry in the exploding iterable & add all of the other columns
        for explode_value in explode_values:
            # Deep copy existing observation
            new_observation = copy.deepcopy(row)
            # Add one (newly flattened) value from exploding iterable
            new_observation[column_to_explode] = explode_value
            # Add to the list of new observations
            new_observations.append(new_observation)
    # Create a DataFrame
    return_df = pd.DataFrame(new_observations)
    # Return
    return return_df


def groupby_weighted_average(df, groupby, columns, weight):
    """
    perform a weighted average on specified columns
    during a groupby operation

    :param df: A dataframe to group
    :type df: pandas.DataFrame
    :param groupby: column(s) to groupby
    :type groupby: str or list
    :param columns: column(s) to average
    :type columns: str or list
    :param weight: column to use as weight
    :type weight: str
    :return: A grouped dataframe with averaged columns
    :rtype: pandas.DataFrame
    """
    if not isinstance(columns, list):
        columns = [columns]
    new_columns = [(c, weight) for c in columns]
    df[new_columns] = pd.concat([df[c] * df[weight] for c in columns], axis=1)
    grouped = df.groupby(groupby)[new_columns].sum().div(df.groupby(groupby)['volume'].sum(), axis=0)
    grouped = grouped.rename(columns={(c, w): c for c, w in grouped.columns})
    return grouped
