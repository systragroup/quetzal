__author__ = 'qchasserieau'

import os
import pandas as pd
from simpledbf import Dbf5
from syspy.io.pandasdbf import exceltodbf_qc

def read_dbf(path, encoding='utf-8'):

    """
    read a .dbf file, temporary write it as a csv and return a pandas.core.frame.DataFrame object.

    :param path: The complete path to the .dbf to read
    :type path: str
    :param encoding: The codec to use when decoding text-based records
    :type encoding: str
    :return: a pandas.DataFrame corresponding to the .dbf file
    :rtype: pandas.core.frame.DataFrame

    .. warnings:: not all encodings are handled by this function
    """

    dbf_instance = Dbf5(path, codec=encoding)
    csv_name = path[:-4] + '_from_dbf'+'.csv'

    # clear the folder from the temporary csv if necessary
    if os.path.isfile(csv_name):
        os.remove(csv_name)

    dbf_instance.to_csv(csv_name)
    df = pd.read_csv(csv_name, sep=',', encoding='Latin-1', na_values='None')
    os.remove(csv_name)
    return df


def write_dbf(df, path, pre_process=True, encoding='cp850'):

    """
    write a .dbf from a pandas.core.frame.DataFrame object

    :param df: the DataFrame to write as a dbf
    :param path: The complete path to the .dbf to write
    :param pre_process: pre-process df with convert_stringy_things_to_string if True

    :type df:  pandas.core.frame.DataFrame
    :type path: str
    :type pre_process: bool

    :return: None
    :rtype: NoneType

    .. warnings:: the length of the .dbf file cannot exceed 1 000 000 lines
    """

    inner_df = convert_stringy_things_to_string(df.copy())if pre_process else df.copy()
    xlsx_name = path.split('.')[0] + '_from_df' + '.xlsx'  # temporary file name
    inner_df.to_excel(xlsx_name, index=False)  # write index as a regular column.
    exceltodbf_qc.exceltodbf(xlsx_name, path, encoding=encoding)
    os.remove(xlsx_name)


def normalize(frame):
    df = frame.copy().astype(float)
    for column in df.columns:
        weight = df[column].interpolate().sum()
        df[column] = df[column].interpolate()/weight
    return df


def convert_bytes_to_string(df, debug=False, encoding='utf-8'):

    """
    returns a pandas.core.frame.DataFrame where all bytes columns are converted to string

    :param df: the DataFrame to convert
    :type df:  pandas.core.frame.DataFrame
    :return: inner_df
    :rtype: pandas.core.frame.DataFrame
    """

    inner_df = df.copy()
    bytes_columns = [column for column in inner_df.columns if type(inner_df[column].iloc[0]) in {bytes, str}]
    if debug == True:
        print('bytes_columns converted to string:' + str(bytes_columns))
    for column in bytes_columns :
        try:
            inner_df[column] = inner_df[column].apply(to_str, args={encoding})
        except AttributeError:
            print('fail: column:', column)
            pass
    return inner_df

def to_str(bor, encoding):
    if type(bor) == str:
        return bor
    else:
        return bor.decode(encoding=encoding, errors='strict')

def convert_stringy_things_to_string(df, debug=False):

    """
    returns a pandas.core.frame.DataFrame where all bytes columns are converted to string

    :param df: the DataFrame to convert
    :type df:  pandas.core.frame.DataFrame
    :return: inner_df
    :rtype: pandas.core.frame.DataFrame
    """

    inner_df = convert_bytes_to_string(df).copy()
    stringy_columns = [column for column in inner_df.columns if type(inner_df[column].iloc[0]) == str ]
    if debug == True:
        print('stringy_columns converted to string:' + str(stringy_columns))
    inner_df[stringy_columns] = inner_df[stringy_columns].astype(str)
    return inner_df

def clean_dbf(path):
    write_dbf(read_dbf(path), path, pre_process=True)