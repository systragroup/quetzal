# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from syspy.syspy_utils import syscolors
import pandas as pd
# Advice: import seaborn at the beginning of your project to create easily nice plots

def clean_seq(x, col):
    x = x.sort_values(col)
    x[col] = np.arange(1, len(x)+1)
    return x

def shift_loadedlinks_alightings(load_df, load_column, alighting_column, boarding_column):
    """
    Shift alighting column in a loadedlinks to get a station-wise df.
    """
    load_df = load_df.reset_index(drop=True).copy()
    load_df = clean_seq(load_df, 'link_sequence')
    last = load_df.loc[load_df['link_sequence']==load_df['link_sequence'].max()].copy()
    last['a'] = last['b']
    last['b'] = ''
    last['link_sequence'] += 1
    last[alighting_column] = 0
    last[load_column] = 0
    last[boarding_column] = 0
    last.index += 1

    # Shift alightings
    load_df = load_df.append(last).reset_index(drop=True)
    temp_a = load_df[alighting_column].copy()
    temp_a.index+=1
    load_df[alighting_column] = temp_a
    load_df[alighting_column] = load_df[alighting_column].fillna(0)
    return load_df


def plot_load_b_a_for_loadedlinks(
        loaded_links, ax=None,
        load_column='load', boarding_column='boardings', alighting_column='alightings',
        width=0.2, label='', shift_alightings=False):
    
    """
    Export load graph for the specified line.

    The user can directly chose the figure size and font size in jupyter notebook with the following lines:
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    """
    load_df = loaded_links.copy()
    
    if ax is None:
         fig, ax = plt.subplots(1, 1)

    ## Prepare dataframe
    load_df = clean_seq(load_df, 'link_sequence')
    if shift_alightings:
        load_df = shift_loadedlinks_alightings(
            load_df, load_column, alighting_column, boarding_column
        )

    # Load
    ax.bar(
        load_df['link_sequence'].values,
        load_df[load_column].values,
        facecolor=syscolors.red_shades[3],
        # edgecolor=syscolors.red_shades[2],
        width=1,
        linewidth=1,
        label=label + ' load',
        align='edge',
        alpha=0.5
    )

    # Boardings
    ax.bar(
        load_df['link_sequence'].values + width / 2,
        load_df[boarding_column].values,
        facecolor=syscolors.red_shades[2],
        width=width,
        label=label + ' boardings',
        align='center',
    )

    # Alightings
    ax.bar(
        load_df['link_sequence'].values - width / 2,
        load_df[alighting_column].values,
        facecolor=syscolors.secondary_colors[5],
        width=width,
        label=label + ' alightings',
        align='center'
    )
    
    plt.xticks(
        load_df['link_sequence'].values,
        load_df['a'].values,
        ha='right',
        rotation=45
    )
    return ax


def create_two_directions_load_b_a_graph(
        load_fwd_bwd, 
        load_column='load', boarding_column='boarding', alighting_column='alighting',
        forward_col_suffix='_fwd', backward_col_suffix='_bwd',
        forward_label='forward', backward_label='backward', 
        legend=True, **kwargs):
    """
    Export load graph for the specified line, with boardings and alightings at each station
    The input load_fwd_bwd must be a dataframe with the columns:
    - 'a': station
    - 'link_sequence': bar plot sequence
    - 'load_fwd': load forward FROM a
    - 'boarding_fwd': boarding forward at a (outgoing forward link)
    - 'alighting_fwd': alighting forward at a (incoming forward link)
    - 'load_bwd': load backward TO a
    - 'boarding_bwd': boarding forward at a (outgoing backward link)
    - 'alighting_bwd': alighting forward at a (incoming backward link)

    The user can directly chose the figure size and font size in jupyter notebook with the following lines:
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    """
    # Create figure
    fig, ax = plt.subplots(1, 1)
    load_fwd_bwd = load_fwd_bwd.copy()

    # FORWARD
    load_column_fwd = load_column + forward_col_suffix
    alighting_column_fwd = alighting_column + forward_col_suffix
    boarding_column_fwd = boarding_column + forward_col_suffix
    ax = plot_load_b_a_for_loadedlinks(
        load_fwd_bwd[
            ['a', 'link_sequence', load_column_fwd, alighting_column_fwd, boarding_column_fwd]
        ], ax,
        load_column=load_column_fwd, alighting_column=alighting_column_fwd,
        boarding_column=boarding_column_fwd,
        label=forward_label, shift_alightings=False, **kwargs
    )

    # BACKWARD
    load_column_bwd = load_column + backward_col_suffix
    alighting_column_bwd = alighting_column + backward_col_suffix
    boarding_column_bwd = boarding_column + backward_col_suffix
    load_fwd_bwd[[load_column_bwd, alighting_column_bwd, boarding_column_bwd]] *= -1
    ax = plot_load_b_a_for_loadedlinks(
        load_fwd_bwd[
            ['a', 'link_sequence', load_column_bwd, alighting_column_bwd, boarding_column_bwd]
        ], ax,
        load_column=load_column_bwd, alighting_column=alighting_column_bwd,
        boarding_column=boarding_column_bwd,
        label=backward_label, shift_alightings=False, **kwargs
    )
    
    # Add zero split line
    plt.axhline(y=0, linewidth=2.5, color='k')

    return fig, ax


def directional_loads_to_station_bidirection_load(
        load_fwd, load_bwd, stations_to_parent_stations={},
        load_column='load', boarding_column='boarding', alighting_column='alighting',
        forward_suffix='_fwd', backward_suffix='_bwd'):
    """
    Take forward and backward loaded links for a line and return a station-oriented load df
    """
    # Get parent station
    load_fwd = load_fwd.replace(stations_to_parent_stations)
    load_bwd = load_bwd.replace(stations_to_parent_stations)

    # Format 
    stations = load_fwd[['a', 'link_sequence']]
    index_max = load_fwd['link_sequence'].max()
    stations = stations.append(
        pd.Series(
            {
                'a': load_fwd.loc[load_fwd['link_sequence']==index_max, 'b'].values[0],
                'link_sequence': index_max + 1
            }
        ),
        ignore_index=True
    )
    stations = clean_seq(stations, 'link_sequence')

    # Fwd load and boarding
    stations = stations.merge(
        load_fwd[['a', load_column, boarding_column]],
        left_on='a',
        right_on='a',
        how='left'
    )
    # Fwd alighting
    stations = stations.merge(
        load_fwd[['b', alighting_column]],
        left_on='a',
        right_on='b',
        how='left',
    )

    # bwd load and alighting
    stations = stations.merge(
        load_bwd[['b', load_column, alighting_column]],
        left_on='a',
        right_on='b',
        how='left',
        suffixes=(forward_suffix, backward_suffix)
    )
    # bwd boarding
    stations = stations.merge(
        load_bwd[['a', boarding_column]],
        left_on='a',
        right_on='a',
        how='left',
        suffixes=(forward_suffix, backward_suffix)
    )
    stations = stations.fillna(0)

    return stations

#### DEPRECATED ####
def save_line_load_graph(
        load_fwd, load_bwd, load_column='volume_pt', image_name='line_load.png', yticks=None,
        title='Line load', legend=True, save_fig=True, clean_sequence=False, *args, **kwargs
        ):
    """
    Export load graph for the specified line.

    The user can directly chose the figure size and font size in jupyter notebook with the following lines:
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    """
    # Clean link sequence if required
    if clean_sequence:
        load_fwd = clean_seq(load_fwd, 'link_sequence')
        load_bwd = clean_seq(load_bwd, 'link_sequence')

    # Creat figure
    fig, ax = plt.subplots(1, 1, **kwargs)

    # forward load
    ax.bar(
        load_fwd['link_sequence'].values,
        load_fwd[load_column].values,
        facecolor=syscolors.red_shades[3],
        # edgecolor=syscolors.red_shades[2],
        width=1,
        linewidth=2,
        label='forward',
        align='edge',
        alpha=0.5
    )

    # backward load
    ax.bar(
        (1+len(load_bwd) - load_bwd['link_sequence']).values,
        -load_bwd[load_column].values,
        facecolor=syscolors.red_shades[2],
        # edgecolor=syscolors.red_shades[1],
        width=1,
        linewidth=2,
        alpha=0.5,
        align='edge',
        label='backward'
    )

    # Add zero split line
    plt.axhline(y=0, linewidth=2.5, color='k')

    # Stations labels: we need to add the terminus station
    plt.xticks(
        np.append(load_fwd['link_sequence'].values,
                  len(load_fwd['link_sequence']) + 1),
        np.append(load_fwd['a'].values, load_fwd[
                  (load_fwd['link_sequence'] == len(load_fwd))]['b'].values[0]),
        ha='right',
        rotation=45
    )

    plt.title(title)
    if legend:
        plt.legend()
    plt.ylabel('Number of passengers')

    if yticks is None:
        rounding = int(np.floor(max(load_bwd[load_column].max(), load_fwd[load_column].max()) ** (1/10)))
        max_value = round(
            max(load_bwd[load_column].max(), load_fwd[load_column].max()), -rounding)
        yticks = np.arange(-max_value, max_value, round(max_value // 5, -rounding))

    plt.yticks(yticks, [int(y) for y in abs(yticks)])

    if save_fig:
        plt.savefig(
            image_name,
            bbox_inches='tight'
        )


def create_line_load_b_a_graph(
        load_fwd, load_bwd=None, image_name='line_load_b_a.png',
        width=0.2, yticks=None, xticks=None, legend=True, forward_label='forward', backward_label='backward',
        title='Line load', save_fig=True, clean_sequence=False):
    """
    Export load graph for the specified line, with boardings and alightings at each station

    The user can directly chose the figure size and font size in jupyter notebook with the following lines:
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    """
    # Clean link sequence if required
    if clean_sequence:
        load_fwd = clean_seq(load_fwd, 'link_sequence')
    # Creat figure
    fig, ax = plt.subplots(1, 1)

    # forward load
    ax.bar(
        load_fwd['link_sequence'].values,
        load_fwd['volume_pt'].values,
        facecolor=syscolors.red_shades[3],
        # edgecolor=syscolors.red_shades[2],
        width=1,
        linewidth=2,
        label=forward_label + ' load',
        align='edge',
        alpha=0.5
    )

    # Forwards boardings-alightings
    ax.bar(
        load_fwd['link_sequence'].values + width / 2,
        load_fwd['boardings'].values,
        facecolor=syscolors.red_shades[2],
        width=width,
        label=forward_label + ' boardings',
        align='center',
    )
    ax.bar(
        np.append(
            load_fwd['link_sequence'].values,
            len(load_fwd['link_sequence']) + 1  # We need to add the final alighting value which is the load
        ) - width / 2,
        np.append(
            load_fwd['alightings'].values, 
            load_fwd[(load_fwd['link_sequence'] == len(load_fwd))]['volume_pt'].values[0]
        ),
        facecolor=syscolors.secondary_colors[5],
        width=width,
        label=forward_label + ' alightings',
        align='center'
    )

    if load_bwd is not None:
        if clean_sequence:
            load_bwd = clean_seq(load_bwd, 'link_sequence') 
        # backward load
        ax.bar(
            (1 + len(load_bwd) - load_bwd['link_sequence']).values,
            -load_bwd['volume_pt'].values,
            facecolor=syscolors.red_shades[2],
            # edgecolor=syscolors.red_shades[1],
            width=1,
            alpha=0.5,
            align='edge',
            label=backward_label + ' load'
        )

        ax.bar(
            (2 + len(load_bwd) - load_bwd['link_sequence']).values + width / 2,
            -load_bwd['boardings'].values,
            facecolor=syscolors.red_shades[1],
            width=width,
            label=backward_label + ' boardings',
            align='center'
        )
        ax.bar(
            np.append(
                (2 + len(load_bwd) - load_bwd['link_sequence'] - width / 2).values,
                1 - width/2
            ),
            - np.append(
                load_bwd['alightings'].values,
                load_bwd[(load_bwd['link_sequence']==len(load_bwd))]['volume_pt'].values[0]
            ),
            facecolor=syscolors.secondary_colors[4],
            width=width,
            align='center',
            label=backward_label + ' alightings'
        )

        # Add zero split line
        plt.axhline(y=0, linewidth=2.5, color='k')

    # Stations labels: we need to add the terminus station
    if xticks is None:
        plt.xticks(
            np.append(load_fwd['link_sequence'].values,
                    len(load_fwd['link_sequence']) + 1),
            np.append(load_fwd['a'].values, load_fwd[
                    (load_fwd['link_sequence'] == len(load_fwd))]['b'].values[0]),
            ha='right',
            rotation=45
        )
    else:
        plt.xticks(
            np.append(load_fwd['link_sequence'].values,
                    len(load_fwd['link_sequence']) + 1),
            xticks,
            ha='right',
            rotation=45
        )

    plt.title(title)
    if legend:
        if load_bwd is not None:
            ncol = 2
        else:
            ncol = 1 
        plt.legend(ncol=ncol)
        plt.ylabel('Number of passengers')

    if yticks is None:
        if load_bwd is not None:
            max_value = round(
                max(load_fwd['volume_pt'].max(), load_fwd['volume_pt'].max()), -2)
            yticks = np.arange(-max_value, max_value,
                               round(max_value // 5, -2))
        else:
            max_value = round(load_fwd['volume_pt'].max(), -2)
            yticks = np.arange(0, max_value, round(max_value // 5, -2))

    plt.yticks(yticks)#, [int(y) for y in abs(yticks)])

    if save_fig:
        plt.savefig(
            image_name,
            bbox_inches='tight')

    return fig, ax