import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from syspy.syspy_utils import syscolors
from branca import colormap as cm
from matplotlib import patches

# Advice: import seaborn at the beginning of your project to create easily nice plots


def clean_seq(x, col):
    x = x.sort_values(col)
    x[col] = np.arange(1, len(x) + 1)
    return x


def sort_links(to_sort, a_name='a', b_name='b', max_iter=50):
    """
    Given a links dataframe for one line and one direction, that may contain branches,
    return the sorted dataframe based on origin and destination stops.
    """
    try:
        to_sort = to_sort.sort_values('link_sequence')
    except KeyError:
        pass
    sorted_line = pd.DataFrame()
    left_to_sort = pd.DataFrame()
    i = 0
    count = 0
    while len(to_sort) > 0 and count < max_iter and i < 1000:
        count += 1
        for index, row in to_sort.iterrows():
            if i == 0:
                sorted_line = sorted_line.append(row, ignore_index=True)
                i += 1
            else:
                a = row[a_name]
                b = row[b_name]
                if len(sorted_line[sorted_line[b_name] == a]) > 0:  # place after
                    name = sorted_line[sorted_line[b_name] == a].index.max()
                    row.name = name
                    sorted_line = sorted_line.append(row)
                    sorted_line.reset_index(drop=True, inplace=True)
                elif len(sorted_line[sorted_line[a_name] == b]) > 0:  # place before
                    name = sorted_line[sorted_line[a_name] == b].index.min()
                    row.name = name
                    to_insert = pd.DataFrame(row).T
                    sorted_line = pd.concat(
                        [sorted_line.iloc[:name], to_insert, sorted_line.iloc[name:]]
                    ).reset_index(drop=True)
                else:
                    left_to_sort = left_to_sort.append(row, ignore_index=True)
                i += 1

        to_sort = left_to_sort.copy()
        left_to_sort = pd.DataFrame()

    if len(left_to_sort) > 0:
        raise Exception('Sorting failed')
    return sorted_line


def shift_loadedlinks_alightings(load_df, load_columns=['load'], alighting_columns=['alightings'], boarding_columns=['boardings']):
    """
    Shift alighting column in a loadedlinks to get a station-wise df.
    """
    load_df = load_df.reset_index(drop=True).copy()
    load_df = clean_seq(load_df, 'link_sequence')
    last = load_df.loc[load_df['link_sequence'] == load_df['link_sequence'].max()].copy()
    last['a'] = last['b']
    last['b'] = ''
    last['link_sequence'] += 1
    for col in load_columns + alighting_columns + boarding_columns:
        last[col] = 0
    last.index += 1

    # Shift alightings
    load_df = load_df.append(last).reset_index(drop=True)
    temp_a = load_df[alighting_columns].copy()
    temp_a.index += 1
    load_df[alighting_columns] = temp_a
    load_df[alighting_columns] = load_df[alighting_columns].fillna(0)
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

    # Prepare dataframe
    load_df = clean_seq(load_df, 'link_sequence')
    if shift_alightings:
        load_df = shift_loadedlinks_alightings(
            load_df, [load_column], [alighting_column], [boarding_column]
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

    ax.set_xticks(load_df['link_sequence'].values)
    ax.set_xticklabels(load_df['a'].values, ha='right', rotation=45)
    return ax


def create_two_directions_load_b_a_graph(
        load_fwd_bwd,
        load_column='load', boarding_column='boardings', alighting_column='alightings',
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
        load_column='load', boarding_column='boardings', alighting_column='alightings',
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
                'a': load_fwd.loc[load_fwd['link_sequence'] == index_max, 'b'].values[0],
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


# DEPRECATED #
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
        (1 + len(load_bwd) - load_bwd['link_sequence']).values,
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

    # Stations labels: we need to add the terminus station
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
        rounding = int(np.floor(max(load_bwd[load_column].max(), load_fwd[load_column].max()) ** (1 / 10)))
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
                1 - width / 2
            ),
            - np.append(
                load_bwd['alightings'].values,
                load_bwd[(load_bwd['link_sequence'] == len(load_bwd))]['volume_pt'].values[0]
            ),
            facecolor=syscolors.secondary_colors[4],
            width=width,
            align='center',
            label=backward_label + ' alightings'
        )

        # Add zero split line
        plt.axhline(y=0, linewidth=2.5, color='k')

    # Stations labels: we need to add the terminus station
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
            np.append(load_fwd['link_sequence'].values, len(load_fwd['link_sequence']) + 1),
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

    plt.yticks(yticks)  # , [int(y) for y in abs(yticks)])

    if save_fig:
        plt.savefig(
            image_name,
            bbox_inches='tight')
    return fig, ax

def compute_line_od_vol(self, line, line_col='route_id', vol_col='volume'):
    '''
    Get the OD Matrix of a specified line  
    '''
    only_pt_los = self.pt_los[~self.pt_los['boarding_links'].isna() & ~self.pt_los['alighting_links'].isna()].copy()

    # Filter LOS containing line
    links_line = self.links[line_col].to_dict()
    line_los = only_pt_los[only_pt_los['boarding_links'].apply(lambda boardings: any([links_line.get(x) == line for x in boardings]))]

    # Create a dataframe of OD volume for line
    line_los['line_od_links'] = line_los.apply(lambda x: list(zip(x['boarding_links'], x['alighting_links'])), axis=1)
    link_a = self.links['a'].to_dict()
    link_b = self.links['b'].to_dict()
    link_line = self.links['route_id'].to_dict()
    line_od = line_los.explode('line_od_links').groupby('line_od_links')[vol_col].sum().reset_index()
    line_od['line_od'] = line_od['line_od_links'].apply(lambda x: (link_a.get(x[0]), link_b.get(x[1])))
    line_od['line'] = line_od['line_od_links'].apply(lambda x: link_line.get(x[0]))
    line_od = line_od[line_od['line'] == line].copy()

    # Line Stations
    links = self.links[self.links['route_id'] == line]
    stations = links[links['direction_id'] == 0].sort_values(by='link_sequence')['a'].values
    stations = np.append(stations, links[links['direction_id'] == 0].sort_values(by='link_sequence')['b'].values[-1])

    # OD volume matrix
    line_od[['a','b']] = pd.DataFrame(line_od['line_od'].tolist(), index=line_od.index)
    line_od = line_od.set_index(['a', 'b'])
    vol = line_od[line_od['line'] == line][vol_col].unstack()

    return pd.DataFrame(vol, columns=stations, index=stations).fillna(0)
    

def single_direction_arc_diagram(ax, stations, vol, max_width = 5.0, min_width = 1.0, 
                                    ymax = 0.0, ymin = 0.0, reverse=False, cmap=None, 
                                    **kwargs):
    '''
    Export arc diagram representing OD of a line
    '''
    max_vol = vol.max()
    min_vol = vol.min()
    plot_stations = stations[::-1] if reverse else stations

    # Default colomap
    if cmap is None:
        cmap = cm.LinearColormap(["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#67000d"], vmin=min_vol, vmax=max_vol)

    # Create arcs
    for i in plot_stations:
        ix = stations.index(i)
        for j in plot_stations[plot_stations.index(i) +1 :]:
            jx = stations.index(j)
            if vol[ix, jx] == 0: continue
            lw = (max_width - min_width)*vol[ix, jx]/(max_vol-min_vol) +min_width
            mxmy = mx, my = [(ix + jx) / 2, 0.0]
            r = abs(ix - mx)

            if reverse & (-r < ymin): ymin = -r
            if (not reverse) & (r > ymax): ymax = r

            arc = patches.Arc(mxmy, 2*r, 2*r, 
                                theta1=0, theta2=180, 
                                color=cmap(vol[ix, jx]),
                                angle=180 if reverse else 0, 
                                linewidth=lw,
                                zorder=vol[ix, jx],
                                **kwargs)

            ax.add_patch(arc)

    # Plot Settings
    ax.set_xlim(-0.1, len(stations) -0.9)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(stations)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )
    plt.tick_params(bottom = False)
    plt.tick_params(top = False)
    ax.set_yticks([])
    plt.axhline(y=0, linewidth=2.5, color='k', zorder=max_vol+1)
    #plt.colorbar(cmap) #TODO
    return ax
