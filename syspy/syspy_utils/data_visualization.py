"""
**data_visualization provides tool for easily manipulate color scales and plot linear regressions**


example:
::

    import data_visualization

    fig = data_visualization.linear_plot(
        vol, 'nautos_before', 'v_1_before',
        label_column='punto_vp_after',
        x_label = 'volumen_medido',
        y_label = 'volumen_modelado',
        title='volumenes medidos y modelados al nivel del punto de conteo (CAR) ',
        fit_intercept=False )

    fig.savefig('Q:/conteos_car_before.png', bbox_inches='tight')

"""
__author__ = 'qchasserieau'

import itertools

import branca.colormap as cm
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import six
from shapely import geometry
from sklearn import linear_model
from tqdm import tqdm

"""
The following colors are mentioned in Systra's graphical charter: \n
red shades \n
grey shades \n
rainbow shades : spot colors, vivid and highly contrasted \n
sorted colors advised for word documents \n
secondary colors \n
"""

# Couleurs d'accompagnement de la charte graphique
rainbow_shades = ["#D22328", "#559BB4", "#91A564", "#DC9100", "#8C4B7D", "#A08C69",
                  "#647D6E", "#5A7382", "#64411E", "#A00037", "#643C5A"]

# Nuances de rouge
# en rgb [(105,18,20),(157,26,30),(210,35,40),(232,119,122),(240,164,166),(247,210,211)]
red_shades = ['#691214', '#9d1a1e', '#d22328', '#e8777a', '#f0a4a6', '#f7d2d3']

# Nuances de gris
# en rgb [(48,48,50),(90,90,90),(127,127,127),(166,165,165),(199,199,200),(227,227,228)]
grey_shades = ['#303032', '#5a5a5a', '#7f7f7f', '#a6a5a5', '#c7c7c8', '#e3e3e4']


# Couleurs ordonné dans le sens des préconisations de la charte graphique
sorted_colors = ['#d22328', '#7f7f7f', '#691214', '#f0a4a6']

# Couleurs secondaires
# en rgb [(100,60,90),(158,27,22),(100,66,30),(100,125,110),(91,115,130),(84,154,179),(219,145,3),(84,160,60)]
secondary_colors = ['#643c5a', '#9e1b16', '#64421e', '#647d6e', '#5b7382', '#549ab3',
                    '#db9103', '#54a03c']

# Couleurs utilisées par Linedraft
linedraft_shades = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def itercolors(color_list, repetition):
    return list(itertools.chain(*[[color] * repetition for color in color_list]))


_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'


def rgb(triplet):
    return _HEXDEC[triplet[1:][0:2]], _HEXDEC[triplet[1:][2:4]], _HEXDEC[triplet[1:][4:6]]


def triplet(rgb, lettercase=LOWERCASE):
    return '#' + (format(rgb[0] << 16 | rgb[1] << 8 | rgb[2], '06' + lettercase)).upper()


def clear(rgb, x=50):
    (r, g, b) = rgb
    _r = round(((100 - x) * r + x * 255) / 100)
    _g = round(((100 - x) * g + x * 255) / 100)
    _b = round(((100 - x) * b + x * 255) / 100)
    return (_r, _g, _b)


def clear_shades():
    return [triplet(clear(rgb(shade))) for shade in rainbow_shades]


d = {
    'marron': 8,
    'orange': 5,
    'rouge': 0,
    'bleue': 1,
    'verte': 2,
    'jaune': 3,
    'violette': 4,
    'rose': 9
}


def in_string(name):
    for c in d.keys():
        if c in name:
            return rainbow_shades[d[c]]
    return rainbow_shades[7]


def width_series(value_series, outer_average_width=5, max_value=None, method='linear'):
    """
    :param value_series: the pd.Series that contain the values
    :param outer_average_width: the average width of the width series to return
    :param max_value: value to use as the maximum when normalizing the series (to focus low values)
    :param method: linear or surface
    :return: width_series: pd.Series that contains the widths corresponding to the values
    :rtype: pd.Series
    """
    max_value = max_value if max_value else np.max(list(value_series.values))
    if method == 'linear':
        serie = value_series.apply(lambda x: x / max_value * outer_average_width)
    elif method == 'surface':
        serie = value_series.apply(lambda x: np.sqrt(x / max_value) * outer_average_width)
    return serie


def color_series(
    value_series,
    colors=[rainbow_shades[3], rainbow_shades[0]],
    index=None,
    max_value=None,
    min_value=None,
    method='linear',
    reversed_colors=False
):
    """
    :param value_series: the pd.Series that contain the values
    :param colors: list containing the colors used to build the color scale ['red', 'blue']
    :param index: list containing the value corresponding to each color
    :param min_value: the values below min_values are given the same color : colors[0]
    :param max_value: value to use as the maximum when normalizing the series (to focus on low values).
    :param method: method used (linear or step)
    :return: color_series: pd.Series that contains the colors corresponding to the values
    :rtype: pd.Series

    example:
    ::
        import pandas as pd

        # common visualization library
        import matplotlib.pyplot as plt
        % matplotlib inline
        plt.rcParams['figure.figsize'] = [17,7]
        plt.rcParams['font.size'] = 20

        # custom visualization library
        from syspy_utils import data_visualization, syscolors

        data = {'a': 1, 'b': 2, 'c': 3, 'd': 5, 'e': 7, 'f': 10}
        series = pd.Series(data)

        color_scale = ('#559BB4', '#D22328')
        color_series = data_visualization.color_series(series, colors=color_scale,  min_value=2, max_value=6)
        dataframe = pd.DataFrame({'value': series, 'color': color_series, 'width': width_series})
        series.plot(kind='bar', color=color_series)

    .. figure:: ./pictures/data_visualization_color_series.png
        :width: 25cm
        :align: center
        :alt: bar plot with color series
        :figclass: align-center

        bar plot with color series
    """
    colors = list(reversed(colors)) if reversed_colors else colors
    max_value = max_value if max_value else np.max(list(value_series.values))
    min_value = min_value if min_value else np.min(list(value_series.values))

    if method == 'linear':
        if index is None:
            index = np.linspace(min_value, max_value, len(colors))
        colormap = cm.LinearColormap(colors, index=index)
    else:
        if index is None:
            index = value_series.quantile(np.linspace(0, 1, len(colors))).values
        colormap = cm.StepColormap(colors, index=index)
    return value_series.apply(lambda x: colormap(max(min(x, max_value), min_value)))


def linear_plot(
    df,
    x_column,
    y_column,
    label_column=None,
    x_label=None,
    y_label=None,
    title='', auto_title=True, fit_intercept=True,
    plot_identity=True,
    beam_prediction=None,
    beam_identity=2,
    dynamic_size=True,
    dynamic_color=True,
    box_size=15,
    slope_kwargs={}
):
    """
    :param df: pd.DataFrame to plot
    :param x_column: column to plot as x in the scatter plot
    :param y_column: column to plot as y in the scatter plot
    :param label_column: if given, the dots of the scatter plot are labeled with this column of df
    :param x_label: x axis label
    :param y_label: y axis label
    :param title: title of the figure
    :param auto_title: if given, the parameters of the regression are added to the title
    :param fit_intercept: if False the linear model fits y = ax + b with b = 0
    :param plot_identity: if True, the identity line y = x is added to the plot
    :param beam_prediction: width of the beam to plot on both side of the prediction line
    :param beam_identity: width of the beam to plot on both side of the identity line
    :param dynamic_size: if True, label sizes are proportional to te value of x*y
    :param dynamic_color: in True, the further from the identity line, the more red the label
    :param box_size: size of the label
    :param slope_kwargs: args of the regression line plot
    :return: the figure

    example:
    ::
        fig = data_visualization.linear_plot(
            vol, 'nautos_before', 'v_1_before',
            label_column='punto_vp_after',
            x_label = 'volumen_medido',
            y_label = 'volumen_modelado',
            title='volumenes medidos y modelados al nivel del punto de conteo (CAR) ',
            fit_intercept=False)

    .. figure:: ./pictures/data_visualization_linear_plot.png
        :width: 25cm
        :align: center
        :alt: linear regression plot
        :figclass: align-center

        linear regression plot

    """
    plt.clf()
    pool = df.copy()

    pool['distance'] = color_series(
        np.absolute((pool[y_column] - pool[x_column]) / (pool[y_column] + pool[x_column])).fillna(0),
        colors=[clear_shades()[1], clear_shades()[0]],
    )
    pool['size'] = width_series(np.absolute(np.power(np.maximum(pool[y_column], pool[x_column]), 0.3)), box_size)

    x_array = np.array(pool[x_column])
    y_array = np.array(pool[y_column])

    plt.scatter(x_array, y_array, color=rainbow_shades[1])

    if label_column:
        for label, x, y, d, c in zip(
            list(pool[label_column]),
            pool[x_column].values[:],
            pool[y_column].values[:],
            pool['size'],
            pool['distance']
        ):

            d = d if dynamic_size else box_size
            c = c if dynamic_color else clear_shades()[1]
            plt.annotate(
                label,
                xy=(x, y), xytext=(-10, 10),
                textcoords='offset points', ha='right', va='bottom', size=d,
                bbox=dict(boxstyle='round, pad=0.5', fc=c, alpha=1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
    regr.fit(x_array[:, np.newaxis], y_array)

    linspace = np.linspace(np.min(x_array), np.max(x_array), 1000)

    (x_label, y_label) = (x_label if x_label else x_column, y_label if y_label else y_column)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if auto_title:
        title += y_label + ' vs ' + x_label
    plt.title('%s \n y = %f x + %f —— R^2 = %f' % (title,
                                                   regr.coef_[0],
                                                   regr.intercept_,
                                                   regr.score(x_array[:, np.newaxis], y_array)))

    _slope_kwargs = {'color': rainbow_shades[0], 'linewidth': 3}
    _slope_kwargs.update(slope_kwargs)

    if plot_identity:
        plt.plot(linspace, linspace, color=rainbow_shades[2])[0]

    if beam_prediction:
        plt.plot(linspace, regr.predict(linspace[:, np.newaxis]) * beam_prediction, **_slope_kwargs)
        plt.plot(linspace, regr.predict(linspace[:, np.newaxis]) / beam_prediction, **_slope_kwargs)

    if beam_identity:
        plt.plot(linspace, linspace * beam_identity, color=rainbow_shades[2], linestyle='dashed')
        plt.plot(linspace, linspace / beam_identity, color=rainbow_shades[2], linestyle='dashed')

    plot = plt.plot(linspace, regr.predict(linspace[:, np.newaxis]), **_slope_kwargs)[0]
    return plot


def render_mpl_table(
    data,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color=red_shades[1],
    row_colors=['#f1f1f2', 'w'],
    edge_color='w',
    bbox=[0, 0, 1, 1],
    header_columns=0,
    figsize=None,
    ax=None,
    dpi=96,
    **kwargs
):
    if figsize:
        col_width = figsize[0] / len(data.T)
        row_height = figsize[1] / (len(data) + 1)

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


spectral = list(reversed([
    '#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'
]))


def add_raster(ax, raster, keep_ax_limits=True, adjust='linear', cmap='gray'):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    rasterio.plot.show(raster, ax=ax, adjust=adjust, cmap=cmap, with_bounds=True)
    if keep_ax_limits:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def add_north(ax, fontsize=14):
    x, y, arrow_length = 0.99, 0.99, 0.05
    ax.annotate(
        'N', xy=(x, y), xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor='black', width=2, headwidth=7),
        ha='center', va='top', fontsize=fontsize, xycoords=ax.transAxes)


def add_scalebar(ax, fontsize=14):
    from matplotlib_scalebar.scalebar import ScaleBar
    ax.add_artist(
        ScaleBar(
            1, location='lower left', pad=1, frameon=False,
            scale_loc='top', font_properties={'size': fontsize},
            length_fraction=0.1,
            height_fraction=0.01
        )
    )


def add_north_and_scalebar(ax, fontsize=14):
    add_north(ax, fontsize)
    add_scalebar(ax, fontsize)


def bandwidth(
    gdf, value_column, max_value=None, power=1, legend=True, legend_values=None, legend_length=1 / 3,
    label_column=None, max_linewidth_meters=100, variable_width=True, line_offset=True, cmap=spectral,
    geographical_bounds=None, label_kwargs={'size': 12}, *args, **kwargs
):
    # TODO:
    # 1. add to plot model
    # 2. add arrows?

    # Can only plot valid LineString
    df = gdf[gdf.length > 0].sort_values(value_column).copy()
    df = df[df.geometry.type == 'LineString']

    if label_column is None:  # Then it is 'label'
        df.drop('label', 1, errors='ignore', inplace=True)
        label_column = 'label'
    # Create base plot
    plot = base_plot(df, geographical_bounds, *args, **kwargs)

    # Handle legend
    if legend:
        if legend_values is None:
            s = df[value_column].copy()
            r = int(np.log10(s.mean()))
            legend_values = [np.round(s.quantile(i / 5), -r) for i in range(6)]
        legend_df = create_legend_geodataframe(
            plot, legend_values=legend_values, relative_width=legend_length,
            label_column=label_column, value_column=value_column
        )
        df = pd.concat([legend_df, df]).reset_index()

    # Plot values
    # Power scale
    df['power'] = np.power(df[value_column], power)
    power_max_value = np.power(max_value, power) if max_value else df['power'].max()
    # Color
    df['color'] = color_series(df['power'], colors=cmap, max_value=power_max_value)
    # Linewidth
    if variable_width:
        df['geographical_width'] = width_series(df['power'], max_linewidth_meters, max_value=power_max_value)
    else:
        df['geographical_width'] = max_linewidth_meters
    df['linewidth'] = linewidth_from_data_units(df['geographical_width'].values, plot)
    # Offset
    if line_offset:
        df['geometry'] = df.apply(
            lambda x: x['geometry'].parallel_offset(x['geographical_width'] * 0.5, 'right'), 1
        )
        df = df[df.geometry.type == 'LineString']
        df = df[df.length > 0]  # offset can create empty LineString
        # For some reason, right offset reverse the coordinates sequence
        # https://github.com/Toblerity/Shapely/issues/284
        df['geometry'] = df.geometry.apply(lambda x: geometry.LineString(x.coords[::-1]))
    # Plot
    df.plot(color=df['color'], linewidth=df['linewidth'], ax=plot)

    # Plot label
    df_label = create_label_dataframe(df, label_column)
    df_label.apply(
        lambda x: plot.annotate(
            s=x[label_column],
            xy=x.geometry.centroid.coords[0],
            xytext=x['label_offset'],
            textcoords='offset pixels',
            rotation=x['label_angle'],
            rotation_mode='anchor',
            ha='center', va=x['va'],
            **label_kwargs
        ),
        axis=1
    )

    plt.xticks([])
    plt.yticks([])
    return plot


def add_basemap(
    ax,
    zoom,
    url='http://tile.stamen.com/terrain-background/tileZ/tileX/tileY.png',
    errors='ignore',
):
    # TODO: move to another file
    try:
        import contextily as ctx
        xmin, xmax, ymin, ymax = ax.axis()
        basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
        ax.imshow(basemap, extent=extent, interpolation='bilinear')
        # restore original x/y limits
        ax.axis((xmin, xmax, ymin, ymax))
    except Exception as e:
        print('could not add basemap:', e)
        if errors != 'ignore':
            assert False


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def create_legend_geodataframe(
        plot, legend_values, relative_width=0.3,
        value_column='value', label_column='label'
):
    """
    Create a GeoDataFrame with legend data, geolocated at the right bottom of the plot.
    - plot: the matplotlib plot that will include the legend
    - legend_values (list of float): list of legend values to include
    - relative_width (float between 0 and 1): legend length (1=entire axis length)
    - value_column: column name for values
    - label_column: column name for labels

    returns:
    GeoDataFrame with columns value_column, label_column, 'geometry'
    """
    # Get plot bounds
    ylims = plot.get_ylim()
    xlims = plot.get_xlim()

    # Place legend
    plot_width = xlims[1] - xlims[0]
    plot_height = ylims[1] - ylims[0]
    rank = 0
    dx = plot_width * relative_width / len(legend_values)
    data = []
    for v in reversed(legend_values):
        g = geometry.LineString([
            (xlims[1] - plot_width / 100 - rank * dx, ylims[0] + plot_height / 100),
            (xlims[1] - plot_width / 100 - (rank + 1) * dx, ylims[0] + plot_height / 100)]
        )
        rank += 1
        data.append([v, g, str(v)])
        legend_df = gpd.GeoDataFrame(data, columns=[value_column, 'geometry', label_column])

    return legend_df


def base_plot(df, geographical_bounds, *args, **kwargs):
    """
    Create a basic plot of a GeoDataFrame with specific geographical bounds.
    """
    def _set_bandwidth_geographical_bounds(plot, xmin, ymin, xmax, ymax, offset=0.01):
        x_offset = (xmax - xmin) * offset
        y_offset = (ymax - ymin) * offset
        plot.set_xlim(xmin - x_offset, xmax + x_offset)
        plot.set_ylim(ymin - y_offset, ymax + y_offset)
    # Light geometry plot
    plot = gpd.GeoDataFrame(df).plot(linewidth=0.1, color='grey', *args, **kwargs)
    # Set bounds geographical bounds
    if geographical_bounds is not None:
        _set_bandwidth_geographical_bounds(plot, *geographical_bounds)
    return plot


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def get_normal(linestring):
    try:
        c = linestring.coords[:]
        v = (-(c[1][1] - c[0][1]), c[1][0] - c[0][0])
        return v / np.sqrt(v[1] * v[1] + v[0] * v[0])
    except IndexError:  # list index out of range
        # it is a point
        return 0
    except NotImplementedError:  # MultiLineString
        c = linestring[0].coords[:]
        v = (-(c[1][1] - c[0][1]), c[1][0] - c[0][0])
        return v / np.sqrt(v[1] * v[1] + v[0] * v[0])


def get_label_angle_alignment_offset(row, linewidth_column='linewidth'):
    try:
        x = [row.geometry.coords[0], row.geometry.coords[-1]]
        angle = (np.arccos((x[1][0] - x[0][0]) / row.geometry.length) * 180 / 3.14) + 180
        angle *= np.sign(x[1][1] - x[0][1])
    except NotImplementedError:  # MultiLineString
        x = row.geometry[0].coords[:]
        angle = (np.arccos((x[1][0] - x[0][0]) / row.geometry.length) * 180 / 3.14) + 180
        angle *= np.sign(x[1][1] - x[0][1])
    except IndexError:
        angle = 0

    va = 'bottom'
    if linewidth_column is None:
        label_offset = 1
    else:
        label_offset = row[linewidth_column]
    return pd.Series(
        {
            'va': va, 'label_angle': angle,
            'label_offset': -label_offset / 4 * row['normal']
        }
    )


def create_label_dataframe(df, label_column='label', linewidth_column='linewidth'):
    cols = ['geometry', label_column]
    if linewidth_column is not None:
        cols += [linewidth_column]
    label_df = df[cols].dropna(subset=[label_column])
    label_df['normal'] = label_df.geometry.apply(get_normal)
    columns = ['label_angle', 'label_offset', 'va']
    label_df[columns] = label_df.apply(get_label_angle_alignment_offset, 1)[columns]
    return label_df
