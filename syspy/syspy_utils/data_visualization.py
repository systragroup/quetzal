# -*- coding: utf-8 -*-

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

import six

import pandas as pd
import numpy as np
from sklearn import linear_model

import matplotlib.pyplot as plt
import branca.colormap as cm

from syspy.syspy_utils.syscolors import rainbow_shades, clear_shades, red_shades


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
        serie = value_series.apply(lambda x: x/max_value*outer_average_width)
    elif method == 'surface':
        serie = value_series.apply(lambda x: np.sqrt(x/max_value)*outer_average_width)
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
        if index == None:
            index = np.linspace(min_value, max_value, len(colors))
        colormap = cm.LinearColormap(colors, index=index)
    else:
        if index == None:
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
        np.absolute((pool[y_column]-pool[x_column])/(pool[y_column]+pool[x_column])).fillna(0),
        colors=[clear_shades[1], clear_shades[0]],
    )
    pool['size'] = width_series(np.absolute(np.power(np.maximum(pool[y_column],pool[x_column]), 0.3)), box_size)

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
            c = c if dynamic_color else clear_shades[1]
            plt.annotate(
                label,
                xy=(x, y), xytext = (-10, 10),
                textcoords = 'offset points', ha = 'right', va = 'bottom', size=d,
                bbox = dict(boxstyle='round, pad=0.5', fc=c, alpha=1),
                arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
    regr.fit(x_array[:,np.newaxis], y_array)

    linspace = np.linspace(np.min(x_array), np.max(x_array), 1000)

    (x_label, y_label) = (x_label if x_label else x_column, y_label if y_label else y_column)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if auto_title:
        title += y_label + ' vs ' + x_label
    plt.title('%s \n y = %f x + %f —— R^2 = %f' % (title,
                                                   regr.coef_[0],
                                                   regr.intercept_,
                                                   regr.score(x_array[:,np.newaxis],y_array)))

    _slope_kwargs={'color':rainbow_shades[0], 'linewidth':3}
    _slope_kwargs.update(slope_kwargs)

    if plot_identity:
        plt.plot(linspace, linspace, color=rainbow_shades[2])[0]

    if beam_prediction:
        plt.plot(linspace, regr.predict(linspace[:, np.newaxis])*beam_prediction, **_slope_kwargs)
        plt.plot(linspace, regr.predict(linspace[:, np.newaxis])/beam_prediction, **_slope_kwargs)

    if beam_identity:
        plt.plot(linspace, linspace*beam_identity, color=rainbow_shades[2], linestyle='dashed')
        plt.plot(linspace, linspace/beam_identity, color=rainbow_shades[2], linestyle='dashed')

    
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
        row_height = figsize[1] / (len(data) +1)
        
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax