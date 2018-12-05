# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def classification(values, classes, ascending=True, bins=None, round_bins=None):
    """
    Return the array of classes of the values, for the given classes
    Inputs:
        values: np.array of values to classify
        classes: ordered list of classe names
        ascending: bool
        bins: boundaries of each classes. Automatically computed with equal width if None
        round_bins: integer to round the inner boundaries. Not rounded if None
        
    Outputs:
        tuple (np.array of the classes of the values, np.array of bins bounds)
        
    Example:
    
        x = np.array([1,2,10])
        classes = [1,2,3]
        classification(x,classes)
        ----
        (array([1, 1, 3]), array([  1. ,   4. ,   7. ,  10.1]))
    """
    
    n_classes = len(classes)
    if bins is None:
        bins = n_classes

    # Create bins
    _, bins_bounds = np.histogram(values, bins=bins)
    # Force max value in interval
    bins_bounds[-1] = bins_bounds[-1] * 1.01
    
    # Round bins: all but first and last
    if round_bins is not None:
            temp = np.round(bins_bounds[1:-1], round_bins)
            temp = np.append(np.array([bins_bounds[0], bins_bounds[-1]]), temp)
            bins_bounds = np.sort(temp)
    
    # Get values index
    if ascending:
        index = np.digitize(values, bins_bounds)
#         print('index:', index)
        return np.array([classes[i-1] for i in index]), bins_bounds
    else: 
        index = np.digitize(values, bins_bounds)
#         print('index:', index)
        return np.array([classes[n_classes-i] for i in index]), bins_bounds


def basic_scoring(x, classes=[1,2,3,4,5], methods={}):
    """
    To apply to a dataframe to get a basic ascending scoring from 1 to 5 with equal intervals.
    """
    classes, bounds =  classification(x, classes)
    result_dict = {x.index[i] : classes[i] for i in range(len(classes))}
    result_dict.update({'bounds': bounds})
    return pd.Series(result_dict)


def background_colormap(val):
    """
    Basic 1 to 5 background colormap for nice dataframe printing or excel export purposes.
    
    Example:
        df.style.applymap(background_colormap)
    """
    color = ['#FB676D', '#FBA977','#FEE987', '#B1D584', '#62C073']
    return 'background-color: %s' % color[val - 1]