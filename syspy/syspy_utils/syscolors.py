# -*- coding: utf-8 -*-



"""
The following colors are mentioned in Systra's graphical charter: \n
red shades \n
grey shades \n
rainbow shades : spot colors, vivid and highly contrasted \n
sorted colors advised for word documents \n
secondary colors \n
"""

import itertools


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
    return list(itertools.chain(*[[color]*repetition for color in color_list]))

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'

def rgb(triplet):
    return _HEXDEC[triplet[1:][0:2]], _HEXDEC[triplet[1:][2:4]], _HEXDEC[triplet[1:][4:6]]

def triplet(rgb, lettercase=LOWERCASE):
    return '#'+(format(rgb[0]<<16 | rgb[1]<<8 | rgb[2], '06'+lettercase)).upper()

def clear(rgb, x=50):
    (r, g, b) = rgb
    _r = round(((100 - x) * r + x * 255) / 100)
    _g = round(((100 - x) * g + x * 255) / 100)
    _b = round(((100 - x) * b + x * 255) / 100)
    return (_r, _g, _b)

def clear_shades():
    return [triplet(clear(rgb(shade))) for shade in rainbow_shades]


d = {
    'marron' : 8,
    'orange' : 5,
    'rouge' : 0,
    'bleue' : 1,
    'verte' : 2,
    'jaune' : 3,
    'violette' : 4,
    'rose': 9
}

def in_string(name):
    for c in d.keys():
        if c in name:
            return rainbow_shades[d[c]]
    return rainbow_shades[7]


