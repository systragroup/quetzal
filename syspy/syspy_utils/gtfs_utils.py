# -*- coding: utf-8 -*-

"""

This modules provides tools for manipulating gtfs files

example::


"""


__author__ = 'mjoly'

import pandas as pd
import json


def get_patterns(gtfs_dir, direction=True, by_route=True) :

    stop_times = pd.read_csv(gtfs_dir+'stop_times.txt')
    trips = pd.read_csv(gtfs_dir+'trips.txt').set_index('trip_id')


    trips_to_patterns = stop_times[['trip_id','stop_id']].groupby('trip_id').aggregate(lambda x : (tuple(x)))
    trips_to_patterns.rename(columns={'stop_id' : 'geo_pattern_links'},inplace = True)


    trips_to_patterns['reversed_geo_pattern_links'] = trips_to_patterns.apply(lambda r: tuple(reversed(r['geo_pattern_links'])),axis=1)
    trips_to_patterns['route_id'] = trips['route_id']


    def pattern_choice(r) :
     if r['geo_pattern_links']<r['reversed_geo_pattern_links'] :
          return r['geo_pattern_links']
     else :
          return r['reversed_geo_pattern_links']



    if direction==False :

        trips_to_patterns['chosen_geo_pattern_links'] = trips_to_patterns.apply(lambda r : pattern_choice(r),axis = 1)
        trips_to_patterns['direction']=0
    else :
        trips_to_patterns['chosen_geo_pattern_links'] = trips_to_patterns['geo_pattern_links']
        trips_to_patterns['direction']=trips_to_patterns.apply(lambda r : 1 + 1*(r['geo_pattern_links']<r['reversed_geo_pattern_links']),axis = 1)
    trips_direction = trips_to_patterns.copy()



    if by_route==True :
        patterns = trips_to_patterns.groupby(['chosen_geo_pattern_links','route_id','direction']).count().reset_index()
        patterns = patterns[['chosen_geo_pattern_links','direction','route_id']]
        patterns.reset_index(inplace=True)
        patterns.rename(columns={'index' : 'pattern_id'},inplace=True)
        trips_to_patterns=trips_to_patterns.reset_index().merge(patterns,on=['chosen_geo_pattern_links','direction','route_id'])[['pattern_id','trip_id']]
    else :
        patterns = trips_to_patterns.groupby(['chosen_geo_pattern_links','direction']).count().reset_index()
        patterns = patterns[['chosen_geo_pattern_links','direction']]
        patterns.reset_index(inplace=True)
        patterns.rename(columns={'index' : 'pattern_id'},inplace=True)
        trips_to_patterns=trips_to_patterns.reset_index().merge(patterns,on=['chosen_geo_pattern_links','direction'])[['pattern_id','trip_id']]

    trips_to_patterns.set_index('trip_id',inplace=True)
    patterns.rename(columns={'chosen_geo_pattern_links':'geo_pattern_links'},inplace=True)
    patterns.set_index('pattern_id',inplace=True)



    return trips_to_patterns, patterns, trips_direction
