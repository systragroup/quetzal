# -*- coding: utf-8 -*-

from quetzal.model import model, transportmodel
import pandas as pd
import numpy as np

def densify(series):
    index = pd.MultiIndex.from_product(
        series.index.levels,
        names=series.index.names
    )
    dense = pd.Series(np.nan, index).fillna(series).fillna(0)
    dense.name=series.name
    return dense

class SummaryModel(transportmodel.TransportModel):

    def summary_earning(self, segments=('root',), inplace=False, dense=False):
        """
        summarize earnings by fare_id and by segment
        """
        df = pd.merge(self.volumes[['origin', 'destination'] + list(segments)], self.pt_los)
        
        for segment in segments:
            df[segment] =  df[segment] * df[(segment, 'probability')]

        
        df = df.dropna(subset=['price_breakdown'])
        df['fare_id_tuple'] = df['fare_id_list'].apply(tuple)
        agg_dict = {segment: 'sum' for segment in segments}
        agg_dict['price_breakdown'] = 'first'
        temp = df.groupby('fare_id_tuple').agg(agg_dict)
        fare_id_set = set.union(*[set(t) for t in temp.index])
        revenue_dict = {
            segment :{f:0 for f in fare_id_set}
            for segment in segments
        }

        def row_revenue(row, segment):
            for key, value in row['price_breakdown'].items():
                revenue_dict[segment][key] += value * row[segment]

        
        for segment in segments:
            _ = temp.apply(row_revenue, axis=1, segment=segment)
            
        stack = pd.DataFrame(revenue_dict).stack()
        stack.index.names = ['fare_id', 'segment']
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_earning = stack
        else:
            return stack

    def summary_path_sum(self, segments=('root',), inplace=False, dense=False):

        """
        focuses on user perception
        processes self.car_los, self.pt_los and self.volume
        summarize 'time', 'in_vehicle_time', 'in_vehicle_length', 
        'count', 'price', 'ntransfers' by segment and route_type
        """

        left = pd.concat([self.car_los, self.pt_los])
        right = self.volumes[['origin', 'destination'] + list(segments)]
        df = pd.merge(left, right, on=['origin', 'destination'])
        
        df.reset_index(drop=True)

        df['count'] = 1
        columns = [
            'time', 'in_vehicle_time', 'in_vehicle_length', 
            'count', 'price', 'ntransfers'
        ]
        idf = df[['route_type']]

        to_concat = []
        for segment in segments:
            df[(segment, 'volume')] = df[(segment, 'probability')] * df[segment]
            pool = pd.DataFrame(
                df[columns].apply(lambda c: c*df[(segment, 'volume')]),
            )
            pool.columns = [(segment, c) for c in columns]
            to_concat.append(pool)
            
        idf = pd.concat([idf] + to_concat, axis=1)
        frame = idf.fillna(0).groupby('route_type').sum().T.sort_index()
        frame.index = pd.MultiIndex.from_tuples(frame.index)
        frame.index.names = ['segment', 'indicator']

        stack = frame.stack()
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_path_sum = stack
        else:
            return stack

    def summary_link_sum(self, segments=('root',), inplace=False, dense=False):

        """
        focuses on network use
        processes self.loaded_links
        summarize 'boardings' and 'length' 
        by segment, route_type, route_id and trip_id
        """
    
        df = self.loaded_links.copy()
        columns = []
        for segment in segments:
            columns += [(segment, c) for c in ['boardings']]

        to_concat = [
            df[columns + ['route_type', 'route_id','trip_id'] ]]
        
        columns = ['length', 'time']
        
        for segment in segments:
            pool = df[columns].apply(lambda c: c*df[segment])
            pool.columns = [(segment, c) for c in columns]
            to_concat.append(pool)
            
        idf = pd.concat(to_concat, axis=1)

        g = idf.groupby(['route_type', 'route_id','trip_id']).sum()
        g.columns = pd.MultiIndex.from_tuples(g.columns)
        stack = g.stack().stack()
        stack.index.names = ['route_type', 'route_id','trip_id', 'indicator', 'segment']
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_sum = stack
        else:
            return stack

    def summary_link_max(self, segments=('root',), inplace=False, dense=False):
        """
        focuses on network use
        processes self.loaded_links
        calculate maximum demand
        by segment, route_type, route_id and trip_id
        """
        df = self.loaded_links
        stack = df[
            ['route_type', 'route_id','trip_id'] + segments
        ].groupby(['route_type', 'route_id','trip_id']).max().stack()
        stack.index.names = ['route_type', 'route_id','trip_id', 'segment']
        stack.name = 'max'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_max = stack
        else:
            return stack
    
    def summary_path_average(self, segments=('root',), inplace=False, dense=False, complete=True):
        s = self.summary_path_sum(segments=segments) if complete else self.stack_path_sum
        us = s.unstack('indicator')
        us = us.apply(lambda c: c/us['count'])
        stack = us.fillna(0).stack()
        stack.name = 'average'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_path_average = stack
        else:
            return stack

    def summary_aggregated_path_average(
        self, segments=('root',), inplace=False, dense=False,  pt_route_types=set(), complete=True):
        """
        focuses on user perception
        by route_type
        """

        stack = self.summary_path_sum(segments=segments) if complete else self.stack_path_sum
        stack = stack.reset_index()
        stack['route_type'] = stack['route_type'].apply(
            lambda rt: 'pt' if rt in pt_route_types else rt)

        total = stack.groupby(
            ['route_type', 'indicator']
        ).sum()

        us = total['sum'].unstack('indicator')
        share = (us['count'] / us['count'].sum())
        us = us.apply(lambda c: c/us['count'])
        us['share'] = share
        stack = us.stack()
        stack.name = 'average'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_aggregated_path_average = stack
        else:
            return stack


