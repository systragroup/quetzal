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

    def summary_earning(self, inplace=False, dense=False):
        """
        summarize earnings by fare_id and by segment
        """
        segments = self.segments
        df = pd.merge(self.volumes[['origin', 'destination'] + list(segments)], self.pt_los)

        for segment in segments:
            df[segment] =  df[segment] * df[(segment, 'probability')]

        df = df.dropna(subset=['price_breakdown'])
        df['fare_id_tuple'] = df['route_fares'].apply(
            lambda d: tuple(d.items())
        ) + df['od_fares'].apply(
            lambda d: tuple(d.items())
        )
        agg_dict = {segment: 'sum' for segment in segments}
        
        agg_dict['route_fares'] = 'first'
        agg_dict['od_fares'] = 'first'
        
        temp = df.groupby('fare_id_tuple').agg(agg_dict)
        
        fare_id_set = set(self.fare_rules['fare_id'])
        fare_revenue_dict = {
            segment :{f:0 for f in fare_id_set}
            for segment in segments
        }
        
        agency_id_set = set(self.fare_attributes['agency_id'])
        agency_revenue_dict = {
            segment :{f:0 for f in agency_id_set}
            for segment in segments
        }
        
        agency_dict = self.fare_attributes.set_index(['fare_id'])['agency_id']

        def row_revenue(row, segment):
            for key, value in row['route_fares'].items():
                fare_revenue_dict[segment][key] += value * row[segment]
                agency_revenue_dict[segment][agency_dict[key]] += value * row[segment]
            for key, value in row['od_fares'].items():
                agency_revenue_dict[segment][key] += value * row[segment]


        for segment in segments:
            _ = temp.apply(row_revenue, axis=1, segment=segment)



        fare_stack = pd.DataFrame(fare_revenue_dict).stack()
        agency_stack = pd.DataFrame(agency_revenue_dict).stack()
        stack = pd.concat([fare_stack, agency_stack])
        stack.index.names = ['fare_id', 'segment']
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_earning = stack
        else:
            return stack

    def summary_path_sum(self, inplace=False, dense=False):

        """
        focuses on user perception
        processes self.car_los, self.pt_los and self.volume
        summarize 'time', 'in_vehicle_time', 'in_vehicle_length', 
        'count', 'price', 'ntransfers' by segment and route_type
        """
        segments = self.segments

        try:
            left = pd.concat([self.car_los, self.pt_los])
        except AttributeError:
            try:
                left =self.pt_los
            except AttributeError:
                left =self.car_los
            
        right = self.volumes[['origin', 'destination'] + list(segments)]
        df = pd.merge(left, right, on=['origin', 'destination'])
        
        df.reset_index(drop=True)

        df['count'] = 1
        columns = [
            'time', 'in_vehicle_time', 'in_vehicle_length', 
            'count', 'price', 'ntransfers', 'length'
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

    def summary_link_sum(self, route_label='route_id', inplace=False, dense=False):

        """
        focuses on network use
        processes self.loaded_links
        summarize 'boardings' and 'length' 
        by segment, route_type, route_id and trip_id
        """
        segments = self.segments
    
        df = self.loaded_links.copy()
        columns = []
        for segment in segments:
            columns += [(segment, c) for c in ['boardings']]

        to_concat = [
            df[columns + ['route_type', route_label,'trip_id'] ]]
        
        columns = ['length', 'time']
        
        for segment in segments:
            pool = df[columns].apply(lambda c: c*df[segment])
            pool.columns = [(segment, c) for c in columns]
            to_concat.append(pool)
            
        idf = pd.concat(to_concat, axis=1)

        g = idf.groupby(['route_type', route_label,'trip_id']).sum()
        g.columns = pd.MultiIndex.from_tuples(g.columns)
        stack = g.stack().stack()
        stack.index.names = ['route_type', route_label,'trip_id', 'indicator', 'segment']
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_sum = stack
        else:
            return stack

    def summary_link_max(self, route_label='route_id', segments=('root',), inplace=False, dense=False):
        """
        focuses on network use
        processes self.loaded_links
        calculate maximum demand
        by segment, route_type, route_id and trip_id
        """
        segments = list(self.segments)
        df = self.loaded_links
        stack = df[
            ['route_type', route_label,'trip_id'] + segments
        ].groupby(['route_type', route_label,'trip_id']).max().stack()
        stack.index.names = ['route_type', route_label,'trip_id', 'segment']
        stack.name = 'max'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_max = stack
        else:
            return stack
    
    def summary_path_average(self, inplace=False, dense=False, complete=True):
        segments = self.segments
        s = self.summary_path_sum() if complete else self.stack_path_sum
        us = s.unstack('indicator')
        us = us.apply(lambda c: c/us['count'])
        stack = us.fillna(0).stack()
        stack.name = 'average'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_path_average = stack
        else:
            return stack

    def summary_aggregated_path_average(self, inplace=False, dense=False,  pt_route_types=set(),complete=True):
        """
        focuses on user perception
        by route_type
        """
        segments = self.segments
        stack = self.summary_path_sum() if complete else self.stack_path_sum
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

    def summary_od(
        self, 
        costs=['price', 'time', 'in_vehicle_time', 'in_vehicle_length', 'ntransfers'],
        pt_route_types=set(), 
        inplace=False
    ):
        segments = self.segments
        try:
            left = pd.concat([self.car_los, self.pt_los])
        except AttributeError:
            try:
                left =self.pt_los
            except AttributeError:
                left =self.car_los

        right = self.volumes[['origin', 'destination'] + list(segments)]
        los = pd.merge(left, right, on=['origin', 'destination'], suffixes=['_old', ''])
        columns = []
        los['mode'] = los['route_type']
        los.loc[los['route_type'].isin(pt_route_types), 'mode'] = 'pt'
        

        for segment in segments:
            los[(segment, 'volume')] = los[(segment, 'probability')] * los[segment]


        for segment in segments:
            columns.append((segment, 'volume'))
            for service in costs:
                column = (segment, service)
                columns.append(column)
                los[column] = los[(segment, 'probability')] * los[service]

        df = los.groupby(['origin', 'destination', 'mode'])[columns].sum()      
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        # add root (weighted mean of segments)

        df[('root', 'volume')] = sum([df[(segment, 'volume')].fillna(0) for segment in segments])

        weighted_sum = sum(
            [
                df[segment][costs].apply(lambda s: s * df[(segment, 'volume')]).fillna(0)
                for segment in segments
            ]      
        )

        for c in costs: 
            df[('root', c)] = (weighted_sum[c] / df[('root', 'volume')]).fillna(0)

        df.columns.names = 'segment', 'sum'

        if inplace:
            self.od_los = df
        else:
            return df


