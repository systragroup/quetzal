import numpy as np
import pandas as pd
from quetzal.model import model, transportmodel


def densify(series):
    index = pd.MultiIndex.from_product(
        series.index.levels,
        names=series.index.names
    )
    dense = pd.Series(np.nan, index).fillna(series).fillna(0)
    dense.name = series.name
    return dense


class SummaryModel(transportmodel.TransportModel):
    def summary_earning(self, inplace=False, dense=False):
        """
        summarize earnings by fare_id and by segment
        """
        df = self.los.copy()

        msg = 'los must have either route_fares or od_fares'
        assert ('od_fares' in df.columns) | ('route_fares' in df.columns), msg
        if 'route_fares' not in df.columns:
            df['route_fares'] = np.nan
        if 'od_fares' not in df.columns:
            df['od_fares'] = np.nan

        df.dropna(subset=['od_fares', 'route_fares'], inplace=True, how='all')
        df[['route_fares', 'od_fares']] = df[['route_fares', 'od_fares']].where(
            df[['route_fares', 'od_fares']].notna(), lambda x: [{}])

        df['fare_id_tuple'] = df['route_fares'].apply(
                lambda d: tuple(d.items())
            ) + df['od_fares'].apply(
                lambda d: tuple(d.items())
            )

        agg_dict = {'volume': sum}
        agg_dict['route_fares'] = 'first'
        agg_dict['od_fares'] = 'first'

        temp = df.groupby('fare_id_tuple').agg(agg_dict)

        agency_dict = self.fare_attributes.set_index(['fare_id'])['agency_id']
        fare_revenue_dict = {f: 0 for f in set(self.fare_rules['fare_id'])}
        agency_revenue_dict = {f: 0 for f in set(self.fare_attributes['agency_id'])}

        for volume, route_fares, od_fares in temp[['volume', 'route_fares', 'od_fares']].values:
            for fare_id, fare in route_fares.items():
                fare_revenue_dict[fare_id] += volume * fare
                agency_revenue_dict[agency_dict[fare_id]] += fare * volume
            for agency_id, fare in od_fares.items():
                agency_revenue_dict[agency_id] += volume * fare

        fare_stack = pd.Series(fare_revenue_dict)
        agency_stack = pd.Series(agency_revenue_dict)
        stack = pd.concat([fare_stack, agency_stack])
        stack.index.name = 'fare_id'
        stack.name = 'sum'
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
        df = self.los.copy()
        df['count'] = 1
        columns = [
            'time', 'in_vehicle_time', 'in_vehicle_length',
            'price', 'ntransfers', 'length', 'count'

        ]
        for c in columns:
            df[c] = df[c] * df['volume']
        stack = df.groupby('route_type')[columns].sum().stack()
        stack.index.names = ['route_type', 'indicator']
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
        df = self.links[['length', 'boardings', 'time', 'route_type', route_label, 'trip_id']]
        stack = df.groupby(['route_type', route_label, 'trip_id']).sum().stack()
        stack.index.names = ['route_type', route_label, 'trip_id', 'indicator']
        stack.name = 'sum'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_sum = stack
        else:
            return stack

    def summary_link_max(
        self, route_label='route_id', segments=('root',), inplace=False, dense=False
    ):
        """
        focuses on network use
        processes self.loaded_links
        calculate maximum demand
        by segment, route_type, route_id and trip_id
        """
        df = self.links.copy()
        stack = df[
            ['route_type', route_label, 'trip_id', 'volume']
        ].groupby(['route_type', route_label, 'trip_id']).max().stack()
        stack.index.names = ['route_type', route_label, 'trip_id', 'segment']
        stack.name = 'max'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_link_max = stack
        else:
            return stack

    def summary_path_average(self, inplace=False, dense=False, complete=True):
        us = self.summary_path_sum().unstack('indicator')
        us = us.apply(lambda c: c / us['count'])
        us = us.drop('count', axis=1)
        stack = us.fillna(0).stack()
        stack.name = 'average'
        stack = densify(stack) if dense else stack
        if inplace:
            self.stack_path_average = stack
        else:
            return stack

    def summary_aggregated_path_average(
        self, inplace=False, dense=False, pt_route_types=set(), complete=True
    ):
        """
        focuses on user perception
        by route_type
        """
        stack = self.summary_path_sum()
        stack = stack.reset_index()
        stack['route_type'] = stack['route_type'].apply(
            lambda rt: 'pt' if rt in pt_route_types else rt)

        total = stack.groupby(
            ['route_type', 'indicator']
        ).sum()

        us = total['sum'].unstack('indicator')
        total = stack.groupby(
            ['route_type', 'indicator']
        ).sum()

        us = total['sum'].unstack('indicator')
        share = (us['count'] / us['count'].sum())
        us = us.apply(lambda c: c / us['count'])
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
                left = self.pt_los
            except AttributeError:
                left = self.car_los

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
