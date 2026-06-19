import unittest
import pandas as pd
import polars as pl
import numpy as np
from quetzal.engine.add_network_common_links import _get_shared_stops

trip1 = pd.DataFrame(
    {
        'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
        'link_sequence': [11, 12, 13, 14, 15, 16, 17],
        'trip_id': 'trip_1',
    }
)
trip2 = pd.DataFrame(
    {
        'stop': ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7', 'node_8'],
        'link_sequence': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'trip_id': 'trip_2',
    }
)
trip3 = pd.DataFrame(
    {
        'stop': ['node_4', 'node_3', 'node_2', 'node_52', 'node_6', 'node_7'],
        'link_sequence': [1, 2, 3, 4, 5, 6],
        'trip_id': 'trip_3',
    }
)

trip4 = pd.DataFrame(
    {
        'stop': ['node_4', 'node_6', 'node_42', 'node_1', 'node_2', 'node_3'],
        'link_sequence': [1, 2, 3, 4, 5, 6],
        'trip_id': 'trip_4',
    }
)
trip_4_short = pd.DataFrame(
    {
        'stop': ['node_4', 'node_6', 'node_42', 'node_1'],
        'link_sequence': [1, 2, 3, 4],
        'trip_id': 'trip_4_short',
    }
)


def _concat_trips(trips):
    stops_list = pd.concat(trips)
    stops_list['index'] = 'link_' + stops_list.index.astype(str)
    stops_list = stops_list.sort_values(['trip_id', 'link_sequence'])
    stop_list_dict = stops_list.groupby('trip_id')['stop'].agg(np.array).to_dict()
    pl_stops_list = pl.DataFrame(stops_list)
    return pl_stops_list, stop_list_dict


stops_list = pd.concat([trip1, trip2, trip3, trip4, trip_4_short])
stops_list['index'] = 'link_' + stops_list.index.astype(str)
stops_list = stops_list.sort_values(['trip_id', 'link_sequence'])
stop_list_dict = stops_list.groupby('trip_id')['stop'].agg(np.array).to_dict()
pl_stops_list = pl.DataFrame(stops_list)


class TestCommonLinks(unittest.TestCase):
    def testCommonStops(self):

        trip1 = pd.DataFrame(
            {
                'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
                'link_sequence': [11, 12, 13, 14, 15, 16, 17],
                'trip_id': 'trip_1',
            }
        )
        trip2 = pd.DataFrame(
            {
                'stop': ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7', 'node_8'],
                'link_sequence': [1, 2, 3, 4, 5, 6, 7, 8, 10],
                'trip_id': 'trip_2',
            }
        )

        pl_stops_list, _ = _concat_trips([trip1, trip2])
        res = _get_shared_stops(pl_stops_list, {'trip_1', 'trip_2'})
        expected = ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7']
        self.assertEqual(list(res), expected)

    def testCommonStopsSkipNonShareNodes(self):
        trip1 = pd.DataFrame(
            {
                'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
                'link_sequence': [11, 12, 13, 14, 15, 16, 17],
                'trip_id': 'trip_1',
            }
        )
        trip2_bis = pd.DataFrame(
            {
                'stop': ['node_0', 'node_1', 'node_4', 'node_5', 'node_6', 'node_7', 'node_8'],
                'link_sequence': [1, 2, 3, 4, 5, 6, 7],
                'trip_id': 'trip_2_bis',
            }
        )

        pl_stops_list, _ = _concat_trips([trip1, trip2_bis])
        res = _get_shared_stops(pl_stops_list, {'trip_1', 'trip_2_bis'})
        expected = ['node_1', 'node_4', 'node_5', 'node_6', 'node_7']
        self.assertEqual(list(res), expected)

    def testCommonStopWithReversedStart(self):
        trip1 = pd.DataFrame(
            {
                'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
                'link_sequence': [11, 12, 13, 14, 15, 16, 17],
                'trip_id': 'trip_1',
            }
        )
        trip3 = pd.DataFrame(
            {
                'stop': ['node_4', 'node_3', 'node_2', 'node_52', 'node_6', 'node_7'],
                'link_sequence': [1, 2, 3, 4, 5, 6],
                'trip_id': 'trip_3',
            }
        )
        pl_stops_list, _ = _concat_trips([trip1, trip3])
        res = _get_shared_stops(pl_stops_list, {'trip_1', 'trip_3'})
        expected = ['node_4', 'node_6', 'node_7']
        self.assertEqual(list(res), expected)
        # TODO the path  ['node_2', 'node_6', 'node_7'] is also valid

    def testCommonStopReturnNonValidPath(self):
        trip1 = pd.DataFrame(
            {
                'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
                'link_sequence': [11, 12, 13, 14, 15, 16, 17],
                'trip_id': 'trip_1',
            }
        )
        trip4 = pd.DataFrame(
            {
                'stop': ['node_4', 'node_6', 'node_42', 'node_1', 'node_2', 'node_3'],
                'link_sequence': [1, 2, 3, 4, 5, 6],
                'trip_id': 'trip_4',
            }
        )
        pl_stops_list, _ = _concat_trips([trip1, trip4])
        res = _get_shared_stops(pl_stops_list, {'trip_1', 'trip_4'})
        expected = ['node_1', 'node_2', 'node_4', 'node_6']
        self.assertEqual(list(res), expected)
        # TODO: this is not valud. it will return nothing in the next step. the order is not right.
        # we want 2 common trips here. 1-2 and 4-6. but no connection between 2-4

    def testCommonStopsWithFinishAtBeginningOfLine(self):
        # just return the common, the last stop is skipped
        trip1 = pd.DataFrame(
            {
                'stop': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'],
                'link_sequence': [11, 12, 13, 14, 15, 16, 17],
                'trip_id': 'trip_1',
            }
        )
        trip_4_short = pd.DataFrame(
            {
                'stop': ['node_4', 'node_6', 'node_42', 'node_1'],
                'link_sequence': [1, 2, 3, 4],
                'trip_id': 'trip_4_short',
            }
        )
        pl_stops_list, _ = _concat_trips([trip1, trip_4_short])
        res = _get_shared_stops(pl_stops_list, {'trip_1', 'trip_4_short'})
        expected = ['node_4', 'node_6']
        self.assertEqual(list(res), expected)

        # trip_set = {'trip_1', 'trip_2', 'trip_4_short'}
        # res = _get_shared_stops(pl_stops_list, trip_set)
        # expected = np.array(['node_4', 'node_6'])
        # self.assertEqual(res, expected)
