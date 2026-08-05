import unittest

import pandas as pd
from quetzal.model import stepmodel


zones = pd.DataFrame({'id': ['z1', 'z2']})
zones.index = zones['id']
zones.index.name = 'index'

volumes = pd.DataFrame(
    {
        'origin': ['z1', 'z2'],
        'destination': ['z2', 'z1'],
        'car': [150, 100],
    }
)
volumes.index.name = 'index'

zone_to_road = pd.DataFrame(
    {
        'a': ['z1', 'a', 'e', 'z2', 'z2'],
        'b': ['a', 'z1', 'z2', 'e', 'c'],
        'time': [100, 100, 100, 100, 100],
        'length': [10, 10, 10, 10, 10],
        'direction': ['access', 'eggress', 'eggress', 'access', 'access'],
    }
)
zone_to_road.index = 'zr_' + zone_to_road.index.astype(str)
zone_to_road.index.name = 'index'


rlinks = pd.DataFrame(
    {
        'a': ['a', 'b', 'b', 'c', 'b', 'd', 'd', 'e', 'e', 'c'],
        'b': ['b', 'a', 'c', 'b', 'd', 'b', 'e', 'd', 'c', 'e'],
        'time': [1, 1, 5, 5, 2, 2, 4, 4, 6, 6],
        'length': [10] * 10,
        'capacity': [50] * 10,
        'vdf': ['free_flow'] * 10,
    }
)

rlinks.index = 'rlink_' + rlinks.index.astype(str)
rlinks.index.name = 'index'


class TestTransportModel(unittest.TestCase):
    def setUp(self):
        sm = stepmodel.StepModel()
        sm.road_links = rlinks
        sm.zone_to_road = zone_to_road
        sm.zones = zones
        sm.volumes = volumes
        self.sm = sm

    def _test_car_los(self, sm, method: str):
        car_los = sm.car_los
        if method == 'aon':
            expected_columns = ['origin', 'destination', 'path', 'edge_path', 'time']
        else:
            expected_columns = ['origin', 'destination', 'path', 'edge_path', 'time', 'segment']

        for col in expected_columns:
            self.assertIn(col, car_los.columns)

        expected_path_1 = ['z1', 'a', 'b', 'd', 'e', 'z2']
        self.assertEqual(car_los['path'][0], expected_path_1)

        expected_path_2 = ['z2', 'c', 'b', 'a', 'z1']
        self.assertEqual(car_los['path'][1], expected_path_2)

    def _test_car_los_time(self, sm, jam_time='jam_time', access_time='jam_time'):
        # test that time is all the paths time with connectors
        time_dict = sm.road_links.set_index(['a', 'b'])[jam_time].to_dict()
        time_dict.update(sm.zone_to_road.set_index(['a', 'b'])[access_time].to_dict())

        test = sm.car_los['edge_path'].apply(lambda ls: sum([time_dict.get(x) for x in ls]))

        self.assertEqual(list(test.values), list(sm.car_los['time'].values))

    def test_msa_step_road_pathfinder(self):
        sm = self.sm
        kwargs = {
            'maxiters': 5,
            'tolerance': 0.01,
            'segments': ['car'],
            'time_column': 'time',
            'access_time': 'time',
            'turn_penalties': None,
            'return_car_los': True,
            'assign_on_connectors': True,
        }
        for method in ['aon', 'bfw', 'msa', 'fw']:
            sm.step_road_pathfinder(method=method, **kwargs)
            self._test_car_los(sm, method)

            if method == 'aon':
                self._test_car_los_time(sm, 'time', 'time')
            else:
                self._test_car_los_time(sm, 'jam_time', 'jam_time')

    def test_expanded_step_road_pathfinder(self):
        sm = self.sm
        kwargs = {
            'maxiters': 5,
            'tolerance': 0.01,
            'segments': ['car'],
            'time_column': 'time',
            'access_time': 'time',
            'turn_penalties': {},
            'return_car_los': True,
            'assign_on_connectors': False,
        }
        for method in ['aon', 'bfw', 'msa', 'fw']:
            sm.step_road_pathfinder(method=method, **kwargs)
            self._test_car_los(sm, method)

            if method == 'aon':
                self._test_car_los_time(sm, 'time', 'time')
            else:
                self._test_car_los_time(sm, 'jam_time', 'time')
                # assert columns are created
                expected_columns = ['jam_time', 'jam_speed', 'flow', ('car', 'cost'), ('car', 'flow')]
                for col in expected_columns:
                    self.assertIn(col, sm.road_links.columns)

                # assert there is volume on links
                self.assertTrue(sm.road_links[('car', 'flow')].sum() > 0)

    def test_expanded_step_road_pathfinder_with_connectors(self):
        sm = self.sm
        kwargs = {
            'maxiters': 5,
            'tolerance': 0.01,
            'segments': ['car'],
            'time_column': 'time',
            'access_time': 'time',
            'turn_penalties': {},
            'return_car_los': True,
            'assign_on_connectors': True,
        }
        for method in ['aon', 'bfw', 'msa', 'fw']:
            sm.step_road_pathfinder(method=method, **kwargs)
            self._test_car_los(sm, method)

            if method == 'aon':
                self._test_car_los_time(sm, 'time', 'time')
            else:
                self._test_car_los_time(sm, 'jam_time', 'jam_time')
                # assert columns are created
                expected_columns = ['jam_time', 'jam_speed', 'flow', ('car', 'cost'), ('car', 'flow')]
                for col in expected_columns:
                    self.assertIn(col, sm.road_links.columns)
                    self.assertIn(col, sm.zone_to_road.columns)

                # assert volumes on connectors respect total volume
                self.assertEqual(sm.zone_to_road[sm.zone_to_road['a'] == 'z1'][('car', 'flow')].sum(), 150)
                self.assertEqual(sm.zone_to_road[sm.zone_to_road['a'] == 'z2'][('car', 'flow')].sum(), 100)

                self.assertEqual(sm.zone_to_road[sm.zone_to_road['b'] == 'z1'][('car', 'flow')].sum(), 100)
                self.assertEqual(sm.zone_to_road[sm.zone_to_road['b'] == 'z2'][('car', 'flow')].sum(), 150)
                # assert there is volume on links
                self.assertTrue(sm.road_links[('car', 'flow')].sum() > 0)
