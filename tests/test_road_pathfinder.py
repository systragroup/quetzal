import unittest

from quetzal.engine.road_pathfinder import (
    init_network,
    init_volumes,
    aon_roadpathfinder,
    get_car_los_time,
    extended_roadpathfinder,
    msa_roadpathfinder,
)
import pandas as pd
from quetzal.model import stepmodel


zones = pd.DataFrame({'id': ['z1', 'z2']})
zones.index = zones['id']
zones.index.name = 'index'

volumes = pd.DataFrame({'origin': ['z1', 'z2'], 'destination': ['z2', 'z1'], 'car': [150, 100]})
volumes.index.name = 'index'

zone_to_road = pd.DataFrame(
    {
        'a': ['z1', 'a', 'e', 'z2', 'z2'],
        'b': ['a', 'z1', 'z2', 'e', 'c'],
        'time': [100, 100, 100, 100, 100],
        'length': [10, 10, 10, 10, 10],
        #'direction': ['access', 'access', 'eggress', 'eggress', 'access'],
    }
)
zone_to_road.index = 'zr_' + zone_to_road.index.astype(str)
zone_to_road.index.name = 'index'


rlinks = pd.DataFrame(
    {
        'a': ['a', 'b', 'b', 'c', 'b', 'd', 'd', 'e', 'e', 'c'],
        'b': ['b', 'a', 'c', 'b', 'd', 'b', 'e', 'd', 'c', 'e'],
        'time': [1, 1, 5, 5, 2, 2, 4, 4, 6, 6],
        'length': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    }
)

rlinks.index = 'rlink_' + rlinks.index.astype(str)
rlinks.index.name = 'index'

from quetzal.engine.vdf import default_bpr, free_flow

num_cores = 1
vdf = {'free_flow': free_flow, 'default_bpr': default_bpr}


class TestRoadPathfinder(unittest.TestCase):
    def setUp(self):
        sm = stepmodel.StepModel()
        rlinks['capacity'] = 50
        rlinks['vdf'] = 'free_flow'
        sm.road_links = rlinks

        sm.zone_to_road = zone_to_road
        sm.zones = zones
        sm.volumes = volumes
        self.sm = sm

    def test_init_network(self):
        method = 'bfw'
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)

        expected_len = len(self.sm.road_links) + len(self.sm.zone_to_road)
        self.assertEqual(len(network), expected_len)

        expected_columns = [
            'a',
            'b',
            'vdf',
            'segments',
            'flow',
            'auxiliary_flow',
            'base_flow',
            ('car', 'flow'),
            ('car', 'auxiliary_flow'),
        ]
        for col in expected_columns:
            self.assertIn(col, network.columns)

    def test_init_network_aon(self):
        method = 'aon'
        segments = []
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        expected_len = len(self.sm.road_links) + len(self.sm.zone_to_road)
        self.assertEqual(len(network), expected_len)

        expected_columns = ['a', 'b', 'time']
        for col in expected_columns:
            self.assertIn(col, network.columns)

    def test_init_volumes(self):
        volumes = init_volumes(self.sm)
        pd.testing.assert_frame_equal(volumes, self.sm.volumes, check_dtype=False)

    def test_init_volumes_return_volumes_if_no_volumes(self):
        sm = self.sm.copy()
        del sm.volumes
        volumes = init_volumes(sm)
        expected_len = len(sm.zones) * len(sm.zones)
        self.assertEqual(len(volumes), expected_len)
        self.assertIn('volume', volumes.columns)

    def _get_aon_los(self):
        method = 'aon'
        segments = []
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self.sm)
        car_los = aon_roadpathfinder(network, volumes, time_column, num_cores)
        return car_los

    def test_aon_pathfinder(self):
        car_los = self._get_aon_los()

        expected_path_1 = ['z1', 'a', 'b', 'd', 'e', 'z2']
        self.assertEqual(car_los['path'][0], expected_path_1)

        expected_path_2 = ['z2', 'c', 'b', 'a', 'z1']
        self.assertEqual(car_los['path'][1], expected_path_2)

    def test_get_car_los_time(self):
        car_los = self._get_aon_los()
        car_los = get_car_los_time(car_los, self.sm.road_links, self.sm.zone_to_road, 'time', 'time')

        self.assertEqual(car_los['time'][0], 207)
        self.assertEqual(car_los['time'][1], 206)

    def _get_msa_roadpathfinder(self, maxiters=10, method='bfw', track_links_list=[]):
        tolerance = 0.01
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = msa_roadpathfinder(
            network,
            volumes,
            segments=segments,
            method=method,
            vdf=vdf,
            maxiters=maxiters,
            tolerance=tolerance,
            track_links_list=track_links_list,
            log=False,
            time_col=time_column,
            num_cores=num_cores,
        )
        return links, car_los, relgap_list

    def test_msa_pathfinder(self):
        for method in ['bfw', 'msa', 'fw']:
            links, car_los, relgap = self._get_msa_roadpathfinder(method=method)
            expected_path_1 = ['z1', 'a', 'b', 'd', 'e', 'z2']
            self.assertEqual(car_los['path'][0], expected_path_1)

            expected_path_2 = ['z2', 'c', 'b', 'a', 'z1']
            self.assertEqual(car_los['path'][1], expected_path_2)

    def test_msa_pathfinder_track_link(self):
        index_dict = self.sm.road_links.reset_index().set_index(['a', 'b'])['index'].to_dict()

        link = index_dict.get(('b', 'd'))
        assert self.sm.road_links.loc[link, 'a'] == 'b'
        assert self.sm.road_links.loc[link, 'b'] == 'd'
        track_links_list = [link]
        links, car_los, relgap = self._get_msa_roadpathfinder(track_links_list=track_links_list)
        # expected_path = ['z1', 'a', 'b', 'd', 'e', 'z2']
        expected_link_path = [*map(index_dict.get, [('a', 'b'), ('b', 'd'), ('d', 'e')])]
        found_path = links[links[link] == 150]['index'].tolist()
        self.assertEqual(expected_link_path, found_path)

    def _get_extended_roadpathfinder(self, method='bfw', track_links_list=[]):
        maxiters = 10
        tolerance = 0.01
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = extended_roadpathfinder(
            network,
            volumes,
            self.sm.zones,
            segments=segments,
            method=method,
            maxiters=maxiters,
            tolerance=tolerance,
            vdf=vdf,
            track_links_list=track_links_list,
            log=False,
            time_col=time_column,
            zone_penalty=ntleg_penalty,
            num_cores=num_cores,
        )
        return links, car_los, relgap_list

    def test_extended_pathfinder(self):
        for method in ['bfw', 'msa', 'fw']:
            links, car_los, relgap = self._get_extended_roadpathfinder(method=method)
            expected_path_1 = ['z1', 'a', 'b', 'd', 'e', 'z2']
            self.assertEqual(car_los['path'][0], expected_path_1)

            expected_path_2 = ['z2', 'c', 'b', 'a', 'z1']
            self.assertEqual(car_los['path'][1], expected_path_2)

    def test_extended_pathfinder_track_link(self):
        index_dict = self.sm.road_links.reset_index().set_index(['a', 'b'])['index'].to_dict()

        link = index_dict.get(('b', 'd'))
        assert self.sm.road_links.loc[link, 'a'] == 'b'
        assert self.sm.road_links.loc[link, 'b'] == 'd'
        track_links_list = [link]
        links, car_los, relgap = self._get_extended_roadpathfinder(track_links_list=track_links_list)
        # expected_path = ['z1', 'a', 'b', 'd', 'e', 'z2']
        expected_link_path = [*map(index_dict.get, [('a', 'b'), ('b', 'd'), ('d', 'e')])]
        found_path = links[links[link] == 150]['index'].tolist()
        self.assertEqual(expected_link_path, found_path)

    def _get_turn_penality_roadpathfinder(self):
        method = 'bfw'
        maxiters = 10
        tolerance = 0.01
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        turn_penalty = 100
        turn_penalties = {'rlink_0': ['rlink_4']}
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = extended_roadpathfinder(
            network,
            volumes,
            self.sm.zones,
            segments=segments,
            method=method,
            maxiters=maxiters,
            tolerance=tolerance,
            vdf=vdf,
            log=False,
            time_col=time_column,
            zone_penalty=ntleg_penalty,
            turn_penalty=turn_penalty,
            track_links_list=[],
            turn_penalties=turn_penalties,
            num_cores=num_cores,
        )
        return links, car_los, relgap_list

    def test_turn_penality_roadpathfinder(self):
        links, car_los, relgap = self._get_turn_penality_roadpathfinder()
        expected_path_1 = ['z1', 'a', 'b', 'c', 'e', 'z2']
        self.assertEqual(car_los['path'][0], expected_path_1)

        expected_path_2 = ['z2', 'c', 'b', 'a', 'z1']
        self.assertEqual(car_los['path'][1], expected_path_2)

        expected_columns = ['flow', 'jam_time']
        for col in expected_columns:
            self.assertIn(col, links.columns)
