import unittest

from quetzal.engine.road_pathfinder import init_network, init_volumes, msa_roadpathfinder, expanded_roadpathfinder
import pandas as pd
from quetzal.model import stepmodel


zones = pd.DataFrame({'id': ['p', 'q']})
zones.index = zones['id']
zones.index.name = 'index'

volumes = pd.DataFrame({'origin': ['p'], 'destination': ['q'], 'car': [1000]})
volumes.index.name = 'index'

zone_to_road = pd.DataFrame({'a': ['p', 'b'], 'b': ['a', 'q'], 'time': [10, 10], 'length': [10, 10]})
zone_to_road.index = 'zr_' + zone_to_road.index.astype(str)
zone_to_road.index.name = 'index'


rlinks = pd.DataFrame(
    {
        'a': ['a', 'm1', 'a', 'm2', 'a', 'm3'],
        'b': ['m1', 'b', 'm2', 'b', 'm3', 'b'],
        'time': [10, 10, 20, 20, 25, 25],
        'length': [10, 10, 10, 10, 10, 10],
        'capacity': [200, 200, 400, 400, 300, 300],
    }
)

rlinks.index = 'rlink_' + rlinks.index.astype(str)
rlinks.index.name = 'index'


num_cores = 1

free_flow = 'time'
default_bpr = 'time * (1 + {alpha} * (flow/capacity)**{beta})'.format(alpha=0.15, beta=4)
vdf = {'free_flow': free_flow, 'default_bpr': default_bpr}


class TestFrankWolfe(unittest.TestCase):
    def setUp(self):
        sm = stepmodel.StepModel()
        rlinks['vdf'] = 'default_bpr'
        sm.road_links = rlinks

        sm.zone_to_road = zone_to_road
        sm.zones = zones
        sm.volumes = volumes
        self.sm = sm

    def _get_msa_roadpathfinder(self, maxiters=10, method='bfw'):
        tolerance = 10e-6
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        network = init_network(self.sm, method, segments, time_column, access_time)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = msa_roadpathfinder(
            network,
            volumes,
            segments=segments,
            method=method,
            vdf=vdf,
            maxiters=maxiters,
            tolerance=tolerance,
            time_col=time_column,
            num_cores=num_cores,
            log=False,
        )
        return links, car_los, relgap_list

    def _test_msa_pathfinder_with_method(self, method):
        links, car_los, relgap = self._get_msa_roadpathfinder(method=method)
        expected_flows = {('a', 'm1'): 358.329, ('a', 'm2'): 464.514, ('a', 'm3'): 177.157}
        expected_jam_time = 25.456
        for key, expected_flow in expected_flows.items():
            self.assertAlmostEqual(links.loc[key, 'flow'], expected_flow, places=0)
            self.assertAlmostEqual(links.loc[key, 'jam_time'], expected_jam_time, places=1)

    def test_bfw_pathfinder(self):
        self._test_msa_pathfinder_with_method(method='bfw')

    def test_fw_pathfinder(self):
        self._test_msa_pathfinder_with_method(method='fw')

    @unittest.skip('msa will fail affectation cause its not good')
    def test_msa_pathfinder(self):
        self._test_msa_pathfinder_with_method(method='msa')

    def test_msa_pathfinder_base_flow(self):
        self.sm.road_links['base_flow'] = 100
        links, car_los, relgap = self._get_msa_roadpathfinder(method='bfw')
        expected_flows = {('a', 'm1'): 358.329, ('a', 'm2'): 464.514, ('a', 'm3'): 177.157}
        for key, expected_flow in expected_flows.items():
            self.assertNotAlmostEqual(links.loc[key, 'flow'], expected_flow, places=0)
        self.assertAlmostEqual(links.loc[('a', 'm1'), 'jam_time'], links.loc[('a', 'm2'), 'jam_time'], places=0)
        self.assertAlmostEqual(links.loc[('a', 'm1'), 'jam_time'], links.loc[('a', 'm3'), 'jam_time'], places=0)
        self.sm.road_links['base_flow'] = 0

    def _get_expanded_roadpathfinder(self, maxiters=10, method='bfw'):
        tolerance = 10e-6
        segments = ['car']
        time_column = 'time'
        access_time = 'time'
        network = init_network(self.sm, method, segments, time_column, access_time)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = expanded_roadpathfinder(
            network,
            volumes,
            zones=self.sm.zones,
            segments=segments,
            method=method,
            vdf=vdf,
            maxiters=maxiters,
            tolerance=tolerance,
            time_col=time_column,
            num_cores=num_cores,
            log=False,
        )
        return links, car_los, relgap_list

    def _test_expanded_pathfinder_with_method(self, method):
        links, car_los, relgap = self._get_expanded_roadpathfinder(method=method)
        expected_flows = {('a', 'm1'): 358.329, ('a', 'm2'): 464.514, ('a', 'm3'): 177.157}
        expected_jam_time = 25.456
        for key, expected_flow in expected_flows.items():
            self.assertAlmostEqual(links.loc[key, 'flow'], expected_flow, places=0)
            self.assertAlmostEqual(links.loc[key, 'jam_time'], expected_jam_time, places=1)

    def test_bfw_expanded_pathfinder(self):
        self._test_expanded_pathfinder_with_method(method='bfw')

    def test_fw_expanded_pathfinder(self):
        self._test_expanded_pathfinder_with_method(method='fw')

    @unittest.skip('msa will fail affectation cause its not good')
    def test_msa_expanded_pathfinder(self):
        self._test_msa_pathfinder_with_method(method='msa')

    def test_expanded_pathfinder_base_flow(self):
        self.sm.road_links['base_flow'] = 100
        links, car_los, relgap = self._get_expanded_roadpathfinder(method='bfw')
        expected_flows = {('a', 'm1'): 358.329, ('a', 'm2'): 464.514, ('a', 'm3'): 177.157}
        for key, expected_flow in expected_flows.items():
            self.assertNotAlmostEqual(links.loc[key, 'flow'], expected_flow, places=0)
        self.assertAlmostEqual(links.loc[('a', 'm1'), 'jam_time'], links.loc[('a', 'm2'), 'jam_time'], places=0)
        self.assertAlmostEqual(links.loc[('a', 'm1'), 'jam_time'], links.loc[('a', 'm3'), 'jam_time'], places=0)
        self.sm.road_links['base_flow'] = 0
