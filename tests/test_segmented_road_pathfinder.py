import unittest

from quetzal.engine.road_pathfinder import init_network, init_volumes, msa_roadpathfinder, expanded_roadpathfinder
from quetzal.engine.msa_plugins import LinksTracker
import pandas as pd
from quetzal.model import stepmodel


zones = pd.DataFrame({'id': ['z1', 'z2']})
zones.index = zones['id']
zones.index.name = 'index'

volumes = pd.DataFrame({'origin': ['z1', 'z2'], 'destination': ['z2', 'z1'], 'car': [150, 100], 'truck': [100, 0]})
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
        'length': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'base_flow': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    }
)

rlinks.index = 'rlink_' + rlinks.index.astype(str)
rlinks.index.name = 'index'


from quetzal.engine.vdf import default_bpr, free_flow

num_cores = 1
vdf = {'free_flow': free_flow, 'default_bpr': default_bpr}


class TestSegmentedRoadPathfinder(unittest.TestCase):
    def setUp(self):
        sm = stepmodel.StepModel()
        rlinks['capacity'] = 50
        rlinks['vdf'] = 'default_bpr'
        sm.road_links = rlinks

        sm.zone_to_road = zone_to_road
        sm.zones = zones
        sm.volumes = volumes
        self.sm = sm

    def _get_msa_roadpathfinder(self, maxiters=10, method='bfw', segments=['car'], **kwargs):
        tolerance = 0.01
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
            log=False,
            time_col=time_column,
            num_cores=num_cores,
            **kwargs,
        )
        return links, car_los, relgap_list

    def test_msa_pathfinder_flow_SHOULD_be_the_sum_of_segments(self):
        self.sm.road_links['segments'] = [set(['car', 'truck']) for _ in range(len(self.sm.road_links))]
        segments = ['car', 'truck']
        self.sm.road_links.loc['rlink_4', 'segments'] = set(['car'])
        links, car_los, relgap = self._get_msa_roadpathfinder(segments=segments)

        flow_agg = links['flow'].values
        cols = [(seg, 'flow') for seg in segments] + ['base_flow']
        flow_segmented = links[cols].sum(axis=1).values
        for a, b in zip(flow_agg, flow_segmented):
            self.assertAlmostEqual(a, b, places=3)

    def test_msa_pathfinder_with_fw_SHOULD_track_links_volumes(self):
        self.sm.road_links['segments'] = [set(['car', 'truck']) for _ in range(len(self.sm.road_links))]
        segments = ['car', 'truck']
        self.sm.road_links.loc['rlink_4', 'segments'] = set(['car'])
        link = ['rlink_2']
        tracker = LinksTracker(link)
        links, car_los, relgap = self._get_msa_roadpathfinder(segments=segments, method='fw', tracker_plugin=tracker)
        links = links.set_index('index')

        tracked_flow = tracker.merge()

        expected_flow = links.loc[link][('car', 'flow')]
        car_flow = tracked_flow['car']
        self.assertAlmostEqual(expected_flow.values[0], car_flow.loc[link, link].values[0][0], places=3)

        expected_flow = links.loc[link][('truck', 'flow')]
        truck_flow = tracked_flow['truck']
        self.assertAlmostEqual(expected_flow.values[0], truck_flow.loc[link, link].values[0][0], places=3)

        expected_flow = links.loc[link, 'flow'] - links.loc[link, 'base_flow']
        tot_flow = car_flow + truck_flow
        self.assertAlmostEqual(expected_flow.values[0], tot_flow.loc[link, link].values[0][0], places=3)

    def test_expanded_pathfinder_car_los_SHOULD_contain_all_segments(self):
        segments = ['car', 'truck']
        links, car_los, relgap = self._get_msa_roadpathfinder(segments=segments, method='bfw')

        expected_columns = ['origin', 'destination', 'path', 'segment']
        for col in expected_columns:
            self.assertIn(col, car_los.columns)
        for seg in segments:
            self.assertIn(seg, car_los['segment'].unique())

    def _get_expanded_roadpathfinder(self, method='bfw', segments=['car'], **kwargs):
        maxiters = 10
        tolerance = 0.1
        time_column = 'time'
        access_time = 'time'
        ntleg_penalty = 100
        network = init_network(self.sm, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self.sm)
        links, car_los, relgap_list = expanded_roadpathfinder(
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
            num_cores=num_cores,
            **kwargs,
        )
        return links, car_los, relgap_list

    def test_expanded_pathfinder_flow_SHOULD_be_the_sum_of_segments(self):
        self.sm.road_links['segments'] = [set(['car', 'truck']) for _ in range(len(self.sm.road_links))]
        segments = ['car', 'truck']
        self.sm.road_links.loc['rlink_4', 'segments'] = set(['car'])
        links, car_los, relgap = self._get_expanded_roadpathfinder(segments=segments, method='bfw')

        flow_agg = links['flow'].values
        cols = [(seg, 'flow') for seg in segments] + ['base_flow']
        flow_segmented = links[cols].sum(axis=1).values
        for a, b in zip(flow_agg, flow_segmented):
            self.assertAlmostEqual(a, b, places=3)

    def test_expanded_pathfinder_with_fw_SHOULD_track_links_volumes(self):
        self.sm.road_links['segments'] = [set(['car', 'truck']) for _ in range(len(self.sm.road_links))]
        segments = ['car', 'truck']
        self.sm.road_links.loc['rlink_4', 'segments'] = set(['car'])
        link = ['rlink_2']
        tracker = LinksTracker(link)
        links, car_los, relgap = self._get_expanded_roadpathfinder(
            segments=segments, method='fw', tracker_plugin=tracker
        )
        links = links.set_index('index')

        tracked_flow = tracker.merge()

        expected_flow = links.loc[link][('car', 'flow')]
        car_flow = tracked_flow['car']
        self.assertAlmostEqual(expected_flow.values[0], car_flow.loc[link, link].values[0][0], places=3)

        expected_flow = links.loc[link][('truck', 'flow')]
        truck_flow = tracked_flow['truck']
        self.assertAlmostEqual(expected_flow.values[0], truck_flow.loc[link, link].values[0][0], places=3)

        expected_flow = links.loc[link, 'flow'] - links.loc[link, 'base_flow']
        tot_flow = car_flow + truck_flow
        self.assertAlmostEqual(expected_flow.values[0], tot_flow.loc[link, link].values[0][0], places=3)

    def test_expanded_pathfinder_car_los_SHOULD_contain_all_segments(self):
        segments = ['car', 'truck']
        links, car_los, relgap = self._get_expanded_roadpathfinder(segments=segments, method='bfw')

        expected_columns = ['origin', 'destination', 'path', 'segment']
        for col in expected_columns:
            self.assertIn(col, car_los.columns)
        for seg in segments:
            self.assertIn(seg, car_los['segment'].unique())
