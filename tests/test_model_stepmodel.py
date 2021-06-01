import json
import os
import sys
import unittest

import pandas as pd


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        # Import syspy
        sys.path.append(r'../syspy')
        from quetzal.model import stepmodel

        # Read json_database object
        self.data_path = r'tests/data/'
        with open(self.data_path + r'json_database.json', 'r') as infile:
            json_database_object = json.load(infile)

        # load model
        self.sm = stepmodel.StepModel(json_database=json_database_object)

    def test_step_ntlegs(self):
        self.sm.step_ntlegs(
            n_ntlegs=5,
            short_leg_speed=2,
            long_leg_speed=10,
            threshold=1000,
            max_ntleg_length=100000
        )

    def test_arg_tracking(self):
        self.sm.step_ntlegs(
            n_ntlegs=5,
            short_leg_speed=2,
            long_leg_speed=10,
            threshold=1000,
            max_ntleg_length=100000
        )
        self.sm.step_ntlegs(use_tracked_args=True)

    def test_step_distribution(self):
        self.sm.step_distribution()

    def test_pathfinder(self):
        self.sm.step_pathfinder()

    def test_step_assignment(self):
        self.sm.step_assignment(
            volume_column='volume_pt',
            boardings=True,
            alightings=True,
            transfers=True
        )

    def test_chekpoints(self):

        self.sm.checkpoints(
            link_checkpoints=(),
            node_checkpoints=('BULLFROG')
        )

    def test_to_hdf(self):
        from quetzal.model import stepmodel
        if not os.path.exists(self.data_path + r'out/'):
            os.mkdir(self.data_path + r'out/')

        # test to_hdf
        filepath = self.data_path + r'out/test_hdf.hdf'
        self.sm.to_hdf(filepath=filepath)

        # test read_hdf
        sm = stepmodel.read_hdf(filepath)
        # test to_hdf + read_hdf work well together
        assert sm.parameters['checkpoints']['kwargs'][
            'node_checkpoints'] == ['BULLFROG']

    def test_json_database_io(self):
        from quetzal.model import stepmodel

        # Read json file
        with open(self.data_path + r'json_database.json', 'r') as infile:
            json_database_object = json.load(infile)
        # load model
        sm2 = stepmodel.StepModel(json_database=json_database_object)

        # Export model  without epsg
        delattr(sm2, 'epsg')
        delattr(sm2, 'coordinates_unit')
        dumped_json_database = sm2.to_json_database()

        # Assert initial json file and exported one are identical
        # msg = 'Exported json object differs from imported one.'
        # assert len(dumped_json_database) == len(json_database_object), msg

    def test_linear_solver(self):
        # preparation
        sm = self.sm
        sm.od_stack = pd.merge(
            sm.pt_los,
            sm.volumes,
            on=['origin', 'destination'],
            suffixes=['_los', '_vol']
        ).sort_values(['origin', 'destination']).reset_index(drop=True)
        sm.od_stack['euclidean_distance'] = 10

        # linear solver
        constrained_links = {'link_12': 0.8, 'link_11': 1}
        linprog_kwargs = {
            'bounds_A': [0.8, 1.2],
            'bounds_emissions': [0.9, 1.1],
            'bounds_tot_emissions': [0.99, 1.01],
            'pas_distance': 200,
            'maxiter': 10000,
            'tolerance': 1e-5
        }

        sm.linear_solver(
            constrained_links=constrained_links,
            linprog_kwargs=linprog_kwargs
        )
        reset = sm.pivot_stack_matrix.sort_values(['origin', 'destination']).reset_index(drop=True)

        def check_bound(value, bounds, tolerance=0):
            return bounds[0] * (1 - tolerance) <= value <= bounds[1] * (1 + tolerance)

        def assert_bound(value, bounds, tolerance=0):
            print(value, bounds)
            assert check_bound(value, bounds, tolerance)

        sm.volumes['pivoted'] = sm.volumes['volume_pt'] * reset['pivot']

        # second assignment
        sm.step_assignment(
            volume_column='pivoted',
            boardings=True,
            alightings=True,
            transfers=True
        )

        # total
        volsum = (sm.volumes['volume_pt'] * reset['pivot']).sum()
        growth = volsum / sm.volumes['volume_pt'].sum()

        # emissions
        attractions = sm.volumes.groupby(['destination']).sum()
        emissions = sm.volumes.groupby(['origin']).sum()

        attractions['growth'] = attractions['pivoted'] / attractions['volume_pt']
        emissions['growth'] = emissions['pivoted'] / emissions['volume_pt']

        # pairwise
        pivoted = sm.volumes.copy()
        pivoted['growth'] = pivoted['pivoted'] / pivoted['volume_pt']

        # total
        assert_bound(
            growth,
            linprog_kwargs['bounds_tot_emissions'],
            tolerance=linprog_kwargs['tolerance']
        )

        # test objective is reached
        for key, value in constrained_links.items():
            bounds = [value, value]
            assert_bound(
                sm.loaded_links['pivoted'][key],
                bounds,
                tolerance=linprog_kwargs['tolerance']
            )

        # emissions
        values = [
            attractions['growth'].min(),
            attractions['growth'].max(),
            emissions['growth'].min(),
            emissions['growth'].max()
        ]

        for value in values:
            assert_bound(
                value,
                linprog_kwargs['bounds_emissions'],
                tolerance=linprog_kwargs['tolerance']
            )

        # pairwise
        values = [
            reset['pivot'].min(),
            reset['pivot'].max()
        ]

        for value in values:
            assert_bound(
                value,
                linprog_kwargs['bounds_A'],
                tolerance=linprog_kwargs['tolerance']
            )


if __name__ == '__main__':
    unittest.main()
