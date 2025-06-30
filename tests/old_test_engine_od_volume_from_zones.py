import sys
from unittest import TestCase, skipIf
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
from quetzal.engine.engine import od_volume_from_zones

sys.path.append(r'../syspy')


origin_power = np.power


@patch('numpy.power')
@patch('syspy.distribution.distribution.CalcDoublyConstrained', autospec=True)
@patch('syspy.skims.skims.euclidean', autospec=True)
@skipIf(True, 'want to skip')
class Test(TestCase):
    def test_zones_only_intrazonal(self, skims_euclidean, distribution_CalcDoublyConstrained, spy_power):
        spy_power.side_effect = origin_power
        values = [[1, 2, 3], [1, 2, 4]]
        zones = pd.DataFrame(values, columns=['emission', 'attraction', 'geometry'])
        values = [[0.0, 0.0, 0.0], [1.0, 0.0, 157249.38127194397], [0.0, 1.0, 157249.38127194397], [1.0, 1.0, 0.0]]
        euclidean_df = pd.DataFrame(values, columns=['origin', 'destination', 'euclidean_distance'])
        skims_euclidean.return_value = euclidean_df

        values = [[0.9, 0.1], [0.1, 0.9]]
        volume_array = np.array(values)
        distribution_CalcDoublyConstrained.return_value = volume_array

        result = od_volume_from_zones(zones)

        expected = pd.DataFrame(
            [[0, 0, 0.9], [0, 1, 0.1], [1, 0, 0.1], [1, 1, 0.9]], columns=['origin', 'destination', 'volume']
        )

        pd.testing.assert_frame_equal(result, expected)
        skims_euclidean.assert_called_once_with(zones, intrazonal=False)
        distribution_CalcDoublyConstrained.assert_called_once_with(
            zones['emission'].values, zones['attraction'].values, ANY
        )
        spy_power.assert_called_once_with(ANY, -2)
