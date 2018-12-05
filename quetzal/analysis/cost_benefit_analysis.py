# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def od_weighted_time_delta(reference_od, scenario_od):
    columns = [
        'origin', 'destination', 'volume',
        'volume_pt', 'volume_car', 'volume_walk',
        'duration_car', 'duration_pt'
    ]

    ref = reference_od[columns].set_index(['origin', 'destination'])
    scen = scenario_od[columns].set_index(['origin', 'destination'])

    constant_pt = np.minimum(ref['volume_pt'], scen['volume_pt'])
    constant_car = np.minimum(ref['volume_car'], scen['volume_car'])

    duration_car_to_pt = scen['duration_pt'] - ref['duration_car']
    duration_pt_to_car = scen['duration_car'] - ref['duration_pt']
    volume_car_to_pt = np.maximum(ref['volume_car'] - scen['volume_car'], 0)
    volume_pt_to_car = np.maximum(scen['volume_car'] - ref['volume_car'], 0)

    delta = scen - ref

    weighted_time_delta = pd.DataFrame(
        {
            'time_constant_pt': constant_pt * delta['duration_pt'],
            'time_constant_car': constant_car * delta['duration_car'],
            'time_car_to_pt': volume_car_to_pt * duration_car_to_pt,
            'time_pt_to_car': volume_pt_to_car * duration_pt_to_car,
            'volume_pt_to_car': volume_pt_to_car,
            'volume_car_to_pt': volume_car_to_pt,
            'duration_pt_to_car': duration_pt_to_car,
            'duration_car_to_pt': duration_car_to_pt,
            'volume_constant_pt': constant_pt,
            'volume_constant_car': constant_car,
            'duration_car': delta['duration_car'],
            'duration_pt': delta['duration_pt']
        }
    )
    ordered_columns = [
        'volume_constant_pt', 'duration_pt', 'time_constant_pt',
        'volume_constant_car', 'duration_car', 'time_constant_car',
        'volume_car_to_pt', 'duration_car_to_pt', 'time_car_to_pt',
        'volume_pt_to_car', 'duration_pt_to_car', 'time_pt_to_car'
    ]

    return weighted_time_delta[ordered_columns]
