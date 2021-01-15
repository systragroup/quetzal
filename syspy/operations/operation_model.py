import pandas as pd
import numpy as np
from copy import deepcopy
from quetzal.io import export_utils


class OperationModel():
    def __init__(
        self, rolling_stock, headway=600, turnback_time=300, minimum_dwell_time=20,
        design_headway=180, loaded_links=pd.DataFrame()
    ):

        self.turnback_time = turnback_time  # duration in secs
        self.minimum_dwell_time = minimum_dwell_time  # duration in secs
        self.design_headway = design_headway  # duration in secs
        self.headway = headway  # duration in secs
        self.rolling_stock = rolling_stock  # Rolling Stock object
        self.loaded_links = loaded_links  # Quetzal StepModel loaded_links attributes

    from syspy.operations.speed_utils import plot_speed_vs_position, plot_speed_vs_time, plot_time_vs_position

    def _init_dwell_times(self):
        """
        Create model dwell_times attribute.
        """
        temp = self.loaded_links.groupby('direction_id').apply(export_utils.shift_loadedlinks_alightings)
        temp = temp.reset_index(drop=True).drop(['b', 'load'], 1)
        self.dwell_times = temp.copy()

    def _init_travel_times(self):
        """
        Create model travel_times attribute.
        """
        self.travel_times = self.loaded_links[['a', 'b', 'link_sequence', 'direction_id', 'length']].copy()

    def compute_dwell_times(self):
        if 'dwell_times' not in self.__dict__:
            self._init_dwell_times()

        self.dwell_times['time'] = self.dwell_times.apply(
            lambda x: self.rolling_stock.get_dwell_time(
                x['boardings'] * self.headway / 3600, x['alightings'] * self.headway / 3600,
                self.minimum_dwell_time
            ),
            1
        )

    def compute_density(self):
        self.loaded_links[['seating', 'standing_density']] = self.loaded_links['load'].apply(
            lambda x: self.rolling_stock.distribute_load(x * self.headway / 3600),
        )
        self.loaded_links['seating'] *= 3600 / self.headway

    def recommended_headway(self, target_max_density=6):
        rs_capacity = self.rolling_stock.compute_capacity(target_max_density)
        return 3600 / (self.loaded_links['load'].max() / rs_capacity)

    def copy(self):
        copy = deepcopy(self)
        return copy

    def compute_tour_indicators(self):
        temp = {}
        temp['tour length'] = self.travel_times.length.sum()
        temp['commercial speed'] = temp['tour length'] / (
            self.travel_times['time'].sum() + self.dwell_times['time'].sum()
        ) * 3.6
        temp['tour duration'] = (
            self.travel_times['time'].sum() + self.dwell_times['time'].sum() + 2 * self.turnback_time
        )
        self.tour_indicators = pd.Series(temp)
        self.tour_indicators.name = 'tour'

    def compute_operations_indicators(self):
        # Requires tour indicators
        t_i = self.tour_indicators
        temp = {}
        temp['headway'] = self.headway
        temp['running trains'] = np.ceil(t_i['tour duration'] / self.headway)
        temp['running train units'] = temp['running trains'] * self.rolling_stock.n_units
        temp['train departure margin'] = temp['running trains'] * self.headway - t_i['tour duration']
        temp['train.km'] = t_i['tour duration'] * 3600 / self.headway / 1000
        temp['train-unit.km'] = temp['train.km'] * self.rolling_stock.n_units

        self.operations_indicators = pd.Series(temp)
        self.operations_indicators.name = 'operations'

    def compute_capacity_indicators(self):
        temp = {}
        temp['max_density'] = self.loaded_links['standing_density'].max()

        for direction in self.loaded_links['direction_id']:
            load_dir = self.loaded_links[self.loaded_links['direction_id'] == direction]
            label = 'share of seating pax.km in direction {}'.format(direction)
            temp[label] = load_dir['seating'].sum() / load_dir['load'].sum()

        temp['design capacity seated'] = self.rolling_stock.seats * 3600 / self.design_headway
        temp['design capacity aw2'] = self.rolling_stock.capacity * 3600 / self.design_headway
        temp['design capacity aw3'] = temp['design capacity seated'] + (
            temp['design capacity aw2'] - temp['design capacity seated']) * 1.5

        self.capacity_indicators = pd.Series(temp)
        self.capacity_indicators.name = 'capacity'

    def group_kpi(self):
        self.kpi = pd.concat(
            [
                pd.DataFrame(self.tour_indicators).unstack(),
                pd.DataFrame(self.operations_indicators).unstack(),
                pd.DataFrame(self.capacity_indicators).unstack(),
            ]
        ).reset_index().rename(
            columns={'level_0': 'category', 'level_1': 'indicator'}
        ).set_index(['category', 'indicator'])

    def compute_kpi(self):
        self.compute_tour_indicators()
        self.compute_operations_indicators()
        self.compute_capacity_indicators()
        self.group_kpi()
