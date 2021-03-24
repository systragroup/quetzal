import pandas as pd


def distribute_load(load, total_capacity_aw2, seating_capacity):
    standing_capacity_aw2 = total_capacity_aw2 - seating_capacity
    seating = min(load, seating_capacity)
    standing = load - seating
    standing_density = standing / standing_capacity_aw2 * 4
    return pd.Series({'seating': seating, 'standing_density': standing_density})


def get_dwell_time(boardings, alightings, door_time, pax_flow_per_sec, min_dwell_time):
    pax_time = (boardings + alightings) / pax_flow_per_sec
    return max(pax_time + door_time, min_dwell_time)


def compute_capacity(n_seats, aw2_capacity, target_density):
    """
    Compute RS capacity for target density
    """
    return (aw2_capacity - n_seats) * target_density / 4 + n_seats


class RollingStockUnit():
    def __init__(
        self, length, weight, max_speed,
        n_doors, door_width, door_time, pax_flow,
        seats, capacity
    ):
        self.length, self.weight, self.max_speed = length, weight, max_speed
        self.n_doors, self.door_width, self.door_time = n_doors, door_width, door_time
        self.pax_flow = pax_flow
        self.seats, self.capacity = seats, capacity

    def distribute_load(self, load):
        return distribute_load(load, self.capacity, self.seats)

    def get_dwell_time(self, boardings_per_headway, alightings_per_headway, min_dwell_time):
        flow = self.n_doors * self.door_width * self.pax_flow
        return get_dwell_time(
            boardings_per_headway, alightings_per_headway,
            self.door_time, flow, min_dwell_time
        )

    def compute_capacity(self, target_density):
        return compute_capacity(self.seats, self.capacity, target_density)


class RollingStock(RollingStockUnit):
    def __init__(self, RSUnit, n_units):
        super().__init__(**RSUnit.__dict__)
        self.n_units = n_units
        self.length *= n_units
        self.weight *= n_units
        self.n_doors *= n_units
        self.seats *= n_units
        self.capacity *= n_units
