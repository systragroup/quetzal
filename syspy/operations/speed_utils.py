import numpy as np
import matplotlib.pyplot as plt

"""
Compute train running time based on the following assumptions:
- point dynamic
- constant acceleration / braking
- given maximum speed (imposed by rolling stock or rail curve)
"""


def compute_v_star(d, a, b):
    return a * (2 * d / (a * (1 + a / b)))**0.5


def compute_t_star(d, a, b, v_star=None):
    return (2 * d / (a * (1 + a / b)))**0.5


def compute_x_star(d, a, b):
    return 0.5 * a * (compute_t_star(d, a, b) ** 2)


def reach_v_max(d, a, b, v_max):
    """
    Estimates if maximal speed v_max (m/s) can be reached over running duration d (m)
    with braking b and acceleration a
    """
    return d >= 0.5 * v_max ** 2 * (1 / a + 1 / b)


def compute_t_total(d, a, b, v_max):
    if reach_v_max(d, a, b, v_max):
        return d / v_max + v_max * 0.5 * (1 / a + 1 / b)
    else:
        return compute_t_star(d, a, b) * (a + b) / b


def format_v_versus_time(d, a, b, v_max, i=10):
    """
    Returns two arrays of lenghth i, one for times and one for speeds, assuming constant acceleration
    followed by constant braking to minimize running duration.
    :params:
        - d (float): running distance (m)
        - a (float): constant acceleration (m/s²)
        - b (float): constant braking (m/s²)
        - v_max (float): max speed (m/s)
    :returns:
        - times (numpy array): time array (s)
        - speeds (numpy array): speed array (m/s)
    """
    if reach_v_max(d, a, b, v_max):
        t_star_1 = v_max / a
        t_star_2 = d / v_max + v_max * 0.5 * (1 / a - 1 / b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star_1, i)
        t2 = np.linspace(t_star_1, t_star_2, i)
        t3 = np.linspace(t_star_2, t_total, i)
        v1 = t1 * a
        v2 = t2 * 0 + v_max
        v3 = (-b * t3 + b * d / v_max + v_max * 0.5 * (b / a + 1))
        times = np.append(np.append(t1, t2), t3)
        speeds = np.append(np.append(v1, v2), v3)
    else:
        t_star = compute_t_star(d, a, b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star, i)
        t2 = np.linspace(t_star, t_total, i)
        v1 = t1 * a
        v2 = ((a + b) * t_star - b * t2)
        times = np.append(t1, t2)
        speeds = np.append(v1, v2)

    return times, speeds


def format_x_versus_time(d, a, b, v_max, i=10):
    """
    Returns two arrays of lenghth i, one for times and one for positions, assuming constant acceleration
    followed by constant braking to minimize running duration.
    :params:
        - d (float): running distance (m)
        - a (float): constant acceleration (m/s²)
        - b (float): constant braking (m/s²)
        - v_max (float): max speed (m/s)
    :returns:
        - times (numpy array): time array (s)
        - positions (numpy array): position array (m)
    """
    if reach_v_max(d, a, b, v_max):
        t_star_1 = v_max / a
        t_star_2 = d / v_max + v_max * 0.5 * (1 / a - 1 / b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star_1, i)
        t2 = np.linspace(t_star_1, t_star_2, i)
        t3 = np.linspace(t_star_2, t_total, i)
        x1 = t1**2 * a / 2
        x2 = v_max * (t2 - v_max / (2 * a))
        x3 = d + b * t_total * t3 - b/2 * (t3**2 + t_total**2)
        times = np.append(np.append(t1, t2), t3)
        positions = np.append(np.append(x1, x2), x3)
    else:
        t_star = compute_t_star(d, a, b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star, i)
        t2 = np.linspace(t_star, t_total, i)
        x1 = t1**2 * a / 2
        x2 = t2 * (-b * t2 / 2 + b * t_total) + d - b / 2 * t_total ** 2
        times = np.append(t1, t2)
        positions = np.append(x1, x2)

    return times, positions


def plot_v_versus_time(d, a, b, v_max_kmh, **kwargs):
    times, speeds = format_v_versus_time(d, a, b, v_max_kmh / 3.6)
    plot = plt.plot(times, speeds, **kwargs)
    plt.xlabel('time (s)')
    plt.ylabel('speed (km/h)')
    return plot


def plot_x_versus_time(d, a, b, v_max_kmh, **kwargs):
    times, positions = format_x_versus_time(d, a, b, v_max_kmh)
    plot = plt.plot(times, positions, **kwargs)
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    return plot


def plot_v_versus_x(d, a, b, v_max_kmh, **kwargs):
    v_max = v_max_kmh / 3.6
    if reach_v_max(d, a, b, v_max):
        t_total = compute_t_total(d, a, b, v_max)
        x_star_1 = v_max**2 / (2 * a)
        x_star_2 = v_max * (t_total - v_max / b - v_max / (2*a))
        t_total = compute_t_total(d, a, b, v_max)
        x1 = np.arange(0, x_star_1, 1)
        x2 = np.arange(x_star_1, x_star_2, 1)
        x3 = np.arange(x_star_2, d, 1)
        v1 = (2 * a * x1) ** 0.5
        v2 = x2 * 0 + v_max
        v3 = (2 * b * (d - x3))**0.5
        plot = plt.plot(np.append(np.append(x1, x2), x3), 3.6 * np.append(np.append(v1, v2), v3), **kwargs)
    else:
        x_star = compute_x_star(d, a, b)
        x1 = np.arange(0, x_star, 1)
        x2 = np.arange(x_star, d, 1)
        v1 = (2 * a * x1) ** 0.5
        v2 = (2 * b * (d - x2))**0.5
        plot = plt.plot(np.append(x1, x2), 3.6 * np.append(v1, v2), **kwargs)

    plt.xlabel('position (m)')
    plt.ylabel('speed (km/h)')

    return plot


def time_vs_position_arrays(df):
    times = np.array([0])
    positions = np.array([0])
    for i, row in df.iterrows():
        times = np.append(times, row['travel_x_t_arrays'][0] + times[-1])
        times = np.append(times, times[-1] + row['time_dwell'])
        positions = np.append(positions, row['travel_x_t_arrays'][1] + positions[-1])
        positions = np.append(positions, positions[-1])
    return times, positions


def plot_time_vs_position(om, direction, ax=None, **plt_kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    # filter direction
    travel = om.travel_times[om.travel_times['direction_id'] == direction]
    dwell = om.dwell_times[om.dwell_times['direction_id'] == direction]

    df = travel.merge(
        dwell[['a', 'time']], on='a', suffixes=['_travel', '_dwell']
    )

    df['travel_x_t_arrays'] = df['length'].apply(lambda x: format_x_versus_time(
        x, om.rolling_stock.acceleration, om.rolling_stock.braking, om.rolling_stock.max_speed)
    )
    df['cum_length'] = df['length'].cumsum() - df['length']

    times, positions = time_vs_position_arrays(df)

    ax.plot(positions, times, **plt_kwargs)
    ax.set_xticks(np.append(df['cum_length'].values, df['length'].cumsum().values[-1]))
    ax.set_xticklabels(np.append(df['a'].values, df['b'].values[-1]), ha='right', rotation=45)

    ax.set_ylabel('Time (s)')

    return ax


def speed_vs_time_arrays(df):
    times = np.array([0])
    speeds = np.array([0])
    for i, row in df.iterrows():
        times = np.append(times, row['travel_v_t_arrays'][0] + times[-1])
        times = np.append(times, times[-1] + row['time_dwell'])
        speeds = np.append(speeds, row['travel_v_t_arrays'][1])
        speeds = np.append(speeds, [0])
    return speeds, times


def plot_speed_vs_time(om, direction, ax=None, **plt_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # filter direction
    travel = om.travel_times[om.travel_times['direction_id'] == direction]
    dwell = om.dwell_times[om.dwell_times['direction_id'] == direction]

    df = travel.merge(dwell[['a', 'time']], on='a', suffixes=['_travel', '_dwell'])

    df['travel_v_t_arrays'] = df['length'].apply(lambda x: format_v_versus_time(
        x, om.rolling_stock.acceleration, om.rolling_stock.braking, om.rolling_stock.max_speed / 3.6)  # km/h to m/s
    )

    speeds, times = speed_vs_time_arrays(df)
    ax.plot(times, speeds * 3.6, **plt_kwargs)  # m/s to km/h
    # ticks
    df['total_time'] = df['time_dwell'] + df['time_travel']
    df['cum_tot_time'] = df['total_time'].cumsum() - df['total_time']
    ax.set_xticks(np.append(df['cum_tot_time'].values, df['total_time'].cumsum().values[-1]))
    ax.set_xticklabels(np.append(df['a'].values, df['b'].values[-1]), ha='right', rotation=45)

    ax.set_ylabel('Speed (km/h)')

    return ax


def speed_vs_position_arrays(df):
    positions = np.array([0])
    speeds = np.array([0])
    for i, row in df.iterrows():
        positions = np.append(positions, row['travel_x_t_arrays'][1] + positions[-1])
        positions = np.append(positions, positions[-1])
        speeds = np.append(speeds, row['travel_v_t_arrays'][1])
        speeds = np.append(speeds, [0])
    return speeds, positions


def plot_speed_vs_position(om, direction, ax=None, **plt_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # filter direction
    travel = om.travel_times[om.travel_times['direction_id'] == direction]
    dwell = om.dwell_times[om.dwell_times['direction_id'] == direction]

    df = travel.merge(dwell[['a', 'time']], on='a', suffixes=['_travel', '_dwell'])

    df['travel_x_t_arrays'] = df['length'].apply(lambda x: format_x_versus_time(
        x, om.rolling_stock.acceleration, om.rolling_stock.braking, om.rolling_stock.max_speed / 3.6)  # km/h to m/s
    )
    df['travel_v_t_arrays'] = df['length'].apply(lambda x: format_v_versus_time(
        x, om.rolling_stock.acceleration, om.rolling_stock.braking, om.rolling_stock.max_speed / 3.6)  # km/h to m/s
    )

    speeds, positions = speed_vs_position_arrays(df)
    ax.plot(positions, speeds * 3.6, **plt_kwargs)  # m/s to km/h
    # ticks
    df['cum_length'] = df['length'].cumsum() - df['length']
    ax.set_xticks(np.append(df['cum_length'].values, df['length'].cumsum().values[-1]))
    ax.set_xticklabels(np.append(df['a'].values, df['b'].values[-1]), ha='right', rotation=45)

    ax.set_ylabel('Speed (km/h)')

    return ax
