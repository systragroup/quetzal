import matplotlib.pyplot as plt
import numpy as np

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
    return d >= 0.5 * v_max**2 * (1 / a + 1 / b)


def compute_t_total(d, a, b, v_max):
    if reach_v_max(d, a, b, v_max):
        return d / v_max + v_max * 0.5 * (1 / a + 1 / b)
    else:
        return compute_t_star(d, a, b) * (a + b) / b


def format_v_versus_time(d, a, b, v_max, i=10):
    if reach_v_max(d, a, b, v_max):
        t_star_1 = v_max / a
        t_star_2 = d / v_max + v_max * 0.5 * (1 / a - 1 / b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star_1, i)
        t2 = np.linspace(t_star_1, t_star_2, i)
        t3 = np.linspace(t_star_2, t_total, i)
        v1 = t1 * a * 3.6
        v2 = t2 * 0 + v_max * 3.6
        v3 = (-b * t3 + b * d / v_max + v_max * 0.5 * (b / a + 1)) * 3.6
        times = np.append(np.append(t1, t2), t3)
        speeds = np.append(np.append(v1, v2), v3)
    else:
        t_star = compute_t_star(d, a, b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star, i)
        t2 = np.linspace(t_star, t_total, i)
        v1 = t1 * a * 3.6
        v2 = ((a + b) * t_star - b * t2) * 3.6
        times = np.append(t1, t2)
        speeds = np.append(v1, v2)
    return times, speeds


def plot_v_versus_time(d, a, b, v_max, **kwargs):
    times, speeds = format_v_versus_time(d, a, b, v_max)
    plot = plt.plot(times, speeds, **kwargs)
    plt.xlabel('time (s)')
    plt.ylabel('speed (km/h)')
    return plot


def format_x_versus_time(d, a, b, v_max, i=10):
    if reach_v_max(d, a, b, v_max):
        t_star_1 = v_max / a
        t_star_2 = d / v_max + v_max * 0.5 * (1 / a - 1 / b)
        t_total = compute_t_total(d, a, b, v_max)
        t1 = np.linspace(0, t_star_1, i)
        t2 = np.linspace(t_star_1, t_star_2, i)
        t3 = np.linspace(t_star_2, t_total, i)
        x1 = t1**2 * a / 2
        x2 = v_max * (t2 - v_max / (2 * a))
        x3 = d + b * t_total * t3 - b / 2 * (t3**2 + t_total**2)
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


def plot_x_versus_time(d, a, b, v_max, **kwargs):
    times, positions = format_x_versus_time(d, a, b, v_max)
    plot = plt.plot(times, positions, **kwargs)
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    return plot


def plot_v_versus_x(d, a, b, v_max, **kwargs):
    if reach_v_max(d, a, b, v_max):
        t_total = compute_t_total(d, a, b, v_max)
        x_star_1 = v_max**2 / (2 * a)
        x_star_2 = v_max * (t_total - v_max / b - v_max / (2 * a))
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

# def all_graphs(d,a,b,v_max):
