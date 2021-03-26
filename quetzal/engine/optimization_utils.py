import numpy as np
import pandas as pd
import scipy


def time_length_matrix(sm):
    # pathfinder
    sm.step_road_pathfinder(all_or_nothing=True)

    # length per road type
    los = sm.car_los[['origin', 'destination', 'time', 'link_path', 'ntlegs']]
    ztr_length = sm.zone_to_road.set_index(['a', 'b'])['distance'].to_dict()
    los['access'] = los['ntlegs'].apply(
        lambda x: sum(map(lambda l: ztr_length.get(l, 0), x))
    )

    def type_split(x, col='type'):
        # TODO speed up
        return sm.road_links.loc[x].groupby(col)['length'].sum()

    temp = los['link_path'].apply(lambda x: type_split(x, col='type')).fillna(0)
    los[temp.columns] = temp
    return los.drop(['link_path', 'ntlegs'], 1)


def optimize(od_classified, od_targets, classes, target_col, classes_bounds, od_weights=None):
    """
    Solve the linear optimization problem:
        min(||(od_classified * x - od_targets)*od_weights||^2)
            with x respecting classes_bounds
    - each sample is one OD
    - each feature/classe is a column of od_classified
    No inequality constraints between features.

    :param od_classified: pd.DataFrame with rows as samples (ODs) and columns 'origin', 'destination', classes
    :param od_targets: pd.DataFrame with rows as targets and columns 'origin', 'destination', target_col
    :param target_col: name of target column
    :param classes: list of classes (i.e. column names of od_classified but origin and destination)
    :param classes_bounds: pd.Series or dict of bounds {classe: [min, max] for classe in classes}
    :param od_weigths: pd.DataFrame of weigth of each OD with columns 'origin', 'destination', 'weight'

    :return dict {classe: value} that solve the optimization problem

    TODO: Add scaling
    """
    # ensure there is no column names collision
    assert(
        set(od_classified.columns).intersection(set(od_targets.columns)) == {'origin', 'destination'}
    ), 'Some columns are shared'

    merged = od_targets.merge(
        od_classified,
        on=['origin', 'destination'],
        how='inner'
    )
    if od_weights is not None:
        merged = merged.merge(
            od_weights,
            on=['origin', 'destination'],
            how='inner'
        )
    else:
        merged['weight'] = 1

    X_df = merged[classes]
    A = X_df.values  # Scale here if necessary

    Y_df = merged[target_col]
    b = Y_df.values  # Scale here if necessary

    # Objective 0.5 *(w(Ax-b)*w(Ax-b)) ** 0.5 + regularization
    def J(x, alpha=0, weights=merged['weight'].values):
        diff = (np.dot(A, x) - b) * weights
        return 0.5 * np.sqrt(np.dot(diff, diff.T)) + alpha * np.sqrt(np.dot(x, x))

    # Sort bounds
    bounds = [classes_bounds[c] for c in classes]
    # Without constraints
    x0 = [(x[0] + x[1]) / 2 for x in bounds]
    results = scipy.optimize.minimize(J, x0, bounds=bounds)

    assert(results.success is True), 'Optimization failed'
    return {classe: v for classe, v in zip(classes, results.x)}


def solve_road_speed_optimization(
    od_classified_lengths, od_duration_targets, road_classes, speed_bounds, od_weights=None
):
    """
    Apply optimize to road speed problem
    """
    od_classified = od_classified_lengths[['origin', 'destination'] + road_classes]
    od_targets = od_duration_targets[['origin', 'destination', 'time']]
    bounds = pd.Series(
        {classe: [1 / x for x in speed_bounds[classe]][::-1] for classe in road_classes}
    )
    r = optimize(od_classified, od_targets, road_classes, 'time', bounds, od_weights)
    return {classe: 1 / r[classe] for classe in road_classes}


def dynamic_road_speed_optimization(sm, target_times, classes, speed_bounds, od_weights=None, max_iter=10):
    # TODO: dumb for now, might deserve better
    for i in range(max_iter):
        tlm = time_length_matrix(sm)
        speeds = solve_road_speed_optimization(tlm, target_times, list(classes), speed_bounds, od_weights)
        print(speeds)
        # Recompute duration
        sm.road_links['speed'] = sm.road_links['type'].replace(speeds)
        sm.road_links['time'] = sm.road_links['length'] / sm.road_links['speed']
        sm.zone_to_road['speed'] = speeds['access']
        sm.zone_to_road['time'] = sm.zone_to_road['distance'] / sm.zone_to_road['speed']
