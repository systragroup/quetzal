import pandas as pd
import numpy as np
from scipy.optimize import linprog


def linearsolver(
    indicator,
    constrained_links,
    od_stack,
    bounds_A,
    bounds_emissions,
    bounds_tot_emissions,
    pas_distance,
    maxiter,
    tolerance
):
    '''
    Cette fonction est le coeur de la méthode linear_solver.
    - Elle construit les contraintes et l'objectif dans build_constraints
    - Elle résoud le problème d'optimisation linéaire ainsi formé, avec les
    bonnes contraintes et paramètres.

    :param indicator (pd.DataFrame): indicator of the model
    :param constrained_links (dict): dict of constrained links with volumes
    :param od_stack (pd.DataFrame): od_stack of the model
    :param bounds_A, bounds_emissions, bounds_tot_emissions, pas_distance:
        parameters for building constraints
    :param maxiter, tolerance (int): maximum iterations and tolerance on
        objective for the linprog function
    :returns pivot_stack_matrix (pd.DataFrame): elements of pivot_stack_matrix.
    '''
    constrained_zip = zip(*constrained_links.items())
    keys, values = tuple(constrained_zip)
    obj, A_ub, b_ub, bound_ub = build_constraints(
        indicator,
        values,
        od_stack,
        bounds_A,
        bounds_emissions,
        bounds_tot_emissions,
        pas_distance
    )
    try:
        l = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bound_ub,
                    options={'maxiter': maxiter, 'bland': True, 'tol': tolerance})
        x = l.x[0:len(indicator)]
        pivot_stack_matrix = pd.merge(
            od_stack[['origin', 'destination']],
            pd.DataFrame(x, columns=['pivot']),
            left_index=True, right_index=True
        )
        return pivot_stack_matrix
    except:
        raise NotImplementedError
        pass


def build_indicator(od_stack, constrained_links):
    '''
    :param od_stack (pd.DataFrame): od_stack of the model
    :param constrained_links (dict): constrained links and their volumes
    :returns indicator (DataFrame):  indicator for the model
    '''
    constrained_zip = zip(*constrained_links.items())
    keys, values = tuple(constrained_zip)
    paths = od_stack['path']
    indicator = pd.DataFrame([
        tuple(int(i in p) for i in keys)
        for p in paths
    ])
    return indicator


def reduce_indicator(big_indicator, cluster_series, volumes):
    '''
    :param big_indicator (pd.DataFrame): indicator of the entire model
    :param cluster_series (pd.Serie): correspondance between zones and clusters
    :param volumes (pd.DataFrame): od_stack of the model
    :returns indicator (DataFrame): reduced indicator for the aggregated model
    '''
    nb_keys = len(big_indicator.columns)
    table = volumes.merge(big_indicator, left_index=True, right_index=True)
    proto = pd.merge(table, pd.DataFrame(cluster_series),
                     left_on='origin', right_index=True)
    proto = pd.merge(proto, pd.DataFrame(cluster_series), left_on='destination',
                     right_index=True, suffixes=['_origin', '_destination'])
    grouped = proto.groupby(
        ['cluster_origin', 'cluster_destination'])[
            [i for i in range(nb_keys)]]

    indicator = pd.DataFrame([
        tuple(np.ma.average(table[k], weights=table['volume_pt'], axis=0)
            for k in range(nb_keys))
        for couple, table in grouped
        ]).fillna(0)
    return indicator


def extrapolate(agg_pivot_stack_matrix, od_stack, cluster_series):
    """
    Extrapolates the model.
    Given the aggregated model and its pivot_stack_matrix, we build the
    pivot_stack_matrix for the whole model.

    :param agg_pivot_stack_matrix: agg_pivot_stack_matrix of the aggregated model
    :param od_stack (pd.DataFrame): od_stack of the entire model
    :param cluster_series (pd.Series)
    :return grouped (pd.DataFrame): pivot_stack_matrix of the whole model
    :return od_stack: od_stack of the whole model, with pivot column updated
    """
    proto = pd.merge(
        od_stack[['origin', 'destination']],
        pd.DataFrame(cluster_series), left_on='origin', right_index=True)
    proto = pd.merge(
        proto, pd.DataFrame(cluster_series), left_on='destination',
        right_index=True, suffixes=['_origin', '_destination'])
    agg_pivot_stack_matrix.columns = ['cluster_origin', 'cluster_destination', 'pivot']
    grouped = pd.merge(
        proto, agg_pivot_stack_matrix,
        on=['cluster_origin', 'cluster_destination'])
    grouped = grouped.sort_values(by=['origin', 'destination'])
    grouped.reset_index(inplace=True)
    grouped.drop(
        ['index', 'cluster_origin', 'cluster_destination'],
        axis=1, inplace=True)

    # putting the pivot in od_stack matrix
    od_stack['pivot'] = grouped['pivot']
    return grouped, od_stack


def build_constraints(
    indicator,
    values_links,
    od_stack,
    bounds_A,
    bounds_emissions,
    bounds_tot_emissions,
    pas_distance
):
    """
    Ici on construit l'objectif et les contraintes du problème d'optimisation
    suivant:
    variable: pivot A
    min_A ||Rc - Rm|| = ||indic*volumes*A - constrained values||
    sous contraintes:
        * - Pg < volumes*A - e/a < Pg ((C1): émissions et attractions par zone conservées)
        * - Pg' < sum(volumes*A) - e/a_tot < Pg' ((C2) émissions et attractions totales conservées)
        * - Pd < <D,X> - distance_moyenne < Pd ((C3) distance moyenne conservée)
        * bound_A_min < A(i) < bound_A_max ((B) bornes du problème) forall i

    Comme l'objectif n'est pas linéaire, on décide d'utiliser la norme 1 et de
    linéariser la valeur absolue en ajoutant une nouvelle variable y:
        (~~ y(i) = |indic*volumes*A - constrained values|(i))
    Le problème devient alors:
        min y
    sous contraintes
        * y(i) >= (indic*volumes*A - constrained values)(i)   forall i (C4)
        * y(i) >= - (indic*volumes*A - constrained values)(i)   forall i (C5)
        * (mêmes contraintes que précedemment en plus)

    Parameters:
    :param indicator: indicator of the model
    :param values_links (tuple): values of constrained links
    :param od_stack (pd.DataFrame): od_stack of the model
    :param bounds_A (list): upper and lower bound of coefficients of the pivot
    :param bounds_emissions (list): upper and lower bound of the multiplicative
        coefficient on emissions and attractions per cluster
    :param bounds_tot_emissions (list): idem but for emisisons and attractions
        of the entire model
    :param pas_distance (int): additive pas on the mean distance
    :return objectif: objective of the function
    :return A_ub: matrix of inequality constraint (left side)
    :return b_ub: vector of inequality constraint (right side)
    """
    
    od_stack = od_stack.sort_values(['origin', 'destination'])
    volumes = od_stack['volume_pt']
    zone_list = sorted(od_stack['origin'])

    I = np.array(indicator)
    V = np.array(volumes)
    # df est en fait l'application linéaire
    # à appliquer aux paramètres d'ajustement
    df = pd.DataFrame(I.transpose()*V)
    X = np.array(volumes)

    # dimension du probleme: 2* nb_od
    # Objectif: min y
    nb_od = len(V)  # taille de A
    nb_zones = len(volumes)
    k = len(values_links)  # taille de y
    objectif = np.array([0 for i in range(nb_od)] + [1 for i in range(k)])

    # Constraintes de linéarisation (C4) et (C5)
    gr = pd.DataFrame(-np.eye(k))
    A_pos = pd.concat([df, gr], axis=1)  # (C4)
    A_neg = pd.concat([-df, gr], axis=1)  # (C5)
    b_sum = pd.DataFrame(np.array(values_links))

    # constraints on attraction and emission per cluster (C1)
    # marche seulement si l'index des zones est un range
    A_emission = pd.DataFrame(
        [
            [int(zone_list[i] == od_stack['origin'][j]) * X[j] for j in range(nb_od)]
            for i in range(nb_zones)]
    )
    A_attraction = pd.DataFrame(
        [
            [int(zone_list[i] == od_stack['destination'][j]) * X[j] for j in range(nb_od)]
            for i in range(nb_zones)]
    )
    A_e = pd.concat(
        [A_emission, pd.DataFrame([[0 for i in range(k)] for j in range(nb_zones)])],
        axis=1
    )
    A_a = pd.concat(
        [A_attraction, pd.DataFrame([[0 for i in range(k)] for j in range(nb_zones)])],
        axis=1
    )
    b_e_pos = bounds_emissions[1] * A_emission.sum(axis=1)
    b_a_pos = bounds_emissions[1] * A_attraction.sum(axis=1)
    b_e_neg = - bounds_emissions[0] * A_emission.sum(axis=1)
    b_a_neg = - bounds_emissions[0] * A_attraction.sum(axis=1)

    # constraints on total emission and attranction (C2)
    A_tot_em = pd.concat(
        [pd.DataFrame([X]), pd.DataFrame([[0 for i in range(k)]])],
        axis=1
    )
    tot_em_pos = bounds_tot_emissions[1] * X.sum()
    tot_em_neg = - bounds_tot_emissions[0] * X.sum()
    b_tot_em = pd.DataFrame([tot_em_pos, tot_em_neg])

    # constraints on distance (C3)
    vect_dist = od_stack['euclidean_distance']
    dist_moy = (vect_dist.values*X).sum()/X.sum()
    G1 = pd.DataFrame([[X[i]*(vect_dist[i] - dist_moy - pas_distance) for i in range(nb_od)]])
    G2 = pd.DataFrame([[X[i]*(-vect_dist[i] + dist_moy - pas_distance) for i in range(nb_od)]])
    G_y = pd.DataFrame([[0 for j in range(k)]])
    G_sup = pd.concat([G1, G_y], axis=1)
    G_inf = pd.concat([G2, G_y], axis=1)

    A_ub = pd.concat(
        [A_pos, A_neg, A_e, A_a, -A_e, -A_a, A_tot_em, -A_tot_em, G_sup, G_inf]
    ).values
    b_ub = pd.concat(
        [b_sum, -b_sum, b_e_pos, b_a_pos, b_e_neg, b_a_neg, b_tot_em, pd.DataFrame([0,0])]
    ).values

    # bounds (B)
    bounds = [bounds_A for i in range(nb_od)] + [(0, None) for i in range(k)]

    return objectif, A_ub, b_ub, bounds
