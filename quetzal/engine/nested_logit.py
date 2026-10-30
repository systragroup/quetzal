from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import polars as pl


def rank_and_filter_paths(
    paths: pd.DataFrame,
    od_cols: list[str] = ['origin', 'destination'],
    by: str = 'utility',
    n_paths_max: int | None = None,
):
    paths = paths.sort_values(by=by, ascending=False)
    paths['rank'] = paths.groupby([*od_cols, 'route_type', 'segment']).cumcount() + 1
    paths['rank'] = paths['rank'].astype(np.int32)

    if n_paths_max is not None:
        paths = paths.loc[paths['rank'] <= n_paths_max]

    return paths


def rank_and_filter_paths_pl(
    paths: pl.DataFrame,
    od_cols: list[str] = ['origin', 'destination'],
    by: str = 'utility',
    n_paths_max: int | None = None,
):
    paths = paths.with_columns(
        pl.col(by).rank(method='ordinal', descending=True).over([*od_cols, 'route_type', 'segment']).alias('rank')
    )
    if n_paths_max is not None:
        paths = paths.filter(pl.col('rank') <= n_paths_max)

    return paths


def nest_utility(utilities: np.ndarray, phi: float = 1):
    if phi == 0:
        phi = 1
    exp_mat = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = np.sum(exp_mat, axis=1).reshape(-1, 1)
    composite_utility = phi * np.log(exponential_s)
    return composite_utility


def incremental_nest_utility(
    delta_utilities: pd.DataFrame | pd.Series,
    probabilities: pd.DataFrame | pd.Series,
):
    addents = np.exp(delta_utilities) * probabilities.divide(probabilities.sum(axis=1), axis=0)
    composite_delta_utility = np.log(addents.sum(axis=1, skipna=True))
    return composite_delta_utility


def incremental_nest_utility_pl(delta_utilities: np.ndarray, probabilities: np.ndarray):
    addents = np.exp(delta_utilities) * probabilities / probabilities.sum(axis=1, keepdims=True)
    composite_delta_utility = np.log(np.nansum(addents, axis=1, keepdims=True))
    return composite_delta_utility


def nest_probabilities(utilities: np.ndarray, phi: float) -> np.ndarray:
    if phi == 0:
        phi = 1
    exp_mat = np.exp(np.multiply(utilities, 1 / phi))
    tot = np.sum(exp_mat, axis=1).reshape(-1, 1)

    return exp_mat / tot


def incremental_nest_probability(
    delta_utilities: pd.DataFrame | pd.Series,
    probabilities: pd.DataFrame | pd.Series,
):
    cond_probabilities = probabilities.divide(probabilities.sum(axis=1), axis=0)
    addents = np.exp(delta_utilities) * cond_probabilities
    denum = (addents).sum(axis=1, skipna=True)
    new_probabilities = np.exp(delta_utilities) * cond_probabilities.divide(denum, axis=0)
    return new_probabilities


def incremental_nest_probability_pl(delta_utilities: np.ndarray, probabilities: np.ndarray):
    cond_probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    addents = np.exp(delta_utilities) * cond_probabilities
    denum = np.nansum(addents, axis=1, keepdims=True)
    new_probabilities = np.exp(delta_utilities) * cond_probabilities / denum
    return new_probabilities


def rank_and_mode_utilities(
    paths: pd.DataFrame,
    ascending_modes: list[str],
    route_types: list[str],
    phi: dict[str, float],
    od_cols: list[str] = ['origin', 'destination'],
    verbose=True,
):
    # rank_utilities
    # a bit faster than set_index() unstack()
    rank_utilities = paths.pivot(index=[*od_cols, 'segment'], columns=['route_type', 'rank'], values='utility')
    rank_utilities.fillna(-np.inf, inplace=True)

    # mode utilities
    mode_utilities = pd.DataFrame(index=rank_utilities.index)
    mode_utilities.columns.name = 'route_type'
    mode_utilities[ascending_modes] = -np.inf

    # aggregate rank_utilities
    for mode in route_types:
        print('path utilities', mode, phi[mode], '->', mode) if verbose else None
        mode_utilities[mode] = nest_utility(rank_utilities[mode].values, phi[mode])

    return rank_utilities, mode_utilities


def rank_and_mode_utilities_pl(
    paths: pl.DataFrame,
    ascending_modes: list[str],
    route_types: list[str],
    phi: dict[str, float],
    index_cols: list[str] = ['origin', 'destination', 'segment'],
    verbose=True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pl.DataFrame]:
    rank_utilities = paths.pivot(on=['route_type', 'rank'], index=index_cols, values='utility').with_row_index('_group')
    groups = rank_utilities.select([*index_cols, '_group'])
    rank_utilities = rank_utilities.drop([*index_cols, '_group']).fill_null(-np.inf)

    n_group = len(groups)
    inf_arr = np.full((n_group, 1), -np.inf)

    rank_utilities_dict = {}
    mode_utilities_dict = {mode: inf_arr for mode in ascending_modes}
    for mode in route_types:
        print('path utilities', mode, phi[mode], '->', mode) if verbose else None
        cols = [col for col in rank_utilities.columns if mode in col]  # get cols like '{"rail",1}','{"rail",2}'
        cols = sorted(cols, key=lambda x: int(x.split(',')[1][:-1]))  # make sure they are orderd with rank [1,2,3,...]
        mat = rank_utilities.select(pl.col(cols)).to_numpy()
        rank_utilities_dict[mode] = mat
        mode_utilities_dict[mode] = nest_utility(mat, phi[mode])
    return rank_utilities_dict, mode_utilities_dict, groups


def symmetrize_utilities(utilities: pd.DataFrame | pd.Series):
    reversed_utilities = utilities.swaplevel(0, 1)
    reversed_utilities.index.names = utilities.index.names
    return utilities.add(reversed_utilities) / 2


def symmetrize_utilities_np(utilities: dict[str, np.ndarray], groups: pl.DataFrame):
    rindexes = groups.with_columns(  # reverse origin and destination
        [
            pl.col('origin').alias('destination'),
            pl.col('destination').alias('origin'),
        ]
    )
    index_cols = groups.drop('_group').columns
    indexes = groups.join(rindexes, on=index_cols, how='left').select(['_group', '_group_right']).sort('_group')
    rindexes = indexes['_group_right'].to_numpy()
    for mode, utility in utilities.items():
        utilities[mode] = (utility + utility[rindexes]) / 2
    return utilities


def plot_nests(nests):
    g = nx.DiGraph(nests)
    root = [n for n in g.nodes if g.in_degree(n) == 0][0]
    depths = nx.single_source_shortest_path_length(g, root)
    pos = {}
    levels = [0] * (max(depths.values()) + 1)
    for key, x in depths.items():
        pos[key] = [levels[x], -x]
        levels[x] += 1

    plot = plt.axes()
    nx.draw(
        g,
        pos=pos,
        node_color='white',
        alpha=1,
        node_size=1000,
        arrows=False,
        edge_color='green',
        font_size=15,
        font_weight='normal',
        labels={k: k for k in g.nodes},
        axes=plot,
    )
    return plot


def nested_logit_from_paths(
    paths: pd.DataFrame,
    od_cols=['origin', 'destination'],
    mode_nests=None,
    phi=None,
    verbose=False,
    decimals=None,  # minimum probability
    n_paths_max=None,
    nchunks=10,
    workers=1,
    return_od_tables=True,
    symmetric=False,
    clip=-np.inf,
    incremental=False,
    ref_paths=None,
    ref_probabilities=None,
):
    groups_col = od_cols[0]  # split by origin so each tasks have all of the paths for a given origin.
    origins = list(paths[groups_col].unique())
    groups = np.array_split(origins, nchunks)

    if workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for group in tqdm(groups):
                chunk = paths[paths[groups_col].isin(group)]
                p = executor.submit(
                    one_block_nested_logit_from_paths,
                    chunk,
                    phi=phi,
                    mode_nests=mode_nests,
                    od_cols=od_cols,
                    decimals=decimals,
                    n_paths_max=n_paths_max,
                    return_od_tables=return_od_tables,
                    symmetric=symmetric,
                    clip=clip,
                    incremental=incremental,
                    ref_paths=ref_paths,
                    ref_probabilities=ref_probabilities,
                )
                results.append(p)

        results = [res.result() for res in results]
        paths = pd.concat([res[0] for res in results])
        mode_utilities = pd.concat([res[1] for res in results])
        mode_probabilities = pd.concat([res[2] for res in results])

    else:
        p_list = []
        mu_list = []
        mp_list = []
        for group in tqdm(groups):
            chunk = paths[paths[groups_col].isin(group)]
            p, mu, mp = one_block_nested_logit_from_paths(
                chunk,
                phi=phi,
                mode_nests=mode_nests,
                od_cols=od_cols,
                decimals=decimals,
                n_paths_max=n_paths_max,
                return_od_tables=return_od_tables,
                symmetric=symmetric,
                clip=clip,
                incremental=incremental,
                ref_paths=ref_paths,
                ref_probabilities=ref_probabilities,
            )
            p_list.append(p)
            mu_list.append(mu)
            mp_list.append(mp)

        paths = pd.concat(p_list)
        mode_utilities = pd.concat(mu_list)
        mode_probabilities = pd.concat(mp_list)
    return paths, mode_utilities, mode_probabilities


def _preparation(mode_nests: dict[str, list[str]], phi: dict[str, float] | None):
    g = nx.DiGraph(mode_nests)
    root = [n for n in g.nodes if g.in_degree(n) == 0][0]
    depths = nx.single_source_shortest_path_length(g, root)

    if phi is None:
        phi = {mode: 1 for mode in g.nodes}

    # fill phi_dict if is not complete (make it collapse)
    def recursive_phi(mode):
        try:
            return phi[mode]
        except KeyError:
            parent = list(nx.neighbors(g.reverse(), mode))[0]
            phi[mode] = recursive_phi(parent)
            return phi[mode]

    phi = {n: recursive_phi(n) for n in g.nodes}

    ascending_modes = []
    for degree in range(max(depths.values()), -1, -1):
        leaf_modes = [n for n in g.nodes if depths[n] == degree]
        ascending_modes += leaf_modes
    neighbors = {}
    for mode in ascending_modes:
        neighbors[mode] = list(nx.neighbors(g, mode))
    return ascending_modes, phi, neighbors


def one_block_nested_logit_from_paths(*args, **kwargs):
    return one_block_nested_logit_from_paths_np(*args, **kwargs)
    # return one_block_nested_logit_from_paths_pl(*args, **kwargs)


def one_block_nested_logit_from_paths_df(
    paths: pd.DataFrame,
    od_cols=['origin', 'destination'],
    mode_nests=None,
    phi=None,
    verbose=False,
    decimals=None,  # minimum probability
    n_paths_max=None,
    return_od_tables=True,
    symmetric=False,
    clip=-np.inf,
    incremental=False,
    ref_paths=None,
    ref_probabilities=None,
):

    if 'segment' not in paths.columns:
        paths['segment'] = 'all'

    if mode_nests is None:
        mode_nests = {'root': list(set(paths['route_type']))}

    ascending_modes, phi, neighbors = _preparation(mode_nests, phi)
    descending_modes = list(reversed(ascending_modes))
    route_types = list(paths['route_type'].unique())

    # set to category for a boost in performance (memory and speed)
    for name in [*od_cols, 'segment']:
        paths[name] = paths[name].astype('category')
    modes_dtype = pd.CategoricalDtype(categories=set([*ascending_modes, *route_types]))
    paths['route_type'] = paths['route_type'].astype(modes_dtype)

    # get ranks and filter n_paths_max
    paths = rank_and_filter_paths(paths, od_cols=od_cols, n_paths_max=n_paths_max)

    #
    # rank utilities and initialize probabilities
    #
    rank_utilities, mode_utilities = rank_and_mode_utilities(
        paths, ascending_modes, route_types, phi, od_cols=od_cols, verbose=verbose
    )

    mode_probabilities = pd.DataFrame(index=rank_utilities.index)
    mode_probabilities.columns.name = 'route_type'

    if incremental:
        ref_paths = rank_and_filter_paths(ref_paths, od_cols=od_cols, n_paths_max=n_paths_max)
        _, ref_mode_utilities = rank_and_mode_utilities(
            ref_paths, ascending_modes, route_types, phi, od_cols=od_cols, verbose=verbose
        )

        for mode in route_types:
            mode_utilities[mode] = (
                mode_utilities[mode].sub(ref_mode_utilities[mode].replace([-np.inf], -100)).fillna(0.0)
            )  # Delta utilitie
            mode_utilities[mode] = mode_utilities[mode].replace([-np.inf], -100)

        ref_probabilities.set_index(mode_probabilities.index.names, inplace=True)

    if symmetric:
        mode_utilities = symmetrize_utilities(mode_utilities)

    mode_utilities = mode_utilities.clip(clip, -clip)

    # propagate utilities to higher modes (bottom -> up) (leaf => root)
    for mode in ascending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode utilities', children, phi[mode], '->', mode) if verbose else None
            if incremental:
                mode_utilities[mode] = incremental_nest_utility(mode_utilities[children], ref_probabilities[children])
            else:  # Direct Logit
                mode_utilities[mode] = nest_utility(mode_utilities[children].values, phi=phi[mode])

    mode_utilities = mode_utilities[descending_modes]

    #
    #  mode probability
    #
    #  propagate probabilities

    mode_probabilities[descending_modes[0]] = 1  # root mode
    for mode in descending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode probabilities', mode, phi[mode], '->', children) if verbose else None
            if incremental:
                partial = incremental_nest_probability(mode_utilities[children], ref_probabilities[children])
            else:
                partial = nest_probabilities(utilities=mode_utilities[children].values, phi=phi[mode])
            mode_probabilities[children] = mode_probabilities[[mode]].values * partial
    mode_probabilities = mode_probabilities[descending_modes].fillna(0)

    #
    # stack probabilities and merge on back on paths
    #
    # rank_probabilities
    rank_probabilities = rank_utilities.copy()
    for mode in route_types:
        print('path probabilities', mode, phi[mode], '->', mode) if verbose else None
        rank_probabilities[mode] = nest_probabilities(rank_utilities[mode].values, phi[mode])

    if decimals is not None:
        rank_probabilities = np.round(rank_probabilities, decimals)
    rank_probabilities = rank_probabilities.replace(0, np.nan)
    rank_probabilities_s = rank_probabilities.stack().stack()
    rank_probabilities_s.name = 'assignment_share'

    mode_probabilities_s = mode_probabilities.stack()
    mode_probabilities_s = mode_probabilities_s.loc[mode_probabilities_s > 0]
    mode_probabilities_s.name = 'modal_split_share'

    rank_probabilities_s = rank_probabilities_s.reset_index()
    mode_probabilities_s = mode_probabilities_s.reset_index()
    mode_probabilities_s['route_type'] = mode_probabilities_s['route_type'].astype(modes_dtype)

    # merge assignment probablities on paths

    merged = pd.merge(rank_probabilities_s, mode_probabilities_s, on=[*od_cols, 'route_type', 'segment'])
    merged['probability'] = merged['assignment_share'] * merged['modal_split_share']

    merge_columns = [*od_cols, 'route_type', 'segment', 'rank']

    paths['index'] = paths.index
    paths = pd.merge(
        paths.drop('probability', axis=1, errors='ignore'),
        merged[[*merge_columns, 'probability']],
        on=merge_columns,
        suffixes=['_old', ''],
        how='left',
    ).set_index('index')

    if not return_od_tables:
        mode_utilities = pd.DataFrame()
        mode_probabilities = pd.DataFrame()

    return paths, mode_utilities.reset_index(), mode_probabilities.reset_index()


def one_block_nested_logit_from_paths_np(
    paths: pd.DataFrame,
    od_cols=['origin', 'destination'],
    mode_nests: dict[str, list[str]] | None = None,
    phi: dict[str, float] | None = None,
    verbose: bool = False,
    decimals: int | None = None,  # minimum probability
    n_paths_max: int | None = None,
    return_od_tables: bool = True,
    symmetric: bool = False,
    clip: float = -np.inf,
    incremental: bool = False,
    ref_paths: pd.DataFrame | None = None,
    ref_probabilities: pd.DataFrame | None = None,
):
    index_cols = [*od_cols, 'segment']
    if 'segment' not in paths.columns:
        paths['segment'] = 'all'

    if mode_nests is None:
        mode_nests = {'root': list(set(paths['route_type']))}

    ascending_modes, phi, neighbors = _preparation(mode_nests, phi)
    descending_modes = list(reversed(ascending_modes))
    route_types = list(paths['route_type'].unique())

    # set to category for a boost in performance (memory and speed)
    for name in [*od_cols, 'segment', 'route_type']:
        paths[name] = paths[name].astype('category')

    paths_pl = pl.DataFrame(paths.reset_index())
    paths_pl = rank_and_filter_paths_pl(paths_pl, od_cols=od_cols, n_paths_max=n_paths_max)
    rank_utilities_dict, mode_utilities_dict, groups = rank_and_mode_utilities_pl(
        paths_pl, ascending_modes, route_types, phi, index_cols, verbose=verbose
    )
    paths_pl = paths_pl.join(groups, on=index_cols, maintain_order='left')
    paths = paths_pl.to_pandas().set_index('index')
    del paths_pl

    if incremental:
        ref_paths_pl = pl.DataFrame(ref_paths)
        ref_prob_pl = pl.DataFrame(ref_probabilities)
        for col in index_cols:
            dtype = groups.schema[col]
            ref_prob_pl = ref_prob_pl.with_columns(pl.col(col).cast(dtype))
            ref_paths_pl = ref_paths_pl.with_columns(pl.col(col).cast(dtype))
        ref_prob_pl = ref_prob_pl.join(groups, on=index_cols, how='right')  # maintain order
        ref_paths_pl = ref_paths_pl.join(groups, on=index_cols, how='right')
        ref_paths_pl = rank_and_filter_paths_pl(ref_paths_pl, od_cols=od_cols, n_paths_max=n_paths_max)
        _, ref_mode_utilities, _ = rank_and_mode_utilities_pl(
            ref_paths_pl, ascending_modes, route_types, phi, index_cols, verbose=verbose
        )
        for mode in route_types:  # Delta utilitie
            ref_utility = ref_mode_utilities[mode]
            ref_utility[np.isinf(ref_utility)] = -100
            utility = mode_utilities_dict[mode]
            utility = utility - ref_utility
            utility[np.isnan(ref_utility)] = 0
            utility[np.isinf(ref_utility)] = -100
            mode_utilities_dict[mode] = utility

    if symmetric:
        mode_utilities_dict = symmetrize_utilities_np(mode_utilities_dict, groups)

    for key, utility in mode_utilities_dict.items():
        mode_utilities_dict[key] = utility.clip(-clip, clip)  # TODO was reversed...why

    # propagate utilities to higher modes (bottom -> up) (leaf => root)
    for mode in ascending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode utilities', children, phi[mode], '->', mode) if verbose else None
            utilities = np.hstack([mode_utilities_dict[ch] for ch in children])
            if incremental:
                mode_utilities_dict[mode] = incremental_nest_utility_pl(utilities, ref_prob_pl[children].to_numpy())
            else:  # Direct Logit
                mode_utilities_dict[mode] = nest_utility(utilities, phi=phi[mode])

    #
    #  mode probability
    #
    #  propagate probabilities

    arr = np.full_like(len(groups), 1)
    mode_probabilities = {mode: arr for mode in descending_modes}

    for mode in descending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode probabilities', mode, phi[mode], '->', children) if verbose else None
            utilities = np.hstack([mode_utilities_dict[ch] for ch in children if ch in mode_utilities_dict])
            if incremental:
                partial = incremental_nest_probability_pl(utilities, ref_prob_pl[children].to_numpy())
            else:
                partial = nest_probabilities(utilities=utilities, phi=phi[mode])
            probabilities = mode_probabilities[mode] * partial
            probabilities[np.isnan(probabilities)] = 0
            for i, child in enumerate(children):
                mode_probabilities[child] = probabilities[:, i, np.newaxis]

    #
    # stack probabilities and merge on back on paths
    #
    # rank_probabilities
    rank_probabilities = {}
    for mode in route_types:
        print('path probabilities', mode, phi[mode], '->', mode) if verbose else None
        probabilities = nest_probabilities(rank_utilities_dict[mode], phi[mode])
        probabilities[np.isnan(probabilities)] = 0
        if decimals is not None:
            probabilities = np.round(probabilities, decimals)
        rank_probabilities[mode] = probabilities

    route_types_dict = {mode: np.int32(i) for i, mode in enumerate(route_types)}
    paths['_mode'] = paths['route_type'].map(route_types_dict).astype(int)
    paths['_rank'] = paths['rank'] - 1

    paths['assignment_share'] = 0
    for mode, probabilities in rank_probabilities.items():
        mask = paths['route_type'] == mode

        paths.loc[mask, 'assignment_share'] = probabilities[
            paths.loc[mask, '_group'].to_numpy(),
            paths.loc[mask, '_rank'].to_numpy(),
        ]

    stacked_probabilities = np.stack([mode_probabilities[mode] for mode in route_types_dict])
    # Retrieve probabilities for each row
    paths['modal_split_share'] = stacked_probabilities[
        paths['_mode'].to_numpy(),
        paths['_group'].to_numpy(),
    ]

    paths['probability'] = paths['assignment_share'] * paths['modal_split_share']
    paths = paths.drop(columns=['_group', '_mode', '_rank', 'assignment_share', 'modal_split_share'])
    if return_od_tables:
        utility_df = groups.select(index_cols).to_pandas()
        probability_df = groups.select(index_cols).to_pandas()
        for mode in descending_modes:
            utility_df[mode] = mode_utilities_dict[mode]
            probability_df[mode] = mode_probabilities[mode]

        return paths, utility_df, probability_df
    else:
        return paths, pd.DataFrame(), pd.DataFrame()
