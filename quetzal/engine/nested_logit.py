from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import polars as pl
from typing import Literal


def rank_and_filter_paths(
    paths: pd.DataFrame,
    index_cols: list[str] = ['origin', 'destination', 'segment'],
    by: str = 'utility',
    n_paths_max: int | None = None,
    engine='polars',
):
    if n_paths_max is not None:
        if engine == 'polars':
            # faster with polars to rank
            cols = [*index_cols, 'route_type']
            pdf = pl.DataFrame(paths[[*cols, by]])
            paths['rank'] = pdf.select(pl.col(by).rank(method='ordinal', descending=True).over(cols)).to_numpy()
        else:
            # sort then cumsum gives us rank
            paths = paths.sort_values(by=by, ascending=False)
            paths['rank'] = paths.groupby([*index_cols, 'route_type']).cumcount() + 1
            paths = paths.sort_index()  # sort back
    else:
        # rank is random here
        paths['rank'] = paths.groupby([*index_cols, 'route_type']).cumcount() + 1
    paths['rank'] = paths['rank'].astype(np.int32)

    if n_paths_max is not None:
        paths = paths.loc[paths['rank'] <= n_paths_max]

    return paths


def nest_utility(utilities: np.ndarray, phi: float = 1):
    if phi == 0:
        phi = 1
    exp_mat = np.exp(np.multiply(utilities, 1 / phi))
    exponential_s = np.sum(exp_mat, axis=1).reshape(-1, 1)
    composite_utility = phi * np.log(exponential_s)
    return composite_utility


def incremental_nest_utility(delta_utilities: np.ndarray, probabilities: np.ndarray):
    addents = np.exp(delta_utilities) * probabilities / probabilities.sum(axis=1, keepdims=True)
    composite_delta_utility = np.log(np.nansum(addents, axis=1, keepdims=True))
    return composite_delta_utility


def nest_probabilities(utilities: np.ndarray, phi: float) -> np.ndarray:
    if phi == 0:
        phi = 1
    exp_mat = np.exp(np.multiply(utilities, 1 / phi))
    tot = np.sum(exp_mat, axis=1).reshape(-1, 1)

    return exp_mat / tot


def incremental_nest_probability(delta_utilities: np.ndarray, probabilities: np.ndarray):
    cond_probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    addents = np.exp(delta_utilities) * cond_probabilities
    denum = np.nansum(addents, axis=1, keepdims=True)
    new_probabilities = np.exp(delta_utilities) * cond_probabilities / denum
    return new_probabilities


def symmetrize_utilities(utilities: dict[str, np.ndarray], group_serie: pd.Series):
    # utility of an od become (u_od + u_do)/2 unchanged if no return path
    reversed_group = group_serie.swaplevel(0, 1)
    reversed_group.index.names = group_serie.index.names
    indexes = group_serie.to_frame().merge(reversed_group, left_index=True, right_index=True).values
    idx = indexes[:, 0]
    reversed_idx = indexes[:, 1]
    for mode, utility in utilities.items():
        utilities[mode][idx] = (utility[idx] + utility[reversed_idx]) / 2
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
    inf=np.inf,
    incremental=False,
    ref_paths=None,
    ref_probabilities=None,
    engine='polars',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_paths_max is None:
        print('rank are random when n_paths_max is None')

    groups_col = od_cols[0]  # split by origin so each tasks have all of the paths for a given origin.
    origins = list(paths[groups_col].unique())
    groups = np.array_split(origins, nchunks)
    _dict = {el: i for i, group in enumerate(groups) for el in group}
    group_mask = paths[groups_col].map(_dict.get)

    if workers > 1:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in tqdm(range(len(groups))):
                p = executor.submit(
                    one_block_nested_logit_from_paths,
                    paths=paths[group_mask == i],
                    phi=phi,
                    verbose=verbose,
                    mode_nests=mode_nests,
                    od_cols=od_cols,
                    decimals=decimals,
                    n_paths_max=n_paths_max,
                    return_od_tables=return_od_tables,
                    symmetric=symmetric,
                    clip=clip,
                    inf=inf,
                    incremental=incremental,
                    ref_paths=ref_paths,
                    ref_probabilities=ref_probabilities,
                    engine=engine,
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
        for i in tqdm(range(len(groups))):
            p, mu, mp = one_block_nested_logit_from_paths(
                paths=paths[group_mask == i],
                phi=phi,
                verbose=verbose,
                mode_nests=mode_nests,
                od_cols=od_cols,
                decimals=decimals,
                n_paths_max=n_paths_max,
                return_od_tables=return_od_tables,
                symmetric=symmetric,
                clip=clip,
                inf=inf,
                incremental=incremental,
                ref_paths=ref_paths,
                ref_probabilities=ref_probabilities,
                engine=engine,
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


def group_paths(paths: pd.DataFrame, index_cols: list[str], engine='polars'):

    if engine == 'polars':
        pdf = pl.DataFrame(paths[index_cols])
        groups = pdf.select(index_cols).unique(maintain_order=True).with_row_index('_group')
        group_serie = groups.to_pandas().set_index(index_cols)['_group']
        paths['_group'] = pdf.join(groups, on=index_cols)['_group'].to_numpy()
    else:
        paths['_group'] = paths.groupby(index_cols, observed=True, sort=False).ngroup()
        group_serie = paths.drop_duplicates('_group').set_index(index_cols)['_group']
    return paths, group_serie


def rank_and_mode_utilities(
    df: pd.DataFrame,
    ascending_modes: list[str],
    phi: dict[str, float],
    inf: float = np.inf,
):

    n_groups = df['_group'].max() + 1
    n_ranks = df['rank'].max()

    inf_arr = np.full((n_groups, 1), -inf, dtype=np.float64)

    rank_utilities_dict = {}
    mode_utilities_dict = {mode: inf_arr for mode in ascending_modes}
    # TODO to filter per mode. we could try to take path['route_type'] get its index (per mode) and filter rows,cols,data with it
    for mode, mode_path in df.groupby('route_type'):
        mat = np.full((n_groups, n_ranks), -inf, dtype=np.float64)
        rows = mode_path['_group'].to_numpy()
        cols = mode_path['rank'].to_numpy() - 1
        data = mode_path['utility'].to_numpy()
        mat[rows, cols] = data
        rank_utilities_dict[mode] = mat
        mode_utilities_dict[mode] = nest_utility(mat, phi[mode])  # type: ignore

    return rank_utilities_dict, mode_utilities_dict


def propagate_mode_utility(
    mode_utilities: dict[str, np.ndarray],
    ascending_modes: list[str],
    neighbors: dict,
    phi: dict[str, float],
    verbose: bool,
):
    # propagate utilities to higher modes (bottom -> up) (leaf => root)
    for mode in ascending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode utilities', children, phi[mode], '->', mode) if verbose else None
            utilities = np.hstack([mode_utilities[ch] for ch in children])
            mode_utilities[mode] = nest_utility(utilities, phi=phi[mode])
    return mode_utilities


def propagate_mode_utility_incremental(
    mode_utilities: dict[str, np.ndarray],
    ascending_modes: list[str],
    neighbors: dict,
    phi: dict[str, float],
    ref_probs: pd.DataFrame,
    verbose: bool,
):
    # propagate utilities to higher modes (bottom -> up) (leaf => root)
    for mode in ascending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode utilities', children, phi[mode], '->', mode) if verbose else None
            utilities = np.hstack([mode_utilities[ch] for ch in children])
            mode_utilities[mode] = incremental_nest_utility(utilities, ref_probs[children].to_numpy())
    return mode_utilities


def mode_probability(
    mode_utilities: dict[str, np.ndarray],
    descending_modes: list[str],
    neighbors: dict,
    phi: dict[str, float],
    verbose: bool,
):
    root = descending_modes[0]
    mode_probabilities = {root: np.full_like(mode_utilities[root], 1)}

    for mode in descending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode probabilities', mode, phi[mode], '->', children) if verbose else None
            utilities = np.hstack([mode_utilities[ch] for ch in children if ch in mode_utilities])
            partial = nest_probabilities(utilities=utilities, phi=phi[mode])
            probabilities = mode_probabilities[mode] * partial
            probabilities[np.isnan(probabilities)] = 0
            for i, child in enumerate(children):
                mode_probabilities[child] = probabilities[:, i, np.newaxis]
    return mode_probabilities


def mode_probability_incremental(
    mode_utilities: dict[str, np.ndarray],
    descending_modes: list[str],
    neighbors: dict,
    phi: dict[str, float],
    ref_probs: pd.DataFrame,
    verbose: bool,
):
    root = descending_modes[0]  # init root at 1, then propagate from there
    mode_probabilities = {root: np.full_like(mode_utilities[root], 1)}
    for mode in descending_modes:
        children = neighbors.get(mode, [])
        if len(children):
            print('mode probabilities', mode, phi[mode], '->', children) if verbose else None
            utilities = np.hstack([mode_utilities[ch] for ch in children if ch in mode_utilities])
            partial = incremental_nest_probability(utilities, ref_probs[children].to_numpy())
            probabilities = mode_probabilities[mode] * partial
            probabilities[np.isnan(probabilities)] = 0
            for i, child in enumerate(children):
                mode_probabilities[child] = probabilities[:, i, np.newaxis]
    return mode_probabilities


def rank_probability(
    rank_utilities: dict[str, np.ndarray],
    phi: dict[str, float],
    decimals: int | None,
    verbose: bool,
):
    rank_probabilities = {}
    for mode, utility in rank_utilities.items():
        print('path probabilities', mode, phi[mode], '->', mode) if verbose else None
        probabilities = nest_probabilities(utility, phi[mode])
        probabilities[np.isnan(probabilities)] = 0
        if decimals is not None:
            probabilities = np.round(probabilities, decimals)
        rank_probabilities[mode] = probabilities
    return rank_probabilities


def one_block_nested_logit_from_paths(
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
    inf=np.inf,
    incremental: bool = False,
    ref_paths: pd.DataFrame | None = None,
    ref_probabilities: pd.DataFrame | None = None,
    engine: Literal['polars', 'pandas'] = 'polars',
):
    index_cols = [*od_cols, 'segment']
    if 'segment' not in paths.columns:
        paths['segment'] = 'all'

    if mode_nests is None:
        mode_nests = {'root': list(set(paths['route_type']))}

    if clip > 0:
        clip = -clip
    if inf < 0:
        inf = -inf

    ascending_modes, phi, neighbors = _preparation(mode_nests, phi)
    descending_modes = list(reversed(ascending_modes))

    # set to category for a boost in performance (memory and speed)
    for name in [*od_cols, 'segment', 'route_type']:
        paths[name] = paths[name].astype('category')

    paths = rank_and_filter_paths(paths, index_cols=index_cols, n_paths_max=n_paths_max, engine=engine)
    paths, group_serie = group_paths(paths, index_cols=index_cols, engine=engine)

    rank_utilities, mode_utilities = rank_and_mode_utilities(paths, ascending_modes, phi, inf=inf)

    if incremental:
        assert ref_probabilities is not None
        assert ref_paths is not None

        ref_probs = ref_probabilities.merge(group_serie, on=index_cols, how='left')
        ref_probs = ref_probs.sort_values('_group')

        ref_paths = rank_and_filter_paths(ref_paths, index_cols=index_cols, n_paths_max=n_paths_max)
        ref_paths = ref_paths.merge(group_serie, on=index_cols, how='left')  # add _group

        _, ref_mode_utilities = rank_and_mode_utilities(ref_paths, ascending_modes, phi, inf=inf)

        for mode, utility in mode_utilities.items():  # Delta utilitie
            delta_utilities = utility - ref_mode_utilities[mode]
            delta_utilities[np.isnan(delta_utilities)] = 0
            mode_utilities[mode] = delta_utilities

    if symmetric:
        mode_utilities = symmetrize_utilities(mode_utilities, group_serie)

    for key, utility in mode_utilities.items():
        mode_utilities[key] = utility.clip(clip, -clip)
    #
    # propagate utilities to higher modes (bottom -> up) (leaf => root)
    #
    if incremental:
        assert ref_probs is not None
        mode_utilities = propagate_mode_utility_incremental(
            mode_utilities, ascending_modes, neighbors, phi, ref_probs, verbose
        )
    else:
        mode_utilities = propagate_mode_utility(mode_utilities, ascending_modes, neighbors, phi, verbose)
    #
    #  propagate probabilities
    #
    if incremental:
        assert ref_probs is not None
        mode_probabilities = mode_probability_incremental(
            mode_utilities, descending_modes, neighbors, phi, ref_probs, verbose
        )
    else:
        mode_probabilities = mode_probability(mode_utilities, descending_modes, neighbors, phi, verbose)

    rank_probabilities = rank_probability(rank_utilities, phi, decimals, verbose)

    #
    # compute paths probability and merge back on path
    #
    route_types = list(paths['route_type'].unique())
    route_types_dict = {mode: np.int32(i) for i, mode in enumerate(route_types)}
    paths['_mode'] = paths['route_type'].map(route_types_dict).astype(int)
    paths['_rank'] = paths['rank'] - 1

    paths['assignment_share'] = 0
    for mode, rank_probs in rank_probabilities.items():
        mask = paths['route_type'] == mode
        _rows = paths.loc[mask, '_group'].to_numpy()
        _cols = paths.loc[mask, '_rank'].to_numpy()
        paths.loc[mask, 'assignment_share'] = rank_probs[_rows, _cols]

    paths['modal_split_share'] = 0
    stacked_probabilities = np.stack([mode_probabilities[mode] for mode in route_types_dict])
    _rows = paths['_mode'].to_numpy()
    _cols = paths['_group'].to_numpy()
    paths['modal_split_share'] = stacked_probabilities[_rows, _cols]

    paths['probability'] = paths['assignment_share'] * paths['modal_split_share']
    paths = paths.drop(columns=['_group', '_mode', '_rank', 'assignment_share', 'modal_split_share'])

    if return_od_tables:
        utility_df = _to_df(group_serie, mode_utilities, descending_modes)
        probability_df = _to_df(group_serie, mode_probabilities, descending_modes)
        return paths, utility_df, probability_df
    else:
        return paths, pd.DataFrame(), pd.DataFrame()


def _to_df(group_serie: pd.Series, mat_dict: dict[str, np.ndarray], modes: list[str]):
    df = group_serie.index.to_frame(index=False)
    order = group_serie.to_numpy()
    for mode in modes:
        df[mode] = mat_dict[mode][order]
    return df
