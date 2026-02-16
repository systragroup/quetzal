import polars as pl
from typing import Dict, List, Any


def analysis_path(df: pl.DataFrame, vertex_type: Dict[Any, int], excluded: List[str] = []):
    # vertex_type={1: zones, 2: nodes, 3: links}
    # excluded: list of attribute we dont want to compute.

    agg_kwargs = {
        'link_path': pl.col('path').filter(pl.col('type') == 3),
        'node_path': pl.col('path').filter(pl.col('type') == 2),
        'boardings': pl.col('path').filter((pl.col('type') == 2) & (pl.col('to') == 3)),
        'alightings': pl.col('path').filter((pl.col('type') == 2) & (pl.col('from') == 3)),
        'boarding_links': pl.col('path').filter((pl.col('type') == 3) & (pl.col('from') != 3)),
        'alighting_links': pl.col('path').filter((pl.col('type') == 3) & (pl.col('to') != 3)),
        'transfers': pl.col('path').filter((pl.col('type') == 2) & (pl.col('from') == 3) & (pl.col('to') == 3)),
        'ntlegs': pl.concat_list([pl.col('path').head(2), pl.col('path').tail(2)]),
        'footpaths': pl.concat_list(
            [
                pl.col('path').filter((pl.col('type') == 2) & (pl.col('to') == 2)),
                pl.col('path').filter((pl.col('type') == 2) & (pl.col('from') == 2)),
            ]
        ),
    }
    # remove expr that are in the exluded_list
    agg_kwargs = {k: expr for k, expr in agg_kwargs.items() if k not in excluded}

    df = df.lazy()
    df = df.explode(['path'])  # explode df on paths
    df = df.with_columns(pl.col('path').replace_strict(vertex_type, default=None).alias('type'))
    df = df.with_columns(pl.col('type').shift(-1).over('index').alias('to'))
    df = df.with_columns(pl.col('type').shift(+1).over('index').alias('from'))
    df = df.group_by('index', maintain_order=True).agg(**agg_kwargs)

    # computed metrics on agg dataframe
    if ('boarding_links' in agg_kwargs.keys()) & ('ntransfers' not in excluded):
        # need to cast as Int32 because len return uint32. 0-1 in uint overflow to like 4294967295
        df = df.with_columns(ntransfers=(pl.col('boarding_links').list.len().cast(pl.Int32) - 1).clip(0))

    if ('link_path' in agg_kwargs.keys()) & ('all_walk' not in excluded):
        df = df.with_columns(all_walk=(pl.col('link_path').list.len() == 0))

    return df.collect(engine='streaming')


def analysis_pt_los(self, walk_on_road: bool = True, excluded: List[str] = []):
    nodes = [*self.nodes.index, *self.road_nodes.index] if walk_on_road else self.nodes.index
    vertex_sets = {1: set(self.zones.index), 2: set(nodes), 3: set(self.links.index)}
    vertex_type = {}
    for vtype, vset in vertex_sets.items():
        for v in vset:
            vertex_type[v] = vtype
    paths_df = analysis_path(self.pt_los['index', 'path'], vertex_type, excluded)

    to_drop = set(paths_df.columns).union(excluded) - {'index'}  # remove existing columns if already exist (we replace)
    self.pt_los = self.pt_los.drop(to_drop, strict=False).join(paths_df, on='index', maintain_order='left')
