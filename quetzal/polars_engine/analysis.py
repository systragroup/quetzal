import polars as pl
from typing import Dict


def analysis_path(df: pl.DataFrame, vertex_type: Dict):
    # {1: zones, 2: nodes, 3: links}

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

    df = df.lazy()
    df = df.explode(['path'])  # explode df on paths
    df = df.with_columns(pl.col('path').replace_strict(vertex_type, default=None).alias('type'))
    df = df.with_columns(pl.col('type').shift(-1).over('index').alias('to'))
    df = df.with_columns(pl.col('type').shift(+1).over('index').alias('from'))
    df = df.group_by('index', maintain_order=True).agg(**agg_kwargs)
    # need to cast as Int32 because len return uint32. 0-1 in uint overflow to like 4294967295
    df = df.with_columns(
        ntransfers=(pl.col('boarding_links').list.len().cast(pl.Int32) - 1).clip(0),
        all_walk=(pl.col('link_path').list.len() == 0),
    )

    return df.collect(engine='streaming')


def analysis_pt_los(self, walk_on_road=True):
    nodes = [*self.nodes.index, *self.road_nodes.index] if walk_on_road else self.nodes.index
    vertex_sets = {1: set(self.zones.index), 2: set(nodes), 3: set(self.links.index)}
    vertex_type = {}
    for vtype, vset in vertex_sets.items():
        for v in vset:
            vertex_type[v] = vtype
    paths_df = analysis_path(self.pt_los['index', 'path'], vertex_type)

    paths_df = paths_df.with_columns(pl.col('boarding_links').list.len().alias('ntransfers'))
    self.pt_los = self.pt_los.join(paths_df, on='index')
    return paths_df
