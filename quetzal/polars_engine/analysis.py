import polars as pl
import pandas as pd
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
        'access': pl.col('path').head(2),  # its way simpler to just add 2 columns than having a list of list for ntlegs
        'eggress': pl.col('path').tail(2),
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


def analysis_pt_time(self, walk_on_road=False):
    assert 'boarding_time' in self.links.columns, 'need boarding_time in self.links'
    # need: link_path,boarding_links,footpaths,access,eggress

    if walk_on_road:
        road_links = self.road_links.copy()
        road_links['time'] = road_links['walk_time']
        road_to_transit = self.road_to_transit.copy()

        footpaths = pd.concat([road_links, road_to_transit, self.footpaths])
        access = pd.concat([self.zone_to_road, self.zone_to_transit])
    else:
        footpaths = self.footpaths
        access = self.zone_to_transit

    footpaths = pl.LazyFrame(footpaths[['a', 'b', 'time']])
    access = pl.LazyFrame(access[['a', 'b', 'time']])

    in_vehicle_time = self.links['time'].to_dict()
    waiting_time = (self.links['headway'] / 2).to_dict()
    boarding_time = self.links['boarding_time'].to_dict()

    exprs = {
        'in_vehicle_time': pl.col('link_path').list.eval(pl.element().replace(in_vehicle_time, default=0)).list.sum(),
        'waiting_time': pl.col('boarding_links').list.eval(pl.element().replace(waiting_time, default=0)).list.sum(),
        'boarding_time': pl.col('boarding_links').list.eval(pl.element().replace(boarding_time, default=0)).list.sum(),
    }
    # drop time columns if they exist
    to_drop = ['in_vehicle_time', 'waiting_time', 'boarding_time', 'footpath_time', 'access_time']
    self.pt_los = self.pt_los.drop(to_drop, strict=False)
    los = self.pt_los.lazy()
    # compute  new times columns
    los = los.with_columns(**exprs)
    # compute footpath_time and access_time

    footpaths_time = _compute_footpaths_time(los, footpaths)
    los = los.join(footpaths_time, on='index', maintain_order='left', how='left')

    # compute ntlegs time as access_time = access_time + eggress_time

    access_time = _compute_access_time(los, access)
    los = los.join(access_time, on='index', maintain_order='left', how='left')

    self.pt_los = los.collect(engine='streaming')


def _compute_footpaths_time(los: pl.LazyFrame, footpaths_time: pl.LazyFrame) -> pl.LazyFrame:
    # los = pdf with ['index', 'footpaths'] (list of list)
    # footpaths_time = pdf with ['a','b','time']
    # return a pdf ['index', 'footpath_time']
    _los = los.select(['index', 'footpaths'])
    _los = _los.explode('footpaths')
    _los = _los.with_columns(a=pl.col('footpaths').list.first(), b=pl.col('footpaths').list.last())
    _los = _los.join(footpaths_time, on=['a', 'b'])
    _los = _los.group_by('index').agg(pl.col('time').sum().alias('footpath_time'))
    return _los.select(['index', 'footpath_time'])


def _compute_access_time(los, access_time):
    # los = pdf with ['index', 'access','eggress'] (simple list)
    # access = pdf with ['a','b','time']
    # return a pdf ['index', 'access_time'] where access_time is access + egress time

    _los = los.select(['index', 'access', 'eggress'])
    # split access to a, b columns and join the access_time on a,b
    _los = _los.with_columns(a=pl.col('access').list.first(), b=pl.col('access').list.last())
    _los = _los.join(access_time, on=['a', 'b']).drop(['a', 'b']).rename({'time': 'access_time'})
    # split eggress to a,b columns and join the egress_time on a,b
    _los = _los.with_columns(a=pl.col('eggress').list.first(), b=pl.col('eggress').list.last())
    _los = _los.join(access_time, on=['a', 'b']).drop(['a', 'b']).rename({'time': 'eggress_time'})
    # access_time = access + eggress
    _los = _los.with_columns((pl.col('access_time') + pl.col('eggress_time')).alias('access_time'))
    return _los.select(['index', 'access_time'])
    # add total access_time to the los
