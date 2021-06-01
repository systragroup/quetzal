import pandas as pd
from syspy.spatial import spatial
from syspy.spatial.geometries import b_crosses_a_to_the_left


def direct(row):
    direct = b_crosses_a_to_the_left(
        row['geometry_screen'],
        row['geometry_link']
    )
    return direct


def intersection_flows(screens, links, flow_column, **nearest_geometry_kwargs):
    screens = screens[['geometry']].copy()
    links = links[['geometry', flow_column]].copy()

    cross = spatial.nearest_geometry(screens, links, **nearest_geometry_kwargs)
    cross = cross[cross['actual_distance'] == 0].copy()

    cross = pd.merge(cross, screens, left_on='ix_one', right_index=True)
    cross = pd.merge(cross, links, left_on='ix_many',
                     right_index=True, suffixes=['_screen', '_link'])

    cross['direct'] = cross.apply(direct, axis=1)
    cross = cross.loc[cross['direct'] is True]
    cross.rename(columns={'ix_one': 'screen', 'ix_many': 'link'}, inplace=True)
    return cross[['screen', 'link', flow_column]]


def intersection_flow(screens, links, flow_column, **kwargs):
    flows = intersection_flows(screens, links, flow_column, **kwargs)
    flow = pd.merge(
        flows.groupby('screen')[[flow_column]].sum(),
        screens[[flow_column]],
        left_index=True,
        right_index=True,
        suffixes=['_link', '_screen']
    )
    flow['screen'] = flow.index
    return flow
