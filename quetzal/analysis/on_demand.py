# -*- coding: utf-8 -*-

def node_path(path):
    return {
        int(p.split('node_')[1])
        for p in path
        if type(p) is str
        and 'node' in p
    }


def select(row, link_checkpoints, node_checkpoints):
    links, nodes = link_checkpoints, node_checkpoints
    bool_nodes = row['node_path'] >= nodes if nodes else True
    bool_links = row['pt_path'] >= links if links else True
    return bool_nodes & bool_links


def checkpoint_demand(
    od_stack,
    link_checkpoints,
    node_checkpoints,
    volume_columns=['volume'],
    suffixe=''
):
    df = od_stack.copy()
    df['select_bool'] = df.apply(
        lambda r: select(r, link_checkpoints, node_checkpoints),
        axis=1
    )

    for column in volume_columns:
        df[column + suffixe] = df[column] * df['select_bool']

    return df

