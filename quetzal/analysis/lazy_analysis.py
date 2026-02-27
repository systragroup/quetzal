import numpy as np
import pandas as pd
import numpy.typing as npt
from quetzal.engine.lazy_path import LazyPaths
from quetzal.engine.fast_utils import remap_int_array, dict_to_lut, get_numba_tuple_dict, remap_tuple_array
from quetzal.engine.lazy_path_utils import sum_jagged_array, analysis_flat_path, min_jagged_array


def lazy_analysis_pt_los(
    lazy_path: LazyPaths,
    pt_los: pd.DataFrame,
    vertex_type: dict[str, int],
    includes=[
        'link_path',
        'node_path',
        'boardings',
        'alightings',
        'boarding_links',
        'alighting_links',
        'ntlegs',
        'footpaths',
        'transfers',
        'all_walk',
        'ntransfers',
    ],
):

    # create a lookup table for the vertex type
    index_node = lazy_path.index_node
    vertex_type = {k: np.int16(vertex_type[v]) for k, v in index_node.items()}
    vertex_lut = dict_to_lut(vertex_type)

    flat_paths, offsets = lazy_path.get_all_paths(pt_los, remap=False, flat=True)

    res = analysis_flat_path(flat_paths, offsets, vertex_lut, includes)

    for name, _path, _offset in res:
        _path = remap_int_array(_path, index_node)
        if name in ['ntlegs', 'footpaths']:  # we have a list of tuple for those
            _path = [tuple(el) for el in _path]
        # stack to a list of list for pandas
        pt_los[name] = [_path[_offset[i] : _offset[i + 1]] for i in range(len(_offset) - 1)]
    # other values
    if ('all_walk' in includes) & ('link_path' in includes):
        pt_los['all_walk'] = pt_los['link_path'].apply(lambda p: len(p) == 0)
    if ('ntransfers' in includes) & ('boarding_links' in includes):
        pt_los['ntransfers'] = pt_los['boarding_links'].apply(lambda x: max(len(x) - 1, 0))
    return pt_los


def lazy_analysis_pt_time(lazy_path: LazyPaths, pt_los: pd.DataFrame, vertex_type, links, access, footpaths):
    #
    # create a lookup table for the vertex type
    node_index = lazy_path.node_index
    index_node = lazy_path.index_node
    vertex_type = {k: np.int16(vertex_type[v]) for k, v in index_node.items()}
    vertex_lut = dict_to_lut(vertex_type)

    flat_paths, offsets = lazy_path.get_all_paths(pt_los, remap=False, flat=True)

    includes = ['link_path', 'boarding_links', 'ntlegs', 'footpaths']

    res = analysis_flat_path(flat_paths, offsets, vertex_lut, includes)

    for name, _path, _offset in res:
        if name == 'ntlegs':
            d = access.set_index(['a', 'b'])['time'].to_dict()
            d = get_numba_tuple_dict(d, node_index)
            path_of_vals = remap_tuple_array(_path, d)
            pt_los['access_time'] = sum_jagged_array(path_of_vals, _offset)

        if name == 'footpaths':
            d = footpaths.set_index(['a', 'b'])['time'].to_dict()
            d = get_numba_tuple_dict(d, node_index)
            path_of_vals = remap_tuple_array(_path, d)
            pt_los['footpath_time'] = sum_jagged_array(path_of_vals, _offset)

        if name == 'link_path':
            d = links['time'].to_dict()
            d = {node_index.get(k): v for k, v in d.items()}
            path_of_vals = remap_int_array(_path, d)
            pt_los['in_vehicle_time'] = sum_jagged_array(path_of_vals, _offset)

        if name == 'boarding_links':
            d = links['headway'].to_dict()
            d = {node_index.get(k): v / 2 for k, v in d.items()}  # ehadway /2
            path_of_vals = remap_int_array(_path, d)
            pt_los['waiting_time'] = sum_jagged_array(path_of_vals, _offset)

            d = links['boarding_time'].to_dict()
            d = {node_index.get(k): v for k, v in d.items()}
            path_of_vals = remap_int_array(_path, d)
            pt_los['boarding_time'] = sum_jagged_array(path_of_vals, _offset)

    cols = ['access_time', 'footpath_time', 'in_vehicle_time', 'waiting_time', 'boarding_time']
    pt_los['time'] = pt_los[cols].T.sum()
    return pt_los


def lazy_analysis_pt_route_type(
    lazy_path: LazyPaths, pt_los: pd.DataFrame, route_type_dict: dict[str, str], hierarchy: list[str]
) -> npt.NDArray[np.str_]:
    # get link_path.
    # remap hierarchy to int (ex: [car, subway, bus, walk] => [0, 1, 2, 3])
    # for each OD. get the minimum route type (so the highest hierarchy) or last one.

    node_index = lazy_path.node_index
    # transform hierarchy to int [car, subway, bus, walk] => [0, 1, 2, 3]
    hierarchy_dict = {mode: np.int16(i) for i, mode in enumerate(hierarchy)}  # {mode: int}
    reverse_hierarchy_dict = {v: k for k, v in hierarchy_dict.items()}
    default = hierarchy_dict.get(hierarchy[-1])  # last hierarchy as in
    # route_type_dict is now link_index (as int) : route_type (as int)
    route_type_dict = {node_index.get(k): hierarchy_dict.get(v, default) for k, v in route_type_dict.items()}

    # 1) get link_path. we need vertex_type, but we only care about links,
    vertex_type = {k: np.int16(3) for k in route_type_dict.keys()}
    vertex_lut = dict_to_lut(vertex_type, max_key=max(node_index.values()) + 1)

    flat_paths, offsets = lazy_path.get_all_paths(pt_los, remap=False, flat=True)
    res = analysis_flat_path(flat_paths, offsets, vertex_lut, ['link_path'])
    _, link_paths, offsets = res[0]
    # 2) get each link route_types
    route_types_paths = remap_int_array(link_paths, route_type_dict)  # here we have list of route_type. as int
    # 3) return the higest route_type for each OD
    type_paths = min_jagged_array(route_types_paths, offsets, default)  # return highest hierarchy or hierarchy[-1]
    type_paths = remap_int_array(type_paths, reverse_hierarchy_dict)  # remap back to string (route_types)
    return type_paths
