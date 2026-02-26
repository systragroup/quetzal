import numpy as np
from typing import List, Dict, Union
from quetzal.engine.fast_utils import compute_offsets, remap_int_array, dict_to_lut
from quetzal.engine.lazy_path_utils import (
    get_path_3d,
    remap_predecessor,
    compute_path_lengths,
    get_flat_path,
    get_paths_hash,
    sum_jagged_array,
    analysis_flat_path,
)

import pandas as pd


class LazyPaths:
    def __init__(self, predecessors, node_index: Dict[str, int], sources: List[str], reverse: bool = False) -> int:
        # TODO: append predecessor to stay 2d? this way we can reuse the same functions.
        # just need to change source_index. thi also allow us to have different source per pathfinder Session.
        self.predecessors = np.array([predecessors])
        self.node_index = node_index  # {link_0: 0, link_1: 1, ...}
        self.sources = sources  # list of zones
        self.reverse = reverse

        self.index_node = {k: v for v, k in node_index.items()}  # {0: link_0, 1: link1, ...}
        self.source_index = dict(zip(sources, range(len(sources))))  # {zone_0:0, zone_1: 1, ...}

    def __repr__(self):
        return f'Lazy paths with {self.predecessors.shape[0]} sessions'

    def __hash__(self):
        return id(self)

    def append(self, predecessor, index_node) -> int:

        new_pred = remap_predecessor(self.predecessors[0], self.node_index, predecessor, index_node)
        self.predecessors = np.vstack([self.predecessors, [new_pred]])
        return self.predecessors.shape[0]

    def _get_int_path(self, o: int, d: int, s: int) -> List[int]:
        return get_path_3d(self.predecessors, o, d, s, self.reverse)

    def _remap_od_list(self, od_list: np.ndarray) -> np.ndarray[int, int]:
        o_list = [*map(self.source_index.get, od_list[:, 0])]
        d_list = [*map(self.node_index.get, od_list[:, 1])]
        s_list = np.array(od_list[:, 2], dtype=np.int32)
        od_index_list = np.stack([o_list, d_list, s_list], axis=1)
        return od_index_list

    def _get_jagged_int_path(self, od_index_list: np.ndarray):
        lengths = compute_path_lengths(od_index_list, self.predecessors)
        offsets = compute_offsets(lengths)
        flat_paths = get_flat_path(od_index_list, self.predecessors, offsets, self.reverse)
        return flat_paths, offsets

    #
    # exposed
    #
    def get_path(self, origin: str, destination: str, session: int, remap: bool = True) -> Union[List[str], List[int]]:
        path = self._get_int_path(self.source_index[origin], self.node_index[destination], session)
        if remap:
            path = [*map(self.index_node.get, path)]
        return path

    def get_all_paths(self, pt_los: pd.DataFrame, remap: bool = True, flat=False) -> list[list]:
        od_list = self._remap_od_list(pt_los[['origin', 'destination', 'lazy_session']].values)
        flat_paths, offsets = self._get_jagged_int_path(od_list)
        if remap:
            flat_paths = remap_int_array(flat_paths, self.index_node)
        if flat:
            return flat_paths, offsets
        else:
            return [flat_paths[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]

    def get_all_paths_hash(self, pt_los: pd.DataFrame) -> list[list]:
        od_list = self._remap_od_list(pt_los[['origin', 'destination', 'lazy_session']].values)
        return get_paths_hash(od_list, self.predecessors)

    def get_all_paths_len(self, pt_los: pd.DataFrame) -> list[list]:
        od_list = self._remap_od_list(pt_los[['origin', 'destination', 'lazy_session']].values)
        return compute_path_lengths(od_list, self.predecessors)

    def apply_dict_on_all_paths(self, pt_los: pd.DataFrame, mapper: Dict[str, float]):
        od_list = self._remap_od_list(pt_los[['origin', 'destination', 'lazy_session']].values)
        flat_paths, offsets = self._get_jagged_int_path(od_list)

        mapper = {self.node_index.get(k): v for k, v in mapper.items()}
        flat_paths = remap_int_array(flat_paths, mapper)
        return sum_jagged_array(flat_paths, offsets)


def concat_lazy_los(los_list) -> tuple[pd.DataFrame, LazyPaths]:
    # concat Lazy_paths object in a single one
    # remove lazy_paths of los
    # add a lazy_session.
    pt_los = pd.concat(los_list, ignore_index=True)
    lazy_path_list = pt_los['path'].unique()
    lazy_session = {p: i for i, p in enumerate(lazy_path_list)}
    pt_los['lazy_session'] = pt_los['path'].map(lazy_session)
    lazy_path = lazy_path_list[0]
    for obj in lazy_path_list[1:]:
        lazy_path.append(obj.predecessors[0], obj.index_node)
    pt_los = pt_los.drop(columns='path')
    return pt_los, lazy_path


def get_vertex_type_dict(links, nodes, centroids) -> Dict[str, int]:
    vertex_sets = {1: set(centroids.index), 2: set(nodes.index), 3: set(links.index)}
    vertex_type = {}
    for vtype, vset in vertex_sets.items():
        for v in vset:
            vertex_type[v] = vtype
    return vertex_type


def lazy_analysis_pt_los(
    lazy_path: LazyPaths,
    pt_los: pd.DataFrame,
    links,
    nodes,
    centroids,
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

    vertex_type = get_vertex_type_dict(links, nodes, centroids)
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


def lazy_analysis_pt_time(lazy_path: LazyPaths, pt_los: pd.DataFrame, links, nodes, centroids, access, footpaths):
    #
    # create a lookup table for the vertex type
    vertex_type = get_vertex_type_dict(links, nodes, centroids)
    index_node = lazy_path.index_node
    node_index = lazy_path.node_index
    vertex_type = {k: np.int16(vertex_type[v]) for k, v in index_node.items()}
    vertex_lut = dict_to_lut(vertex_type)

    flat_paths, offsets = lazy_path.get_all_paths(pt_los, remap=False, flat=True)

    includes = ['link_path', 'boarding_links', 'ntlegs', 'footpaths']

    res = analysis_flat_path(flat_paths, offsets, vertex_lut, includes)

    for name, _path, _offset in res:
        if name == 'link_path':
            d = links['time'].to_dict()
            d = {node_index.get(k): v for k, v in d.items()}
            result = remap_int_array(_path, d)
            pt_los['in_vehicle_time'] = [result[_offset[i] : _offset[i + 1]].sum() for i in range(len(_offset) - 1)]
    return pt_los
