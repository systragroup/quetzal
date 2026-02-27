import numpy as np
import pandas as pd
from typing import List, Dict, Union
from quetzal.engine.fast_utils import (
    compute_offsets,
    remap_int_array,
    dict_to_lut,
    get_path,
    compute_path_lengths,
    get_flat_path,
)
from quetzal.engine.lazy_path_utils import get_paths_hash, sum_jagged_array


class LazyPaths:
    def __init__(self, predecessors, node_index: Dict[str, int], sources: List[str], reverse: bool = False) -> int:
        self.predecessors = predecessors
        self.node_index = node_index  # {link_0: 0, link_1: 1, ...}
        self.reverse = reverse

        self.index_node = {k: v for v, k in node_index.items()}  # {0: link_0, 1: link1, ...}
        self.source_index = dict(zip(sources, range(len(sources))))  # {zone_0:0, zone_1: 1, ...}
        # when there are multiple pathfinder session. concat predecessor on origin (axis=0)
        # so we need to add an offset, ex: session 2 first origin will be 1200 and not 0 because its predecessor start at column 1200
        self.source_offset = {0: 0}  # {session_id: offset}

    def __repr__(self):
        return f'Lazy paths with {len(self.source_offset.keys())} sessions'

    def __hash__(self):
        return id(self)

    def append(self, predecessor, index_node, session_id: int) -> int:
        new_pred = remap_predecessor(self.predecessors, self.node_index, predecessor, index_node)
        self.source_offset[session_id] = len(self.predecessors)
        self.predecessors = np.concatenate([self.predecessors, new_pred], axis=0)
        return self.predecessors.shape[0]

    def _get_int_path(self, o: int, d: int) -> List[int]:
        return get_path(self.predecessors, o, d, self.reverse)

    def _remap_od_list(self, od_list: np.ndarray) -> np.ndarray[int, int]:
        o_list = np.array([*map(self.source_index.get, od_list[:, 0])], dtype=np.int32)
        d_list = np.array([*map(self.node_index.get, od_list[:, 1])], dtype=np.int32)
        offset = np.array([*map(self.source_offset.get, od_list[:, 2])], dtype=np.int32)
        od_index_list = np.stack([o_list + offset, d_list], axis=1)
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
        path = self._get_int_path(self.source_index[origin] + self.source_offset[session], self.node_index[destination])
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


def remap_predecessor(full_pred, full_dict, pruned_pred, pruned_dict):
    # return pruned_pred in the indexing of full_pred (row-wise)
    pruned_pred = pruned_pred.copy()
    pruned_to_full = {k: full_dict[v] for k, v in pruned_dict.items()}

    # create a lookup table :  prune to full index
    pruned_to_full_map = dict_to_lut(pruned_to_full)
    # remap value in pruned_predecessor to the one in full_pred
    mask = pruned_pred != -9999  # need to mask if we want to keep -9999,else it will look at index -9999 in lut
    pruned_pred[mask] = pruned_to_full_map[pruned_pred[mask]]

    # reorder rows to the full predecessor indexing. (and shape it as the full_pred rows.)
    col = pruned_pred.shape[0]
    rows = full_pred.shape[1]
    new_predecessor = np.full((col, rows), -9999, dtype=np.int32)
    full_indexes = np.array(list(pruned_to_full.values()))
    pruned_indexes = np.array(list(pruned_to_full.keys()))
    new_predecessor[:, full_indexes] = pruned_pred[:, pruned_indexes]

    return new_predecessor


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
        session_id = lazy_session.get(obj)
        lazy_path.append(obj.predecessors, obj.index_node, session_id)
    pt_los = pt_los.drop(columns='path')
    return pt_los, lazy_path
