import numpy as np
import numba as nb
import numpy.typing as npt


@nb.njit()
def get_path_3d(predecessors, i, j, s, reverse=False):
    path = []
    k = j
    while k != -9999:
        path.append(k)
        k = predecessors[s, i, k]
    if reverse:
        return path
    else:
        return path[::-1]


@nb.njit(parallel=True)
def compute_path_lengths(od_list, predecessors):
    # return the length of each path in predecessor.
    n = len(od_list)
    lengths = np.zeros(n, dtype=np.int32)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s)
        lengths[i] = len(path)
    return lengths


@nb.njit(parallel=True)
def get_flat_path(od_list, predecessors, offsets, reverse=False) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    total_len = offsets[-1]
    flat_paths = np.zeros(total_len, dtype=np.int32)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s, reverse)
        start = offsets[i]
        for j in range(len(path)):
            flat_paths[start + j] = path[j]
    return flat_paths


@nb.njit(parallel=True)
def get_paths_hash(od_list, predecessors) -> np.ndarray:
    # get all paths as a single flat array.
    n = len(od_list)
    hash_list = np.empty(n, dtype=np.int64)
    for i in nb.prange(n):
        o, d, s = od_list[i]
        path = get_path_3d(predecessors, o, d, s)
        h = 0
        for x in path:
            h ^= hash(x) + 0x9E3779B9 + (h << 6) + (h >> 2)  # boost::hash_combine
        hash_list[i] = h

    return hash_list


@nb.njit()
def sum_jagged_array(flat_paths, offsets):
    row_sums = np.zeros(len(offsets) - 1, dtype=flat_paths.dtype)
    for i in nb.prange(len(offsets) - 1):
        row_sums[i] = flat_paths[offsets[i] : offsets[i + 1]].sum()
    return row_sums


def remap_predecessor(full_pred, full_dict, pruned_pred, pruned_dict):
    pruned_pred = pruned_pred.copy()
    pruned_to_full = {k: full_dict[v] for k, v in pruned_dict.items()}

    # create a lookup table :  prune to full index
    pruned_to_full_map = np.empty(pruned_pred.shape[1], dtype=np.int32)
    full_indexes = np.array(list(pruned_to_full.values()))
    pruned_indexes = np.array(list(pruned_to_full.keys()))
    pruned_to_full_map[pruned_indexes] = full_indexes

    # remap value in pruned_predecessor to the one in full_pred
    mask = pruned_pred != -9999  # need to mask if we want to keep -9999,else it will look at index -9999 in lut
    pruned_pred[mask] = pruned_to_full_map[pruned_pred[mask]]

    # reorder rows to the full predecessor indexing. (and shape it as the full_pred)
    new_predecessor = np.full_like(full_pred, -9999, dtype=np.int32)
    new_predecessor[:, full_indexes] = pruned_pred[:, pruned_indexes]

    # assert full_paths.source_index == pruned_paths.source_index, 'TODO need to remap source too.'

    return new_predecessor


@nb.njit(parallel=True)
def get_mask_offsets(offsets, mask):
    N = len(offsets)
    lengths = np.zeros(N, dtype=np.int32)
    for i in nb.prange(N - 1):
        start = offsets[i]
        end = offsets[i + 1]
        lengths[i + 1] = sum(mask[start:end])

    return np.cumsum(lengths)


def get_flat_path_from_mask(paths: npt.NDArray[np.int_], offsets: npt.NDArray[np.int_], mask: npt.NDArray[np.bool]):
    # filter flat_path with a mask and return it with its new offsets
    new_paths = paths[mask]
    new_offsets = get_mask_offsets(offsets, mask)
    return new_paths, new_offsets


def _get_link_path(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = node_vtype == 3
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_node_path(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = node_vtype == 2
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_boardings(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = (node_vtype == 2) & (to_vtype == 3)
    mask[-1] = False
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_alightings(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = (node_vtype == 2) & (from_vtype == 3)
    mask[-1] = False
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_boarding_links(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = (node_vtype == 3) & (from_vtype != 3)
    mask[-1] = False
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_alighting_links(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = (node_vtype == 3) & (to_vtype != 3)
    mask[-1] = False
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_transfers(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask = (node_vtype == 2) & (from_vtype == 3) & (to_vtype == 3)
    mask[-1] = False
    new_paths, new_offsets = get_flat_path_from_mask(paths, offsets, mask)
    return new_paths, new_offsets


def _get_footpaths(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask_a = (node_vtype == 2) & (to_vtype == 2)
    mask_b = (node_vtype == 2) & (from_vtype == 2)
    mask_a[-1] = False
    mask_b[-1] = False

    new_paths = np.column_stack([paths[mask_a], paths[mask_b]])
    new_offsets = get_mask_offsets(offsets, mask_a)

    return new_paths, new_offsets


def _get_ntlegs(paths, offsets, node_vtype, from_vtype, to_vtype):
    mask_a = ((node_vtype == 1) & (to_vtype == 2)) | ((node_vtype == 2) & (to_vtype == 1))
    mask_b = ((node_vtype == 2) & (from_vtype == 1)) | ((node_vtype == 1) & (from_vtype == 2))

    new_paths = np.column_stack([paths[mask_a], paths[mask_b]])
    new_offsets = get_mask_offsets(offsets, mask_a)

    return new_paths, new_offsets


# vertex_type={1: zones, 2: nodes, 3: links}
def analysis_flat_path(
    paths: npt.NDArray[np.int_],
    offsets: npt.NDArray[np.int_],
    vertex_lut: npt.NDArray[np.int_],
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
    ],
):
    node_vtype = vertex_lut[paths]
    to_vtype = np.roll(node_vtype, -1)
    from_vtype = np.roll(node_vtype, +1)
    output = []
    if 'link_path' in includes:
        new_paths, new_offsets = _get_link_path(paths, offsets, node_vtype, from_vtype=from_vtype, to_vtype=to_vtype)
        output.append(('link_path', new_paths, new_offsets))

    if 'node_path' in includes:
        new_paths, new_offsets = _get_node_path(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('node_path', new_paths, new_offsets))

    if 'boardings' in includes:
        new_paths, new_offsets = _get_boardings(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('boardings', new_paths, new_offsets))

    if 'alightings' in includes:
        new_paths, new_offsets = _get_alightings(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('alightings', new_paths, new_offsets))

    if 'boarding_links' in includes:
        new_paths, new_offsets = _get_boarding_links(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('boarding_links', new_paths, new_offsets))

    if 'alighting_links' in includes:
        new_paths, new_offsets = _get_alighting_links(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('alighting_links', new_paths, new_offsets))

    if 'ntlegs' in includes:
        new_paths, new_offsets = _get_ntlegs(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('ntlegs', new_paths, new_offsets))

    if 'footpaths' in includes:
        new_paths, new_offsets = _get_footpaths(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('footpaths', new_paths, new_offsets))

    if 'transfers' in includes:
        new_paths, new_offsets = _get_transfers(paths, offsets, node_vtype, from_vtype, to_vtype)
        output.append(('transfers', new_paths, new_offsets))

    return output


# @nb.njit()
# def analysis_flat_path(path, offsets, vertex_type):
#     num_od = len(offsets) - 1

#     link_path = []
#     node_path = []
#     boardings = []
#     alightings = []
#     boarding_links = []
#     alighting_links = []
#     ntlegs = []
#     footpaths = []
#     transfers = []

#     node_path_offset = [0]
#     link_path_offset = [0]
#     boardings_offset = [0]
#     alightings_offset = [0]
#     boarding_links_offset = [0]
#     alighting_links_offset = [0]
#     ntlegs_offset = [0]
#     footpaths_offset = [0]
#     transfers_offset = [0]

#     for i in range(num_od):
#         start = offsets[i]
#         end = offsets[i + 1]
#         for j in range(start + 1, end - 1):
#             from_node = path[j - 1]
#             node = path[j]
#             to_node = path[j + 1]

#             from_vtype = vertex_type[from_node]
#             vtype = vertex_type[node]
#             to_vtype = vertex_type[to_node]

#             if vtype == 3:
#                 link_path.append(node)
#                 if to_vtype != 3:
#                     alighting_links.append(node)
#                 if from_vtype != 3:
#                     boarding_links.append(node)
#             elif vtype == 2:
#                 node_path.append(node)
#                 if from_vtype == 3:
#                     alightings.append(node)
#                 elif from_vtype == 1:
#                     ntlegs.append((from_node, node))  # remap to ntlegs
#                 elif from_vtype == 2:
#                     footpaths.append((from_node, node))  # remap to foothpath indexes
#                 if to_vtype == 3:
#                     boardings.append(node)
#                 elif to_vtype == 1:
#                     ntlegs.append((node, to_node))
#                 if (from_vtype == 3) & (to_vtype == 3):
#                     transfers.append(node)

#         # transfers = [n for n in boardings if n in alightings]
#         link_path_offset.append(len(link_path))
#         node_path_offset.append(len(node_path))
#         boardings_offset.append(len(boardings))
#         alightings_offset.append(len(alightings))
#         boarding_links_offset.append(len(boarding_links))
#         alighting_links_offset.append(len(alighting_links))
#         ntlegs_offset.append(len(ntlegs))
#         footpaths_offset.append(len(footpaths))
#         transfers_offset.append(len(transfers))

#     return (
#         ('link_path', np.array(link_path), link_path_offset),
#         ('node_path', np.array(node_path), node_path_offset),
#         ('boardings', np.array(boardings), boardings_offset),
#         ('alightings', np.array(alightings), alightings_offset),
#         ('boarding_links', np.array(boarding_links), boarding_links_offset),
#         ('alighting_links', np.array(alighting_links), alighting_links_offset),
#         ('ntlegs', np.array(ntlegs), ntlegs_offset),
#         ('footpaths', np.array(footpaths), footpaths_offset),
#         ('transfers', np.array(transfers), transfers_offset),
#     )
