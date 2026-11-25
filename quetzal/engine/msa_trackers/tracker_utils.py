from typing import NamedTuple
from scipy.sparse import csr_matrix
import numpy as np
import numpy.typing as npt
import pandas as pd


class TrackedVolume(NamedTuple):
    iteration: int
    seg: str
    volumes: csr_matrix


class TrackedWeight(NamedTuple):
    iteration: int
    phi: float
    beta: list[float]


def sparse_mat_to_df(mat: csr_matrix, row_labels: list, col_labels: list) -> pd.DataFrame:
    return pd.DataFrame(mat.toarray(), index=row_labels, columns=col_labels)


def apply_biconjugated_frank_wolfe(mat_datas: list[TrackedVolume], weights) -> dict[str, csr_matrix]:
    # read info [iteration,segment,file]
    segments = {el.seg for el in mat_datas}
    # read weights [phi,beta]
    phi_dict = {w.iteration: w.phi for w in weights}
    beta_dict = {w.iteration: w.beta for w in weights}
    res: dict[str, csr_matrix] = {}
    for seg in segments:
        filtered = [mat for mat in mat_datas if mat.seg == seg]
        res[seg] = apply_biconjugated_frank_wolfe_per_segment(filtered, phi_dict, beta_dict)
    return res


def apply_biconjugated_frank_wolfe_per_segment(
    mat_datas: list[TrackedVolume], phi_dict: dict, beta_dict: dict[int, list]
) -> csr_matrix:
    flow = csr_matrix(0)
    sk_1 = csr_matrix(0)
    sk_2 = csr_matrix(0)
    for mat_data in mat_datas:
        it = mat_data.iteration
        phi = phi_dict[it]
        beta = beta_dict[it]
        aux_flow = mat_data.volumes
        if it > 2:
            aux_flow = beta[0] * aux_flow + beta[1] * sk_1 + beta[2] * sk_2
        sk_2 = sk_1.copy() if sk_1 is not None else None
        sk_1 = aux_flow.copy()

        if it == 0:
            flow = aux_flow.copy()
        else:
            flow = (1 - phi) * flow + phi * aux_flow

    return flow


def tracked_od_to_matrix(
    od_is_used: list[list[int]] | npt.NDArray[np.bool_], odv: np.ndarray, origins: np.ndarray, sparse_od_dict
) -> csr_matrix:
    index_zone = {i: k for i, k in enumerate(origins)}
    rows, cols, data = [], [], []
    for i, is_used in enumerate(od_is_used):
        filtered = odv[is_used]
        origins_index = filtered[:, 0]
        sparse_destinations = filtered[:, 1].astype(np.int32)
        volumes = filtered[:, 2]
        sparse_origins = [*map(index_zone.get, origins_index)]
        od = [*zip(sparse_origins, sparse_destinations)]
        index = [*map(sparse_od_dict.get, od)]
        rows.extend(index)
        cols.extend([i] * len(index))
        data.extend(volumes)

    return csr_matrix((data, (rows, cols)), shape=(len(sparse_od_dict), len(od_is_used)))
