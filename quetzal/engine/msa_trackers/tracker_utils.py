from typing import NamedTuple
from scipy.sparse import csr_matrix
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


def apply_biconjugated_frank_wolfe(
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
