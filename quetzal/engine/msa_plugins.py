import pandas as pd
import os
import json
from typing import List, Dict, Union
import numba as nb
from quetzal.engine.pathfinder_utils import get_node_path
from quetzal.io.hdf_io import to_zippedpickle, read_zippedpickle
import tempfile
import shutil


@nb.njit(locals={'predecessors': nb.int32[:, ::1]}, parallel=True)
def assign_tracked_volumes_parallel(odv, predecessors, volumes_list, key_list):
    for i in nb.prange(len(key_list)):
        assign_tracked_volume(odv, predecessors, volumes_list[i], key_list[i])
    return volumes_list


@nb.njit(locals={'predecessors': nb.int32[:, ::1]})  # parallel=> not thread safe. do not!
def assign_tracked_volume(odv, predecessors, volumes, track_index):
    # volumes is a numba dict with all the key initialized
    is_tuple = isinstance(track_index, tuple)
    for i in range(len(odv)):
        origin = odv[i, 0]
        destination = odv[i, 1]
        v = odv[i, 2]
        if v > 0:
            path = get_node_path(predecessors, origin, destination)
            if is_tuple:  # not expanded: we have tuple (a,b). else its a sparse index (link directly)
                path = list(zip(path[:-1], path[1:]))
            if track_index in path:
                for key in path:
                    volumes[key] += v
    return volumes


class LinksTracker:
    def __init__(self, track_links_list: List[str] = [], save_folder=None):
        # list of road_link index to track
        # use self.merge() at the end.

        self.track_links_list = track_links_list
        self._save_folder = save_folder
        self.path = save_folder

        self.weights = pd.DataFrame()
        self.info = pd.DataFrame()
        self.tracked_df: Dict[str, pd.DataFrame] = {}

    def init(
        self,
        links_sparse_index: Union[List[int], List[tuple[int, int]]],
        links_to_sparse: Union[Dict[str, int], Dict[str, tuple[int, int]]],
    ):
        self.links_sparse_index = links_sparse_index
        self.sparse_links_list = [*map(links_to_sparse.get, self.track_links_list)]
        # self.reversed_index_dict = {v: k for k, v in index_dict.items()}
        # self.reversed_column_dict = {v: k for k, v in column_dict.items()}
        self.sparse_to_links = {v: k for k, v in links_to_sparse.items()}
        # export data
        if self():
            self.makedir()

    def makedir(self):
        if self._save_folder is None:
            self.path = tempfile.mkdtemp()
            print('Temp dir:', self.path)
        elif not os.path.exists(self._save_folder):  # __call__()
            self.path = self._save_folder
            os.makedirs(self.path, exist_ok=True)

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return len(self.track_links_list) > 0

    def __del__(self):  # when object is deleted delete temp folder
        if self._save_folder is None:
            shutil.rmtree(self.path, ignore_errors=True)  # auto-clean on object deletion

    def init_df(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.links_sparse_index, columns=self.sparse_links_list).fillna(0)

    def assign_volume_on_links(self, ab_volumes, odv, pred, seg):
        volumes = [ab_volumes.copy() for _ in self.sparse_links_list]
        volumes = assign_tracked_volumes_parallel(odv, pred, volumes, self.sparse_links_list)
        self.add_volumes(volumes, seg)

    def add_volumes(self, volumes, seg):
        df = self.init_df()
        for link_index, vols in zip(self.sparse_links_list, volumes):
            df[link_index] = vols
        self.tracked_df[seg] = df

    def add_weights(self, phi, beta, relgap, it):
        new_line = pd.DataFrame([{'iteration': it, 'phi': phi, 'beta': beta, 'relgap': relgap}])
        self.weights = pd.concat([self.weights, new_line], ignore_index=True)

    def add_file_info(self, seg, name, it):
        new_line = pd.DataFrame([{'iteration': it, 'segment': seg, 'file': name}])
        self.info = pd.concat([self.info, new_line], ignore_index=True)

    def save(self, it):
        for seg, df in self.tracked_df.items():
            df = df.rename(columns=self.sparse_to_links)
            df.index = df.index.map(self.sparse_to_links.get)
            df.index.name = 'index'
            name = f'tracked_volume_{seg}_{it}.zippedpickle'
            to_zippedpickle(df, os.path.join(self.path, name))
            self.add_file_info(seg, name, it)

        self.weights.to_csv(os.path.join(self.path, 'weights.csv'), index=False)
        self.info.to_csv(os.path.join(self.path, 'info.csv'), index=False)
        self.tracked_df = {}

    def merge(self) -> Dict[str, pd.DataFrame]:
        # apply frank wolfe for each iteration on each segments
        return _merge(path=self.path)


def _merge(path: str = 'road_pathfinder'):
    # read info [iteration,segment,file]
    info = pd.read_csv(os.path.join(path, 'info.csv'))
    iterations = info['iteration'].sort_values().unique()
    segments = info['segment'].unique()
    file_dict = info.set_index(['iteration', 'segment'])['file'].to_dict()

    # read weights [phi,beta]
    weights = pd.read_csv(os.path.join(path, 'weights.csv'))
    phi_dict = weights.set_index('iteration')['phi'].to_dict()

    weights['beta'] = weights['beta'].apply(json.loads)
    beta_dict = weights.set_index('iteration')['beta'].to_dict()

    def apply_biconjugated_frank_wolfe():
        flow = pd.DataFrame()
        sk_1 = pd.DataFrame()
        sk_2 = pd.DataFrame()
        for it in iterations:
            phi = phi_dict.get(it)
            beta = beta_dict.get(it)
            filename = file_dict.get((it, seg))

            aux_flow = read_zippedpickle(os.path.join(path, filename))
            if it > 2:
                aux_flow = beta[0] * aux_flow + beta[1] * sk_1 + beta[2] * sk_2

            sk_2 = sk_1.copy()
            sk_1 = aux_flow.copy()

            if it == 0:
                flow = aux_flow.copy()
            else:
                flow = (1 - phi) * flow + phi * aux_flow
        return flow

    res = {}
    for seg in segments:
        res[seg] = apply_biconjugated_frank_wolfe()
    return res

    # bfw should work as its  [1,0,0] if fw is use
    # def apply_frank_wolfe():
    #     flow = pd.DataFrame()
    #     for it in iterations:
    #         filename = file_dict.get((it, seg))
    #         aux_flow = read_zippedpickle(os.path.join(path, filename))
    #         phi = phi_dict.get(it)
    #         if it == 0:
    #             flow = aux_flow.copy()
    #         else:
    #             flow = (1 - phi) * flow + phi * aux_flow
    #     return flow
