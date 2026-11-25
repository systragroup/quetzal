from quetzal.engine.msa_trackers.tracker import Tracker
import numpy as np
import pandas as pd
from quetzal.engine.fast_utils import get_jagged_path
from quetzal.engine.msa_trackers.tracker_utils import TrackedWeight

# need to name the class the same as the namedTuple name for pickle.


class PathTracker(Tracker):
    index_to_sparse: dict[str, int]
    sparse_to_index: dict[int, str]

    def __init__(self):
        self.weights: list[TrackedWeight] = []
        self.tracked_df: list[pd.DataFrame] = []

    def init(
        self,
        links_to_sparse: dict[str, int],
        *args,
        **kwargs,
    ):
        self.index_to_sparse = links_to_sparse
        self.sparse_to_index = {v: k for k, v in self.index_to_sparse.items()}

    def __call__(self) -> bool:  # when calling the instance. check if we track links or no.
        return True

    def assign(self, odv, pred, seg, it, *args, **kwargs):
        # just save everything.
        od_list = odv[:, 0:2].astype(np.int32)
        flat_list = get_jagged_path(od_list, pred)
        # get origin and destination from path
        values = flat_list.values.to_numpy()
        offsets = flat_list.offsets.to_numpy()
        origins = values[offsets[:-1]]
        destination = values[offsets[1:] - 1]

        df = pd.DataFrame(
            {
                'iteration': np.int16(it),
                'segment': seg,
                'origin': origins,
                'destination': destination,
                'volume': odv[:, 2],
                'path': pd.Series(flat_list, dtype=pd.ArrowDtype(flat_list.type)),  # save as pyarrow in df
            }
        )

        self.tracked_df.append(df)

    def add_weights(self, phi, beta, it):
        self.weights.append(TrackedWeight(iteration=it, phi=phi, beta=beta))

    def merge(self):
        df = pd.concat([df for df in self.tracked_df])
        df['segment'] = df['segment'].astype('category')
        return df
