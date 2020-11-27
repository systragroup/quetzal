
import pandas as pd
from tqdm import tqdm
import importlib
import pickle


class PickleProtocol:
    def __init__(self, level):
        self.previous = pickle.HIGHEST_PROTOCOL
        self.level = level

    def __enter__(self):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.level

    def __exit__(self, *exc):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.previous


def pickle_protocol(level):
    return PickleProtocol(level)


def write_hdf_to_buffer(frames, level=4):
    with pickle_protocol(level):
        with pd.HDFStore(
                "quetzal.h5", mode="a", driver="H5FD_CORE",
                driver_core_backing_store=0
                ) as out:
            iterator = tqdm(frames.items())
            for key, df in iterator:
                iterator.desc = key
                out[key] = df
            return out._handle.get_file_image()
