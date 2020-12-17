
import pandas as pd
from tqdm import tqdm
import importlib
import pickle
import zlib
import uuid

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


def write_hdf_to_buffer(frames, level=4, complevel=None):
    with pickle_protocol(level):
        with pd.HDFStore(
                "quetzal.h5",
                mode="a",
                driver="H5FD_CORE",
                driver_core_backing_store=0,
                complevel=complevel,
                ) as out:
            iterator = tqdm(frames.items())
            for key, df in iterator:
                iterator.desc = key
                out[key] = df
            return out._handle.get_file_image()


def frame_to_zip(frame, filepath, level=4, complevel=None):
    with pickle_protocol(level):
        with pd.HDFStore(
                "quetzal-%s.h5" % str(uuid.uuid4()),
                mode="a",
                driver="H5FD_CORE",
                driver_core_backing_store=0,
                complevel=complevel,
                ) as out:
            out['frame'] = frame
            buffer = out._handle.get_file_image()
        smallbuffer = zlib.compress(buffer)
        with open(filepath, 'wb') as file:
            file.write(smallbuffer)


def zip_to_frame(filepath):
    with open(filepath, 'rb') as file:
        data = file.read()
        bigbyte = zlib.decompress(data)

    with pd.HDFStore(
            "quetzal-%s.h5" % str(uuid.uuid4()),
            mode="r",
            driver="H5FD_CORE",
            driver_core_backing_store=0,
            driver_core_image=bigbyte
        ) as store:
        return store['frame']