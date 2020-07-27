
import pandas as pd
from tqdm import tqdm

def write_hdf_to_buffer(frames):
    with pd.HDFStore(
            "quetzal.h5", mode="a", driver="H5FD_CORE",
            driver_core_backing_store=0
            ) as out:
        iterator = tqdm(frames.items())
        for key, df in iterator:
            iterator.desc = key
            out[key] = df
        return out._handle.get_file_image()
