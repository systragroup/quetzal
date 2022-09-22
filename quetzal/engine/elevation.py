import numpy as np
import os
from osgeo import gdal
from pathlib import Path
import rasterio
import scipy
from hashlib import sha1


def _query_raster_nearest(nodes, filepath, band=1):
    """
    Query a raster for values at coordinates in a DataFrame's x/y columns.
    Parameters
    ----------
    nodes : pandas.DataFrame
        DataFrame indexed by node ID and with two columns: x and y
    filepath : string or pathlib.Path
        path to the raster file or VRT to query
    band : int
        which raster band to query
    Returns
    -------
    nodes_values : zip
        zipped node IDs and corresponding raster values
    """
    # must open raster file here: cannot pickle it to pass in multiprocessing
    with rasterio.open(filepath) as raster:
        values = np.array(tuple(raster.sample(nodes.values, band)), dtype=float).squeeze()
        values[values == raster.nodata] = np.nan
        return zip(nodes.index, values)


def _query_raster_interp(nodes, filepath, band=1, method='linear'):

    with rasterio.open(filepath) as src:
        band1 = src.read(band)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    
    lons= np.array(xs)
    lats = np.array(ys)
    f = scipy.interpolate.interp2d(
        lons[0, :],
        lats[:, 0],
        band1,
        kind=method
    )
    
    values = [f(r[0], r[1])[0] for r in nodes.values]
    # values = nodes.apply(lambda r: f(r.x, r.y)[0], 1)

    return list(zip(nodes.index, values))


def query_raster(nodes, filepath, band=1, method='nearest'):
    """
    Query a raster for values at coordinates in a DataFrame's x/y columns.
    Parameters
    ----------
    nodes : pandas.DataFrame
        DataFrame indexed by node ID and with two columns: x and y
    filepath : string or pathlib.Path
        path to the raster file or VRT to query
    band : int
        which raster band to query
    Returns
    -------
    nodes_values : zip
        zipped node IDs and corresponding raster values
    """
    if method=='nearest':
        return _query_raster_nearest(nodes, filepath, band)
    else:
        return _query_raster_interp(nodes, filepath, band, method)


def merge_rasters_virtually(filepath):

    if not isinstance(filepath, (str, Path)):
        filepaths = [str(p) for p in filepath]
        sha = sha1(str(filepaths).encode("utf-8")).hexdigest()
        filepath = os.path.join(os.path.dirname(filepaths[0]), f".osmnx_{sha}.vrt")
        gdal.BuildVRT(filepath, filepaths).FlushCache()
    
    return filepath