echo Installing...

call python -m pip install wheels-cp38-win_amd64/GDAL-3.3.2-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/pyproj-3.2.1-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/Fiona-1.8.20-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/Shapely-1.7.1-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/geopandas-0.10.2-py2.py3-none-any.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/rasterio-1.2.10-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install -e . --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_pip.txt 
