if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

@echo off
cd "%~dp0"

SET /P env_name=enter an environment name (default = quetzal_env):

call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false
call conda install -y nb_conda_kernels

IF NOT DEFINED env_name SET "env_name=quetzal_env"

if not "X%CONDA_DEFAULT_ENV%" == "X%env_name%" (
  conda info -e | findstr %env_name% > NUL
  if errorlevel 1 (
    call conda create -n %env_name% -y python=3.8
    timeout 3 /nobreak > NUL
  )
  call activate %env_name%
)

@echo on
echo Installing...

call python -m pip install wheels-cp38-win_amd64/GDAL-3.3.2-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/pyproj-3.2.1-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/Fiona-1.8.20-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/Shapely-1.7.1-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/geopandas-0.10.2-py2.py3-none-any.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install wheels-cp38-win_amd64/rasterio-1.2.10-cp38-cp38-win_amd64.whl --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
call python -m pip install -e . --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_pip.txt 
