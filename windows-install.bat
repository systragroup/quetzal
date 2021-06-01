@echo off
cd "%~dp0"

IF NOT EXIST requirements_win.txt (
  findstr /V "ipykernel pytables bitarray geopandas shapely contextily" requirements.txt > requirements_win.txt
)

call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false
call conda install -y nb_conda_kernels

SET /P env_name=enter an environment name (default = quetzal_env):
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
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a
call %conda_cmd%
call python -m pip install -e . -r requirements_win.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
call conda install -y -c conda-forge rtree=0.9.3
