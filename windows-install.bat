@echo off
cd "%~dp0"

call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false

SET /P env_name=enter an environment name (default = quetzal_env):
IF NOT DEFINED env_name SET "env_name=quetzal_env"

if not "X%CONDA_DEFAULT_ENV%" == "X%env_name%" (
  conda info -e | findstr %env_name% > NUL
  if errorlevel 1 (
    call conda create -n %env_name% -y python=3.7
    timeout 3 /nobreak > NUL
  )
  call activate %env_name%
)
echo Installing...
call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a

@echo on
call conda install -y ipykernel
call conda install -y pytables
call conda install -y geopandas
call conda install -y -c conda-forge contextily
call %conda_cmd%
call pip install -e . --trusted-host pypi.org --trusted-host files.pythonhosted.org