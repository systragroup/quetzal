@echo off
cd "%~dp0"

if not "X%CONDA_DEFAULT_ENV%" == "Xquetzal_env" (
  conda info -e | findstr quetzal_env > NUL
  if errorlevel 1 (
    call conda create -n quetzal_env -y python=3.7
    timeout 3 /nobreak > NUL
  )
  call activate quetzal_env
)

echo Installing...
call set SSL_NO_VERIFY=1
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a

@echo on
call conda install -y ipykernel
call conda install -y pytables
call conda install -y -c conda-forge geopandas
call %conda_cmd%
pip install -e . --trusted-host pypi.org --trusted-host files.pythonhosted.org