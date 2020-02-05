@echo off
cd "%~dp0"
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