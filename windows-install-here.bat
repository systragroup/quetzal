@echo off
cd "%~dp0"

IF NOT EXIST requirements_win.txt (
  findstr /V "ipykernel pytables bitarray geopandas shapely contextily" requirements.txt > requirements_win.txt
)

call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false
call conda install -y nb_conda_kernels

@echo on
echo Installing...
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a
call %conda_cmd%
call pip install -e . -r requirements_win.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
call conda install -y -c conda-forge rtree=0.9.3
