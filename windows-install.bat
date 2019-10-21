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
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a
@echo on
call %conda_cmd%
pip install -e .
