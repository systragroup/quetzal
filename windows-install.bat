@echo off
cd "%~dp0"
if not "X%CONDA_DEFAULT_ENV%" == "Xquetzal" (
  conda info -e | findstr quetzal > NUL
  if errorlevel 1 (
    call conda create -n quetzal -y python=3.6
    timeout 3 /nobreak > NUL
  )
  call activate quetzal
)
echo Installing...
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a
@echo on
call %conda_cmd%
pip install -e .
