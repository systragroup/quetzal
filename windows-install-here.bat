@echo off
cd "%~dp0"
echo Installing...
set conda_cmd=
for /f "delims=" %%a in ('python get_conda_install_args.py') do set conda_cmd=%%a
@echo on
call %conda_cmd%
pip install -e .
