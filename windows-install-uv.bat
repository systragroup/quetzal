@echo off
cd "%~dp0"

set SSL_NO_VERIFY=1

SET /P env_name=enter an environment name (default = quetzal_env):
IF NOT DEFINED env_name SET "env_name=quetzal_env"

echo Installing...

where uv >NUL 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
set "UV_PROJECT_ENVIRONMENT=.venvs\%env_name%"
call uv sync
call uv run python -m ipykernel install --user --name=%env_name%

echo Done!
@pause