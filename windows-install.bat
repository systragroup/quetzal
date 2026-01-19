@echo on
cd "%~dp0"

call set SSL_NO_VERIFY=1
call conda config --set ssl_verify false

SET /P env_name=enter an environment name (default = quetzal_env):
IF NOT DEFINED env_name SET "env_name=quetzal_env"

if not "X%CONDA_DEFAULT_ENV%" == "X%env_name%" (
  conda info -e | findstr %env_name% > NUL
  if errorlevel 1 (
    call conda create -n %env_name% -y python=3.12
    timeout 3 /nobreak > NUL
  )
  call activate %env_name%
)

@echo on
echo Installing...

call python -m pip install poetry
call poetry install
call python -m ipykernel install --user --name=%env_name%

echo Done!
@pause