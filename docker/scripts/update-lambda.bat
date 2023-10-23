@echo off
setlocal enabledelayedexpansion

set "QUETZAL_ROOT=..\..\.."

if "%~1"=="" (
  echo "%0 requires 1 argument <model folder>"
  exit /b -1
)

set "MODEL_FOLDER=%~1"
shift

set ENV_FILE=%QUETZAL_ROOT%\%MODEL_FOLDER%\.env
:: Load model .env
:: Loop through the lines in the .env file and set environment variables
for /f "delims=" %%a in (%ENV_FILE%) do (
    set %%a
)
echo %AWS_ECR_REPO_NAME%
:: Prompt user for a tag
for /f %%i in ('aws ecr describe-images --repository-name %AWS_ECR_REPO_NAME% ^
    --query "sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]"') do set "last_tag=%%i"

set /p TAG="Enter a docker TAG (last: !last_tag!): "

:: Push Image and update lambda
call push-image.bat "%MODEL_FOLDER%" "%TAG%"