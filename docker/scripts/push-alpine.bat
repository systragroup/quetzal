@echo off
setlocal enabledelayedexpansion


REM Connect to ECR
FOR /F "tokens=* USEBACKQ" %%F IN (`aws sts get-caller-identity --query "Account" --output text`) DO (
SET aws_account=%%F
)
FOR /F "tokens=* USEBACKQ" %%F IN (`aws configure get region`) DO (
SET aws_region=%%F
)

aws ecr get-login-password --region %aws_region%  | docker login --username AWS --password-stdin %aws_account%.dkr.ecr.%aws_region%.amazonaws.com

docker pull alpine
docker tag alpine 142023388927.dkr.ecr.ca-central-1.amazonaws.com/quetzal-port-moresby:dummy
docker push 142023388927.dkr.ecr.ca-central-1.amazonaws.com/quetzal-port-moresby:dummy

endlocal