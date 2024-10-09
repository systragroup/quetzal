set "repositoryUrl=%~1"


REM Connect to ECR
FOR /F "tokens=* USEBACKQ" %%F IN (`aws sts get-caller-identity --query "Account" --output text`) DO (
SET aws_account=%%F
)
FOR /F "tokens=* USEBACKQ" %%F IN (`aws configure get region`) DO (
SET aws_region=%%F
)


aws ecr get-login-password --region %aws_region%  | docker login --username AWS --password-stdin %aws_account%.dkr.ecr.%aws_region%.amazonaws.com


docker pull alpine
docker tag alpine "%repositoryUrl%:DUMMY"
docker push "%repositoryUrl%:DUMMY"