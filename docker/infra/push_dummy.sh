declare repositoryUrl=$1 && shift


# Connect to AWS ECR
aws_account=$(aws sts get-caller-identity | jq '.Account' | sed 's/"//g')
aws_region=$(aws configure get region)

aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin \
  $aws_account.dkr.ecr.$aws_region.amazonaws.com


docker pull alpine
docker tag alpine "$repositoryUrl:DUMMY"
docker push "$repositoryUrl:DUMMY"