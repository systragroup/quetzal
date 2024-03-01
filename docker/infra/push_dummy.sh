declare repositoryUrl=$1 && shift
declare AWS_ECR_REPO_NAME=$1 && shift

# Prompt user for a tag
last_tag=$(aws ecr describe-images --repository-name $AWS_ECR_REPO_NAME \
    --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]')

if [  "$last_tag" != "null" ]; then
    echo "Repo not empty. do not push dummy image."
    exit 
fi

# Connect to AWS ECR
aws_account=$(aws sts get-caller-identity | jq '.Account' | sed 's/"//g')
aws_region=$(aws configure get region)

aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin \
  $aws_account.dkr.ecr.$aws_region.amazonaws.com


docker pull alpine
docker tag alpine "$repositoryUrl:DUMMY"
docker push "$repositoryUrl:DUMMY"