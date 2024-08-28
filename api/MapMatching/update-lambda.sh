


declare AWS_ECR_REPO_NAME=quetzal-mapmatching-api
declare AWS_LAMBDA_FUNCTION_NAME=quetzal-mapmatching-api
declare FOLDER_NAME=MapMatching
declare QUETZAL_ROOT=../..


# Prompt user for a tag
last_tag=$(aws ecr describe-images --repository-name $AWS_ECR_REPO_NAME \
    --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]')

echo "Enter a docker TAG (last: $last_tag)":
read TAG

cd $QUETZAL_ROOT
# Build docker image
docker build -f api/$FOLDER_NAME/Dockerfile  -t $AWS_ECR_REPO_NAME:$TAG .

# Connect to AWS ECR
aws_account=$(aws sts get-caller-identity | jq '.Account' | sed 's/"//g')
aws_region=$(aws configure get region)

aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin \
  $aws_account.dkr.ecr.$aws_region.amazonaws.com

#Tag docker
docker tag $AWS_ECR_REPO_NAME:$TAG $aws_account.dkr.ecr.$aws_region.amazonaws.com/$AWS_ECR_REPO_NAME:$TAG

#Push docker to aws
docker push $aws_account.dkr.ecr.$aws_region.amazonaws.com/$AWS_ECR_REPO_NAME:$TAG

#update Lambda
aws lambda update-function-code --region $aws_region --function-name  $AWS_LAMBDA_FUNCTION_NAME \
    --image-uri $aws_account.dkr.ecr.$aws_region.amazonaws.com/$AWS_LAMBDA_FUNCTION_NAME:$TAG > /dev/null

echo "updating lambda function ..."

aws lambda wait function-updated --region $aws_region --function-name $AWS_LAMBDA_FUNCTION_NAME

echo "success"