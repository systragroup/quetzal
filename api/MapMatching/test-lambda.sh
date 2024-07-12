
declare AWS_ECR_REPO_NAME=quetzal-mapmatching-api
declare AWS_LAMBDA_FUNCTION_NAME=quetzal-mapmatching-api
declare FOLDER_NAME=MapMatching
declare QUETZAL_ROOT=../..
source test.env



cd $QUETZAL_ROOT
# Build docker image
docker build -f api/$FOLDER_NAME/Dockerfile  -t $AWS_ECR_REPO_NAME:test .

echo ready

docker run -p 9000:8080 --env-file 'api/MapMatching/test.env' $AWS_ECR_REPO_NAME:test

