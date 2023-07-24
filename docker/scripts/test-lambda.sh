declare QUETZAL_ROOT=../../..

if [ $# -lt 2 ]
then
  echo "$0 requires 2 argument <model folder> <tag>"
  exit -1;
fi

declare MODEL_FOLDER=$1 && shift
declare TAG=$1 && shift

# Change working directory
cd $QUETZAL_ROOT

# Load model .env
source $MODEL_FOLDER/.env

# Build docker image
docker build --build-arg QUETZAL_MODEL_NAME=$QUETZAL_MODEL_NAME \
  -t $AWS_ECR_REPO_NAME:$TAG \
  -f $MODEL_FOLDER/Dockerfile .


echo 'Open a new teminal and run a command like this for your model'
echo curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"notebook_path": "notebooks/model/model.ipynb", "scenario_path_S3": "test/", "launcher_arg": {"scenario": "test", "training_folder": "/tmp","params": {"some_param":"value"}},"metadata": {"user_email": "lamda_test@test.com"}}'


docker run -p 9000:8080 -e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" -e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"  -e "BUCKET_NAME=$AWS_BUCKET_NAME" -e "AWS_LAMBDA_FUNCTION_MEMORY_SIZE=5000" $AWS_ECR_REPO_NAME:test


