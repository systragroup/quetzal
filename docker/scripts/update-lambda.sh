#!/bin/bash

# *******************************************************
# Ce script permet de mettre a jour le docker sur
# le repo ECR d'AWS et la fonction lambda.
#*******************************************************
declare QUETZAL_ROOT=../../..

if [ $# -lt 1 ]
then
  echo "$0 requires 1 argument <model folder>"
  exit -1;
fi

declare MODEL_FOLDER=$1 && shift

# Load model .env
source $QUETZAL_ROOT/$MODEL_FOLDER/.env

# Prompt user for a tag
last_tag=$(aws ecr describe-images --repository-name $AWS_ECR_REPO_NAME \
    --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]')

echo "Enter a docker TAG (last: $last_tag)":
read TAG

# Push Image and update lambda
./push-image.sh $MODEL_FOLDER $TAG






