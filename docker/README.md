# Prerequisites
## AWS-CLI
You need the **AWS Command Line Interface** and **authorized credentials** to send a dockerized model to the AWS Elestic Container Registery. 

Install AWS CLI using [this guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

After this, configure your profile default profile:

    $ aws configure
    AWS Access Key ID [None]: <YOUR_ACCESS_KEY>
    AWS Secret Access Key [None]: <YOUT_SECRET_KEY>
    Default region name [None]: ca-central-1
    Default output format [None]: json

**NOTE:** If your don't have credentials or necessary permissions. Please contact the AWS admin. 

## DOCKER

Install **Docker** using [this guide](https://docs.docker.com/get-docker/)

# Configuring a model for ECR

Configuring a model for ECR need help from the AWS Admin

## Create ECR Repository (AWS Admin only)

Create a ECR repository with a name similar to the model name. There is no particular settings.

## Configure model

1. Copy files from this template directory to the root of model directory and remove .template extensions

2. Fill the environnement variable file `.env`. `QUETZAL_MODEL_NAME` (should be the same as the the model folder). 

4. Create the requirements.txt for the model. We recommand using [pip chill](https://pypi.org/project/pip-chill/).

5. Fill the `Dockerfile.dockerignore`. Inputs that are provided by quenedi and outputs are not necessary in the image (I.E. files that will be set in the `.quenedi.config.json` later). Note that Docker Build will be run from directory higher than the model. You should add the model folder path to your ignored path (Exemple: inputs -> quetzal_model/inputs) 

3. Push the first image to the ECR Repository using the following command (from the quetzal docker script folder):

    ```./push-image.sh <model_folder> inital```

## Create Lambda Function (AWS Admin only)

Create a Python 3.8 Lambda function from inital image and create an new role for this function and add permission to write on the S3 bucket. Configure function ressources + add `BUCKET_NAME` environnement variable

## Config file

Fill the config file `quenedi.config.json` with corresponding values. 

## Create S3 Bucket (AWS Admin only)

Create an S3 bucket and add CORS policy from another bucket (such as quetzal-paris)

## Step function

Modify the step fonction configuration file `step-functions.json` according to model steps.

## Create Step Function (AWS Admin only)

Copy role policies from quetzal-paris

## Create Cognito Role (AWS Admin only)

Custom trust entity + policy copied from quetzal-paris

## End Configuration
1. Add config to S3 config using script

    ```python update-s3-config.py```

2. Add model files for base scenario to s3

    ```python update-s3-model-files.py```

3. Push first version of the model
    
     ```./update-lambda.sh```

# Update a model on ECR

You need AWS permissions to update a model on ECR. You can ask for those permissions to the AWS Admin.


