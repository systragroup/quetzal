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

# TERRAFORM

1. **Create a new .tfvars file** with the name of your model `environement/<your_model_name>.tfvars` <br>
 * replace `<your_model_name>`  with the model name, ex: quetzal-paris <br>
 * the name must be unique in the AWS region (ca-central-1) (s3 bucket limitation)

```JSON
quetzal_model_name      = "<your_model_name>"
lambda_memory_size      = 4016
lambda_time_limit       = 300
lambda_storage_size     = 4016
```
*NOTE*: Ressources configuration may differ depending on model requierements. time_+limit in secs and memory in mb


2. **Go to the infra folder** `(quetzal/docker/infra)`
```bash
cd infra
```

3. **Create a new workspace**. Each model share the same architecture and must be in separated workspace

```bash
terraform workspace new <your_model_name>
```

4. **Select your workspace and initialize it**. this will sync your local copy with the deployed terraform state

```bash
terraform workspace select <your_model_name>
```
```bash
terraform init
```

5.  **Plan your deployment**. This will create a plan of deployment. if it is a new deployment, make sure everything is created and **nothing is destroy** <br>

```bash
terraform plan -var-file="environments/<your_model_name>.tfvars"
```

6.  **Apply your deployment**. Make sure the plan is the same as in the previous step and press yes  <br>

```bash
terraform apply -var-file="environments/<your_model_name>.tfvars"
```

That's it! terraform created:
* **S3 bucket** named <your_model_name> (empty)
* **ECR** repo to store the model docker image (with dummy docker image)
* **Lambda function** with access to the S3 bucket and cloudwatch (running dummy docker image)
* **Step function** to launch the lambda function from the Api Gateway
* **IAM role and policy** to add to the cognito user group (for user to acces the model when authenticated)


# DOCKER

Install **Docker** using [this guide](https://docs.docker.com/get-docker/)

# Configuring a model for ECR

Configuring a model for ECR need help from the AWS Admin

## Configure model

1. **Copy files** from this `template directory` to the **root of the model directory and remove .template extensions**
2. Fill the environnement variable file `.env` .
    * `QUETZAL_MODEL_NAME` should be the same as the the model folder.
    * Everything else should be the same as `<your_model_name>` in terraform
3. Create the requirements.txt for the model. We recommand using [pip chill](https://pypi.org/project/pip-chill/).
4. Modify the step fonction configuration file `step-functions.json` according to model steps.
5. Fill the `Dockerfile.dockerignore`. Inputs that are provided by quenedi and outputs are not necessary in the image (I.E. files that will be set in the `.quenedi.config.json`). Note that Docker Build will be run from directory higher than the model. You should add the model folder path to your ignored path (Exemple: inputs -> quetzal_model/inputs)
6. Go to the quetzal docker script folder `(quetzal/docker/scipts)`
7. Build and push the first image to the ECR Repository using the following command 

   ``./push-image.sh <model_folder_name> initial``

    Or, in windows:
    
    ``push-image.bat <model_folder_name> initial``




## Create Cognito User group (AWS Admin only)
* Create new Cognito user group in quetzal user pool (Cognito Console -> User pool -> Quetzal -> Groups -> Create Group).
    * Enter a group name.
    * Select the role created by terraform (Cognito_quetzal_pool_<model-name>).
* Update congito_group_access.json in quetzal-config bucket to add available bucket to group.
* you can add the policy and add the bucket (cognito_group_access.json) to existing group too.

## End Configuration



1. Add Step function workflow (step-function.json)

   ``python update-s3-config.py <model_folder>``

2. Add model files for base scenario to s3 (Optional)

   ``python update-s3-model-files.py <model_folder> <scenario1> <scenario2>``

3. Push first version of the model

   ``./update-lambda.sh``

# Update a model on ECR

You need AWS permissions to update a model on ECR. You can ask for those permissions to the AWS Admin.
