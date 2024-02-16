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

## Docker

Install **Docker** using [this guide](https://docs.docker.com/get-docker/)

## Terraform

Install **Terraform** using [this guide](https://developer.hashicorp.com/terraform/downloads)

# INFRA (TERRAFORM)

1. **Create a new .tfvars file** with the name of your model `docker/infra/environement/<your_model_name>.tfvars` 

* replace `<your_model_name>`  with the model name, ex: quetzal-paris `<br>`
* the name must be unique in the AWS region (ca-central-1) (s3 bucket limitation)

```
    quetzal_model_name      = "<your_model_name>"
    lambda_memory_size      = 4016
    lambda_time_limit       = 300
    lambda_storage_size     = 4016
```

*NOTE*: Ressources configuration may differ depending on model requierements. time in secs and memory in mb

2. **Go to the infra folder** `(docker/infra)`

```bash
cd docker/infra
```

3. **Create a new workspace**. Each model share the same architecture and must be in separated workspace\

```bash
terraform init
```

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

5. **Plan your deployment**. This will create a plan of deployment. if it is a new deployment, make sure everything is created and **nothing is destroy** `<br>`
   The plan should read  : `Plan: 16 to add, 0 to change, 0 to destroy.`

```bash
terraform plan -var-file="environments/<your_model_name>.tfvars"
```

or on windows make sure to <b>open docker desktop</b> and run

```bash
terraform plan -var-file="environments/<your_model_name>.tfvars" -var os="windows" 
```

6. **Apply your deployment**. Make sure the plan is the same as in the previous step and press yes  `<br>`
   Again, the plan should read  : `Plan: 17 to add, 0 to change, 0 to destroy.`

```bash
terraform apply -var-file="environments/<your_model_name>.tfvars"
```

or on windows make sure to <b>open docker desktop</b> and run

```bash
terraform apply -var-file="environments/<your_model_name>.tfvars" -var os="windows" 
```

That's it! terraform created:

* **S3 bucket** named <your_model_name> (empty)
* **ECR** repo to store the model docker image (with dummy docker image)
* **Lambda function** with access to the S3 bucket and cloudwatch (running dummy docker image)
* **Step function** to launch the lambda function from the Api Gateway
* **IAM role and policy** to add to the cognito user group (for user to acces the model when authenticated)

# Model (DOCKER)

# Configuring a model for ECR

Quenedi need its inputs in the following folders :

* inputs/pt/links.geojson
* inputs/pt/nodes.geojson
* inputs/road/links.geojson
* inputs/road/nodes.geojson
* inputs/od/od.geojson
* inputs/params.json

Note that those are optional. You can have a model without PT, or without road, or without ODs...

## Deploying Model

1. **Copy files** from this `template directory` to the **root of the model directory**
2. Fill the environnement variable file `.env` .

   * `QUETZAL_MODEL_NAME` should be the same as the the **model folder** name.
   * Everything else should be the same as `<your_model_name>` in terraform
3. Create the `requirements.txt` for the model. We recommand using [pip chill](https://pypi.org/project/pip-chill/).
   * you an also use the one provided in this `template directory`
4. Modify the step fonction configuration file `step-functions.json` according to model steps.
5. Fill the `Dockerfile.dockerignore`. Inputs that are provided by quenedi and outputs are not necessary in the image. Note that Docker Build will be run from directory higher than the model. You should add the model folder path to your ignored path (Exemple: inputs -> QUETZAL_MODEL_NAME/inputs)

   **note**: you need a .git in your model for the docker to work but you can ignore the quetzal .git
6. Go to this quetzal docker script folder `(quetzal/docker/scipts)`
7. Build and push the first image to the ECR Repository using the following command

   ```bash
   ./push-image.sh <model_folder_name> initial
   ```

   Or, in windows, make sure Docker desktop is running and run:

   ```bash
   push-image.bat <model_folder_name> initial
   ```
8. Add Step function workflow (step 4: step-function.json in the root directory of your model)

   ```bash
   python update-function-config.py <model_folder>
   ```
9. Add model files for base scenario to s3 **(first scenario should be base)**

   ```bash
   python update-S3-model-files.py <model_folder> <scenario1> <scenario2>
   ```

Note: this script will copy all files from `<model_folder>/scenarios/<scenario1>/` to S3. <br>
for example. with quetzal_test and a base scenario we would have in quetzal_test: `scenarios/base/inputs/pt/links.geojson` and so on

## Create Cognito User group (Optional) (AWS Admin only)

* Create new Cognito user group in quetzal user pool (Cognito Console -> User pool -> Quetzal -> Groups -> Create Group).
  * Enter a group name.
  * Select the role created by terraform (Cognito_quetzal_pool_`<model-name>`).
* You can then add user to the cognito user group in the AWS web interface

* Update cognito_group_access.json in quetzal-config bucket to add available bucket (model) to group. 
   * ex: `<cognito_user_group>` : [`<model-name>`]
   * note: this is necessary as there are no other way for the front to know which models (buckets) are accessible.

## Add Access to existing Cognito User group (if last step skipped) (AWS Admin only)

* Find the IAM role associate to the Cognito user Group (ex: Cognito_quetzal_pool_`<cognito_user_group>`) under IAM>Roles
* In the Permissions tab. click "Add persmissions" then attach policies.
* Select the appropriated policy create by terraform (s3_read_put_`<model-name>`)
  
* Update cognito_group_access.json in quetzal-config bucket to add available bucket (model) to group. 
   * ex: `<cognito_user_group>` : [`<model-name>`]
   * note: this is necessary as there are no other way for the front to know which models (buckets) are accessible.

## Done !

# Update a model on ECR

You need AWS permissions to update a model on ECR. You can ask for those permissions to the AWS Admin.

``./update-lambda.sh <model_folder_name>``

Or, in windows, make sure Docker desktop is running and run:

``update-lambda.bat <model_folder_name>``

# destroy Terraform workspace (for AWS admin)

`terraform workspace select <your_model_name>`

`terraform destroy`

`terraform workspace delete <your_model_name>`

This will fail for S3 and ECR because they are not empty.
empty S3 bucket and ECR. you may need to remove the policy in the cognito group too

# Test lambda function locally

you can test the lambda function docker locally. but you will need to create a s3 folder with your files (ex: test/), and add your AWS crdential to the .env file.

```
QUETZAL_MODEL_NAME=<model_folder>
AWS_ECR_REPO_NAME=<model-name>
AWS_LAMBDA_FUNCTION_NAME=<model-name>
AWS_BUCKET_NAME=<model-name>
AWS_ACCESS_KEY_ID= <your access key>
AWS_SECRET_ACCESS_KEY= <your secret key>
```

after that. you can run the test script to build and run the docker locally

``./test-lambda.sh <model_folder_name> test ``

here. test is the docker tag.

Finally, run this command in a new terminal with the appropriate values.

```bash
 curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"notebook_path": "notebooks/model/model.ipynb", "scenario_path_S3": "test/", "launcher_arg": {"training_folder": "/tmp","params": {"some_param":"value"}},"metadata": {"user_email": "lambda_test@test.com"}}'
```

# debug a docker container

running an interactive shell to explore the docker container

 ``docker run -it --rm --entrypoint /bin/bash <docker_name>:<tag>``

 by default, you will be in `/var/task` which is where all your files (main.py for instance)
 the command `du -ah --max-depth=1 | sort -n` is usefull to see the size of each dir

# Knowned issue

## terraform destroy

ECR  will not be destroy as it is not empty. We need to empty and then destroy ECR as the last step. last step because Lambda depend on an image tag on ecr. if ECR is empty lambda will fail to destroy.

## jupyter-nbconvert KeyError: 'template_paths'

The entrypoint of the dockerfile convert .ipynb to .py files. For some reason. this will not work if there is no .git in the quetzal_model.

## Lambda.Unknown Task timed out (tqdm)

tqdm doesn't work on lambda when the loop is too long (a priori ~1000 iterations).
gives back a timeout error with no log. 

This could also be a timeout issue. lambda have 5 minutes to completes its task and it took more for example.