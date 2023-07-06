
provider "aws" {
    region = var.aws_region
}

module "s3" {
    source = "./modules/storage"
    bucket_name = var.quetzal_model_name       
}

module "ecr" {
    source = "./modules/ecr"
    repo_name = var.quetzal_model_name       
}

module "lambda" {
    source = "./modules/lambda"
    function_name = var.quetzal_model_name
    ecr_repo_name = var.quetzal_model_name  
    bucket_name = var.quetzal_model_name
    role_name = "lambda-${var.quetzal_model_name}-role"
    memory_size = 4016
    time_limit = 5 * 60
    storage_size  = 4016
}
module "step_function" {
    source = "./modules/step_function"
    step_function_name = var.quetzal_model_name       
    step_function_role_name="sfn-${var.quetzal_model_name}-role"
    lambda_function_name = var.quetzal_model_name     
}