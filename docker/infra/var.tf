variable "aws_region" {
  description = "Deployment region (e.g.: ca-central-1)."
  type        = string
  default     = "ca-central-1"
}
variable "quetzal_model_name" {
  description = "Name for S3 bucket and lambda function"
  type        = string
  default     = "quetzal-test"
}

locals {
  quetzal_tags = {
    "cost:project" : "quetzal",
    "cost:name" : "${var.quetzal_model_name}"
  }
}

variable "cognito_identity_pool_id" {
  description = "cognito_identity_pool_id for IAM policy"
  type        = string
  default     = "ca-central-1:b6298c0d-1089-4287-8770-4e9803847671"
}

variable "lambda_memory_size" {
  description = "Lambda function ram in mb"
  default     = 4016
  type        = number
}

variable "lambda_time_limit" {
  description = "Lambda function time limit in seconds"
  default     = 300
  type        = number
}

variable "lambda_storage_size" {
  description = "Lambda function ephemeral storage size in mb"
  default     = 4016
  type        = number
}