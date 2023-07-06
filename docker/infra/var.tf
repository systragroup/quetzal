variable "aws_region" {
  description = "Deployment region (e.g.: ca-central-1)."
  type        = string
  default     = "ca-central-1"
}
variable "quetzal_model_name" {
  description = "Name for S3 bucket and lambda function"
  type        = string
  default     = "quetzal-tf-test"
}

