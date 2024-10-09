variable "user_role_name" {
  description   = "user role Name"
  type          = string
}

variable "admin_role_name" {
  description   = "admin role Name only create one time and policies from each workspace added."
  type          = string
  default       = "Cognito_quetzal_pool_admin"
}

variable "s3_policy_name" {
  description   = "s3 read put policy name to access this model S3 bucket (from the front with auth)"
  type          = string
}

variable "bucket_name" {
  description   = "s3 bucket name to access with this policy"
  type          = string
}

variable "cognito_identity_pool_id" {
  description   = "cognito_identity_pool_id for the policies"
  type          = string
}