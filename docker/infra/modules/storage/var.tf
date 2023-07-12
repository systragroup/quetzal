
variable "bucket_name" {
    description = "Name of the S3_bucket"
    type        = string
}

variable "acl_value" {
    default     = "private"
}