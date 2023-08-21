
variable "bucket_name" {
    description = "Name of the S3_bucket"
    type        = string
}
variable "tags" {
    description = "Tags"
    type        = map
    default     = {"cost:project"="quetzal"}
}

variable "acl_value" {
    default     = "private"
}