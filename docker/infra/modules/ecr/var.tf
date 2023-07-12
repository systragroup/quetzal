variable "repo_name" {
    description = "Name of the ECR repo"
    type        = string
}

variable "encryption_type" {
    description = "The encryption type to use for the repository. Valid values are AES256 or KMS"
    default     = "AES256"
    type        = string

}

variable "scan" {
    description = "scan docker on push"
    default     = false
    type        = bool
}
