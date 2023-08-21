variable "step_function_name" {
  description   = "step function  name"
  type          = string
}

variable "step_function_role_name" {
  description   = "step function role name"
  type          = string
}

variable "lambda_function_name" {
  description   = "lambda function  name"
  type          = string
}

variable "tags" {
    description = "Tags"
    type        = map
    default     = {"cost:project"="quetzal"}
}

variable "state_machine_definition" {
  description = "New state machine definition"
  type        = string
  default     = <<EOF
      {
        "Comment": "A state machine definition",
        "StartAt": "FirstState",
        "States": {
          "FirstState": {
            "Type": "Pass",
            "Result": "Hello, Step Functions!",
            "End": true
          }
        }
      }
    EOF
}