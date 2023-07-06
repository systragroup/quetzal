# 1) create an IAM role for the step function with Trusted Entities
resource "aws_iam_role" "iam_for_sfn" {
  name               = var.step_function_role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# 3) create inline policy to exec lambda function
resource "aws_iam_role_policy" "s3_policy" {
  name = "LambdaInvoke-${var.lambda_function_name}"
  role = aws_iam_role.iam_for_sfn.name
  policy = data.aws_iam_policy_document.sfn_lambda_policy.json
}


resource "aws_sfn_state_machine" "sfn_state_machine" {
  name     = var.step_function_name
  role_arn = aws_iam_role.iam_for_sfn.arn

  definition = <<EOF
{
  "Comment": "A Hello World example of the Amazon States Language using an AWS Lambda Function",
  "StartAt": "HelloWorld",
  "States": {
    "HelloWorld": {
      "Type": "Task",
      "Resource": "${data.aws_lambda_function.lambda.arn}",
      "End": true
    }
  }
}
EOF
}