# to get lambda function ARN
data "aws_lambda_function" "lambda" {
  function_name = var.lambda_function_name
}

# Role trusted policy
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}


# Lambda invoke policy
data "aws_iam_policy_document" "sfn_lambda_policy" {
    version = "2012-10-17"
    statement   {
        effect = "Allow"
        actions = ["lambda:InvokeFunction"]
        resources = ["${data.aws_lambda_function.lambda.arn}:*"]
    }
    statement   {
        effect = "Allow"
        actions = ["lambda:InvokeFunction"]
        resources = ["${data.aws_lambda_function.lambda.arn}"]
    }
}


