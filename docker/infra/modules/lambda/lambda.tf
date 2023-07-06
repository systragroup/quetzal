
# 1) create an IAM role for the function
resource "aws_iam_role" "iam_for_lambda" {
  name               = var.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# 2) create Log Group for cloud watch
resource "aws_cloudwatch_log_group" "log_group" {
  name              = "/aws/lambda/${var.function_name}"
  retention_in_days = 14
}

# 3) create lambda CloudWatch Logging policy
resource "aws_iam_policy" "lambda_logging" {
  name        = "lambda_logging_${var.function_name}"
  description = "IAM policy for logging from a lambda"
  policy      = data.aws_iam_policy_document.lambda_logging.json
}

# 4) attach CloudWatch logging policy to the role
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.iam_for_lambda.name
  policy_arn = aws_iam_policy.lambda_logging.arn
}


# 5) create inline policy to access the s3 Bucket
resource "aws_iam_role_policy" "s3_policy" {
  name = "S3PutGetObject_${var.function_name}"
  role = aws_iam_role.iam_for_lambda.name
  policy = data.aws_iam_policy_document.s3_policy.json
}

# 6) create the Lambda function with dummy image from ECR
resource "aws_lambda_function" "test_lambda" {
    # If the file is not in the current working directory you will need to include a
    # path.module in the filename.
    function_name       = var.function_name
    role                = aws_iam_role.iam_for_lambda.arn
    architectures       = ["x86_64"] 
    package_type        = "Image"
    image_uri           = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com/${var.ecr_repo_name}@${data.aws_ecr_image.latest.id}"

    memory_size         = var.memory_size
    timeout             = var.time_limit
    ephemeral_storage {
        size            = var.storage_size
    }
    environment {
        variables = {
            BUCKET_NAME = var.bucket_name
    }
  }


    depends_on = [
            aws_iam_role_policy_attachment.lambda_logs,
            aws_cloudwatch_log_group.log_group,
    ]
}