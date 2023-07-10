
data "aws_caller_identity" "current" {}
data "aws_region" "current" {} # data.aws_region.current.name
data "aws_ecr_image" "latest" {
  repository_name   = var.ecr_repo_name
  most_recent       = true
}

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

# policy to log on cloudwatch. using the just created log group arn
data "aws_iam_policy_document" "lambda_logging" {
    version = "2012-10-17"
    statement   {
        effect = "Allow"
        actions = ["logs:CreateLogGroup"]
        resources = [
            "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
            ]
    }
    statement {
        effect = "Allow"
        actions = [
            "logs:CreateLogStream",
            "logs:PutLogEvents"
        ]
        resources = [
            "${aws_cloudwatch_log_group.log_group.arn}:*"
        ]
    }
}

# policy to read and write on the s3 bucket
data "aws_iam_policy_document" "s3_policy" {
    version = "2012-10-17"
	statement	{
        effect= "Allow"
        actions= [
            "s3:ListAllMyBuckets",
            "s3:GetBucketLocation"
        ]
        resources= ["*"]
    }
	statement	{
        effect= "Allow"
        actions= ["s3:ListBucket"]
        resources= ["arn:aws:s3:::${var.function_name}"]
    }
	statement	{
        effect= "Allow"
        actions= [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject"
        ]
        resources= ["arn:aws:s3:::${var.function_name}/*"]
    }
	
    
}

