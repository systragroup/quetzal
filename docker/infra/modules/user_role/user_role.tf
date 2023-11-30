# 1) create an IAM role for auth user with Trusted Entities
resource "aws_iam_role" "iam_for_user" {
  name               = var.user_role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# 2) create S3 policy for the bucket
resource "aws_iam_policy" "user_s3_policy" {
  name        = var.s3_policy_name
  description = "IAM policy to access a model S3 bucket"
  policy      = data.aws_iam_policy_document.user_s3_policy.json
}

# 3) attach s3 policy to the role
resource "aws_iam_role_policy_attachment" "user_s3_policy" {
  role       = aws_iam_role.iam_for_user.name
  policy_arn = aws_iam_policy.user_s3_policy.arn
}

# 4) attach quetzal_config read bucket policy
resource "aws_iam_role_policy_attachment" "user_config_policy" {
  role       = aws_iam_role.iam_for_user.name
  policy_arn = data.aws_iam_policy.config.arn
}
