data "aws_iam_policy" "config" {
  name = "s3_read_quetzal_config"
}

data "aws_iam_policy" "osm" {
  name = "s3_read_put_quetzal-osm"
}

data "aws_iam_policy" "matrixroadcaster" {
  name = "s3_read_put_matrixroadcaster"
}

# Role trusted policy
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Federated"
      identifiers = ["cognito-identity.amazonaws.com"]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition{
      test = "ForAnyValue:StringEquals"
      variable = "cognito-identity.amazonaws.com:aud"
      values = [var.cognito_identity_pool_id]
    }

    condition{
      test = "ForAnyValue:StringLike"
      variable = "cognito-identity.amazonaws.com:amr"
      values = ["authenticated"]
    }
  }
}


# policy to read and write on the s3 bucket
data "aws_iam_policy_document" "user_s3_policy" {
    version = "2012-10-17"
	statement	{
        effect= "Allow"
        actions= ["s3:ListBucket"]
        resources= ["arn:aws:s3:::${var.bucket_name}"]
    }
	statement	{
        effect= "Allow"
        actions= [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject"
        ]
        resources= ["arn:aws:s3:::${var.bucket_name}/*"]
    }
	statement	{
        effect= "Deny"
        actions= [
            "s3:PutObject",
            "s3:DeleteObject"
        ]
        resources= ["arn:aws:s3:::${var.bucket_name}/base/*"]
    }
    
}

