resource "aws_ecr_repository" "repository" {
  name                 = var.repo_name
  image_tag_mutability = "MUTABLE"
    tags               = var.tags


  image_scanning_configuration {
    scan_on_push        = var.scan
  }
  encryption_configuration{
      encryption_type   = var.encryption_type

  }

  #https://stackoverflow.com/questions/69907325/terraform-aws-lambda-function-requires-docker-image-in-ecr/74395215#74395215
  # create a dummy docker image. we need it to create a lambda function
  
}

resource "null_resource" "docker_packaging" {
	
	  provisioner "local-exec" {
    # This is a 1-time execution to put a dummy image into the ECR repo, so 
    #    terraform provisioning works on the lambda function. Otherwise there is
    #    a chicken-egg scenario where the lambda can't be provisioned because no
    #    image exists in the ECR
    command     = <<EOF
      docker login ${data.aws_ecr_authorization_token.token.proxy_endpoint} -u AWS -p ${data.aws_ecr_authorization_token.token.password}
      docker pull alpine
      docker tag alpine ${aws_ecr_repository.repository.repository_url}:DUMMY
      docker push ${aws_ecr_repository.repository.repository_url}:DUMMY
      EOF
  }

	  depends_on = [
	    aws_ecr_repository.repository,
	  ]
}