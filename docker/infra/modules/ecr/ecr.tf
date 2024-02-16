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
}

resource "null_resource" "image" {

 #https://stackoverflow.com/questions/69907325/terraform-aws-lambda-function-requires-docker-image-in-ecr/74395215#74395215
  # create a dummy docker image. we need it to create a lambda function
  provisioner "local-exec" {
    # This is a 1-time execution to put a dummy image into the ECR repo, so 
    #    terraform provisioning works on the lambda function. Otherwise there is
    #    a chicken-egg scenario where the lambda can't be provisioned because no
    #    image exists in the ECR
    when        = create
    command = var.os == "windows"? "cmd /c push_dummy.bat ${aws_ecr_repository.repository.repository_url}" : " bash push_dummy.sh ${aws_ecr_repository.repository.repository_url}"
  }
  

}