provider "aws" {
  region = "us-east-1"
}

# ECR Repository
resource "aws_ecr_repository" "mnist_project" {
  name = "mnist_project"
}

# ECS Cluster
resource "aws_ecs_cluster" "mnist_cluster" {
  name = "mnist_cluster"  # Added the name argument
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution_role" {
  name               = "ecsTaskExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Attach ECS task policies for pulling images and logging
resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  role       = aws_iam_role.ecs_task_execution_role.name
}

# ECS Task Definition
resource "aws_ecs_task_definition" "mnist_task" {
  family                   = "mnist_task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  memory                   = "512"
  cpu                      = "256"

  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([{
    name      = "mnist_container"
    image     = "${aws_ecr_repository.mnist_project.repository_url}:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
    }]
  }])
}

# ECS Service
resource "aws_ecs_service" "mnist_service" {
  name            = "mnist_service"
  cluster         = aws_ecs_cluster.mnist_cluster.id
  task_definition = aws_ecs_task_definition.mnist_task.arn
  desired_count   = 1

  launch_type = "FARGATE"

  network_configuration {
    subnets         = ["subnet-49d73705"]             # Replace with your subnet
    security_groups = ["sg-0ca2e889bc24313bc"]       # Replace with your security group
    assign_public_ip = true
  }
}
# Local-exec Provisioner to Push Docker Image
resource "null_resource" "push_image" {
  depends_on = [aws_ecr_repository.mnist_project]

  provisioner "local-exec" {
    command = <<EOT
      #!/bin/bash

      # Variables
      AWS_ACCOUNT_ID=284101495229 # Replace with your AWS account ID
      REGION=us-east-1
      REPOSITORY_NAME=mnist_project
      IMAGE_NAME=mnist_project-linux-amd64

      # Authenticate Docker to ECR
      aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

      # Tag the existing Docker image
      docker tag $IMAGE_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

      # Push the Docker image to ECR
      docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest
    EOT

    interpreter = ["bash", "-c"]
  }
}