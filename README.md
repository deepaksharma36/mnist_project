# MNIST Classification Project

This project trains a neural network to classify MNIST digits and deploys it with FastAPI for live inference requests.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/deepaksharma36/mnist_project.git
   cd mnist_project
   ```

2. Build the Docker Image:
   ```
   docker build -t mnist_project .
   ```
3. Run the docker image and lounch the Inference Service on Local Host
    ```
    docker run -p 8000:8000 mnist_project
    ```
4. Run inference on the service through FAST-API(on localhost)
    ```
    # if trained model is unavailable then dvc pipeline will be invoked during the setup 
    curl -X POST "http://127.0.0.1:8000/predict" -F "image=@sample.png"
    ```


5. Deploying the service on the Cloud using Amazon Eelestic Container Service

## Prerequisites

- AWS account with IAM permissions to create resources (ECR, ECS, IAM, etc.)
- AWS CLI installed and configured
- Docker installed and running
- Terraform installed

## Project Overview

The Terraform configuration performs the following:

1. Creates an Amazon ECR repository for Docker image storage.
2. Deploys an ECS Cluster with Fargate launch type.
3. Defines an IAM role for ECS Task execution.
4. Defines an ECS Task that uses the Docker image from ECR.
5. Deploys an ECS Service to run the task on Fargate.
6. Uses a `local-exec` provisioner to push the Docker image from your local environment to ECR.

## Setup Instructions
create docker image
```
docker build --platform linux/amd64,linux/arm64 -t <your-image-name> .
```

Configure AWS CLI
```
#Ensure the AWS CLI is installed and configured with appropriate credentials:
aws configure
```
Modify teraform variables
```
edit main.tf by updating AWS account ID, subnet ID, security group

```
Init and deploy with Terraform
```
terraform init
terraform apply
```

Access the cloud service (The service is live on the below ip address)
```
Once deployed, the ECS service will run on a container with a public IP assigned.
curl -X POST "http://3.88.138.16:8000/predict" -F "image=@sample.png"

```
## run dvc pipline through cli
```
dvc repro

```

