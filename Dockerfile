# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /mnist_project

# Set the PYTHONPATH
ENV PYTHONPATH="/mnist_project/src"

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for DVC
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container
COPY . .

# Expose the port for FastAPI
EXPOSE 8000

# As of now the data will be downloaded locally and then processed by the dvc pipeline
# in future a copy of the data can be maintained on S3 for pulling directly
# RUN dvc pull

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
