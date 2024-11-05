import os
import subprocess
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from models.model import SimpleNN 
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import time

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1") 

def log_metric(metric_name, value, unit="Count"):
    try:
        cloudwatch.put_metric_data(
            Namespace="FastAPIApp",
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Unit": unit,
                    "Value": value,
                }
            ],
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("CloudWatch logging error:", e)


app = FastAPI()

# Path to the model file
MODEL_PATH = 'src/models/model.pth'

class InputData(BaseModel):
    input: list  # This can be adjusted according to your input data format

def run_dvc_pipeline():
    """Run the DVC pipeline to ensure the model is generated."""
    try:
        # Run the DVC repro command to reproduce the pipeline
        subprocess.run(['dvc', 'repro'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running DVC pipeline: {e}")
        raise RuntimeError("Failed to run DVC pipeline.")

def load_model():
    """Load the model from the specified path."""
    model = SimpleNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image: UploadFile) -> torch.Tensor:
    """Preprocess the uploaded image to the format expected by the model."""
    image_data = Image.open(image.file)  # Convert to grayscale
    image_data = image_data.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image_data) / 255.0  # Normalize to [0, 1]
    image_tensor = torch.tensor(image_array, dtype=torch.float32).view(-1, 28 * 28)  # Flatten
    return image_tensor

# Check if the model exists; if not, run the DVC pipeline
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Running DVC pipeline to generate the model...")
    run_dvc_pipeline()

# Load the model
model = load_model()

@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    # Log metrics to CloudWatch
    log_metric("RequestLatency", latency, unit="Seconds")
    log_metric("RequestCount", 1)
    if response.status_code >= 400:
        log_metric("ErrorCount", 1)

    return response


@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    """Make predictions on the uploaded image file."""
    try:
        input_tensor = preprocess_image(image)  # Preprocess the image
        with torch.no_grad():  # No need to track gradients during inference
            output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        return {'predicted_class': predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
