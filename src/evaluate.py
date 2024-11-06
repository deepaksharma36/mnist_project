
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.model import SimpleNN
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load parameters from params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the test dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),
])

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            logger.debug(f"Batch {batch_idx + 1}/{len(test_loader)} - Correct predictions: {correct}/{total}")

    accuracy = correct / total * 100
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

def main():
    # Load the model
    logger.info("Loading model...")
    model = SimpleNN()
    model.load_state_dict(torch.load('src/models/model.pth'))  # Load the trained weights
    model.to(device)

    logger.info("Loading test dataset...")
    # Load the test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.ImageFolder("data/mnist/processed/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=True)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
