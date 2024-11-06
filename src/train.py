# src/train.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision import datasets, transforms
from models.model import SimpleNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_dir, model_path):
    logger.info("Starting training process.")
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, drop_last=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    logger.info(f"Training parameters: epochs={params['epochs']}, batch_size={params['batch_size']}, learning_rate={params['learning_rate']}")

    for epoch in range(params["epochs"]):
        epoch_loss = 0
        for batch_idx , (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item()}")
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{params['epochs']} - Average Loss: {avg_loss}")


    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")

if __name__ == "__main__":
    train("data/mnist/processed/train", "src/models/model.pth")
