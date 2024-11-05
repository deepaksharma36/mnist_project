# src/train.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.model import SimpleNN

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_dir, model_path):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, drop_last=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    for epoch in range(params["epochs"]):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train("data/mnist/processed/train", "src/models/model.pth")
