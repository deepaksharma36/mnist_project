import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 3, 128)  # Input layer: 28x28 pixels flattened
        self.fc2 = nn.Linear(128, 64)        # Hidden layer
        self.fc3 = nn.Linear(64, 10)         # Output layer: 10 classes for digits 0-9

    def forward(self, x):
        # Flatten the input tensor
        #x = self.flatten(x)
        x = x.view(-1, 3 * 28 * 28)  # Reshape the input to a flat vector
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)          # Output layer (logits)
        return x

if __name__ == "__main__":
    # Test the model
    model = SimpleNN()
    print(model)

    # Create a dummy input tensor with batch size 1 and 1 channel (28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(output)
