import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        
        # Initial convolutional layer conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # Output: 28x28x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer with 2x2 kernel

        # Parallel convolutional layers conv2_1 and conv2_2
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: 14x14x64
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: 14x14x64
        
        # Convolutional layer to match the number of channels for the skip connection
        self.conv1_skip = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)  # Output: 14x14x64
        
        # Sequential convolutional layers conv3_1 and conv3_2
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)  # Output: 7x7x512
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)  # Output: 7x7x512

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 7 * 7, 1000)  # Input from flattened conv3_2
        self.fc2 = nn.Linear(1000, 500)
        self.output = nn.Linear(500, 10)  # Output layer with 10 classes

    def forward(self, x):
        # Forward through conv1 and pooling
        x = F.relu(self.conv1(x))  # Output: 28x28x32
        x = self.pool(x)  # Output: 14x14x32
        
        # Forward through parallel layers conv2_1 and conv2_2
        x2_1 = F.relu(self.conv2_1(x))  # Output: 14x14x64
        x2_2 = F.relu(self.conv2_2(x))  # Output: 14x14x64

        # Connection from conv1 to conv2_2
        # Skip connection from conv1 to conv2_2
        x_skip = F.relu(self.conv1_skip(x))  # Transform conv1 output to match conv2_2 channels
        x2_2 = x2_2 + x_skip  # Element-wise addition (skip connection from conv1 to conv2_2)
        
        # Concatenate outputs from conv2_1 and conv2_2
        x = torch.cat((x2_1, x2_2), dim=1)  # Output: 14x14x128
        x = self.pool(x)  # Output: 7x7x128

        # Forward through sequential layers conv3_1 and conv3_2
        x3_1 = F.relu(self.conv3_1(x))  # Output: 7x7x512
        x3_2 = F.relu(self.conv3_2(x))  # Output: 7x7x512
                
        # Combine the outputs of conv3_1 and conv3_2 by concatenation
        x = torch.cat((x3_1, x3_2), dim=1)  

        # Flatten the output from conv3_2 for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Forward through fully connected layers
        x = F.relu(self.fc1(x))  # Output: 1000
        x = F.relu(self.fc2(x))  # Output: 500
        x = self.output(x)  # Output: 10
        
        return x

if __name__ == "__main__":
    # Test the model
    # Initialize and print model to verify architecture
    model = CustomNetwork()
    print(model)

    # Create a dummy input tensor with batch size 1 and 1 channel (28x28)
    dummy_input = torch.randn(1, 3, 28, 28)
    output = model(dummy_input)
    print(output)
