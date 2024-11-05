import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# this script is downloading data from the a web location.
# we will use dvc add to version the raw data so if this changes then we can switch to older versions if needed
def download_mnist_data(data_dir="data/mnist"):
    """
    Downloads the MNIST dataset and saves it to the specified directory.
    
    Parameters:
    - data_dir: Directory where the MNIST dataset will be stored.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the transformation for the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the dataset
    ])

    # Download the training and test datasets
    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    print("MNIST dataset downloaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

if __name__ == "__main__":
    download_mnist_data()
