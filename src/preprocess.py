import os
import gzip
import logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_mnist_images(file_path, size):
    with gzip.open(file_path, 'rb') as f:
        f.read(16)  # Skip the header
        buf = f.read(28 * 28 * size)  # Read the images
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        logging.info(f"Extracted {size} images from {file_path}")
        return data.reshape(size, 28, 28)

def extract_mnist_labels(file_path, size):
    logging.info(f"Extracting labels from {file_path}")
    with gzip.open(file_path, 'rb') as f:
        f.read(8)  # Skip the header
        buf = f.read(size)  # Read the labels
        return np.frombuffer(buf, dtype=np.uint8)

def save_images(data, labels, output_dir):
    logging.info(f"Saving images to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(data)):
        label = labels[i]
        label_dir = os.path.join(output_dir, str(label))

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        image_path = os.path.join(label_dir, f'image_{i}.png')
        image = Image.fromarray((data[i] * 255).astype(np.uint8))
        image.save(image_path)
    logging.info(f"Saved {len(data)} images to {output_dir}")


if __name__ == "__main__":
    raw_data_dir = 'data/mnist/MNIST/raw'
    processed_data_dir_train = 'data/mnist/processed/train'
    processed_data_dir_test = 'data/mnist/processed/test'

    train_images_file = os.path.join(raw_data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_file = os.path.join(raw_data_dir, 'train-labels-idx1-ubyte.gz')

    # Extract and save the images and labels
    train_images = extract_mnist_images(train_images_file, 60000)
    train_labels = extract_mnist_labels(train_labels_file, 60000)

    save_images(train_images, train_labels, processed_data_dir_train)

    test_images_file = os.path.join(raw_data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_file = os.path.join(raw_data_dir, 't10k-labels-idx1-ubyte.gz')

    # Extract and save the images and labels
    test_images = extract_mnist_images(test_images_file, 10000)
    test_labels = extract_mnist_labels(test_labels_file, 10000)

    save_images(test_images, test_labels, processed_data_dir_test)
    print("Preprocessing completed!")
