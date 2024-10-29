import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset


class BinarizedMNIST(Dataset):
    """Binarized MNIST dataset."""

    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    FILENAMES = {
        "train": "binarized_mnist_train.amat",
        "validation": "binarized_mnist_valid.amat",
        "test": "binarized_mnist_test.amat",
    }

    def __init__(self, split="train", transform=None, download=True, root="data"):
        """
        Args:
            split (str): One of 'train', 'validation', or 'test'.
            transform (callable, optional): Optional transform to be applied on an image.
            download (bool): If True, downloads the data if it doesn't exist in the root directory.
            root (str): Root directory where the dataset will be stored.
        """
        assert split in [
            "train",
            "validation",
            "test",
        ], "Split must be 'train', 'validation', or 'test'"
        self.split = split
        self.transform = transform
        self.root = os.path.abspath(os.path.join(root, "BinarizedMNIST"))

        if download:
            self.download_data()

        # Load data
        data_path = os.path.join(self.root, self.FILENAMES[split])
        self.data = np.loadtxt(data_path, delimiter=" ", dtype=np.uint8)
        self.data = self.data.reshape(-1, 1, 28, 28)  # Each image is 1x28x28
        self.data = self.data.transpose((0, 1, 3, 2)).copy()  # Transpose img like in EMNIST

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve an image by index and apply any transforms."""
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
        return image

    def download_data(self):
        """Download the dataset if not already present."""
        os.makedirs(self.root, exist_ok=True)
        for filename in self.FILENAMES.values():
            file_path = os.path.join(self.root, filename)
            if not os.path.isfile(file_path):
                print(f"Downloading {filename}...")
                urlretrieve(self.URL + filename, file_path)
                print(f"Downloaded {filename}.")
