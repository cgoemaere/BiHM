from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .binarizedmnist_dataset import BinarizedMNIST
from .binarizedmnist_visualization_callback import BinarizedMNISTVisualizationCallback


class BinarizedMNISTDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for Binarized MNIST dataset."""

    dataset_name = "BinarizedMNIST"
    known_shapes = {"img": (1, 28, 28)}
    prediction_callback = BinarizedMNISTVisualizationCallback

    def __init__(self, batch_size=64):
        """
        Args:
            batch_size (int): Batch size for DataLoaders.
        """
        super().__init__()
        self.data_dir = "data"
        self.batch_size = batch_size

    def prepare_data(self):
        """Download the dataset (if not already downloaded)."""
        BinarizedMNIST(split="train", download=True, root=self.data_dir)
        BinarizedMNIST(split="validation", download=True, root=self.data_dir)
        BinarizedMNIST(split="test", download=True, root=self.data_dir)

    def setup(self, stage: str):
        """
        Setup dataset for different stages (train, validation, test).

        Args:
            stage (str): Optional, either 'fit' or 'test'. Used to control setup for various stages.
        """
        if stage == "fit":
            self.train_set = BinarizedMNIST(split="train", root=self.data_dir)
            self.val_set = BinarizedMNIST(split="validation", root=self.data_dir)

        elif stage == "test" or stage == "predict":
            self.test_set = BinarizedMNIST(split="test", root=self.data_dir)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Transforms batch after being placed on device
        Same as the 'transform' argument in torchvision datasets, but batched.
        """
        return {"img": batch}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)


# Example usage
if __name__ == "__main__":
    dm = BinarizedMNISTDataModule(batch_size=64)
    dm.prepare_data()  # Downloads data if not already downloaded
    dm.setup("fit")  # Prepares train and validation sets
    dm.setup("test")  # Prepares test set

    # Check data loader shapes
    for batch in dm.train_dataloader():
        print(batch.shape)  # Should output (batch_size, 1, 28, 28)
        print(batch.min(), batch.max())  # Should output (0., 1.)
        break
