import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.dataset import SegmentationDataset
from src.transforms import get_val_aug, get_train_aug

class SegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for segmentation data.

    Args:
        hparams (dict): A dictionary containing hyperparameters and configuration for the data module.

    """
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.image_size = hparams["params"]["image_size"]
        self.images_dir = self.hparams["data"]["images_dir"]
        self.batch_size = self.hparams["params"]["batch_size"]

    def setup(self, stage=None) -> None:
        """
        Sets up the data module.

        Args:
            stage (str, optional): The current stage (e.g., "fit", "test"). Defaults to None.

        """
        train_df_path = self.hparams["data"]["train_csv_path"]
        train_transforms = get_train_aug(self.image_size)
        val_df_path = self.hparams["data"]["val_csv_path"]
        val_transforms = get_val_aug(self.image_size)

        self.train_data = SegmentationDataset(df_path=train_df_path, image_dir=self.images_dir, transforms=train_transforms)
        self.val_data = SegmentationDataset(df_path=val_df_path, image_dir=self.images_dir, transforms=val_transforms)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training data loader.

        Returns:
            torch.utils.data.DataLoader: The training data loader.

        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            shuffle=True
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation data loader.

        Returns:
            torch.utils.data.DataLoader: The validation data loader.

        """
        train_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            shuffle=False
        )
        return train_loader

