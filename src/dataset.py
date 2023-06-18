import os
import glob

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.utils import masks_as_image
from src.transforms import get_val_aug

class SegmentationDataset(Dataset):
    """
    Custom dataset class for segmentation data.

    Args:
        df_path (str): The file path to the CSV containing the dataset information.
        image_dir (str): The directory path where the images are located.
        transforms (callable, optional): Optional transformations to apply to the images and masks. Defaults to None.

    """
    def __init__(
        self,
        df_path: pd.DataFrame,
        image_dir: str,
        transforms=None,
    ):
        self.df = pd.read_csv(df_path)
        self.df["ImageId"] = self.df["ImageId"].map(lambda x: os.path.join(image_dir, x))
        self.dataset = list(self.df.groupby("ImageId"))
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = sample[0]
        image_df = sample[1]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = masks_as_image(image_df["EncodedPixels"].values)
        #mask =  np.expand_dims(mask, 0)


        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.float(), mask.float()

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    train_dataset = SegmentationDataset(df_path="data/train_ship_segmentations_v2_short.csv", image_dir="data/train_v2_short", transforms=get_val_aug(256))
    print(train_dataset[0][0].shape, train_dataset[0][1].shape)
