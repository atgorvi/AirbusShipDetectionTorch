import os
import torch

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from src.dataset import SegmentationDataset
from src.transforms import get_val_aug, get_train_aug
from src.utils import object_from_dict, average
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU


class SegmentationPipeline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams.update(hparams)

        self.model = object_from_dict(hparams["model"])

        # for image segmentation dice loss could be the best first choice
        self.criterion = object_from_dict(hparams["criterion"])
        self.metric = object_from_dict(hparams["metric"])

    def forward(self, image):
        mask = self.model(image)
        return mask

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=self.model.parameters(),
        )
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer,
                                     max_lr=self.hparams["optimizer"]["lr"],
                                     total_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        #y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"train_loss": loss, "train_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        #y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"val_loss": loss, "val_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

