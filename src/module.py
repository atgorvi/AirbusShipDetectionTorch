import os
import torch

import pytorch_lightning as pl

from src.utils import object_from_dict
from typing import Tuple, Dict, List


class SegmentationPipeline(pl.LightningModule):
    """
    A PyTorch Lightning module for a segmentation pipeline.

    Args:
        hparams (dict): A dictionary containing hyperparameters for the segmentation pipeline.

    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams.update(hparams)

        # Create model
        self.model = object_from_dict(hparams["model"])

        # Define loss function
        self.criterion = object_from_dict(hparams["criterion"])
        # Define metric
        self.metric = object_from_dict(hparams["metric"])

    def forward(self, image) -> torch.Tensor:
        mask = self.model(image)
        return mask

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
        """
        Configures the optimizer and scheduler for training.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[dict]]: The optimizer and scheduler configuration.

        """
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=self.model.parameters(),
        )
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer,
                                     max_lr=self.hparams["optimizer"]["lr"],
                                     total_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Training step for the segmentation pipeline.

        Args:
            train_batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the input image tensor and the target mask tensor.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The training loss.

        """
        x, y = train_batch
        #y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"train_loss": loss, "train_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, valid_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Validation step for the segmentation pipeline.

        Args:
            valid_batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the input image tensor and the target mask tensor.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the validation loss and metrics.

        """
        x, y = valid_batch

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"val_loss": loss, "val_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

