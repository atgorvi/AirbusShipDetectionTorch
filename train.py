import yaml

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from src.data_module import SegmentationDataModule
from src.module import SegmentationPipeline
from src.utils import object_from_dict


def main():
    with open("configs/config_FPN_efficientnet-b3_256x256.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pl.seed_everything(hparams["seed"])

    epochs = hparams["params"]["epochs"]
    project_name = hparams["project_name"]
    experiment_name = hparams["experiment_name"]

    wandb_logger = WandbLogger(project=project_name, name=experiment_name)
    stopper = object_from_dict(hparams["callbacks"]["stopper"])
    lr_monitor = object_from_dict(hparams["callbacks"]["lr_monitor"])
    checkpoint = object_from_dict(hparams["callbacks"]["checkpoint"])
    callbacks = [stopper, lr_monitor, checkpoint]

    # Create modules for training
    datamodule = SegmentationDataModule(hparams)
    model_pipeline = SegmentationPipeline(hparams)

    # Start training
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(model_pipeline, datamodule)


if __name__ == "__main__":
    main()

