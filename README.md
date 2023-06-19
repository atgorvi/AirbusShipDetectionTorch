# Airbus ships detection
Aim is to create simple binary segmentation pipeline to locate ships in images.

## Install
```sh
git clone https://github.com/atgorvi/AirbusShipDetectionTorch.git
cd AirbusShipDetectionTorch

pip install -r requirements.txt
```
## Data
[Original dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data) 
contains labeled images for train, unlabeled for test, .csv wit masks and .csv with sample submission.
Goal of this task is to detect ships as accurate as possible using binary segmentation. 

Data prepared with notebooks/data_prep.ipynb

- Original image resolution: 768x768.
- Image resolution during training: 256x256.
- Number of images used in this task: 38455 for train, 12816 for validation

## Result
Model has **~60 IoU** on val set

## Training
You can run training command with:
```sh
python train.py --config {PATH/TO/CONFIG}
```
**Architecture**: I tried Unet and FPN architectures. FPN seems to work slight better (~1%). \
**Encoder**: resnet18 seem to have similar IoU with efficientnetb3 in my few runs. \
**Loss**: Dice \
**Opimizer**: Adam (lr=0.0005) \
**Metric**: IoU (threshold=0.5) \
**Number of epochs**: 20 

My training config: configs/config_FPN_resnet_18_256x256.yaml

## Inference example
You can run demo.py for inference:
#### Before running download checkpoint from [Model weights](https://drive.google.com/file/d/1s6m-5aNJfV_7WbPUrQB8xBeSp6RiSlJh/view?usp=sharing) and put to checkpoints forlder
```sh
python demo.py --image {PATH/TO/IMAGE} --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```
## Gradio demo, just run and put image to box
You can run gradio demo.py for inference:
```sh
python gradio_demo.py --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```

## Random image from test set:
#### Original image:
![Original image](https://github.com/atgorvi/AirbusShipDetectionTorch/blob/3247b3e7303bd20da2e486073694808f935c7ece/data/inference/original.jpg)
#### Predicted mask:
![Predicted mask](https://github.com/atgorvi/AirbusShipDetectionTorch/blob/3247b3e7303bd20da2e486073694808f935c7ece/data/inference/pred_mask.jpg)
#### Image with mask overlay:
![Image with mask](https://github.com/atgorvi/AirbusShipDetectionTorch/blob/3247b3e7303bd20da2e486073694808f935c7ece/data/inference/masked.jpg)

## Further improvements
1. Try another architecture like Linknet or another encoder.
2. Use ensemble of models, firstly classify ship/no ship and then create mask 3.Try another loss function or combine them.
3. Try different loss function or combine them.
4. Try another optimizer and scheduler.
5. Play with image resolution which require more computational power.
6. Find the best learning rate.
7. If we don't want to "label" more data we can "pseudo label" it and use for training.
8. Use postprocessing techniques such as dilation to improve masks quality.
9. Play with more advanced augmentations, geometrical transform seem to be good choice.
10. Tune model with very low lr and ReduceOnPlateau scheduler.
