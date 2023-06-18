# Airbus ships detection
Aim is to create simple binary segmentation pipeline to locate ships in images using tf.keras and Unet architecture.

## Install
```sh
git clone 
cd 

pip install -r requirements.txt
```
## Data
[Original dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data) 
contains labeled images for train, unlabeled for test, .csv wit masks and .csv with sample submission.
Goal of this task is to detect ships as accurate as possible using binary segmentation. 

Data prepared with notebooks/data_prep.ipynb

- Original image resolution: 768x768.
- Image resolution during training: 256x256.
- Number of images used in this task: 12583 for train, 4195 for validation, 100 for test set.

## Result
Model has **~57 IoU** on val set

## Training
You can run training command with:
```sh
python train.py --config {PATH/TO/CONFIG}
```
**Architecture**: I tried Unet and FPN architectures. FPN seems to work slight better (~1%). \
**Encoder**: resnet18 was ~2% better than efficientnetb3 in my few runs. \
**Loss**: Dice \
**Opimizer**: Adam (lr=0.0005) \
**Metric**: IoU (threshold=0.5) \
**Number of epochs**: 30 

My training config: configs/config.yaml

My training config: configs/config.yaml

## Inference example
You can run demo.py for inference:
```sh
python demo.py --image {PATH/TO/IMAGE} --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```
## Gradio demo, just run and put image to box
You can run gradio demo.py for inference:
```sh
python demo.py --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```

Random image from test set: \
Original image:
//TODO

## Further improvements
1. Try another architecture like DeepLabV3+, FPN, Linknet or another encoder.
2. Use ensemble of models, firstly classify ship/no ship and then create mask 3.Try another loss function or combine them.
3. Try different loss function or combine them.
4. Try another optimizer and scheduler.
5. Play with image resolution which require more computational power.
6. Find the best learning rate.
7. If we don't want to "label" more data we can "pseudo label" it and use for training.
8. Use postprocessing techniques such as dilation to improve masks quality.
9. Play with more advanced augmentations, geometrical transform seem to be good choice.
