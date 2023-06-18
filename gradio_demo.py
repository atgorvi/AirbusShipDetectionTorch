import os
import yaml
import argparse
import cv2
import torch

import gradio as gr
import numpy as np

from typing import Dict
from skimage import color, segmentation
from src.utils import object_from_dict, state_dict_from_disk, visualize
from src.transforms import get_val_aug, get_test_aug

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model_ckpt", type=str, help="Path to the model ckpt", required=True
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config", required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)


    return args, hparams

if __name__ == "__main__":
    args, hparams = main()

    def process_image(image):
        image_size = hparams["params"]["image_size"]
        model = object_from_dict(hparams["model"])

        transform = get_test_aug(image_size)
        image_data = transform(image)
        input = image_data.unsqueeze(0)

        corrections: Dict[str, str] = {"model.": ""}

        state_dict = state_dict_from_disk(
            file_path=args.model_ckpt,
            rename_in_layers=corrections,
        )
        model.load_state_dict(state_dict)

        model.eval()
        with torch.no_grad():
            result = model(input)

        result = result.squeeze().cpu().numpy().round()

        segmentation_result = cv2.resize(
            result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        label2rgb = color.label2rgb(segmentation_result, image)
        img_with_contours = segmentation.mark_boundaries(
            image, segmentation_result, mode="thick"
        )
        img_with_mask = cv2.addWeighted(
            image,
            1,
            (cv2.cvtColor(segmentation_result, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(
                np.uint8
            ),
            0.5,
            0,
        )

        return segmentation_result, img_with_mask

    def run_segmentation(input_image):
        output_image = process_image(input_image)
        return output_image

    gr.Interface(run_segmentation, inputs=gr.Image(label="Input image"),
                 outputs=[gr.Image(label="Segmentation mask"), gr.Image(label="Segmentation mask")]).launch(debug=True)

