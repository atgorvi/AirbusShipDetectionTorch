import os
import yaml
import argparse
import cv2
import torch

import numpy as np

from typing import Dict
from skimage import color, segmentation
from src.utils import object_from_dict, state_dict_from_disk, visualize
from src.transforms import get_val_aug, get_test_aug


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--image", type=str, help="Path to the image.", required=True
    )
    parser.add_argument(
        "-m", "--model_ckpt", type=str, help="Path to the model ckpt", required=True
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config", required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(args.config) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    image_size = hparams["params"]["image_size"]

    # Load model
    model = object_from_dict(hparams["model"])
    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(
        file_path=args.model_ckpt,
        rename_in_layers=corrections,
    )
    model.load_state_dict(state_dict)

    # Prepare input
    transform = get_test_aug(image_size)
    image_data = transform(image)
    input = image_data.unsqueeze(0)

    # Get predictions mask
    model.eval()
    with torch.no_grad():
        result = model(input)

    result = result.squeeze().cpu().numpy().round()

    # Visualize and save results
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

    gt_mask_path = args.image.replace("images", "masks")
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    save_dir = "data/inference"
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, "original.jpg"), image)
    cv2.imwrite(os.path.join(save_dir, "pred_mask.jpg"), 255 * segmentation_result)
    cv2.imwrite(os.path.join(save_dir, "masked.jpg"), img_with_mask)
    #cv2.imwrite(os.path.join(save_dir, "gt_mask.jpg"), gt_mask)

    visualize(
        original_image=image,
        predicted_mask=segmentation_result,
        # label2rgb=label2rgb,
        # image_with_contour=img_with_contours,
        image_with_mask=img_with_mask,
        #ground_truth_mask=gt_mask,
    )


if __name__ == "__main__":
    main()