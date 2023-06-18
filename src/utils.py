import os
import re
import pydoc
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional, Union, List
from skimage.util import montage
from skimage.morphology import binary_opening, disk, label


def average(outputs: list, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()

def montage_rgb(x: np.ndarray) -> np.ndarray:
    return np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


def multi_rle_encode(img, **kwargs):
    """
    Encode connected regions as separated masks
    """

    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels == k, axis=2), **kwargs) for k in np.unique(labels[labels > 0])]
    else:
        return [rle_encode(labels == k, **kwargs) for k in np.unique(labels[labels > 0])]


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if np.max(img) < min_max_threshold:
        return ''  # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  # ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)) -> np.ndarray:
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list) -> np.ndarray:
    """Take the individual ship masks and create a single mask array for all ships"""
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list) -> np.ndarray:
    """Take the individual ship masks and create a color mask array for each ships"""
    all_masks = np.zeros((768, 768), dtype=np.float)
    scale = lambda x: (len(in_mask_list) + x + 1) / (len(in_mask_list) * 2)  # scale the heatmap image to shift
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask)
    return all_masks

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)

def rename_layers(
    state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]
) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict

def average(outputs: list, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()

def tensor_to_image(tens):
    array = tens.squeeze(0).permute(1, 2, 0).numpy() * 255
    array = array.astype(np.uint8)
    pil_img = Image.fromarray(array)
    return pil_img

def mask_tensor_to_image(tens):
    array = tens.permute(1, 2, 0).numpy() * 255
    array = np.squeeze(array.astype(np.uint8), -1)
    pil_img = Image.fromarray(array)
    return pil_img

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    pass