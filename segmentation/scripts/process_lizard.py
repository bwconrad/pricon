"""
Script for generating image and mask PNG files from Lizard dataset's NumPy array
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="lizard-processed",
        help="Output path of processed dataset",
    )
    args = parser.parse_args()

    data_dir = args.input
    output_dir = args.output

    # Load files
    image_info = pd.read_csv(os.path.join(data_dir, "info.csv"), index_col="Filename")
    patch_info = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))[
        "patch_info"
    ].tolist()
    images = np.load(os.path.join(data_dir, "images.npy"))
    masks = np.load(os.path.join(data_dir, "labels.npy"))[:, :, :, 1]

    # Save as images
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "fold1", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold1", "sem_masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold2", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold2", "sem_masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold3", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold3", "sem_masks"), exist_ok=True)

    for img, m, name in tqdm(zip(images, masks, patch_info), total=len(patch_info)):
        # Get the fold number for the patch
        name_pre = name[: name.find("-")]
        fold = image_info.loc[name_pre]["Split"]

        # Save image and mask
        cv2.imwrite(
            os.path.join(output_dir, f"fold{fold}", "images", name + ".png"), img
        )
        cv2.imwrite(
            os.path.join(output_dir, f"fold{fold}", "sem_masks", name + ".png"), m
        )
