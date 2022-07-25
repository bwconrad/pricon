"""
Script for generating image and mask PNG files from Pannuke dataset's NumPy array
"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pannuke-processed",
        help="Output path of processed dataset",
    )
    args = parser.parse_args()

    data_dir = args.input
    output_dir = args.output

    for i, fold in enumerate(os.listdir(data_dir)):
        # Define paths
        f_name = f"fold{i+1}"
        img_path = os.path.join(data_dir, fold, "images", f_name, "images.npy")
        type_path = os.path.join(data_dir, fold, "images", f_name, "types.npy")
        mask_path = os.path.join(data_dir, fold, "masks", f_name, "masks.npy")

        # Load numpy arrays
        masks = np.load(file=mask_path, mmap_mode="r")
        images = np.load(file=img_path, mmap_mode="r")
        types = np.load(file=type_path)

        # Create output directories
        if not os.path.exists(output_dir):
            os.makedirs(os.path.join(output_dir, f_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, f_name, "sem_masks"), exist_ok=True)

        # Save as images
        for k in trange(
            images.shape[0],
            desc=f"Saving images for {fold}",
            total=images.shape[0],
        ):
            raw_image = images[k].astype(np.uint8)
            raw_mask = masks[k]
            sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
            tissue_type = types[k]

            # save file in op dir
            Image.fromarray(sem_mask).save(
                os.path.join(
                    output_dir,
                    f_name,
                    "sem_masks",
                    f"sem_{tissue_type}_{f_name}_{k:05d}.png",
                )
            )
            Image.fromarray(raw_image).save(
                os.path.join(
                    output_dir,
                    f_name,
                    "images",
                    f"img_{tissue_type}_{f_name}_{k:05d}.png",
                )
            )
