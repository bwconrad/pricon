"""
Script to create Lizard dataset per fold ground truth mask arrays for evaluation
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.nn.functional import one_hot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to processed dataset"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="lizard-targets",
        help="Output path of target files",
    )
    args = parser.parse_args()

    data_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(data_dir):
        print(f"Processing {f}...")

        paths = sorted(glob(os.path.join(data_dir, f, "sem_masks", "*")))
        masks = []
        for p in paths:
            masks.append(cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.int64))

        m = np.stack(masks)
        m = one_hot(torch.tensor(m), num_classes=7).numpy()

        np.save(os.path.join(output_dir, f"targets-{f}.npy"), m)
