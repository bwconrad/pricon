"""
Script to generate pseudo-label masks for a dataset with a trained segmentation model
"""

import argparse
import glob
import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.utils.data as data
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        default="unet",
        help="Segmentation model architecture",
    )
    parser.add_argument(
        "--encoder",
        "-e",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Encoder architecture",
    )
    parser.add_argument(
        "--weights", "-w", type=str, required=True, help="Path to pretrained weights"
    )
    parser.add_argument(
        "--images", "-i", type=str, required=True, help="Path to image directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path of output directory"
    )
    parser.add_argument(
        "--n_classes", "-n", type=int, default=6, help="Number of segmentation classes"
    )

    args = parser.parse_args()

    return args


class Dataset(data.Dataset):
    def __init__(self, path):
        super().__init__()

        self.transforms = A.Compose(
            [
                Normalize(0.5, 0.5),
                ToTensorV2(),
            ],
        )

        self.img_paths = sorted(glob.glob(os.path.join(path, "*")))
        print(f"Loaded {len(self.img_paths)} images from {path}")

    def __getitem__(self, index):
        # Load the image and mask
        path = self.img_paths[index]
        img = cv2.imread(path)  # type:ignore
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type:ignore

        # Apply augmentations
        transformed = self.transforms(image=img)
        img = transformed["image"] / 255

        return img, path

    def __len__(self):
        return len(self.img_paths)


def main(arch, encoder, weights, n_classes, img_path, out_path):
    # Initialize segmentation network
    net = smp.create_model(
        arch,
        encoder_name=encoder,
        in_channels=3,
        classes=n_classes - 1 if n_classes == 2 else n_classes,
        encoder_weights=None,
    )
    net = net.cuda().eval()

    # Load weights
    state_dict = torch.load(weights)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict, strict=True)
    print(f"Loaded segmentation network weights from {weights}")

    # Load data
    dataset = Dataset(img_path)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create output directory
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for img, paths in tqdm(dataloader):
        # Pass batch through network
        img = img.cuda()
        with torch.no_grad():
            masks = net(img)

            if n_classes == 2:
                masks = F.logsigmoid(masks).exp()
                masks = torch.concat([1 - masks, masks], dim=1)

            masks = masks.argmax(dim=1)

        # Save outputs to disk
        for m, path in zip(masks, paths):
            out = os.path.join(
                out_path, os.path.splitext(os.path.basename(path))[0] + ".png"
            )
            Image.fromarray(m.detach().cpu().numpy().astype(np.uint8)).save(out)


if __name__ == "__main__":
    args = get_args()
    main(
        args.arch, args.encoder, args.weights, args.n_classes, args.images, args.output
    )
