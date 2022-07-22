import os
import random
from glob import glob
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms.functional import (InterpolationMode, hflip,
                                               resized_crop, rotate)


class ImageMaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        n_classes: int,
        batch_size: int = 256,
        workers: int = 4,
        crop_size: int = 224,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        brightness: float = 0.8,
        contrast: float = 0.8,
        saturation: float = 0.8,
        hue: float = 0.2,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        gaussian_prob: float = 0.0,
        mean: Sequence[float] = [0.5, 0.5, 0.5],
        std: Sequence[float] = [0.5, 0.5, 0.5],
    ):
        """Basic data module

        Args:
            data_dir: Path to image directory
            mask_dir: Path to mask directory
            n_classes: Number of segmentation mask classes
            batch_size: Number of batch samples
            workers: Number of data workers
            crop_size: Size of image crop
            min_scale: Minimum crop scale
            max_scale: Maximum crop scale
            brightness: Brightness intensity
            contast: Contast intensity
            saturation: Saturation intensity
            hue: Hue intensity
            color_jitter_prob: Probability of applying color jitter
            gray_scale_prob: Probability of converting to grayscale
            flip_prob: Probability of applying horizontal flip
            rotation_prob: Probability of applying rotation
            gaussian_prob: Probability of applying Gausian blurring
            mean: Image normalization means
            std: Image normalization standard deviations
        """
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.workers = workers

        self.transforms = MultiViewTransform(
            ImageMaskTransform(
                crop_size=crop_size,
                min_scale=min_scale,
                max_scale=max_scale,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                color_jitter_prob=color_jitter_prob,
                gray_scale_prob=gray_scale_prob,
                gaussian_prob=gaussian_prob,
                flip_prob=flip_prob,
                rotation_prob=rotation_prob,
                mean=mean,
                std=std,
                n_classes=n_classes,
            )
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            self.dataset = ImageMaskDataset(
                self.image_dir, self.mask_dir, self.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transforms: Callable):
        """Joint image and mask dataset from directories

        Args:
            image_dir: Path to image directory
            mask_dir: Path to mask directory
            transforms: Augmentation pipeline
        """
        super().__init__()

        # Assuming image and mask filenames are the same (or at least sort into the correct order)
        self.image_paths = sorted(
            [f for f in glob(f"{image_dir}/**/*", recursive=True) if os.path.isfile(f)]
        )
        self.mask_paths = sorted(
            [f for f in glob(f"{mask_dir}/**/*", recursive=True) if os.path.isfile(f)]
        )
        assert len(self.image_paths) == len(self.mask_paths)

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
        print(f"Loaded {len(self.mask_paths)} masks from {mask_dir}")

        self.transforms = transforms

    def __getitem__(self, index: int):
        # Load image and mask
        img = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        # Apply augmentations
        img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.image_paths)


class MultiViewTransform:
    def __init__(self, transforms: Callable, n_views: int = 2):
        """Wrapper class to apply transforms on the same image + mask n times

        Args:
            transforms: Image and mask augmentation pipeline
            n_views: Number of augmented views to return
        """
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, img, mask):
        imgs = []
        masks = []

        for _ in range(self.n_views):
            i, m = self.transforms(img, mask)
            imgs.append(i)
            masks.append(m)

        return imgs, masks


class RandomResizedCropFlipRotateWithMask(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale: Sequence[float] = (0.2, 1.0),
        ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        interpolation_mask: InterpolationMode = InterpolationMode.NEAREST,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
    ):
        """Perform the same random resized crop, horizontal flip and
        90 degree rotation on an image and mask

        Args:
            size: Crop size
            scale: Min and max crop scale
            ratio: Min and max crop aspect ratio
            interpolation: Image interpolation mode
            interpolation_mask: Mask interpolation mode
            flip_prob: Probability of applying horizontal flip
            rotation_prob: Probability of applying rotation
        """
        super().__init__(size, scale, ratio, interpolation)
        self.interpolation_mask = interpolation_mask
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob

    def forward(self, img: torch.Tensor, mask: torch.Tensor):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)  # type: ignore

        # Crop
        img = resized_crop(img, i, j, h, w, self.size, self.interpolation)  # type: ignore
        mask = resized_crop(mask, i, j, h, w, self.size, self.interpolation_mask)  # type: ignore

        # Horizontal flip
        if random.random() > self.flip_prob:
            img = hflip(img)
            mask = hflip(mask)

        # 90 degree rotation
        if random.random() > self.rotation_prob:
            angle = random.choice([90, 180, 270])
            img = rotate(img, angle)
            mask = rotate(mask, angle)

        return img, mask


class ImageMaskTransform:
    def __init__(
        self,
        n_classes: int,
        crop_size: int = 224,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        brightness: float = 0.8,
        contrast: float = 0.8,
        saturation: float = 0.8,
        hue: float = 0.2,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        gaussian_prob: float = 0.0,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Augmentation pipeline for image and mask pairs

        Args:
            n_classes: Number of segmentation mask classes
            crop_size: Size of image crop
            min_scale: Minimum crop scale
            max_scale: Maximum crop scale
            brightness: Brightness intensity
            contast: Contast intensity
            saturation: Saturation intensity
            hue: Hue intensity
            color_jitter_prob: Probability of applying color jitter
            gray_scale_prob: Probability of converting to grayscale
            gaussian_prob: Probability of applying Gausian blurring
            flip_prob: Probability of applying horizontal flip
            rotation_prob: Probability of applying rotation
            mean: Image normalization means
            std: Image normalization standard deviations
        """
        super().__init__()

        self.img_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23)], p=gaussian_prob
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze(0) * 255),
                transforms.Lambda(
                    lambda x: F.one_hot(x.long(), num_classes=n_classes).permute(
                        2, 0, 1
                    )
                ),
            ]
        )

        self.shared_transforms = RandomResizedCropFlipRotateWithMask(
            size=crop_size,
            scale=(min_scale, max_scale),
            flip_prob=flip_prob,
            rotation_prob=rotation_prob,
        )

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Apply same geometric augmentations to image and mask
        img, mask = self.shared_transforms(img, mask)

        # Apply individual augmentations
        img = self.img_transforms(img)
        mask = self.mask_transforms(mask)

        return img, mask


if __name__ == "__main__":
    from torchvision.utils import save_image

    dm = DataModule("data/tcga-patches", "data/tcga-masks", n_classes=6)
    dm.setup()
    dl = dm.train_dataloader()

    for b in dl:
        img1, img2, m1, m2 = b[0][0], b[0][1], b[1][0], b[1][1]
        print(len(b[0]))
        print(len(b[1]))
        print(b[0][0].size())
        print(b[0][1].size())
        print(b[1][0].size())
        print(b[1][1].size())

        break

    save_image(img1[:4], "0.png", normalize=True)
    save_image(m1[:4, 0].unsqueeze(1).float(), "1.png")
    save_image(img2[:4], "2.png", normalize=True)
    save_image(m2[:4, 0].unsqueeze(1).float(), "3.png")
