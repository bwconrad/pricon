import glob
import os
from typing import List, Optional

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch.utils.data as data
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.geometric import RandomRotate90, Resize
from albumentations.augmentations.transforms import (Flip, GaussianBlur,
                                                     GaussNoise, Normalize)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Optional[str] = None,
        train_fold: Optional[List[str]] = None,
        test_fold: Optional[str] = None,
        image_dir: str = "images",
        mask_dir: str = "sem_masks",
        train_size: int = 0,
        test_size: int = 0,
        batch_size: int = 16,
        workers: int = 4,
    ):
        """Multiclass Segmentation Datamodule

        Args:
            root: Path to root with train, val, test directories
            train_fold: Paths to training fold directories
            test_fold: Path to test fold directory
            image_dir: Name of image directory
            mask_dir: Name of mask directory
            train_size: Size of image crop during training. 0 = no crop
            test_size: Size of image crop during testing/validation. 0 = no crop
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        assert root or (train_fold and test_fold)

        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers
        self.train_fold = train_fold
        self.test_fold = test_fold
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Augmentations
        self.transforms_train = A.Compose(
            [
                RandomCrop(train_size, train_size) if train_size > 0 else None,
                GaussianBlur(blur_limit=[3, 7], p=0.5),
                GaussNoise(var_limit=25.0, p=0.5),
                RandomRotate90(p=0.5),
                Flip(p=0.5),
                Normalize(0.5, 0.5),
                ToTensorV2(),
            ],
        )
        self.transforms_test = A.Compose(
            [
                Resize(test_size, test_size) if test_size > 0 else None,
                Normalize(0.5, 0.5),
                ToTensorV2(),
            ],
        )

    def setup(self, stage="fit"):
        if self.root:
            if stage == "fit":
                self.train_dataset = SegmentationDataset(
                    os.path.join(self.root, "train"),
                    transforms=self.transforms_train,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )
                self.val_dataset = SegmentationDataset(
                    os.path.join(self.root, "val"),
                    transforms=self.transforms_test,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )
            elif stage == "test":
                self.test_dataset = SegmentationDataset(
                    os.path.join(self.root, "test"),
                    transforms=self.transforms_test,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )
        else:
            if stage == "fit":
                self.train_dataset = SegmentationDataset(
                    self.train_fold,
                    transforms=self.transforms_train,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )
                self.val_dataset = SegmentationDataset(
                    self.test_fold,
                    transforms=self.transforms_test,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )
            elif stage == "test":
                self.test_dataset = SegmentationDataset(
                    self.test_fold,
                    transforms=self.transforms_test,
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class SegmentationDataset(data.Dataset):
    def __init__(self, path, transforms, image_dir="images", mask_dir="sem_masks"):
        super().__init__()

        self.transforms = transforms

        # Get paths
        if isinstance(path, list):
            self.img_paths = []
            self.mask_paths = []
            for p in path:
                self.img_paths.extend(
                    sorted(glob.glob(os.path.join(p, image_dir, "*")))
                )
                self.mask_paths.extend(
                    sorted(glob.glob(os.path.join(p, mask_dir, "*")))
                )
        else:
            self.img_paths = sorted(glob.glob(os.path.join(path, image_dir, "*")))
            self.mask_paths = sorted(glob.glob(os.path.join(path, mask_dir, "*")))

        print(f"Loaded {len(self.img_paths)} images from {path}")

    def __getitem__(self, index):
        # Load the image and mask
        img = cv2.imread(self.img_paths[index])  # type:ignore
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type:ignore
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)  # type:ignore

        # Apply augmentations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"] / 255
        mask = transformed["mask"].long()

        return img, mask

    def __len__(self):
        return len(self.img_paths)
