import random
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .encoder import resnet18, resnet50
from .loss import simclr_loss_func
from .optim import LARSWrapper

ENCODERS = {"resnet18": resnet18, "resnet50": resnet50}


class PRIConModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.2,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        lars: bool = True,
        encoder: str = "resnet18",
        channel_last: bool = False,
        output_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.1,
        n_points: int = 32,
        alpha_point: float = 0.7,
        alpha_region: float = 0.75,
        alpha_image: float = 0.15,
        resume_checkpoint: Optional[str] = None,
        imagenet_init: bool = False,
    ) -> None:
        """Point Region Image-level Contrast Model

        Args:
            lr: Learning rate
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            lars: Use LARS optimizer
            encoder: Encoder/backbone network architecture (resnet18 | resnet50)
            channel_last: Change to channel last memory format for possible training speed up
            output_dim: Projector output dimension
            hidden_dim: Projector hidden layer dimension
            temperature: Temperature parameter for contrastive loss
            n_points: Point loss number of sampled features
            alpha_point: Point loss weight
            alpha_region: Region loss weight
            alpha_image: Image loss weight
            resume_checkpoint: Path of checkpoint to resume training from for cyclic training
            imagenet_init: Initialize encoder weights with ImageNet supervised
        """
        assert alpha_point > 0 or alpha_region > 0 or alpha_image > 0

        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lars = lars
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.channel_last = channel_last
        self.temperature = temperature
        self.n_points = n_points
        self.alpha_point = alpha_point
        self.alpha_region = alpha_region
        self.alpha_image = alpha_image

        # Initalize model
        self.encoder = ENCODERS[encoder](pretrained=imagenet_init)
        features_dim = self.encoder.inplanes

        self.projector = nn.Sequential(
            nn.Conv2d(features_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

        # Load model weights if given
        if resume_checkpoint:
            print(f"Resuming from {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint)["state_dict"]

            # Fix key names
            encoder_model = {}
            projector_model = {}
            for k, v in ckpt.items():
                if k.startswith("encoder"):
                    k = k.replace("encoder" + ".", "")
                    encoder_model[k] = v
                elif k.startswith("projector"):
                    k = k.replace("projector" + ".", "")
                    projector_model[k] = v

            # Load weights
            self.encoder.load_state_dict(encoder_model, strict=False)
            self.projector.load_state_dict(projector_model, strict=False)

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channel_last:
            print("Using channel last memory format")
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def get_valid_classes(mask1: torch.Tensor, mask2: torch.Tensor) -> List[List[int]]:
        bs = mask1.shape[1]
        classes = [[] for _ in range(bs)]
        for n in range(bs):
            for i, (m1, m2) in enumerate(zip(mask1[:, n], mask2[:, n])):
                if not (torch.all(m1 == 0) or torch.all(m2 == 0)):
                    classes[n].append(i)

        return classes

    @staticmethod
    def mask_pooling(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        z1_dense: torch.Tensor,
        z2_dense: torch.Tensor,
        classes: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Mask pooling
        """

        z1_pooled, z2_pooled, selected = [], [], []
        for i, classes_i in enumerate(classes):
            z1 = z1_dense[i]
            z2 = z2_dense[i]
            m1 = mask1[classes_i, i]
            m2 = mask2[classes_i, i]

            z1_pooled.append(
                (torch.einsum("ij,kj->ki", z1, m1).T / torch.einsum("ij->i", m1)).T
            )
            z2_pooled.append(
                (torch.einsum("ij,kj->ki", z2, m2).T / torch.einsum("ij->i", m2)).T
            )
            selected.extend(classes_i)

        z1_pooled = torch.cat(z1_pooled)
        z2_pooled = torch.cat(z2_pooled)

        return z1_pooled, z2_pooled, selected

    @staticmethod
    def sample_index_pairs(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        n_points: int,
        classes: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample n_points index pairs
        """

        bs = mask1.shape[1]
        indices1, indices2, selected = [], [], []
        bs_range = torch.arange(bs)
        for _ in range(n_points):
            i = [random.choice(c) for c in classes]  # Sample a class for each element
            indices1.append(torch.multinomial(mask1[i, bs_range], 1).permute(1, 0))
            indices2.append(torch.multinomial(mask2[i, bs_range], 1).permute(1, 0))
            selected.extend(i)

        indices1 = torch.cat(indices1)
        indices2 = torch.cat(indices2)

        return indices1, indices2, selected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_last:
            x = x.to(memory_format=torch.channels_last)  # type: ignore

        return self.encoder(x)

    def training_step(
        self, batch: Tuple[List[torch.Tensor], List[torch.Tensor]], _
    ) -> torch.Tensor:
        imgs, [mask1, mask2] = batch

        # --- Extract image features --- #
        # Pass through encoder
        feats = [self.forward(img) for img in imgs]

        # Pass through projector
        z1, z2 = [self.projector(feat).flatten(2) for feat in feats]

        # -- Prepare masks -- #
        # Resize masks
        feat_res = feats[0].size(2)
        mask1 = (
            F.adaptive_avg_pool2d(mask1.float(), feat_res).flatten(2).permute(1, 0, 2)
        )
        mask2 = (
            F.adaptive_avg_pool2d(mask2.float(), feat_res).flatten(2).permute(1, 0, 2)
        )

        # Get indices of non-empty classes per batch element
        valid_classes = self.get_valid_classes(mask1, mask2)

        # --- Calculate losses --- #
        loss = 0

        # Image loss
        if self.alpha_point > 0:
            # Sample point pairs
            idxs1, idxs2, classes_pts = self.sample_index_pairs(
                mask1, mask2, self.n_points, valid_classes
            )

            # Get corresponding features for the points
            b = z1.shape[0]
            range = torch.arange(b)
            z1_pts, z2_pts = [], []
            for idx1, idx2 in zip(idxs1, idxs2):
                z1_pts.append(z1[range, :, idx1[:]])
                z2_pts.append(z2[range, :, idx2[:]])
            z_pts = torch.concat(z1_pts + z2_pts)

            # Calculate loss
            classes_pts = torch.tensor(
                classes_pts, device=z_pts.device, dtype=torch.int64
            ).repeat(2)
            loss_point = simclr_loss_func(
                z_pts,
                indexes=classes_pts,
                temperature=self.temperature,
            )

            loss += loss_point * self.alpha_point
            self.log("loss_point", loss_point, on_epoch=True, sync_dist=True)

        # Region loss
        if self.alpha_region > 0:
            # Aggregate region features
            z1_reg, z2_reg, classes_reg = self.mask_pooling(
                mask1, mask2, z1, z2, valid_classes
            )
            z_reg = torch.cat([z1_reg, z2_reg])

            # Calculate loss
            classes_reg = torch.tensor(
                classes_reg, device=z_reg.device, dtype=torch.int64
            ).repeat(2)
            loss_region = simclr_loss_func(
                z_reg,
                indexes=classes_reg,
                temperature=self.temperature,
            )

            loss += loss_region * self.alpha_region
            self.log("loss_region", loss_region, on_epoch=True, sync_dist=True)

        # Image loss
        if self.alpha_image > 0:
            # Global pool dense features
            z1_image = torch.flatten(F.adaptive_avg_pool1d(z1, 1), 1)
            z2_image = torch.flatten(F.adaptive_avg_pool1d(z2, 1), 1)
            z_image = torch.cat([z1_image, z2_image])

            # Calculate loss
            indexes = torch.arange(z1.shape[0], device=self.device).repeat(2)
            loss_image = simclr_loss_func(
                z_image,
                indexes=indexes,
                temperature=self.temperature,
            )

            self.log("loss_image", loss_image, on_epoch=True, sync_dist=True)
            loss += loss_image * self.alpha_image

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],  # type:ignore
            prog_bar=True,
        )

        return loss  # type:ignore

    def configure_optimizers(self) -> dict:
        params = [
            {"name": "backbone", "params": self.encoder.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
        ]

        optimizer = SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        if self.lars:
            optimizer = LARSWrapper(optimizer)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches  # type:ignore
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
