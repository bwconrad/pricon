import os
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn.functional import cross_entropy, one_hot
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification.jaccard import JaccardIndex


def freeze(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return


def unfreeze(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = True
    return


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = "unet",
        encoder: str = "resnet18",
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        in_channels: int = 3,
        n_classes: int = 6,
        encoder_weights: Optional[str] = None,
        freeze_encoder: bool = False,
        unfreeze_after: int = -1,
    ):
        """Multiclass Segmentation Model

        Args:
            arch: Name of segmentation model architecture
            encoder: Name of encoder for segmentation model
            lr: Learning rate
            betas: Adam betas
            weight_decay: Weight decay
            in_channels: Number of channels in input images
            n_classes: Number of classes
            encoder_weights: Path to pretrained encoder weights
            freeze_encoder: Whether to freeze the encoder weights
            unfreeze_after: After what epoch to unfreeze encoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.encoder = encoder
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.encoder_weights = encoder_weights
        self.freeze_encoder = freeze_encoder
        self.unfreeze_after = unfreeze_after

        # Network
        self.net = smp.create_model(
            self.arch,
            encoder_name=self.encoder,
            in_channels=self.in_channels,
            classes=self.n_classes,
            encoder_weights=None
            if self.encoder_weights != "supervised"
            else "imagenet",
        )

        # Load weights
        if self.encoder_weights and self.encoder_weights != "supervised":
            state_dict = torch.load(self.encoder_weights)
            self.net.encoder.load_state_dict(state_dict, strict=True)
            print(f"Loaded encoder weights from {self.encoder_weights}")
        elif self.encoder_weights == "supervised":
            print(f"Loaded Imagenet encoder weights")

        # Freeze encoder
        if self.freeze_encoder:
            freeze(self.net)

        # Metrics
        self.train_iou = JaccardIndex(num_classes=self.n_classes)
        self.val_iou = JaccardIndex(num_classes=self.n_classes)
        self.test_iou = JaccardIndex(num_classes=self.n_classes)

    def forward(self, x):
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        if self.freeze_encoder and self.current_epoch == self.unfreeze_after:
            print(f"\nUnfreezing encoder weights after {self.unfreeze_after} epochs\n")
            unfreeze(self.net)

    def shared_step(self, batch, mode="train"):
        x, y = batch

        # Get loss
        pred = self(x)
        loss = cross_entropy(pred, y)

        # Get metrics
        iou = getattr(self, f"{mode}_iou")(pred.argmax(1), y)

        # Log
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_iou", iou)

        if mode == "test":
            return pred.float().cpu()
        else:
            return loss

    def training_step(self, batch, _):
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],  # type:ignore
            prog_bar=True,
        )
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        # Save predictions
        print("Saving predictions...")
        masks = []
        for output in outputs:
            idxs = output.argmax(dim=1)
            masks.append(one_hot(idxs, num_classes=self.n_classes))
        preds = torch.cat(masks).numpy()
        try:
            np.save(
                os.path.join(
                    self.logger.save_dir,
                    self.logger.name,
                    "version_" + str(self.logger.version)
                    if not isinstance(self.logger, WandbLogger)
                    else str(self.logger.version),
                    "preds.npy",
                ),
                preds,
            )
        except:
            np.save("preds.npy", preds)

    def configure_optimizers(self):
        optimizer = Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.trainer.estimated_stepping_batches  # type:ignore
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]
