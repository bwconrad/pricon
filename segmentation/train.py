import os
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import SegmentationDataModule
from src.model import SegmentationModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = SegmentationModel
dm_class = SegmentationDataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data")
parser.add_lightning_class_args(model_class, "model")

# Pannuke evaluation args
parser.add_argument(
    "--test",
    type=str,
    choices=["false", "pannuke", "lizard", "pannuke_breast"],
    help="After training testing mode",
    default="false",
)
parser.add_argument(
    "--test_types_path",
    type=str,
    help="Path to test types file",
    default="eval/types/types-f3.npy.",
)
parser.add_argument(
    "--test_targets_path",
    type=str,
    help="Path to test targets file",
    default="eval/targets/targets-f3.npy.",
)

args = parser.parse_args()

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_iou",
    mode="max",
    filename="best-{epoch:02d}-{val_iou:.3f}",
)

# Setup trainer
logger = init_logger(args)
dm = dm_class(**args["data"])
model = model_class(**args["model"])
trainer = pl.Trainer.from_argparse_args(
    Namespace(**args), logger=logger, callbacks=[checkpoint_callback]
)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)

# Test
if args["test"] != "false":
    # Import correct evaluation script
    if args["test"] == "pannuke":
        from eval.eval_pannuke import calculate_pq
    elif args["test"] == "pannuke_breast":
        from eval.eval_pannuke_breast import calculate_pq
    else:
        from eval.eval_lizard import calculate_pq

    results = trainer.test(model, dm)[0]

    pq, _ = calculate_pq(
        true_path=args["test_targets_path"],
        pred_path=os.path.join(
            trainer.logger.save_dir,
            trainer.logger.name,
            "version_" + str(trainer.logger.version)
            if not args["logger_type"] == "wandb"
            else str(trainer.logger.version),
            "preds.npy",
        ),
        save_path=os.path.join(
            trainer.logger.save_dir,
            trainer.logger.name,
            "version_" + str(trainer.logger.version)
            if not args["logger_type"] == "wandb"
            else str(trainer.logger.version),
        ),
        types_path=args["test_types_path"],
    )

    trainer.logger.log_metrics({"test_iou": results["test_iou"], "test_pq": pq})
