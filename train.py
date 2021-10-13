import argparse
from typing import Optional
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
from dataset import StereotypeDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class StereotypeDetector(pl.LightningModule):
    def __init__(self, args: argparse):
        super(StereotypeDetector, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, num_labels=args.num_labels
        )
        self.criterion = nn.MSELoss()

    def forward(self, batch) -> torch.Tensor:
        for key in batch[0]:
            batch[0][key] = batch[0][key].squeeze()
        outputs = self.model(**batch[0])
        loss = self.criterion(outputs.logits, batch[1])
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss = self(batch)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, eps=self.args.eps)
        scheduler = MultiplicativeLR(
            optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = StereotypeDataset(
            tokenizer=self.tokenizer,
            data_dir=self.args.data_dir,
            stage="train",
            num_labels=self.args.num_labels,
        )

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        valid_dataset = StereotypeDataset(
            tokenizer=self.tokenizer,
            data_dir=self.args.data_dir,
            stage="valid",
            num_labels=self.args.num_labels,
        )

        return DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tunib/electra-ko-base")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_labels", type=int, default=7)
    return parser


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = _get_parser()
    args = parser.parse_args()
    pl.seed_everything(1234)
    wandb_logger = WandbLogger(name=args.model_name, project="StereoType Detector")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        EarlyStopping(monitor="valid_loss", patience=args.patience),
        ModelCheckpoint(
            monitor="valid_loss",
            dirpath="ckpt",
            filename="epoch={epoch}-valid_loss={valid_loss}",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
        ),
        lr_monitor,
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        gpus=args.gpus,
        deterministic=True,
    )

    model = StereotypeDetector(args=args)

    trainer.fit(model)