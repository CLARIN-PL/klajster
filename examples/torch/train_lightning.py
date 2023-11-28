import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from model import pyramidnet

parser = argparse.ArgumentParser(description="cifar10 classification models")
parser.add_argument("--lr", default=0.1, help="")
parser.add_argument("--batch_size", type=int, default=768, help="")
parser.add_argument("--num_workers", type=int, default=4, help="")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=None, help="")
parser.add_argument("--num_nodes", type=int, default=1, help="")
parser.add_argument("--rootdir", type=str, help="")


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, rootdir: str):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rootdir = rootdir

        self.train_data = None
        self.transforms = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def setup(self, stage: str) -> None:
        self.train_data = CIFAR10(
            root=self.rootdir,
            train=True,
            download=True,
            transform=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


class Model(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.net = pyramidnet()
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

    def training_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        inputs, targets = batch

        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = outputs.max(1)
        acc = 100 * (predicted == targets).sum().item() / targets.size(0)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)

        return {"loss": loss, "accuracy": acc}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.gpu_devices])

    dm = CIFARDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rootdir=args.rootdir,
    )
    model = Model(lr=args.lr)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=len(args.gpu_devices),
        strategy="ddp",
        max_epochs=5,
        num_nodes=args.num_nodes,
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
