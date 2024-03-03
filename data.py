from argparse import Namespace
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
import lightning as L

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


class CIFARDataModule(L.LightningDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__()
        self.opt = opt
        
        self.train_transform = v2.Compose(
            [
                v2.RandomRotation(90),
                v2.RandomHorizontalFlip(p=0.7),
                v2.RandomVerticalFlip(p=0.7),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            ]
        )

        self.val_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            ]
            
        )
        
    def setup(self, stage: str) -> None:
        self.train_dataset = CIFAR10(
            self.opt.data_dir,
            transform=self.train_transform
        )

        self.val_dataset = CIFAR10(
            self.opt.data_dir,
            train=False,
            transform=self.val_transform
        )
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.workers,
            persistent_workers=True,
            pin_memory=True
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1.5 * self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.workers,
            persistent_workers=True,
            pin_memory=True
        )
