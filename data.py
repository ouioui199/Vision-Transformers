from argparse import Namespace
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
import lightning as L

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, InterpolationMode
from torchvision import transforms


class CIFARDataModule(L.LightningDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__()
        self.opt = opt
        
        # self.train_transform = v2.Compose(
        #     [
        #         v2.RandomRotation(90),
        #         v2.RandomHorizontalFlip(p=0.7),
        #         v2.RandomVerticalFlip(p=0.7),
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        #     ]
        # )

        # self.val_transform = v2.Compose(
        #     [
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #         v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        #     ]
        # )
        
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
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
            batch_size=int(1.5*self.opt.batch_size),
            shuffle=False,
            num_workers=self.opt.workers,
            persistent_workers=True,
            pin_memory=True
        )


def im_to_patch(im, patch_size, flatten_channels: bool = True):
    B, C, H, W = im.shape
    assert H // patch_size == 0 and W // patch_size == 0, f"Image height and width are {H, W}, which is not a multiple of the patch size"
    
    im = im.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    im = im.permute(0, 2, 4, 1, 3, 5)
    im = im.flatten(1, 2)
    
    if flatten_channels:
        return im.flatten(2, 4)
    else:
        return im