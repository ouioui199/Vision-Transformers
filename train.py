from argparse import Namespace
from pathlib import Path

from torch import set_float32_matmul_precision

from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from data import CIFARDataModule
from model import BaseViT, ViT, HybridTransformer
from utils import argparser, CustomProgressBar, TBLogger

# set_float32_matmul_precision('high')

def launch_training(opt: Namespace) -> None:
    data_module = CIFARDataModule(opt)
    model_kwargs = {
        "embed_dim": 256, #embedded dim of the transformer model
        "hidden_dim": 512, #hidden dim of mlp layer in attention block
        "num_heads": 8, #number of self attention head
        "num_layers": 6, #number of attention layers
        "patch_size": 4, #size of an image patch after splitting
        "num_channels": opt.in_channels, #number of input image channel
        "num_patches": 64, #number of patches splitted from image
        "num_classes": 10, #number of class for the classification task
        "dropout": 0.2,
    }
    network = BaseViT(opt, model_kwargs)

    trainer = Trainer(
        max_epochs=opt.epochs,
        num_sanity_val_steps=0,
        benchmark=True,
        callbacks=[
            CustomProgressBar(), 
            EarlyStopping(
                monitor='val_loss', 
                verbose=True, 
                patience=opt.early_stopping
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                monitor='val_Accuracy', 
                verbose=True, 
                mode='max', 
                dirpath=Path('weights_storage/version_' + str(opt.version))
            )
        ],
        logger=[
            TBLogger("training_logs", name=None, version=opt.version, sub_dir='train'),
            TBLogger("training_logs", name=None, version=opt.version, sub_dir='valid')
        ]
    )
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(network, train_dataloaders=train_dataloader, num_training=50)
    # network.hparams.lr = lr_finder.suggestion()

    trainer.fit(network, datamodule=data_module)

if __name__ == '__main__':
    parser = argparser()
    opt = parser.parse_args()
    
    launch_training(opt)