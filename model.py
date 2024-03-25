from argparse import Namespace
from typing import Dict, Union, Tuple, Optional
import lightning as L

import torch, timm
from torch import nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from torchmetrics.classification import Accuracy

from data import im_to_patch


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout: float = 0.0) -> None:
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        inp_x = self.layer_norm(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm(x))
        
        return x
    

class VisionTranformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        patch_size: int,
        num_patches: int,
        dropout: float = 0.0
        ) -> None:
        
        super().__init__()
        
        self.patch_size = patch_size
        
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(
                embed_dim,
                hidden_dim, 
                num_heads,
                dropout=dropout
            ) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.rand(1, 1 + num_patches, embed_dim))
        
    def forward(self, x: Tensor) -> Tensor:
        x = im_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T+1]
        
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        cls = x[0]
        return self.mlp_head(cls)
        
        
class BaseViT(L.LightningModule):
    def __init__(self, opt: Namespace, model_kwargs: Optional[Dict]) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        
        self.opt = opt
        self.lr = opt.start_lr
        self.model = VisionTranformer(**model_kwargs)
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = AdamW(params=self.parameters(), lr=self.lr)
        optim_dict = {'optimizer': optimizer}
        if self.opt.lr_plateau:
            optim_dict['lr_scheduler'] = {
                'scheduler': ReduceLROnPlateau(
                    optimizer, 
                    patience=self.opt.scheduler_patience, 
                    verbose=True, 
                    threshold=0.005,
                    threshold_mode='abs'
                    ),
                'monitor': 'val_loss'
            }
        else:
            lambda1 = lambda epoch: self.opt.rate_decay ** epoch
            optim_dict['lr_scheduler'] = LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
            
        return optim_dict
    
    def _calculate_loss(self, batch: Tensor) -> Tuple[Tensor]:
        ims, labels = batch
        preds = self(ims)
        
        loss = self.loss(preds, labels)
        acc = self.accuracy(preds, labels)
        
        return loss, acc
    
    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        train_out = self._calculate_loss(batch)
        return {'loss': train_out[0], 'metrics': train_out[1]}
    
    def on_train_batch_end(self, outputs: Dict[str, Tensor], batch: Tensor, batch_idx: int) -> None:
        self.log('step_loss', outputs['loss'], prog_bar=True)
        self.log('step_metrics', outputs['metrics'])
        
        if not self.train_step_outputs:
            self.train_step_outputs = {
                'step_loss': [outputs['loss']],
                'step_metrics': [outputs['metrics']]
            }
        else:
            self.train_step_outputs['step_loss'].append(outputs['loss'])
            self.train_step_outputs['step_metrics'].append(outputs['metrics'])
            
    def on_train_epoch_end(self) -> None:
        _log_dict = {
            'Loss/loss': torch.tensor(self.train_step_outputs['step_loss']).mean(),
            'Metrics/accuracy': torch.tensor(self.train_step_outputs['step_metrics']).mean()
        }
        
        self.loggers[0].log_metrics(_log_dict, self.current_epoch)
        
    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        valid_out = self._calculate_loss(batch)
        return {'val_loss': valid_out[0], 'metrics': valid_out[1]}
        
    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log('step_val_loss', outputs['val_loss'], prog_bar=True)
        self.log('step_val_metrics', outputs['metrics'])
        
        if not self.valid_step_outputs:
            self.valid_step_outputs = {
                'step_val_loss': [outputs['val_loss']],
                'step_val_metrics': [outputs['metrics']]
            }
        else:
            self.valid_step_outputs['step_val_loss'].append(outputs['val_loss'])
            self.valid_step_outputs['step_val_metrics'].append(outputs['metrics'])
    
    def on_validation_epoch_end(self) -> None:
        
        mean_loss_value = torch.tensor(self.valid_step_outputs['step_val_loss']).mean()
        mean_metrics_value = torch.tensor(self.valid_step_outputs['step_val_metrics']).mean()
        
        _log_dict = {
            'Loss/loss': mean_loss_value,
            'Metrics/accuracy': mean_metrics_value
        }
        
        self.loggers[1].log_metrics(_log_dict, self.current_epoch)
        
        self.log('val_loss', mean_loss_value)
        self.log('val_Accuracy', mean_metrics_value)
        
        
class ViT(BaseViT):
    
    def __init__(self, opt: Namespace, model_kwargs: Optional[Dict]) -> None:
        super().__init__(opt, model_kwargs)
        
        self.model = timm.create_model(
            # 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 
            'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
            # pretrained=True,
            in_chans=self.opt.in_channels,
            img_size=self.opt.model_input_size,
            num_classes=10
        )
        
    def forward(self, x):
        return self.model(x)