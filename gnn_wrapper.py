import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf

from gnn_model import corr_gnn


def get_optimizer (cfg, model: torch.nn.Module):
    param_group = list(model.parameters())
    if cfg["type"] == "adam":
        return torch.optim.Adam(
            param_group,
            lr = cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"]
            )
        )
    elif cfg["type"] == "adamw":
        return torch.optim.AdamW(
            param_group,
            lr = cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"]
            )
        )
    else:
        raise ValueError (f"Unknwown optimizer type: {cfg['type']}")

def get_lr_scheduler (cfg, optimizer: torch.optim.Optimizer):
    if cfg["type"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor = cfg["factor"],
            patience = cfg["patience"],
            min_lr = cfg["min_lr"]
        )
    else:
        raise ValueError(f"Unknown lr scheduler type : {cfg['type']}")

class corr_gnn_wrapper(pl.LightningModule):
    def __init__ (self, config):
        super().__init__()
        self.save_hyperparameters({
            "config" : OmegaConf.to_container(config)
        })
        self.model = corr_gnn(
            in_dimension=config.model.in_dimension,
            hidden_dimension=config.model.hidden_dimension,
            out_dimension=config.model.out_dimension,
            T=config.model.T
        )
        
    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        if "scheduler" in self.config.train:
            scheduler = get_lr_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler,
                "monitor" : "val/loss",
            }
        return optimizer
    
    # Training steps
    def training_step(self, batch, batch_idx):

        pred = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )
        batch.y = batch.y.view(-1, 1)
        loss = F.mse_loss(pred, batch.y)
        batch_size = batch.num_graphs

        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
        )

        return loss
    
    # Validation steps
    def validation_step(self, batch, batch_idx):
        
        pred = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )
        batch.y = batch.y.view(-1, 1)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)
        batch_size = batch.num_graphs

        self.log(
            "val/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size,
            sync_dist=True
        )

        self.log(
            "val/mae", mae,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size,
            sync_dist=True
        )

        return loss