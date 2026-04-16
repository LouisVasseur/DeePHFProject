import sys

sys.path.append(".")
import os
from datetime import datetime

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
from gnn_wrapper import corr_gnn_wrapper
from gnn_dataloader import pyg_loader

@click.command()
@click.option("--config-path", "-c", type=str, default="config_train_conv_gnn.yml")
@click.option("--exp-name", "-n", type=str, default="debug")
@click.option("--resume", "-r", type=str, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", "-b", type=int, default=32)
@click.option("--train-data-path", type=str, default=None)
@click.option("--val-data-path", type=str, default=None)
@click.option("--num-workers", type=int, default=None)
@click.option("--val-freq", type=int, default=None, help="Override val_check_interval.")
@click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--num-sanity-val-steps", type=int, default=1)
@click.option(
    "--log-dir", type=click.Path(dir_okay=True, file_okay=False), default="./logs"
)
def main(
    config_path: str,
    exp_name: str,
    resume: str,
    seed: int,
    batch_size: int,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    val_freq: int,
    num_nodes: int,
    num_sanity_val_steps: int,
    log_dir: str,
):
    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    cfg = OmegaConf.load(config_path)

    if resume is not None:
        run_name = resume
    else:
        run_name = f"{exp_name}_{datetime.now().strftime('%Y-%m%d-%H%M')}"

    os.environ["WANDB_RUN_ID"] = run_name
    logger = loggers.WandbLogger(project="deephf", name=run_name, save_dir=log_dir)
    save_dir = os.path.join(log_dir, run_name, "checkpoints")

    ckpt_path = None
    if os.path.exists(save_dir):
        filenames = [f for f in os.listdir(save_dir) if f.endswith(".ckpt")]
        if filenames:
            last_filename = sorted(
                filenames,
                key=lambda x: int(x.replace(".ckpt", "").replace("step=", "")),
            )[-1]
            ckpt_path = os.path.join(save_dir, last_filename)
            print(f"Resuming from {ckpt_path}")

    model = corr_gnn_wrapper(config=cfg)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True),
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=cfg.train.max_grad_norm,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=-1,
                filename="{step}",
                every_n_train_steps=cfg.train.val_freq,
                save_on_train_epoch_end=False,
            ),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        val_check_interval=cfg.train.val_freq,
    )

    train_dataloader = pyg_loader(dataset_path=train_data_path,
                               shuffle=True,
                               batch_size=batch_size,
                               num_workers=num_workers)
    
    val_dataloader = pyg_loader(dataset_path=val_data_path,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers)
    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
   


