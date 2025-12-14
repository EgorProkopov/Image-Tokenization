import os

from typing import Optional
from omegaconf import DictConfig 

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from clearml import Task
from clearml import Logger as ClearMLLogger

from src.lightning_modules.classification_lightning_modules import CustomClassificationLightningModule
from src.utils import set_seed


def _init_clearml_task(
        project_name: str,
        task_name: str,
) -> Optional[Task]:
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
    ) 
    # TODO: логирование конфигов?
    return task


def train_classification(
    model: CustomClassificationLightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,

    training_config: DictConfig,
):
    cleaml_project_name = training_config['clearml']['project_name']
    clearml_task_name = training_config['clearml']['task_name']

    seed = training_config["seed"]

    accelerator = training_config['training']['accelerator']
    devices = training_config['training']['devices']
    strategy = training_config['training']['strategy']
    max_epochs = training_config['training']['max_epochs']


    log_every_n_steps = training_config['logging']['log_every_n_steps']
    val_check_interval = training_config['logging']['val_check_interval']
    checkpoints_dir = training_config['logging']['checkpoints_dir']
    log_dir = training_config['logging']['log_dir']

    set_seed(seed=seed)

    clearml_task = _init_clearml_task(
        project_name=cleaml_project_name,
        task_name=clearml_task_name
    )

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,          
        name=clearml_task_name,    
        version=None,              
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(checkpoints_dir, cleaml_project_name, clearml_task_name),
        filename="{epoch}-{step}",
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
    ]

    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=tb_logger,  # TODO: добавить логгер  
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        default_root_dir=log_dir,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer

