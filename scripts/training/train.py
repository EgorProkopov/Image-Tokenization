import os

from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

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


def prepare_dataloaders(
    image_size: int = 224,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    num_workers: int = 4,
    dataset_name: str = "benjamin-paine/imagenet-1k",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val dataloaders from the HuggingFace dataset.

    ImageNet-1k: benjamin-paine/imagenet-1k
    ImageNet-100: clane9/imagenet-100
    EuroSAT RGB-10: blanchon/EuroSAT_RGB
    """

    dataset = load_dataset(dataset_name)

    train_split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    val_split_name = "validation" if "validation" in dataset else "val"
    if val_split_name not in dataset:
        raise ValueError(f"Validation split not found in dataset '{dataset_name}'.")

    train_hf_ds = dataset[train_split_name]
    val_hf_ds = dataset[val_split_name]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    
    )
    train_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            image = sample["image"].convert("RGB")
            label = torch.tensor(sample["label"], dtype=torch.long)
            return self.transform(image), label

    train_dataset = HFDataset(train_hf_ds, train_transform)
    val_dataset = HFDataset(val_hf_ds, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


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

