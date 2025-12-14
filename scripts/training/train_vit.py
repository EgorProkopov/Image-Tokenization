import os
import dotenv

from typing import Tuple
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

from scripts.training.train import train_classification
from src.models.vit import ViTLightingModule


def prepare_dataloaders(
    image_size: int = 224,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    num_workers: int = 4,
    dataset_name: str = "benjamin-paine/imagenet-1k",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build ImageNet-1k train/val dataloaders from the HuggingFace dataset.
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


def train_vit(
    model_config: DictConfig,
    tokenizer_config: DictConfig,
    training_config: DictConfig
):
    #TODO: добавить ассерты между конфигами на совпадение одинаковых ключей

    model_hparams = {
        'embedding_dim': model_config['embedding_dim'],
        'qkv_dim': model_config['qkv_dim'],
        'mlp_hidden_size': model_config['mlp_hidden_size'],
        'n_layers': model_config['n_layers'],
        'n_heads': model_config['n_heads'],
        'n_classes': model_config['n_classes']
    }

    tokenizer_hparams = {
        'image_size': tokenizer_config['image_size'],
        'patch_size': tokenizer_config['patch_size'],
        'in_channels': tokenizer_config['in_channels']
    }

    cross_entropy_criterion = torch.nn.CrossEntropyLoss()

    lr = training_config['training']['lr']
    # TODO: ADD LR schedule

    model = ViTLightingModule(
        model_hparams=model_hparams,
        tokenizer_hparams=tokenizer_hparams,
        criterion=cross_entropy_criterion,
        lr=lr,
        log_step=training_config['logging']['log_every_n_steps']
    )

    train_dataloader, val_dataloader = prepare_dataloaders(
        image_size=tokenizer_config['image_size'],
        train_batch_size=training_config['training']['train_batch_size'],
        val_batch_size=training_config['training']['val_batch_size'],
        num_workers=training_config['training']['num_workers']
    )

    train_classification(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=training_config
    )


if __name__ == "__main__":
    dotenv.load_dotenv()

    CONFIGS_DIR = os.getenv("CONFIGS_DIR")
    model_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "models", "vit_base.yaml"))
    tokenizer_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "tokenizers", "vit.yaml"))
    training_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "training", "vit.yaml"))

    train_vit(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        training_config=training_config
    )