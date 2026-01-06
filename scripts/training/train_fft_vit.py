import os
import dotenv

from omegaconf import DictConfig, OmegaConf

import torch

from scripts.training.train import train_classification, prepare_dataloaders
from src.models.fft_vit import FFTViTLightingModule


def train_fft_vit(
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
        'in_channels': tokenizer_config['in_channels'],
        'filter_size': tokenizer_config['filter_size'],
        'norm_type': tokenizer_config['norm_type']
    }

    cross_entropy_criterion = torch.nn.CrossEntropyLoss()

    lr = training_config['training']['lr']
    max_epochs = training_config['training']['max_epochs']
    warmup_steps = training_config['training']['warmup_steps']
    lr_gamma = training_config['training']['lr_gamma']

    model = FFTViTLightingModule(
        model_hparams=model_hparams,
        tokenizer_hparams=tokenizer_hparams,
        criterion=cross_entropy_criterion,
        lr=lr,
        log_step=training_config['logging']['log_every_n_steps'],
        max_epochs=max_epochs,
        warmup_steps=warmup_steps,
        lr_gamma=lr_gamma,
    )

    train_dataloader, val_dataloader = prepare_dataloaders(
        image_size=tokenizer_config['image_size'],
        train_batch_size=training_config['training']['train_batch_size'],
        val_batch_size=training_config['training']['val_batch_size'],
        num_workers=training_config['training']['num_workers'],
        dataset_name="clane9/imagenet-100"
    )

    train_classification(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=training_config,
        clearml_configs={
            "model_config": model_config,
            "tokenizer_config": tokenizer_config,
            "training_config": training_config,
        }
    )


if __name__ == "__main__":
    dotenv.load_dotenv()

    CONFIGS_DIR = os.getenv("CONFIGS_DIR")
    model_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "models", "vit_base.yaml"))
    tokenizer_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "tokenizers", "fft.yaml"))
    training_config = OmegaConf.load(os.path.join(CONFIGS_DIR, "training", "fft_vit.yaml"))

    train_fft_vit(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        training_config=training_config
    )
