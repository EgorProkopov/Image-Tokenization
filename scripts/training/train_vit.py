import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.training.train import train_classification
from src.models.vit import ViTLightingModule
from src.utils import set_seed


def _parse_devices(devices: Optional[str]) -> Optional[Union[int, str, Sequence[int]]]:
    if devices is None:
        return None
    if devices.lower() == "auto":
        return "auto"
    if devices.isdigit():
        return int(devices)
    parsed = [int(item.strip()) for item in devices.split(",") if item.strip()]
    return parsed or None


def build_dataloaders(
    train_dir: Path,
    val_dir: Path,
    image_size: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir.as_posix(), transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir.as_posix(), transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, len(train_dataset.classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/vit.yaml",
        help="Path to training config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/models/vit.yaml",
        help="Path to model config.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to training images in ImageFolder format.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to validation images in ImageFolder format.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="image-tokenization",
        help="ClearML project name.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="vit-classification",
        help="ClearML task name.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of classes if it cannot be inferred from the dataset.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Lightning accelerator override (cpu, gpu, mps, auto).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Device specification (e.g. '0', '0,1', or 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=239,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default="checkpoints/vit",
        help="Where to store checkpoints and logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load(args.model_config)

    train_hparams = OmegaConf.to_container(train_cfg.get("train_hparams", {}), resolve=True)
    logging_cfg = OmegaConf.to_container(train_cfg.get("logging", {}), resolve=True)
    model_hparams = OmegaConf.to_container(model_cfg.get("model", {}), resolve=True)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Train or validation directory does not exist.")

    devices = _parse_devices(args.devices) or train_hparams.get("devices")
    accelerator = args.accelerator or train_hparams.get("accelerator", "auto")
    num_workers = int(train_hparams.get("num_workers", 0))

    train_loader, val_loader, inferred_classes = build_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=int(model_hparams["image_size"]),
        train_batch_size=int(train_hparams["train_batch_size"]),
        val_batch_size=int(train_hparams["val_batch_size"]),
        num_workers=num_workers,
    )

    model_hparams["n_classes"] = args.num_classes or model_hparams.get("n_classes") or inferred_classes

    criterion = torch.nn.CrossEntropyLoss()
    model = ViTLightingModule(
        model_hparams=model_hparams,
        criterion=criterion,
        lr=float(train_hparams["lr"]),
        log_step=int(logging_cfg["logging_train_step"]),
    )

    hyperparams = {
        "model": model_hparams,
        "train_hparams": train_hparams,
        "logging": logging_cfg,
        "data": {"train_dir": str(train_dir), "val_dir": str(val_dir)},
        "seed": args.seed,
    }

    train_classification(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=int(train_hparams["max_epoch"]),
        logging_train_step=int(logging_cfg["logging_train_step"]),
        logging_eval_step=int(logging_cfg["logging_eval_step"]),
        checkpoint_step=int(logging_cfg["checkpoint_step"]),
        clearml_project=args.project_name,
        clearml_task_name=args.task_name,
        hyperparams=hyperparams,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=args.default_root_dir,
    )


if __name__ == "__main__":
    main()
