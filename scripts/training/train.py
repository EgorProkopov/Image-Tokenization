from typing import Any, Dict, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import ClearMLLogger
from torch.utils.data import DataLoader

from scripts.evaluation.eval import evaluate_classification
from src.lightning_modules.classification_lightning_modules import (
    CustomClassificationLightningModule,
)


def _init_clearml_logger(
    project_name: str, task_name: str, hyperparams: Optional[Dict[str, Any]] = None
) -> Tuple[Task, ClearMLLogger]:
    """
    Inits ClearML Task object and Logger object with hyperparameters attached.
    """
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = ClearMLLogger(task=task)
    if hyperparams:
        logger.log_hyperparams(hyperparams)
        task.connect(hyperparams)
    return task, logger


def train_classification(
    model: CustomClassificationLightningModule,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    *,
    max_epochs: int,
    logging_train_step: int,
    logging_eval_step: int,
    checkpoint_step: int,
    clearml_project: str,
    clearml_task_name: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    accelerator: str = "auto",
    devices: Optional[Union[int, str, Sequence[int]]] = None,
    default_root_dir: Optional[str] = None,
    enable_progress_bar: bool = True,
) -> Tuple[pl.Trainer, Optional[Dict[str, Any]]]:
    """
    Trains a classification model
    """
    _, logger = _init_clearml_logger(
        project_name=clearml_project, task_name=clearml_task_name, hyperparams=hyperparams
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=checkpoint_step,
        save_top_k=-1,
        save_last=True,
        filename="step-{step}",
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=logging_train_step,
        val_check_interval=logging_eval_step,
        enable_checkpointing=True,
        default_root_dir=default_root_dir,
        enable_progress_bar=enable_progress_bar,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    evaluation = evaluate_classification(trainer, model, val_dataloader)
    return trainer, evaluation
