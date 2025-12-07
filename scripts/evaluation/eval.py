# TODO: REWORK

from typing import Any, Dict, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader


def evaluate_classification(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    dataloader: Optional[DataLoader],
) -> Optional[Dict[str, Any]]:
    """
    Run validation for a classification model.
    """
    if dataloader is None:
        return None

    results = trainer.validate(model=model, dataloaders=dataloader, verbose=False)
    if isinstance(results, list) and results:
        return results[0]
    return {}
