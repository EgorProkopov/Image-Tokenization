import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score


class CustomClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        criterion,
        lr,
        n_classes,
        log_step=1000,
        max_epochs=1,
        warmup_steps=0,
        lr_gamma=0.80,
    ):
        """
        Base class for custom classification lightning models

        Args:
         - model:  ML model
         - criterion: loss function
         - lr: learning rate
         - n_classes: num_classes
         - max_epochs: total epochs for exponential decay schedule
         - warmup_steps: number of warmup steps before decay
         - lr_gamma: exponential decay factor
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.log_step = log_step
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.lr_gamma = lr_gamma

        self.train_accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.train_precision = Precision(num_classes=n_classes, average='macro', task="multiclass")
        self.train_recall = Recall(num_classes=n_classes, average='macro', task="multiclass")
        self.train_f1 = F1Score(num_classes=n_classes, average='macro', task="multiclass")

        self.val_accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.val_precision = Precision(num_classes=n_classes, average='macro', task="multiclass")
        self.val_recall = Recall(num_classes=n_classes, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=n_classes, average='macro', task="multiclass")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        self.model = self.model.train()

        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)

        preds = torch.argmax(outputs, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        if self.global_step % self.log_step == 0 and self.global_step != 0:
            acc = self.train_accuracy.compute()
            prec = self.train_precision.compute()
            rec = self.train_recall.compute()
            f1 = self.train_f1.compute()

            self.log("train_accuracy", acc, prog_bar=True)
            self.log("train_precision", prec, prog_bar=True)
            self.log("train_recall", rec, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)

            self.train_accuracy.reset()
            self.train_precision.reset()
            self.train_recall.reset()
            self.train_f1.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]
        self.model = self.model.eval()

        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1 = self.val_f1.compute()

        self.log("val_accuracy_epoch", acc, prog_bar=True)
        self.log("val_precision_epoch", prec, prog_bar=True)
        self.log("val_recall_epoch", rec, prog_bar=True)
        self.log("val_f1_epoch", f1, prog_bar=True)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """
        AdamW with linear warmup and exponential decay
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.max_epochs <= 0:
            return optimizer

        schedulers = []
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps
            )
            schedulers.append({
                "scheduler": warmup,
                "interval": "step",
                "frequency": 1,
                "name": "linear_warmup_lr",
            })

        exp_decay = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.lr_gamma
        )
        schedulers.append({
            "scheduler": exp_decay,
            "interval": "epoch",
            "frequency": 1,
            "name": "exponential_lr",
        })

        return [optimizer], schedulers

    def on_before_optimizer_step(self, optimizer):
        """
        Logs global gradient norm.
        """
        if self.global_step == 0 or self.global_step % self.log_step != 0:
            return

        grads = [
            p.grad.detach()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not grads:
            return

        stacked_norms = torch.stack([g.norm(2) for g in grads])
        total_norm = torch.norm(stacked_norms, 2)
        self.log(
            "grad_norm",
            total_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
