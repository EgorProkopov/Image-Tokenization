import torch
import torch.nn as nn

from src.tokenizers.fft.mfft_tokenizer import MFFTTokenizer
from src.models.vit import TransformerEncoder
from src.lightning_modules.classification_lightning_modules import CustomClassificationLightningModule


class MFFTViT(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            filter_size: int = 128,
            energy_ratio: float = 0.900,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = MFFTTokenizer(
            in_channels=in_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            filter_size=filter_size,
            energy_ratio=energy_ratio
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        mfft_output = self.tokenizer(tensor)
        tokens = mfft_output["tokens"]
        filter_size = mfft_output["filter_size"]
        x = self.transformer_encoder(tokens)
        x = x[:, 0]
        logits = self.classifier(x)
        return {"logits": logits, "filter_size": filter_size}


class MFFTViTLightingModule(CustomClassificationLightningModule):
    def __init__(
        self,
        model_hparams,
        tokenizer_hparams,
        criterion,
        lr,
        log_step=1000,
        max_epochs=1,
        warmup_steps=0,
        lr_gamma=0.80,
    ):
        model = MFFTViT(
            in_channels=tokenizer_hparams["in_channels"],
            pixel_unshuffle_scale_factors=tokenizer_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            filter_size=tokenizer_hparams["filter_size"],
            energy_ratio=tokenizer_hparams["energy_ratio"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"],
        )
        super().__init__(
            model,
            criterion,
            lr,
            n_classes=model_hparams["n_classes"],
            log_step=log_step,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            lr_gamma=lr_gamma,
        )
        self.save_hyperparameters()

    def forward(self, x):
        mfft_output = self.model(x)
        return mfft_output

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        mfft_output = self.forward(images)
        logits = mfft_output["logits"]
        filter_size = mfft_output["filter_size"]

        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_filter_size", filter_size, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
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

        images, labels = batch

        mfft_output = self.forward(images)
        logits = mfft_output["logits"]
        filter_size = mfft_output["filter_size"]

        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_filter_size", filter_size, prog_bar=True)
        return loss
