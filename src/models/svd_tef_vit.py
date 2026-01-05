import torch
import torch.nn as nn

from src.lightning_modules.classification_lightning_modules import CustomClassificationLightningModule
from src.losses.gated_losses import GatedL1Loss
from src.models.vit import TransformerEncoder
from src.tokenizers.tef.svd_tef_tokenizer import SVDTEFTokenizer


class SVDTEFViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 768,
        pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
        selection_mode: str = "full",
        top_k: int = 25,
        dispersion_threshold: float = 0.900,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 1024,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = SVDTEFTokenizer(
            in_channels=in_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            selection_mode=selection_mode,
            dispersion_threshold=dispersion_threshold,
            top_k=top_k
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        svd_tef_output = self.tokenizer(tensor)
        tokens = svd_tef_output['tokens']
        scores = svd_tef_output['scores']

        x = self.transformer_encoder(tokens)
        x = x[:, 0]
        logits = self.classifier(x)
        return {
            'logits': logits,
            'scores': scores
        }


class SVDTEFViTLightningModule(CustomClassificationLightningModule):
    def __init__(
        self,
        model_hparams,
        tokenizer_hparams,
        training_hparams,
        criterion,
        lr,
        log_step=1000,
        max_epochs=1,
        warmup_steps=0,
        lr_gamma=0.80,
    ):
        model = SVDTEFViT(
            in_channels=tokenizer_hparams["in_channels"],
            pixel_unshuffle_scale_factors=tokenizer_hparams["pixel_unshuffle_scale_factors"],
            selection_mode=tokenizer_hparams["selection_mode"],
            top_k=tokenizer_hparams["top_k"],
            dispersion_threshold=tokenizer_hparams["dispersion_threshold"],
            embedding_dim=model_hparams["embedding_dim"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"],
        )
        model = torch.compile(model=model)
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

        aux_criterion_name = training_hparams["auxiliary_criterion"]
        if aux_criterion_name == "gated_l1_loss":
            self.alpha = training_hparams["auxiliary_alpha"]
            self.aux_criterion = GatedL1Loss()
        else:
            raise ValueError(f"Unknown auxiliary criterion: {aux_criterion_name}, supported: [gated_l1_loss]")

        self.save_hyperparameters()

    def forward(self, x):
        svd_tef_output = self.model(x)
        return svd_tef_output
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        svd_tef_output = self.forward(images)
        logits = svd_tef_output["logits"]
        scores = svd_tef_output["scores"]
        gates = torch.nn.functional.sigmoid(scores)

        ce_loss = self.criterion(logits, labels)
        auxiliary_loss = self.alpha * self.aux_criterion(gates)
        loss = ce_loss + auxiliary_loss

        self.log("train_total_loss", loss, prog_bar=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True)
        self.log("train_auxiliary_loss", auxiliary_loss, prog_bar=True)

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
        images, labels = batch

        svd_tef_output = self.forward(images)
        logits = svd_tef_output["logits"]
        scores = svd_tef_output["scores"]
        gates = torch.nn.functional.sigmoid(scores)

        ce_loss = self.criterion(logits, labels)
        auxiliary_loss = self.alpha * self.aux_criterion(gates)
        loss = ce_loss + auxiliary_loss

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_total_loss", loss, prog_bar=False)
        self.log("val_ce_loss", ce_loss, prog_bar=False)
        self.log("val_auxiliary_loss", auxiliary_loss, prog_bar=False)
        return loss