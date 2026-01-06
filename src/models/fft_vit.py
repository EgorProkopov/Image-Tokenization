import torch
import torch.nn as nn

from src.lightning_modules.classification_lightning_modules import CustomClassificationLightningModule
from src.tokenizers.fft.fft_tokenizer import FFTTokenizer
from src.models.vit import TransformerEncoder


class FFTViT(nn.Module):
    def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            filter_size: int = 64,
            embedding_dim: int = 768,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
            norm_type: str ='l-infinity'
    ):
        super().__init__()

        self.tokenizer = FFTTokenizer(
            image_size=image_size, 
            in_channels=in_channels, 
            embedding_dim=embedding_dim, 
            filter_size=filter_size,
            patch_size=patch_size,
            norm_type=norm_type
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
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits
    

class FFTViTLightingModule(CustomClassificationLightningModule):
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
        model = FFTViT(
            image_size=tokenizer_hparams["image_size"],
            filter_size=tokenizer_hparams["filter_size"],
            patch_size=tokenizer_hparams["patch_size"],
            in_channels=tokenizer_hparams["in_channels"],
            norm_type=tokenizer_hparams["norm_type"],

            embedding_dim=model_hparams["embedding_dim"],
            num_bins=model_hparams["num_bins"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
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
            lr_gamma=lr_gamma
        )
        self.save_hyperparameters()
