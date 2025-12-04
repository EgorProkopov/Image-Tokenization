import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 64,
	):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self._embedder = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, tensor):
        x = self._embedder(tensor)
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class ViTTokenization(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_embedder = PatchProjection(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = (image_size // patch_size) ** 2

        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, tensor):
        x = self.patch_embedder(tensor)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embeddings
        return x
