import math

import torch
import torch.nn as nn
import torch.fft as fft
from typing import Optional, List, Dict, Any

from src.tokenizers.positional_encoding import PositionalEncoding


class FFTLowFreqFilter(nn.Module):
    def __init__(
        self,
        filter_size: int = 0,
        energy_ratio: float = 0.9,
        downscale_factors: Optional[List[int]] = None,
        eps: float = 1e-8
    ):
        super().__init__()
        self.filter_size = filter_size
        self.energy_ratio = energy_ratio

        self.downscale_factors = downscale_factors or []
        self.total_downscale = math.prod(self.downscale_factors)
        self.eps = eps

    def compute_filter_size(self, power: torch.Tensor) -> int:
        B, W, H = power.shape
        cumsum_h = torch.cumsum(power, dim=1)        # [B, W, H]
        integral = torch.cumsum(cumsum_h, dim=2)     # [B, W, H]
        total_energy = integral[:, -1, -1]           # [B]
        max_k = min(W, H)
        cx, cy = W // 2, H // 2

        for fs in range(self.total_downscale, max_k + 1, self.total_downscale):
            half = fs // 2
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + fs, y0 + fs

            A = integral[:, x1-1, y1-1]
            B_ = integral[:, x0-1, y1-1] if x0 > 0 else 0
            C = integral[:, x1-1, y0-1] if y0 > 0 else 0
            D = integral[:, x0-1, y0-1] if x0 > 0 and y0 > 0 else 0
            low_energy = A - B_ - C + D  # [B]
            if torch.all(low_energy / (total_energy + self.eps) >= self.energy_ratio):
                return fs

        return (max_k // self.total_downscale) * self.total_downscale or self.total_downscale

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        F = fft.fft2(x, norm='ortho')              # [B, 3, W, H]
        F = fft.fftshift(F, dim=(-2, -1))
        real, imag = F.real, F.imag

        signed_log = lambda v: torch.sign(v) * torch.log1p(v.abs() + self.eps)
        log_real = signed_log(real)
        log_imag = signed_log(imag)

        fs = self.filter_size
        if fs == 0:
            power_spectrum = log_real.pow(2) + log_imag.pow(2)      # [B, 3, W, H]
            energy = power_spectrum.sum(dim=1)              # [B, W, H]
            fs = self.compute_filter_size(energy)

        freq_cat = torch.cat([log_real, log_imag], dim=1)  # [B, 6, W, H]

        B, C, W, H = freq_cat.shape
        half = fs // 2
        cx, cy = W // 2, H // 2
        x0, y0 = cx - half, cy - half
        low_freq = freq_cat[:, :, x0:x0 + fs, y0:y0 + fs]

        return {"tensor": low_freq, "filter_size": fs}


class MFFTTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        pixel_unshuffle_scale_factors: List[int] = [2, 2, 2, 2],
        embedding_dim: int = 768,
        filter_size: int = 0,
        energy_ratio: float = 0.9
    ):
        super().__init__()

        self.low_freq_filter = FFTLowFreqFilter(
            filter_size=filter_size,
            energy_ratio=energy_ratio,
            downscale_factors=pixel_unshuffle_scale_factors
        )
        self.in_channels = in_channels
        self.pixel_unshuffle_scale_factors = pixel_unshuffle_scale_factors
        self.embedding_dim = embedding_dim

        layers = []
        current_channels = in_channels * 2
        for scale in self.pixel_unshuffle_scale_factors:
            layers += [
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels),
                nn.LeakyReLU(),
                nn.PixelUnshuffle(downscale_factor=scale)
            ]
            current_channels *= scale ** 2
        self.feature_extractor = nn.Sequential(*layers)

        self.projection = nn.Conv2d(
            current_channels, current_channels, kernel_size=1, bias=False
        )
        self.linear_projection = nn.Linear(current_channels, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def _add_cls_token(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.to(tokens.device).expand(tokens.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)

    def _add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.positional_encoding(tokens)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        low = self.low_freq_filter(x)
        low_x, fs = low["tensor"], low["filter_size"]

        feat = self.feature_extractor(low_x)
        feat = self.projection(feat)
        feat = feat.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = feat.shape
        tokens = feat.view(B, H * W, C)
        tokens = self.linear_projection(tokens)
        tokens = self._add_cls_token(tokens)
        tokens = self._add_positional_encoding(tokens)

        return {"tokens": tokens, "filter_size": fs}

