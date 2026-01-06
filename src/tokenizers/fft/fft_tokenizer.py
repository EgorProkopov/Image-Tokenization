import torch
import torch.fft
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding
from src.tokenizers.vit.vit_tokenizer import PatchProjection


class FFTTokenizer(nn.Module):
    """
    FFT-based tokenizer that converts images into token embeddings.

    Pipeline: FFT -> low-pass filter -> crop -> power spectrum -> patch projection.
    The resulting patch tokens are prepended with a CLS token and enriched with
    sinusoidal positional encoding.
    """
    def __init__(
            self,
            image_size: int = 224, 
            in_channels: int = 3, 
            embedding_dim: int = 768, 
            filter_size: int = 96,
            patch_size: int = 16,
            norm_type: str ='l-infinity'
        ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels

        self.embedding_dim = embedding_dim
        self.filter_size = filter_size

        self.norm_type = norm_type

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

        self.patch_projection = PatchProjection(
            patch_size=patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embedding_dim,
        )


    def forward(self, images):
        """
        :param images: tensor [B, in_channels, image_size, image_size]
        :return: tokens tensor [B, num_tokens+1, embedding_dim] (плюс CLS-токен)
        """

        fft_images_shifted = self.compute_fft(images)
        filtered_fft_images = self.apply_low_pass_filter(
            fft_images_shifted, 
            self.filter_size, 
            norm_type=self.norm_type
        )
        cropped_fft_images = self.crop_fft_filtered(filtered_fft_images, self.filter_size)
        power_spectrum = self.power_spectrum(cropped_fft_images)
        # power_spectrum shape: [batch_size, in_channels, 2 * filter_size, 2 * filter_size]

        tokens = self.patch_projection(power_spectrum)

        B = images.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens_pe = self.positional_encoding(tokens)
        return tokens_pe

    @staticmethod
    def compute_fft(image):
        """
        :param image: Input image shape is [batch_size, in_channels, image_size, image_size]
        :return:
        """
        fft_image = torch.fft.fft2(image)
        fft_image_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))
        return fft_image_shifted

    @staticmethod
    def apply_low_pass_filter(fft_image_shifted, filter_size, norm_type='l-infinity'):
        device = fft_image_shifted.device
        batch_size, in_channels, height, width = fft_image_shifted.shape
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))

        center_y, center_x = height // 2, width // 2
        distance_x = torch.abs(x - center_x)
        distance_y = torch.abs(y - center_y)

        if norm_type == 'l2':
            distance = torch.sqrt(distance_x ** 2 + distance_y ** 2)
            low_pass_mask = distance <= filter_size
        elif norm_type == 'l-infinity':
            low_pass_mask = (distance_x <= filter_size) & (distance_y <= filter_size)
        else:
            raise ValueError("Invalid norm type. Choose 'l2' or 'l-infinity'.")

        low_pass_mask = low_pass_mask.to(torch.complex64)
        low_pass_mask = low_pass_mask.to(device)
        fft_filtered = fft_image_shifted * low_pass_mask
        return fft_filtered

    @staticmethod
    def crop_fft_filtered(fft_filtered, filter_size):
        batch_size, in_channels, height, width = fft_filtered.shape
        center_y, center_x = height // 2, width // 2

        cropped_fft = fft_filtered[:, :, center_y - filter_size:center_y + filter_size,
                      center_x - filter_size:center_x + filter_size]
        return cropped_fft

    @staticmethod
    def power_spectrum(fft_tensor):
        power_spec = torch.log1p(torch.abs(fft_tensor)).float()
        return power_spec

    @staticmethod
    def restore_image_from_fft(fft_filtered):
        fft_filtered_shifted_back = torch.fft.ifftshift(fft_filtered)
        reconstructed_image = torch.fft.ifft2(fft_filtered_shifted_back)
        return torch.abs(reconstructed_image)
