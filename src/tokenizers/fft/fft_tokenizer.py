import torch
import torch.fft
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding


class FFTTokenizer(nn.Module):
    def __init__(self, image_size, num_channels=3, num_bins=16, embedding_dim=768, filter_size=96):
        super().__init__()

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_bins = num_bins

        self.embedding_dim = embedding_dim
        self.filter_size = filter_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

        total_pixels = (2 * self.filter_size) ** 2
        self.bin_size = total_pixels // self.num_bins

        self.sorted_linear_projection = nn.Linear(self.num_channels * self.bin_size, embedding_dim)


    def forward(self, images):
        """
        :param images: тензор [B, num_channels, image_size, image_size]
        :return: тензор токенов [B, num_tokens+1, embedding_dim] (плюс CLS-токен)
        """

        fft_images_shifted = self.compute_fft(images)
        filtered_fft_images = self.apply_low_pass_filter(fft_images_shifted, self.filter_size, norm_type='l-infinity')
        cropped_fft_images = self.crop_fft_filtered(filtered_fft_images, self.filter_size)
        power_spectrum = self.power_spectrum(cropped_fft_images)
        # power_spectrum shape: [batch_size, num_channels, 2 * filter_size, 2 * filter_size]

        tokens = self.tokenize_power_spectrum_by_sorting(power_spectrum, self.num_bins)
        # tokens shape: [batch_size, num_bins, embedding_dim]

        B = images.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        tokens = tokens + self.positional_encoding(tokens)
        return tokens

    def tokenize_power_spectrum_by_sorting(self, power_spectrum, num_bins):
        B, C, H, W = power_spectrum.shape
        total_pixels = H * W
        device = power_spectrum.device

        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        center_y, center_x = H / 2, W / 2
        distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)  # [H, W]
        distances_flat = distances.view(-1)  # [total_pixels]

        sorted_indices = distances_flat.argsort()  # [total_pixels]

        power_flat = power_spectrum.view(B, C, total_pixels)
        power_sorted = power_flat[:, :, sorted_indices]  # [batch_size, num_channels, total_pixels]

        bin_size = self.bin_size
        tokens_list = []
        for b in range(num_bins):
            start = b * bin_size
            end = start + bin_size
            bin_values = power_sorted[:, :, start:end]
            bin_flat = bin_values.reshape(B, -1)
            tokens_list.append(bin_flat)

        tokens = torch.stack(tokens_list, dim=1)
        tokens = self.sorted_linear_projection(tokens)
        return tokens

    @staticmethod
    def compute_fft(image):
        """
        :param image: Input image shape is [batch_size, num_channels, image_size, image_size]
        :return:
        """
        fft_image = torch.fft.fft2(image)
        fft_image_shifted = torch.fft.fftshift(fft_image)
        return fft_image_shifted

    @staticmethod
    def apply_low_pass_filter(fft_image_shifted, filter_size, norm_type='l2'):
        """
        fft_image_shifted shape is [batch_size, num_channels, image_size, image_size]
        :param fft_image_shifted:
        :param filter_size:
        :param norm_type: could be l2 or l-infinity
        :return:
        """
        device = fft_image_shifted.device
        batch_size, num_channels, height, width = fft_image_shifted.shape
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
        batch_size, num_channels, height, width = fft_filtered.shape
        center_y, center_x = height // 2, width // 2

        cropped_fft = fft_filtered[:, :, center_y - filter_size:center_y + filter_size,
                      center_x - filter_size:center_x + filter_size]
        return cropped_fft

    @staticmethod
    def power_spectrum(fft_tensor):
        power_spec = torch.abs(fft_tensor).float()
        return power_spec

    @staticmethod
    def restore_image_from_fft(fft_filtered):
        fft_filtered_shifted_back = torch.fft.ifftshift(fft_filtered)
        reconstructed_image = torch.fft.ifft2(fft_filtered_shifted_back)
        return torch.abs(reconstructed_image)
