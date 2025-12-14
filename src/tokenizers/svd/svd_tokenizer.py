import torch
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding


class SVDLinearTokenizer(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim, dispersion=0.9, full_matrices=True):
        super().__init__()

        self.dispersion = dispersion
        self.full_matrices = full_matrices

        self.embedder = nn.Linear(in_features=image_size * 2 * num_channels, out_features=embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def get_approx_svd(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Input image shape: [channels, image_width, image_height]
        """

        U, S, Vh = torch.linalg.svd(image, full_matrices=self.full_matrices)
        V = Vh.mH

        S_lin = S.sum(dim=0, keepdim=True)

        S_cumsum = torch.cumsum(S_lin, dim=1)
        total_sum = S_lin.sum()
        threshold = self.dispersion * total_sum

        mask = (S_cumsum >= threshold).squeeze()
        mask = torch.where(mask)[0]

        if mask.numel() > 0:
            rank = mask[0].item() + 1
        else:
            rank = S_lin.shape[1]

        approx_U, approx_S, approx_V = U[:, :, :rank], S[:, :rank], V[:, :, :rank]
        # print(rank)
        return approx_U, approx_S, approx_V, rank

    def batched_get_approx_svd(self, batch: torch.Tensor):
        return list(map(self.get_approx_svd, batch))  # Apply SVD to each image in batch

    def get_raw_embeddings(self, approx_U: torch.Tensor, approx_S: torch.Tensor, approx_V: torch.Tensor) -> torch.Tensor:
        """
        Convert the approximate SVD outputs to a raw embeddings (token sequence).
        Input shapes:
          approx_U: [channels, image_size, rank]
          approx_S: [channels, rank]
          approx_V: [channels, image_size, rank]
        Returns:
          raw_embedding: Tensor of shape [tokens, token_dim]
          where tokens = channels * rank and token_dim = image_size * 2.
        """

        # New shapes: [channels, rank, image_size]
        approx_U = approx_U.transpose(1, 2)
        approx_V = approx_V.transpose(1, 2)

        approx_S_root = torch.sqrt(approx_S)

        approx_US = approx_U * approx_S_root.unsqueeze(-1)  # shape: [channels, rank, image_size]
        approx_SV = approx_V * approx_S_root.unsqueeze(-1)  # shape: [channels, rank, image_size]

        # Shapes: [rank, channels, image_size]
        approx_US = approx_US.transpose(0, 1)
        approx_SV = approx_SV.transpose(0, 1)

        #  Shape: [rank, channels, image_size * 2]
        raw_embedding = torch.cat([approx_US, approx_SV], dim=-1)

        # Shape: [rank, image_size * 2 * channels]
        channels = raw_embedding.shape[1]
        rank = raw_embedding.shape[0]
        token_dim = raw_embedding.shape[-1]
        raw_embedding = raw_embedding.reshape(rank, token_dim * channels)

        return raw_embedding

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Input image shape: [channels, image_size, image_size]

        Output embeddings shape: [rank * channels (len of tokens sequence), image_size * 2]
        """
        approx_U, approx_S, approx_V, _ = self.get_approx_svd(image)
        return self.get_raw_embeddings(approx_U, approx_S, approx_V)

    def pad_tokens(self, raw_embebeddings, max_tokens):
        n_tokens = raw_embebeddings.shape[0]
        token_dim = raw_embebeddings.shape[-1]
        if n_tokens < max_tokens:
            pad_tensor = torch.zeros(max_tokens - n_tokens, token_dim, device=raw_embebeddings.device)
            return torch.cat([raw_embebeddings, pad_tensor], dim=0)
        return raw_embebeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input image shape: [batch_size, channels, image_size, image_size]
        Output sequence shape: [batch_size, rank * channels (sequence length), token_dim]
        """
        raw_embeddings_list = list(map(self.process_image, images))
        token_counts = list(map(lambda x: x.shape[0], raw_embeddings_list))
        max_tokens = max(token_counts)
        padded_embeddings_list = list(map(self.pad_tokens, raw_embeddings_list, [max_tokens] * len(raw_embeddings_list)))
        batched_raw_embeddings = torch.stack(padded_embeddings_list, dim=0)

        embeddings = self.embedder(batched_raw_embeddings)

        batch_size = images.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # [batch_size, max_tokens+1, embedding_size]

        positional_encoding = self.positional_encoding(embeddings)
        embeddings = embeddings + positional_encoding
        return embeddings

    @staticmethod
    def reconstruct_image(approx_U, approx_S, approx_V):
        US = approx_U * approx_S.unsqueeze(1)  # Shape: [channels, image_size, rank]
        Vt = approx_V.transpose(1, 2)
        approx_image = torch.bmm(US, Vt)

        return approx_image


class SVDSquareTokenizer(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim, dispersion=0.9, full_matrices=True):
        super().__init__()

        self.dispersion = dispersion
        self.full_matrices = full_matrices

        self.embedder = nn.Linear(in_features=image_size * 2 * num_channels, out_features=embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def get_approx_svd(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Input image shape: [channels, image_width, image_height]
        """

        U, S, Vh = torch.linalg.svd(image, full_matrices=self.full_matrices)
        V = Vh.mH

        S_squared = torch.square(S)
        S_squared = S_squared.sum(dim=0, keepdim=True)

        S_cumsum = torch.cumsum(S_squared, dim=1)
        total_sum = S_squared.sum()
        threshold = self.dispersion * total_sum

        mask = (S_cumsum >= threshold).squeeze()
        mask = torch.where(mask)[0]

        if mask.numel() > 0:
            rank = mask[0].item() + 1
        else:
            rank = S_squared.shape[1]

        approx_U, approx_S, approx_V = U[:, :, :rank], S[:, :rank], V[:, :, :rank]

        return approx_U, approx_S, approx_V, rank

    def batched_get_approx_svd(self, batch: torch.Tensor):
        return list(map(self.get_approx_svd, batch))  # Apply SVD to each image in batch

    def get_raw_embeddings(self, approx_U: torch.Tensor, approx_S: torch.Tensor, approx_V: torch.Tensor) -> torch.Tensor:
        """
        Convert the approximate SVD outputs to a raw embeddings (token sequence).
        Input shapes:
          approx_U: [channels, image_size, rank]
          approx_S: [channels, rank]
          approx_V: [channels, image_size, rank]
        Returns:
          raw_embedding: Tensor of shape [tokens, token_dim]
          where tokens = channels * rank and token_dim = image_size * 2.
        """

        # Shapes: [channels, rank, image_size]
        approx_U = approx_U.transpose(1, 2)
        approx_V = approx_V.transpose(1, 2)

        approx_S_root = torch.sqrt(approx_S)

        approx_US = approx_U * approx_S_root.unsqueeze(-1)  # shape: [channels, rank, image_size]
        approx_SV = approx_V * approx_S_root.unsqueeze(-1)  # shape: [channels, rank, image_size]

        # Shapes: [rank, channels, image_size]
        approx_US = approx_US.transpose(0, 1)
        approx_SV = approx_SV.transpose(0, 1)

        #  Shape: [rank, channels, image_size * 2]
        raw_embedding = torch.cat([approx_US, approx_SV], dim=-1)

        # Shape: [rank, image_size * 2 * channels]
        channels = raw_embedding.shape[1]
        rank = raw_embedding.shape[0]
        token_dim = raw_embedding.shape[-1]
        raw_embedding = raw_embedding.reshape(rank, token_dim * channels)

        print(raw_embedding.shape)

        return raw_embedding

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Input image shape: [channels, image_size, image_size]
        Output embeddings shape: [rank * channels (len of tokens sequence), image_size * 2]
        """
        approx_U, approx_S, approx_V, _ = self.get_approx_svd(image)
        return self.get_raw_embeddings(approx_U, approx_S, approx_V)

    def pad_tokens(self, raw_embebeddings, max_tokens):
        n_tokens = raw_embebeddings.shape[0]
        token_dim = raw_embebeddings.shape[-1]
        if n_tokens < max_tokens:
            pad_tensor = torch.zeros(max_tokens - n_tokens, token_dim, device=raw_embebeddings.device)
            return torch.cat([raw_embebeddings, pad_tensor], dim=0)
        return raw_embebeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input image shape: [batch_size, channels, image_size, image_size]
        Output sequence shape: [batch_size, rank * channels (sequence length), token_dim]
        """
        raw_embeddings_list = list(map(self.process_image, images))
        token_counts = list(map(lambda x: x.shape[0], raw_embeddings_list))
        max_tokens = max(token_counts)
        padded_embeddings_list = list(map(self.pad_tokens, raw_embeddings_list, [max_tokens] * len(raw_embeddings_list)))
        batched_raw_embeddings = torch.stack(padded_embeddings_list, dim=0)

        embeddings = self.embedder(batched_raw_embeddings)

        batch_size = images.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # Shape: [batch_size, max_tokens+1, embedding_size]

        positional_encoding = self.positional_encoding(embeddings)
        embeddings = embeddings + positional_encoding
        return embeddings

    @staticmethod
    def reconstruct_image(approx_U, approx_S, approx_V):
        US = approx_U * approx_S.unsqueeze(1)  # Shape: [channels, image_size, rank]
        Vt = approx_V.transpose(1, 2)
        approx_image = torch.bmm(US, Vt)

        return approx_image


