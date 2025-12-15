import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # TODO: заменить на собственную реализацию паддинга

from src.tokenizers.positional_encoding import PositionalEncoding


class SVDNetworkTokenizer(nn.Module):
    def __init__(
            self,
            in_channels: int =3,
            pixel_unshuffle_scale_factors: list =[2,2,2,2],
            embedding_dim: int =768
    ):
        super().__init__()

        self.in_channels = in_channels
        self.pixel_unshuffle_scale_factors = pixel_unshuffle_scale_factors
        self.embedding_dim = embedding_dim

        # U feature extractor block initialization
        self.u_feature_extractor = nn.ModuleList()
        current_channels = self.in_channels
        for scale in self.pixel_unshuffle_scale_factors:
            self.u_feature_extractor.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                ),
            )
            self.u_feature_extractor.append(nn.BatchNorm2d(current_channels))
            self.u_feature_extractor.append(nn.LeakyReLU())
            self.u_feature_extractor.append(nn.PixelUnshuffle(downscale_factor=scale))
            current_channels = current_channels * (scale ** 2)
        self.u_feature_extractor = nn.Sequential(*self.u_feature_extractor)

        self.projection_u = nn.Conv2d(
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        
        # V feature extractor block initialization
        self.v_feature_extractor = nn.ModuleList()
        current_channels = self.in_channels
        for scale in self.pixel_unshuffle_scale_factors:
            self.v_feature_extractor.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                )
            )
            self.v_feature_extractor.append(nn.BatchNorm2d(current_channels))
            self.v_feature_extractor.append(nn.LeakyReLU())
            self.v_feature_extractor.append(nn.PixelUnshuffle(downscale_factor=scale))
            current_channels = current_channels * (scale ** 2)
        self.v_feature_extractor = nn.Sequential(*self.v_feature_extractor)

        self.projection_v = nn.Conv2d(
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        # final projection
        self.linear_projection = nn.Linear(
            in_features=current_channels * 2,
            out_features=self.embedding_dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.positional_encoding = PositionalEncoding(self.embedding_dim)

    def _get_raw_tokens(self, x):
        """
        Calculates raw u and v features (in tokens shapes)
        """

        # TODO: add residual connections
        raw_u = self.u_feature_extractor(x)
        raw_v = self.v_feature_extractor(x)

        projected_u = self.projection_u(raw_u)
        projected_v = self.projection_v(raw_v)

        projected_u = projected_u.permute(0, 2, 3, 1).contiguous()
        projected_v = projected_v.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = projected_u.shape
        tokens_u = torch.reshape(projected_u, (B, H * W, C))
        tokens_v = torch.reshape(projected_v, (B, H * W, C))

        raw_tokens = torch.cat([tokens_u, tokens_v], dim=-1)
        return raw_tokens
    
    def _add_cls_token(self, tokens):
        """
        Concats cls token with projected tokens
        """
        batch_size = tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens
    
    def _add_positional_encoding(self, tokens):
        """
        Adds sinusoidal positional encoding to projected tokens
        """
        positional_encoding = self.positional_encoding(tokens)
        tokens = tokens + positional_encoding
        return tokens
    
    def forward(self, x):
        raw_tokens = self._get_raw_tokens(x)

        tokens = self.linear_projection(raw_tokens)
        tokens = self._add_cls_token(tokens)
        tokens = self._add_positional_encoding(tokens)

        return tokens
    

class SVDTEFTokenizer(SVDNetworkTokenizer):
    def __init__(
        self,
        in_channels: int = 3,
        pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
        embedding_dim: int = 768,
        selection_mode: str = "full",
        top_k: int = None,
        dispersion_threshold: float = 0.9,
    ):
        super().__init__(in_channels, pixel_unshuffle_scale_factors, embedding_dim)

        self.selection_mode = selection_mode
        self.top_k = top_k
        self.dispersion_threshold = dispersion_threshold

        modes = {"full", "top-k", "dispersion"}

        assert selection_mode in modes, f"selection_mode must be one of {modes}"
        if selection_mode == "top-k":
            assert top_k is not None and top_k > 0, "top-k value must be positive"
        if selection_mode == "dispersion":
            assert 0 < dispersion_threshold <= 1.0, "dispersion must be in (0, 1]"

        # Token Estimation Function (TEF) approximated by neural network
        self.mlp_scorer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def _get_filter_mask(self, scores: torch.Tensor) -> torch.Tensor:
        B, N = scores.shape
        gates = torch.sigmoid(scores)  # [B, N]

        if self.training or self.selection_mode == "full":
            return torch.ones_like(gates, dtype=torch.bool)

        if self.selection_mode == "top-k":
            k = min(self.top_k, N)
            idx = scores.topk(k, dim=1).indices  # [B, k]
            mask = torch.zeros_like(gates, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            return mask

        gates_sorted, idx_sorted = gates.sort(dim=1, descending=True)   # [B, N]
        cumsum = gates_sorted.cumsum(dim=1)                             # [B, N]
        total = gates_sorted.sum(dim=1, keepdim=True)                   # [B, 1]
        thresh = self.dispersion_threshold * total                      # [B, 1]
        keep_sorted = cumsum <= thresh                                  # [B, N]
        mask = torch.zeros_like(gates, dtype=torch.bool)
        mask.scatter_(1, idx_sorted, keep_sorted)
        return mask  # [B, N]
    
    def forward(self, x: torch.Tensor):
        raw_tokens = self._get_raw_tokens(x)            # [B, N, C]
        tokens = self.linear_projection(raw_tokens)     # [B, N, E]

        scores = self.mlp_scorer(tokens).squeeze(-1)    # [B, N]

        gated = tokens * torch.sigmoid(scores).unsqueeze(-1)  # [B, N, E]

        B, N, E = gated.shape
        cls_tokens = self.cls_token.expand(B, 1, E)    # [B, 1, E]
        gated_with_cls = torch.cat([cls_tokens, gated], dim=1)  # [B, N+1, E]

        seq_pe = self._add_positional_encoding(gated_with_cls)

        mask = self._get_filter_mask(scores)               # [B, N]
        mask_with_cls = torch.cat([
            torch.ones((B, 1), dtype=torch.bool, device=mask.device),
            mask
        ], dim=1)

        filtered_list = [seq_pe[b][mask_with_cls[b]] for b in range(B)]  # list of [M_b, E]
        padded_tokens = pad_sequence(filtered_list, batch_first=True)  # [B, M_max, E]

        return {"tokens": padded_tokens, "scores": scores}

