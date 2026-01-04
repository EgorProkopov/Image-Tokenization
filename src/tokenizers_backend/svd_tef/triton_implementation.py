import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # TODO: заменить на собственную реализацию паддинга

from src.tokenizers.positional_encoding import PositionalEncoding
from src.tokenizers_backend.svd_tef.torch_implementation import SVDNetworkTorch

from src.tokenizers_backend.svd_tef.triton_kernels import filter_and_pad


class SVDTEFTriton(SVDNetworkTorch):
    """
    SVD tokenizer with a learnable Token Estimation Function (TEF) for
    token gating and optional filtering at inference.

    Realized with triton kernel for tokens scatter (filter_and_pad) and torch backend at other cases. 

    Args:
        in_channels: Number of input image channels.
        pixel_unshuffle_scale_factors: Per-stage downscale factors for the
            pixel-unshuffle blocks in each branch.
        embedding_dim: Dimension of the output token embeddings.
        selection_mode: Token selection strategy at inference; one of
            {"full", "top-k", "dispersion"}.
        top_k: Number of tokens to keep when selection_mode="top-k".
        dispersion_threshold: Cumulative gate mass to keep when
            selection_mode="dispersion".
    """
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

        
        padded_tokens = filter_and_pad(seq_pe, mask_with_cls)

        return {"tokens": padded_tokens, "scores": scores}
