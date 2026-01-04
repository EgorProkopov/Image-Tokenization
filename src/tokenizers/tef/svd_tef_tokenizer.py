import torch.nn as nn

from src.tokenizers_backend.svd_tef.torch_implementation import SVDNetworkTorch, SVDTEFTorch


class SVDNetworkTokenizer(nn.Module):
    """
    Tokenizer wrapper around SVDNetworkTorch backend.
    """
    def __init__(
        self,
        in_channels: int = 3,
        pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
        embedding_dim: int = 768,
        backend: str = "torch",
    ):
        super().__init__()

        if backend != "torch":
            raise ValueError("SVDNetworkTokenizer supports only torch backend")

        self.backend = SVDNetworkTorch(
            in_channels=in_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        return self.backend(x)


class SVDTEFTokenizer(nn.Module):
    """
    Tokenizer wrapper around SVDTEF backends (torch or triton).
    """
    def __init__(
        self,
        in_channels: int = 3,
        pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
        embedding_dim: int = 768,
        selection_mode: str = "full",
        top_k: int = None,
        dispersion_threshold: float = 0.9,
        backend: str = "torch",
    ):
        super().__init__()

        if backend == "torch":
            self.backend = SVDTEFTorch(
                in_channels=in_channels,
                pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
                embedding_dim=embedding_dim,
                selection_mode=selection_mode,
                top_k=top_k,
                dispersion_threshold=dispersion_threshold,
            )
        elif backend == "triton":
            try:
                from src.tokenizers_backend.svd_tef.triton_implementation import SVDTEFTriton
            except ImportError as exc:
                raise ImportError("Triton backend requested, but triton is not available") from exc

            self.backend = SVDTEFTriton(
                in_channels=in_channels,
                pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
                embedding_dim=embedding_dim,
                selection_mode=selection_mode,
                top_k=top_k,
                dispersion_threshold=dispersion_threshold,
            )
        else:
            raise ValueError("backend must be one of {'torch', 'triton'}")

    def forward(self, x):
        return self.backend(x)
