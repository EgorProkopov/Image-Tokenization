### BENCHMARK RESULTS ###
# Device: cuda:0, dtype: torch.float32
# Input: batch=32, channels=3, size=256x256
# Selection mode: full, top-k=None, dispersion-threshold=0.9
# Tokens shape (torch):  (32, 257, 768)
# Tokens shape (triton): (32, 257, 768)
# Max |torch - triton| diff: 0.000000
# Torch  latency: 9.516 ms/iter (avg over 50)
# Triton latency: 8.569 ms/iter (avg over 50)


import time
from typing import Dict, Tuple
import random

import torch

from src.tokenizers_backend.svd_tef.torch_implementation import SVDTEFTorch

try:
    from src.tokenizers_backend.svd_tef.triton_implementation import SVDTEFTriton
except ImportError as exc:  # pragma: no cover - triton is optional at runtime
    SVDTEFTriton = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

BENCH_BATCH_SIZE = 32
BENCH_IMAGE_SIZE = 256
BENCH_CHANNELS = 3
BENCH_EMBEDDING_DIM = 768
BENCH_SELECTION_MODE = "full"  # "full", "top-k", "dispersion"
BENCH_TOP_K = None
BENCH_DISPERSION_THRESHOLD = 0.9
BENCH_WARMUP = 10
BENCH_ITERS = 50
BENCH_DTYPE = "float32"  # "float32", "float16", "bfloat16"
BENCH_DEVICE = "cuda:0"
BENCH_SEED = 42


def set_random_state(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(inputs)
        synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    return elapsed / max(iters, 1)


def compare_outputs(
    torch_out: Dict[str, torch.Tensor],
    triton_out: Dict[str, torch.Tensor],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    torch_tokens = torch_out["tokens"]
    triton_tokens = triton_out["tokens"]
    torch_shape = tuple(torch_tokens.shape)
    triton_shape = tuple(triton_tokens.shape)
    max_diff = float("nan")
    if torch_shape == triton_shape:
        max_diff = (torch_tokens - triton_tokens).abs().max().item()
    return torch_shape, triton_shape, max_diff


def main() -> None:
    if BENCH_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark (triton kernels).")
    if BENCH_SELECTION_MODE == "top-k" and BENCH_TOP_K is None:
        raise SystemExit("--top-k must be set when selection-mode=top-k")

    if SVDTEFTriton is None:
        raise SystemExit(
            f"Triton backend not available: {_TRITON_IMPORT_ERROR}"
        )

    device = torch.device(BENCH_DEVICE)
    dtype = DTYPES[BENCH_DTYPE]
    set_random_state(BENCH_SEED, device)

    inputs = torch.randn(
        BENCH_BATCH_SIZE,
        BENCH_CHANNELS,
        BENCH_IMAGE_SIZE,
        BENCH_IMAGE_SIZE,
        device=device,
        dtype=dtype,
    )

    common_kwargs = dict(
        in_channels=BENCH_CHANNELS,
        embedding_dim=BENCH_EMBEDDING_DIM,
        selection_mode=BENCH_SELECTION_MODE,
        top_k=BENCH_TOP_K,
        dispersion_threshold=BENCH_DISPERSION_THRESHOLD,
    )

    torch_model = SVDTEFTorch(**common_kwargs)
    triton_model = SVDTEFTriton(**common_kwargs)
    triton_model.load_state_dict(torch_model.state_dict())
    torch_model = torch_model.to(device=device, dtype=dtype)
    triton_model = triton_model.to(device=device, dtype=dtype)

    with torch.inference_mode():
        torch_out = torch_model(inputs)
        triton_out = triton_model(inputs)
    torch_shape, triton_shape, max_diff = compare_outputs(torch_out, triton_out)

    torch_latency = measure_latency(torch_model, inputs, BENCH_WARMUP, BENCH_ITERS, device)
    triton_latency = measure_latency(triton_model, inputs, BENCH_WARMUP, BENCH_ITERS, device)

    print(f"Device: {device}, dtype: {dtype}")
    print(
        f"Input: batch={BENCH_BATCH_SIZE}, channels={BENCH_CHANNELS}, "
        f"size={BENCH_IMAGE_SIZE}x{BENCH_IMAGE_SIZE}"
    )
    print(
        "Selection mode: "
        f"{BENCH_SELECTION_MODE}, top-k={BENCH_TOP_K}, "
        f"dispersion-threshold={BENCH_DISPERSION_THRESHOLD}"
    )
    print(f"Tokens shape (torch):  {torch_shape}")
    print(f"Tokens shape (triton): {triton_shape}")
    if torch_shape == triton_shape:
        print(f"Max |torch - triton| diff: {max_diff:.6f}")
    else:
        print("Shapes differ, skipping diff computation.")
    print(f"Torch  latency: {torch_latency * 1000:.3f} ms/iter (avg over {BENCH_ITERS})")
    print(f"Triton latency: {triton_latency * 1000:.3f} ms/iter (avg over {BENCH_ITERS})")


if __name__ == "__main__":
    main()
