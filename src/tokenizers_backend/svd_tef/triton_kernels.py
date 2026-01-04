import torch

import triton
import triton.language as tl


@triton.jit
def filter_and_pad_kernel(
    X_ptr,                   # [B, S, E]  contigous
    MASK_ptr,                # [B, S] contiguous
    IDX_ptr,                 # [B, S] contiguous
    Y_ptr,                   # [B, M_max, E] contiguous (filled with zeroes at beginning)

    S: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    BLOCK_E: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    pe = tl.program_id(2)

    # mask [b, s]
    mask = tl.load(MASK_ptr + b * S + s).to(tl.int1)         # mask [b, s]
    
    dst = tl.load(IDX_ptr + b * S + s).to(tl.int32)
    dst = tl.where(mask, dst, 0)

    # embedding offsets for current tile
    e = pe * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e < E

    x_base = (b * S + s) * E
    x = tl.load(X_ptr + x_base + e, mask=e_mask, other=0.0)

    y_base = (b * M + dst) * E  # moves to the unfiltered token

    tl.store(Y_ptr + y_base + e, x, mask=e_mask & mask)



def filter_and_pad(
    x: torch.Tensor,
    mask: torch.Tensor,
    block_e: int = 128,
    num_warps: int = 4
):
    """
    Equals to this python-torch code:
        filtered_list = [seq_pe[b][mask_with_cls[b]] for b in range(B)]  # list of [M_b, E]
        padded_tokens = pad_sequence(filtered_list, batch_first=True)  # [B, M_max, E]
    """

    # TODO: add doctring for inputs and output

    assert x.is_cuda and mask.is_cuda, "x and mask should be on same devices"
    assert x.is_contiguous() and mask.is_contiguous(), "x and mask should be contiguous"
    assert x.dim() == 3 and mask.dim() == 2, "x dim should be equal 3 and mask dim should be equal 2"
    assert mask.dtype == torch.bool
    B, S, E = x.shape
    assert mask.shape == (B, S)

    # idx[b][s] - new token s position after removing some tokens
    idx = mask.to(torch.int32).cumsum(dim=1) - 1   # [B, S]
    lengths = mask.sum(dim=1, dtype=torch.int32)   # [B]

    M_max = int(lengths.max().item())
    out = x.new_zeros((B, M_max, E))

    if M_max == 0:
        return out, lengths

    grid = (B, S, triton.cdiv(E, block_e))
    filter_and_pad_kernel[grid](
        x, mask, idx, out,
        S=S, E=E, M=M_max,
        BLOCK_E=block_e,
        num_warps=num_warps
    )

    return out
