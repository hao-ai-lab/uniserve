"""Shared attention-mask helpers."""
from __future__ import annotations

import torch

__all__ = [
    'create_causal_mask',
    'create_block_causal_mask',
    'build_commit_attention_mask',
]


def create_causal_mask(seq_len: int, *, device: torch.device | str) -> torch.Tensor:
    """Standard lower-triangular additive causal mask of shape ``[seq_len, seq_len]``.

    ``0.0`` where a query may attend (j <= i), ``-inf`` above the diagonal.
    """
    causal = torch.ones(int(seq_len), int(seq_len), device=device, dtype=torch.bool).tril()
    return torch.where(causal, 0.0, float("-inf"))


def create_block_causal_mask(index: torch.Tensor) -> torch.Tensor:
    length = index.size(0)
    idx_i = index.unsqueeze(1).expand(length, length)
    idx_j = index.unsqueeze(0).expand(length, length)
    arange = torch.arange(length, device=index.device)
    allowed = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))
    # ``torch.where`` accepts Python scalars for the branches, so no per-call
    # scalar tensors are allocated; the result dtype follows the scalars.
    return torch.where(allowed[None, None, :, :], 0.0, float("-inf"))


def build_commit_attention_mask(
    *,
    num_image_tokens: int,
    past_len: int,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Attention mask for appending a generated image back into the text cache.

    The generated image tokens are followed by the image-end marker. During this
    commit step the image tokens must not attend to that marker, while the marker
    can attend normally to the image tokens and the prior prefix.
    """

    num_image_tokens = int(num_image_tokens)
    past_len = int(past_len)
    if num_image_tokens < 0 or past_len < 0:
        raise ValueError("num_image_tokens and past_len must be non-negative")
    target_len = num_image_tokens + 1
    mask = torch.zeros(
        1,
        1,
        target_len,
        past_len + target_len,
        device=device,
        dtype=dtype or torch.float32,
    )
    if num_image_tokens:
        mask[0, 0, :num_image_tokens, past_len + num_image_tokens] = float("-inf")
    return mask
