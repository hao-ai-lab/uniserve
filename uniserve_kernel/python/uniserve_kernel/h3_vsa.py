"""SM100a block-sparse attention surface for the fixed FastH3 profile."""

from __future__ import annotations

import torch

_IMPORT_ERROR: BaseException | None = None
_sm100a = None

try:  # pragma: no cover - deployment-only CUDA provider.
    from fastvideo_kernel import block_sparse_attn_sm100a as _sm100a
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error


def available() -> bool:
    if _sm100a is None or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major == 10 and bool(getattr(_sm100a, "_HAS_VSA_SM100A", False))


@torch.library.custom_op(
    "uniserve_kernel::h3_vsa_block_sparse",
    mutates_args=(),
)
def _block_sparse_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> torch.Tensor:
    if not available():
        raise RuntimeError("FastH3 SM100a sparse attention is unavailable") from _IMPORT_ERROR
    q_bhsd = query.transpose(0, 1).unsqueeze(0).contiguous()
    k_bhsd = key.transpose(0, 1).unsqueeze(0).contiguous()
    v_bhsd = value.transpose(0, 1).unsqueeze(0).contiguous()
    output, _ = _sm100a.block_sparse_attn_sm100a(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        mask_block_indices.unsqueeze(0).to(torch.int32).contiguous(),
        mask_block_count.unsqueeze(0).to(torch.int32).contiguous(),
        valid_sizes.to(torch.int32).contiguous(),
        need_lse=False,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


@_block_sparse_custom.register_fake
def _block_sparse_custom_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> torch.Tensor:
    del key, value, mask_block_count, mask_block_indices, valid_sizes
    return torch.empty_like(query)


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Invoke the native forward-only Blackwell sparse kernel with H3 metadata."""

    if not available():
        raise RuntimeError("FastH3 SM100a sparse attention is unavailable") from _IMPORT_ERROR
    if tile_size != 64 or query.shape[-1] != 128:
        raise ValueError("FastH3 VSA requires tile 64 and head dimension 128")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
        raise ValueError("FastH3 VSA expects matching [sequence, heads, 128] Q/K/V")
    if valid_sizes.ndim != 1 or valid_sizes.numel() * tile_size != query.shape[0]:
        raise ValueError("FastH3 VSA valid-size metadata does not match the sequence")
    return _block_sparse_custom(
        query,
        key,
        value,
        mask_block_count,
        mask_block_indices,
        valid_sizes,
    )


__all__ = ["available", "block_sparse_attention"]
