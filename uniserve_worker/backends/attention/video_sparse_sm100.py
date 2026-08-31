"""SM100a provider binding for shared block-sparse video attention."""

from __future__ import annotations

from typing import Any

import torch

from ...nn.mesh import SymmetricMemoryWorkspace
from ...ops.video_sparse import compose_to_head_shards, pack_qkv
from . import video_sparse_cute

_IMPORT_ERROR: BaseException | None = None
_provider: Any | None
try:  # pragma: no cover - deployment-only CUDA provider.
    from fastvideo_kernel import block_sparse_attn_sm100a as _provider_module
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error
    _provider = None
else:  # pragma: no cover
    _provider = _provider_module


def available() -> bool:
    if _provider is None or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return (
        major == 10
        and bool(getattr(_provider, "_HAS_VSA_SM100A", False))
        and video_sparse_cute.available()
    )


def import_error() -> BaseException | None:
    return _IMPORT_ERROR or video_sparse_cute.import_error()


@torch.library.custom_op(
    "uniserve_worker::video_sparse_attention_sm100",
    mutates_args=("attention_output", "outputs"),
)
def _block_sparse_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    prefix_tiles: int,
) -> None:
    if not available() or _provider is None:
        raise RuntimeError(
            "SM100a sparse video attention is unavailable"
        ) from import_error()
    if video_sparse_cute.should_use(
        rows=int(query.shape[0]),
        prefix_tiles=prefix_tiles,
    ):
        attended = video_sparse_cute.block_sparse_attention(
            query,
            key,
            value,
            attention_output,
            mask_block_indices,
            mask_block_count,
            valid_sizes,
        )
    else:
        packed = pack_qkv(query, key, value)
        attended, _ = _provider.block_sparse_attn_sm100a(
            packed[0].unsqueeze(0),
            packed[1].unsqueeze(0),
            packed[2].unsqueeze(0),
            mask_block_indices.unsqueeze(0),
            mask_block_count.unsqueeze(0),
            valid_sizes,
            need_lse=False,
        )
    compose_to_head_shards(attended, gate, compressed, tuple(outputs), source_rank)


@_block_sparse_custom.register_fake
def _block_sparse_custom_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    prefix_tiles: int,
) -> None:
    del (
        query,
        key,
        value,
        mask_block_count,
        mask_block_indices,
        valid_sizes,
        gate,
        compressed,
        attention_output,
        outputs,
        source_rank,
        prefix_tiles,
    )


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    tile_size: int,
    prefix_tiles: int,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    exchange: SymmetricMemoryWorkspace,
    exchange_outputs: tuple[torch.Tensor, ...],
    exchange_sync_input: torch.Tensor,
    exchange_sync_output: torch.Tensor,
) -> torch.Tensor:
    if not available():
        raise RuntimeError(
            "SM100a sparse video attention is unavailable"
        ) from import_error()
    if tile_size != 64 or query.shape[-1] != 128:
        raise ValueError(
            "sparse video attention requires tile 64 and head dimension 128"
        )
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
        raise ValueError(
            "sparse video attention expects matching [sequence, heads, 128] Q/K/V"
        )
    if valid_sizes.ndim != 1 or valid_sizes.numel() * tile_size != query.shape[0]:
        raise ValueError("sparse video attention metadata does not match the sequence")
    if gate.shape != query.shape or compressed.shape != (
        query.shape[1],
        valid_sizes.numel(),
        query.shape[2],
    ):
        raise ValueError(
            "sparse video compression buffers do not match attention geometry"
        )
    if (
        attention_output.shape != query.shape
        or attention_output.dtype != query.dtype
        or attention_output.device != query.device
    ):
        raise ValueError("sparse video attention output does not match Q/K/V")
    expected_output_shape = (
        query.shape[0] // exchange.size,
        query.shape[1] * exchange.size,
        query.shape[2],
    )
    if any(
        output.shape != expected_output_shape
        or output.dtype != query.dtype
        or output.device != query.device
        for output in exchange_outputs
    ):
        raise ValueError("symmetric exchange outputs do not match attention geometry")
    _block_sparse_custom(
        query,
        key,
        value,
        mask_block_count,
        mask_block_indices,
        valid_sizes,
        gate,
        compressed,
        attention_output,
        list(exchange_outputs),
        exchange.rank,
        prefix_tiles,
    )
    exchange.fence(exchange_sync_input, exchange_sync_output)
    return exchange_outputs[exchange.rank]


__all__ = ["available", "block_sparse_attention", "import_error"]
