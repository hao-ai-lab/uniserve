"""SM100a provider binding for shared block-sparse video attention."""

from __future__ import annotations

from typing import Any

import torch

from uniserve.attention import video_sparse_cute
from uniserve.nn.parallel_attention import AttentionOutputTargets
from uniserve.ops.video_sparse import compose_to_head_shards, pack_qkv, unpack_add_compression
from uniserve.ops.video_sparse_rows import SparseAttentionPattern

_IMPORT_ERROR: BaseException | None = None
_provider: Any | None
try:  # pragma: no cover - worker_config-only CUDA provider.
    from uniserve_kernel import sparse_attention as _provider_module
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error
    _provider = None
else:  # pragma: no cover
    _provider = _provider_module


def available(device: torch.device | None = None) -> bool:
    """Return whether the SM100 sparse-attention extension is registered."""

    if _provider is None or not torch.cuda.is_available():
        return False
    return _provider.available(device) and video_sparse_cute.available(device)


def import_error() -> BaseException | None:
    """Return the exception that prevented SM100 extension registration, if any."""

    return _IMPORT_ERROR or video_sparse_cute.import_error()


@torch.library.custom_op(
    "uniserve::video_sparse_attention_sm100",
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
    """Dispatch sparse attention and write dense-compressed rank-local output shards."""

    if not available(query.device) or _provider is None:
        raise RuntimeError("SM100a sparse video attention is unavailable") from import_error()
    # The provider's size boundary covers the key sequence traversed by each
    # query, including when query rows are distributed across devices.
    if video_sparse_cute.should_use(
        rows=key.shape[0],
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
        # Queries are head-major; K/V retain independently strided storage so
        # peer-backed keys do not materialize a full local replica.
        if query.shape[0] == key.shape[0]:
            packed = tuple(pack_qkv(query, key, value).unbind(0))
        else:
            packed = (
                query.transpose(0, 1).contiguous(),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        attended = _provider.block_sparse_attention(
            *(tensor.unsqueeze(0) for tensor in packed),
            mask_block_indices,
            mask_block_count,
            valid_sizes,
        )
    # A single-rank output can fuse compression locally; multi-rank execution
    # composes head shards through the declared source coordinate.
    if len(outputs) == 1:
        unpack_add_compression(attended, gate, compressed, outputs[0])
    else:
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
    """Infer custom-op output aliases and geometry without executing the SM100 kernel."""

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
    pattern: SparseAttentionPattern,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    targets: AttentionOutputTargets,
) -> torch.Tensor:
    """Execute SM100 block-sparse attention using per-query block counts and indices."""

    if not available(query.device):
        raise RuntimeError("SM100a sparse video attention is unavailable") from import_error()
    if tile_size != 64 or query.shape[-1] != 128:
        raise ValueError("sparse video attention requires tile 64 and head dimension 128")
    if key.shape != value.shape or query.shape[1:] != key.shape[1:] or query.ndim != 3:
        raise ValueError("sparse video attention expects matching K/V and Q/K head geometry")
    if valid_sizes.ndim != 1 or valid_sizes.numel() * tile_size != key.shape[0]:
        raise ValueError("sparse video attention metadata does not match the sequence")
    if gate.shape != query.shape or compressed.shape != (
        query.shape[1],
        query.shape[0] // tile_size,
        query.shape[2],
    ):
        raise ValueError("sparse video compression buffers do not match attention geometry")
    if (
        attention_output.shape != query.shape
        or attention_output.dtype != query.dtype
        or attention_output.device != query.device
    ):
        raise ValueError("sparse video attention output does not match Q/K/V")
    expected_output_shape = (
        query.shape[0] // len(targets.buffers),
        query.shape[1] * len(targets.buffers),
        query.shape[2],
    )
    if any(
        output.shape != expected_output_shape
        or output.dtype != query.dtype
        or output.device != query.device
        for output in targets.buffers
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
        list(targets.buffers),
        targets.source_rank,
        pattern.dense_prefix_tiles,
    )
    return targets.buffers[targets.source_rank]


__all__ = ["available", "block_sparse_attention", "import_error"]
