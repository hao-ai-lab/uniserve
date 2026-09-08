"""Architecture-portable Triton provider for block-64 video sparse attention."""

from __future__ import annotations

import math

import torch

from ...nn.parallel_attention import AttentionOutputTargets
from ...ops.video_sparse import compose_to_head_shards, unpack_add_compression
from ..triton import triton_available

try:  # pragma: no cover - worker_config-only CUDA provider.
    import triton
    import triton.language as tl
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR: BaseException | None = error
    triton = None
    tl = None
else:  # pragma: no cover
    _IMPORT_ERROR = None

_TILE = 64
_HEAD_DIM = 128
_SOFTMAX_SCALE = 1.0 / math.sqrt(_HEAD_DIM)
_LOG2E = math.log2(math.e)


if triton is not None:

    @triton.jit
    def _block_sparse_attention_kernel(
        query,
        key,
        value,
        output,
        block_indices,
        block_counts,
        valid_sizes,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_head: tl.constexpr,
        output_stride_row: tl.constexpr,
        output_stride_head: tl.constexpr,
        indices_stride_head: tl.constexpr,
        indices_stride_tile: tl.constexpr,
        counts_stride_head: tl.constexpr,
        query_rows: tl.constexpr,
        selected_width: tl.constexpr,
        softmax_scale: tl.constexpr,
        log2e: tl.constexpr,
        tile_size: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        head_dim: tl.constexpr,
        pipeline_stages: tl.constexpr,
    ):
        """Apply a head-wise sparse block map using online softmax accumulation."""

        query_program = tl.program_id(0)
        head = tl.program_id(1)
        query_rows_offset = query_program * block_m + tl.arange(0, block_m)
        query_tile = (query_program * block_m) // tile_size
        columns = tl.arange(0, head_dim)
        query_mask = query_rows_offset[:, None] < query_rows
        query_values = tl.load(
            query
            + query_rows_offset[:, None] * query_stride_row
            + head * query_stride_head
            + columns[None, :],
            mask=query_mask,
            other=0.0,
        )

        normalizer_max = tl.full((block_m,), -float("inf"), tl.float32)
        normalizer_sum = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, head_dim), tl.float32)
        selected_count = tl.load(
            block_counts + head * counts_stride_head + query_tile
        )
        selected_count = tl.minimum(selected_count, selected_width)

        for selected_offset in tl.range(0, selected_count, num_stages=pipeline_stages):
            key_tile = tl.load(
                block_indices
                + head * indices_stride_head
                + query_tile * indices_stride_tile
                + selected_offset
            )
            valid_rows = tl.load(valid_sizes + key_tile)
            key_rows_offset = key_tile * tile_size + tl.arange(0, block_n)
            key_mask = tl.arange(0, block_n) < valid_rows
            key_values = tl.load(
                key
                + key_rows_offset[:, None] * key_stride_row
                + head * key_stride_head
                + columns[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            logits = tl.dot(query_values, tl.trans(key_values)) * softmax_scale
            logits = tl.where(
                query_mask & key_mask[None, :],
                logits,
                -float("inf"),
            )

            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(normalizer_max, block_max)
            correction = tl.exp2((normalizer_max - next_max) * log2e)
            probabilities = tl.exp2((logits - next_max[:, None]) * log2e)
            block_sum = tl.sum(probabilities, axis=1)

            value_values = tl.load(
                value
                + key_rows_offset[:, None] * value_stride_row
                + head * value_stride_head
                + columns[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator = (
                accumulator * correction[:, None]
                + tl.dot(probabilities.to(tl.bfloat16), value_values)
            )
            normalizer_sum = normalizer_sum * correction + block_sum
            normalizer_max = next_max

        normalized = tl.where(
            normalizer_sum[:, None] > 0.0,
            accumulator / normalizer_sum[:, None],
            0.0,
        )
        tl.store(
            output
            + query_rows_offset[:, None] * output_stride_row
            + head * output_stride_head
            + columns[None, :],
            normalized,
            mask=query_mask,
        )


def available(device: torch.device | None = None) -> bool:
    """Return whether Triton can compile kernels for the requested CUDA device."""

    if triton is None or not torch.cuda.is_available():
        return False
    selected = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return triton_available(selected)


def import_error() -> BaseException | None:
    """Return the error that prevented Triton registration, if any."""

    return _IMPORT_ERROR


def _validate_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> None:
    """Validate the public sparse-attention tensor and metadata contract."""

    if query.ndim != 3 or query.shape[1] < 1 or query.shape[2] != _HEAD_DIM:
        raise ValueError("Triton sparse attention requires [sequence, heads, 128] Q/K/V")
    if key.shape != value.shape or key.shape[1:] != query.shape[1:]:
        raise ValueError("Triton sparse attention requires matching K/V and Q/K head geometry")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise ValueError("Triton sparse attention requires BF16 Q/K/V")
    if not query.is_cuda or any(
        tensor.device != query.device for tensor in (key, value, output)
    ):
        raise ValueError("Triton sparse attention tensors must share one CUDA device")
    if output.shape != query.shape or output.dtype != query.dtype or not output.is_contiguous():
        raise ValueError("Triton sparse attention output must be contiguous BF16 with Q's shape")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, value, output)):
        raise ValueError("Triton sparse attention requires a contiguous head dimension")
    if query.shape[0] % _TILE or key.shape[0] % _TILE:
        raise ValueError("Triton sparse attention rows must be a multiple of 64")

    query_tiles = query.shape[0] // _TILE
    if (
        block_indices.ndim != 3
        or tuple(block_indices.shape[:2]) != (query.shape[1], query_tiles)
        or block_indices.shape[2] < 1
        or block_counts.shape != (query.shape[1], query_tiles)
        or valid_sizes.shape != (key.shape[0] // _TILE,)
    ):
        raise ValueError("Triton sparse attention metadata does not match Q/K/V")
    if any(
        tensor.dtype != torch.int32
        or tensor.device != query.device
        or not tensor.is_contiguous()
        for tensor in (block_indices, block_counts, valid_sizes)
    ):
        raise ValueError("Triton sparse attention metadata must be contiguous CUDA int32")


def _launch_config(device: torch.device) -> tuple[int, int, int]:
    """Choose the resident query tile and software pipeline for one architecture."""

    major, _minor = torch.cuda.get_device_capability(device)
    if major == 9:
        return 64, 8, 3
    return 32, 4, 2


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a mutable head-wise block map without host-side planning."""

    if not available(query.device):
        raise RuntimeError("Triton sparse video attention is unavailable") from _IMPORT_ERROR
    _validate_attention(
        query,
        key,
        value,
        output,
        block_indices,
        block_counts,
        valid_sizes,
    )
    assert triton is not None
    block_m, num_warps, pipeline_stages = _launch_config(query.device)
    query_rows, heads, _width = (int(size) for size in query.shape)
    _block_sparse_attention_kernel[(triton.cdiv(query_rows, block_m), heads)](
        query,
        key,
        value,
        output,
        block_indices,
        block_counts,
        valid_sizes,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        int(block_indices.stride(0)),
        int(block_indices.stride(1)),
        int(block_counts.stride(0)),
        query_rows,
        int(block_indices.shape[2]),
        _SOFTMAX_SCALE,
        _LOG2E,
        _TILE,
        block_m,
        _TILE,
        _HEAD_DIM,
        pipeline_stages,
        num_warps=num_warps,
        num_stages=pipeline_stages,
    )
    return output.unsqueeze(0).transpose(1, 2)


@torch.library.custom_op(
    "uniserve_worker::video_sparse_attention_triton",
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
) -> None:
    """Write sparse attention with compression into rank-local output shards."""

    attended = block_sparse_attention(
        query,
        key,
        value,
        attention_output,
        mask_block_indices,
        mask_block_count,
        valid_sizes,
    )
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
) -> None:
    """Declare the output mutation contract for fake tensor execution."""

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
    )


def _validate_outputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_block_count: torch.Tensor,
    mask_block_indices: torch.Tensor,
    valid_sizes: torch.Tensor,
    tile_size: int,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    attention_output: torch.Tensor,
    targets: AttentionOutputTargets,
) -> None:
    """Validate sparse attention, compression, and exchange output geometry."""

    if tile_size != _TILE:
        raise ValueError("sparse video attention requires tile 64")
    _validate_attention(
        query,
        key,
        value,
        attention_output,
        mask_block_indices,
        mask_block_count,
        valid_sizes,
    )
    if gate.shape != query.shape or compressed.shape != (
        query.shape[1],
        query.shape[0] // _TILE,
        query.shape[2],
    ):
        raise ValueError("sparse video compression buffers do not match attention geometry")
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


def execute_sparse_attention(
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
    targets: AttentionOutputTargets,
) -> torch.Tensor:
    """Execute block-sparse attention and route composed rank-local output."""

    del prefix_tiles
    if not available(query.device):
        raise RuntimeError("Triton sparse video attention is unavailable") from _IMPORT_ERROR
    _validate_outputs(
        query,
        key,
        value,
        mask_block_count=mask_block_count,
        mask_block_indices=mask_block_indices,
        valid_sizes=valid_sizes,
        tile_size=tile_size,
        gate=gate,
        compressed=compressed,
        attention_output=attention_output,
        targets=targets,
    )
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
    )
    return targets.buffers[targets.source_rank]


__all__ = [
    "available",
    "block_sparse_attention",
    "execute_sparse_attention",
    "import_error",
]
