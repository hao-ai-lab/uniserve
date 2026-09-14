"""Numerical shapes and stable input views for CUDA graph execution."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch

from uniserve.attention.metadata import AttentionMetadata, AttentionMode
from uniserve.attention.selection import AttentionSelection
from uniserve.math import bucketed_length
from uniserve.model.tensors import TokenSelection, packed_tensor_views
from uniserve.runtime.cuda_graph import CudaGraph, GraphExecutionError
from uniserve_worker.execution.batch import ExecutionOutput, InputBatch
from uniserve_worker.protocol.batch import ForwardMode, PipelineStage

from .model_entry import tensor_signature
from .sampling import SamplerOutput

logger = logging.getLogger(__name__)
TOKEN_CONTINUATION_BIT = 1 << 31
_GRAPH_BINDINGS = itertools.count(1)


@dataclass(frozen=True, slots=True)
class BatchGraph:
    """A worker batch's captured inputs and the numerical executable using them."""

    graph: CudaGraph[tuple[ExecutionOutput, SamplerOutput | None]]
    inputs: InputBatch
    input_leaves: tuple[torch.Tensor, ...]
    attention_leaves: tuple[torch.Tensor, ...]


class _GraphMiss(RuntimeError):
    """Signals that a requested CUDA graph signature has no captured executable."""

    pass


@dataclass(frozen=True, slots=True)
class DiffusionShape:
    """A denoise request shape, including its physical guidance branch count."""

    rows: int
    height: int
    width: int
    cfg_branches: int


@dataclass(frozen=True, slots=True)
class PrefillShape:
    """A padded prefill shape and the minimum live rows exercising its bucket."""

    token_bucket: int
    row_bucket: int
    live_rows: int


def select_flow_captures(
    shapes: Sequence[tuple[int, int]],
    request_counts: Sequence[int],
    cfg_branches: Sequence[int],
    *,
    max_operations: int,
    max_tokens: int,
    per_image_capacity: int,
    latent_capacity: int,
    physical_tokens: Callable[[int, int], int],
    image_tokens: Callable[[int, int], int],
) -> tuple[DiffusionShape, ...]:
    """Intersect requested capture shapes with actual staging and latent bounds."""

    return tuple(
        DiffusionShape(rows, height, width, branches)
        for height, width in shapes
        for rows in request_counts
        for branches in cfg_branches
        if 0 < rows <= max_operations
        and rows * physical_tokens(height, width) * branches <= max_tokens
        and image_tokens(height, width) <= per_image_capacity
        and rows * image_tokens(height, width) <= latent_capacity
    )


def select_prefill_captures(
    token_sizes: Sequence[int],
    row_sizes: Sequence[int],
    *,
    max_rows: int,
    max_tokens: int,
) -> tuple[PrefillShape, ...]:
    """Enumerate exactly the padded prefill buckets within the execution bounds."""

    buckets: list[PrefillShape] = []
    minimum_rows = 1
    for row_bucket in sorted({int(value) for value in row_sizes if int(value) > 1}):
        if minimum_rows > int(max_rows):
            break
        minimum_tokens = minimum_rows if minimum_rows == 1 else minimum_rows + 1
        for token_bucket in sorted(
            {int(value) for value in token_sizes if minimum_tokens <= int(value) <= int(max_tokens)}
        ):
            buckets.append(PrefillShape(token_bucket, row_bucket, minimum_rows))
        minimum_rows = row_bucket
    return tuple(buckets)


def _decode_shape(
    batch: InputBatch,
    batch_sizes: tuple[int, ...],
    block_size: int,
    context_blocks: int,
) -> tuple[int, int] | None:
    """Select a graph decode bucket and derive padded rows, blocks, and token counts."""

    if batch.attention.attention_mode is not AttentionMode.PAGED_DECODE or not batch_sizes:
        return None
    rows = batch.row_count
    bucket = next((value for value in batch_sizes if value >= rows), None)
    if bucket is None or batch.input_ids is None or batch.positions is None:
        return None
    if (
        batch.token_row_indices != tuple(range(rows))
        or batch.attention.query_lens_cpu != (1,) * rows
        or int(batch.input_ids.numel()) != rows
        or int(batch.positions.shape[-1]) != rows
        or len(set(batch.token_selections)) != 1
    ):
        return None
    if batch.attention.block_table is None:
        return None
    live_width = int(batch.attention.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    reserved_width = max(1, (int(bucket) + int(block_size) - 1) // int(block_size))
    return int(bucket), max(live_width, int(context_blocks), reserved_width)


def _prefill_shape(
    batch: InputBatch,
    token_sizes: tuple[int, ...],
    block_size: int,
    row_sizes: tuple[int, ...],
    context_blocks: int,
) -> tuple[PrefillShape, int] | None:
    """Select a graph prefill bucket and derive padded query and cache geometry."""

    if batch.attention.attention_mode is not AttentionMode.PAGED_VARLEN or not token_sizes:
        return None
    rows = batch.row_count
    row_bucket = next((value for value in row_sizes if value > rows), None)
    query_lens = tuple(int(value) for value in batch.attention.query_lens_cpu)
    if (
        row_bucket is None
        or (rows >= row_sizes[0] and all(value == 1 for value in query_lens))
        or batch.input_ids is None
        or batch.positions is None
        or batch.token_row_indices != tuple(range(rows))
        or len(query_lens) != rows
        or any(value < 1 for value in query_lens)
        or int(batch.input_ids.numel()) != sum(query_lens)
        or int(batch.positions.shape[-1]) != sum(query_lens)
    ):
        return None
    live_tokens = sum(query_lens)
    token_bucket = next((value for value in token_sizes if value >= live_tokens), None)
    if token_bucket is None:
        return None
    if batch.attention.block_table is None:
        return None
    live_width = int(batch.attention.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    previous = max((value for value in token_sizes if value < token_bucket), default=0)
    maximum_padding = int(token_bucket) - int(previous)
    reserved_width = max(1, (maximum_padding + int(block_size) - 1) // int(block_size))
    width = max(live_width, int(context_blocks), reserved_width)
    return PrefillShape(int(token_bucket), int(row_bucket), rows), width


def _pad_decode_batch(
    batch: InputBatch,
    bucket: int,
    width: int,
    block_size: int,
) -> InputBatch:
    """Copy a live decode batch into fixed-row graph buffers and synthesize padding rows."""

    padding = bucket - batch.row_count
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None if batch.embedding_mask is None else _fixed_view(batch.embedding_mask, (bucket,))
    )
    if batch.attention.block_table is None or batch.attention.seq_lens is None:
        raise _GraphMiss("paged decode has incomplete tensors")
    return replace(
        batch,
        attention=replace(
            batch.attention,
            prefix_lens=_fixed_view(batch.attention.prefix_lens, (bucket,)),
            query_lens=_fixed_view(batch.attention.query_lens, (bucket,)),
            out_cache_loc=_fixed_view(batch.attention.out_cache_loc, (bucket,)),
            block_table=_fixed_view(batch.attention.block_table, (bucket, width)),
            seq_lens=_fixed_view(batch.attention.seq_lens, (bucket,)),
            prefix_lens_cpu=(*batch.attention.prefix_lens_cpu, *(0 for _ in range(padding))),
            seq_lens_cpu=(*batch.attention.seq_lens_cpu, *(1 for _ in range(padding))),
            query_lens_cpu=(*batch.attention.query_lens_cpu, *(1 for _ in range(padding))),
            max_seqlen_k=width * int(block_size),
        ),
        row_count=bucket,
        request_pool_indices=_fixed_view(batch.request_pool_indices, (bucket,)),
        decode_force_finish=None
        if batch.decode_force_finish is None
        else _fixed_view(batch.decode_force_finish, (bucket,)),
        token_row_indices=tuple(range(bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * bucket,
    )


def _pad_prefill_batch(
    batch: InputBatch, shape: PrefillShape, width: int, block_size: int
) -> InputBatch:
    """Extend a paged-prefill batch into fixed graph buckets using inert rows and tokens."""

    live_rows = batch.row_count
    padding = shape.token_bucket - sum(batch.attention.query_lens_cpu)
    dummy_rows = shape.row_bucket - live_rows

    # Fixed-address views expose the capture bucket without reallocating the
    # staging tensors that back the live prefix.
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (shape.token_bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), shape.token_bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (shape.token_bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None
        if batch.embedding_mask is None
        else _fixed_view(batch.embedding_mask, (shape.token_bucket,))
    )
    if (
        batch.attention.block_table is None
        or batch.attention.seq_lens is None
        or batch.attention.cu_seqlens_q is None
        or batch.attention.cu_seqlens_k is None
        or batch.attention.output_indices is None
    ):
        raise _GraphMiss("paged prefill has incomplete tensors")
    block_table = _fixed_view(
        batch.attention.block_table,
        (shape.row_bucket, width),
    )
    cache_seqlens = _fixed_view(batch.attention.prefix_lens, (shape.row_bucket,))
    query_lens = _fixed_view(batch.attention.query_lens, (shape.row_bucket,))
    kv_seqlens = _fixed_view(batch.attention.seq_lens, (shape.row_bucket,))
    cu_seqlens_q = _fixed_view(batch.attention.cu_seqlens_q, (shape.row_bucket + 1,))
    cu_seqlens_k = _fixed_view(batch.attention.cu_seqlens_k, (shape.row_bucket + 1,))
    output_indices = _fixed_view(batch.attention.output_indices, (shape.row_bucket,))

    # Dummy rows advertise no cached or query tokens. Any token-axis padding is
    # assigned to the first dummy row so cumulative lengths still terminate at
    # the graph's fixed token bucket.
    cache_seqlens[live_rows:].zero_()
    query_lens[live_rows:].zero_()
    kv_seqlens[live_rows:].zero_()
    if padding:
        query_lens[live_rows : live_rows + 1].fill_(padding)
        kv_seqlens[live_rows : live_rows + 1].fill_(padding)
    cu_seqlens_q[live_rows + 1 :].fill_(shape.token_bucket)
    padded_kv_tokens = sum(int(value) for value in batch.attention.seq_lens_cpu) + padding
    cu_seqlens_k[live_rows + 1 :].fill_(padded_kv_tokens)
    output_indices[live_rows:].zero_()
    if padding:
        output_indices[live_rows : live_rows + 1].fill_(shape.token_bucket - 1)
    dummy_query_lens = (padding, *(0 for _ in range(dummy_rows - 1)))

    # CPU mirrors must encode the same geometry because backend planning reads
    # them independently of the device-side cumulative arrays.
    return replace(
        batch,
        attention=replace(
            batch.attention,
            prefix_lens=cache_seqlens,
            query_lens=query_lens,
            out_cache_loc=_fixed_view(batch.attention.out_cache_loc, (shape.token_bucket,)),
            block_table=block_table,
            seq_lens=kv_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            output_indices=output_indices,
            prefix_lens_cpu=(*batch.attention.prefix_lens_cpu, *(0 for _ in range(dummy_rows))),
            query_lens_cpu=(*batch.attention.query_lens_cpu, *dummy_query_lens),
            seq_lens_cpu=(*batch.attention.seq_lens_cpu, *dummy_query_lens),
            max_seqlen_q=bucketed_length(shape.token_bucket),
            max_seqlen_k=width * int(block_size),
        ),
        row_count=shape.row_bucket,
        request_pool_indices=_fixed_view(
            batch.request_pool_indices,
            (shape.row_bucket,),
        ),
        token_row_indices=tuple(range(shape.row_bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * shape.row_bucket,
    )


def _decode_signature(batch: InputBatch, bucket: int, width: int) -> tuple[object, ...]:
    """Build a decode graph signature from padded geometry and tensor contracts."""

    return (
        "paged_decode_bucket",
        bucket,
        width,
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.attention.causal),
        bool(batch.attention.has_cache_writes),
    )


def _prefill_signature(
    batch: InputBatch, shape: PrefillShape, width: int, block_size: int
) -> tuple[object, ...]:
    """Build a prefill graph signature from padded rows, tokens, and cache geometry."""

    return (
        "paged_prefill_bucket",
        shape.row_bucket,
        shape.token_bucket,
        width,
        bucketed_length(shape.token_bucket),
        width * int(block_size),
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.attention.causal),
        bool(batch.attention.has_cache_writes),
    )


def _batch_tensor_signature(batch: InputBatch) -> tuple[object, ...]:
    """Describe all batch tensor leaves by dtype, shape, stride, and device."""

    assert batch.input_ids is not None and batch.positions is not None
    return (
        str(batch.input_ids.dtype),
        batch.input_ids.device.type,
        batch.positions.ndim,
        int(batch.positions.shape[0]) if batch.positions.ndim == 2 else 1,
        str(batch.positions.dtype),
        batch.input_embeddings is not None,
        None if batch.input_embeddings is None else str(batch.input_embeddings.dtype),
    )


def _exact_signature(batch: InputBatch) -> tuple[object, ...]:
    """Build a hashable signature for all graph-observable batch geometry."""

    # AR and denoising invoke the same model.forward entry. Row geometry and
    # selections below distinguish their executed math; encoders and the VAE
    # decoder use separate neural entry points even when tensor shapes match.
    entry = (
        "forward"
        if isinstance(batch.forward_mode, ForwardMode)
        or batch.forward_mode is PipelineStage.DENOISING
        else batch.forward_mode.value
    )
    return (
        entry,
        batch.row_count,
        batch.token_row_indices,
        batch.flow_row_indices,
        batch.attention.attention_mode.value,
        batch.attention.query_lens_cpu,
        tuple(value.value for value in batch.token_selections),
        batch.flow_image_tokens,
        batch.flow_heights,
        batch.flow_widths,
        tuple(value is not None for value in batch.flow_conditioning),
        tensor_signature(tuple(_batch_tensors(batch))),
        batch.attention.route_spans,
        batch.attention.causal_rows_cpu,
        bool(batch.attention.has_cache_writes),
        batch.attention.max_seqlen_q,
        batch.attention.max_seqlen_k,
    )


def _normalize_exact_batch(
    batch: InputBatch,
    *,
    context_blocks: int,
    block_size: int,
) -> InputBatch:
    """Give exact packed graphs their startup-fixed KV table geometry.

    Packed diffusion calls use request-variable KV prefix lengths, but the
    lane input buffer already owns a maximum-width, zero-scrubbed block
    table.  Capturing the active request-width view makes otherwise identical
    startup and serving calls different graph shapes.  Widening that view here
    keeps the physical kernel launch fixed while ``seqused_k`` and the other
    staged tensors carry the live lengths on every replay.
    """

    if batch.attention.attention_mode is not AttentionMode.PACKED or context_blocks <= 0:
        return batch
    if batch.attention.block_table is None:
        raise _GraphMiss("packed attention has no block table")
    if int(batch.attention.block_table.shape[1]) > context_blocks:
        raise _GraphMiss("packed attention exceeds the configured context width")
    return replace(
        batch,
        attention=replace(
            batch.attention,
            block_table=_fixed_view(
                batch.attention.block_table,
                (int(batch.attention.block_table.shape[0]), int(context_blocks)),
            ),
            max_seqlen_k=int(context_blocks) * int(block_size),
        ),
    )


def _graph_batch(
    batch: InputBatch,
    binding: int,
    *,
    own_inputs: bool,
) -> InputBatch:
    """Clone a batch into graph-owned inputs or bind its existing static tensors."""

    graph_batch = _clone_batch(batch) if own_inputs else batch
    return replace(
        graph_batch,
        binding=int(binding),
        cuda_graph_capture=True,
    )


def _clone_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.clone(memory_format=torch.preserve_format)


def _clone_batch(batch: InputBatch) -> InputBatch:
    """Own copies of the fixed model-input columns while borrowing immutable geometry."""

    attention = batch.attention
    return replace(
        batch,
        attention=replace(
            attention,
            prefix_lens=attention.prefix_lens.clone(),
            query_lens=attention.query_lens.clone(),
            out_cache_loc=attention.out_cache_loc.clone(),
            block_table=_clone_optional(attention.block_table),
            seq_lens=_clone_optional(attention.seq_lens),
            cu_seqlens_q=_clone_optional(attention.cu_seqlens_q),
            cu_seqlens_k=_clone_optional(attention.cu_seqlens_k),
            output_indices=_clone_optional(attention.output_indices),
            attention_indexes=_clone_optional(attention.attention_indexes),
            visible_end=_clone_optional(attention.visible_end),
        ),
        request_pool_indices=batch.request_pool_indices.clone(),
        decode_force_finish=_clone_optional(batch.decode_force_finish),
        input_ids=_clone_optional(batch.input_ids),
        input_embeddings=_clone_optional(batch.input_embeddings),
        embedding_mask=_clone_optional(batch.embedding_mask),
        positions=_clone_optional(batch.positions),
        flow_positions=tuple(value.clone() for value in batch.flow_positions),
        flow_timesteps=tuple(value.clone() for value in batch.flow_timesteps),
        flow_latents=tuple(value.clone() for value in batch.flow_latents),
        flow_conditioning=tuple(
            None
            if value is None
            else replace(
                value,
                pixels=value.pixels.clone(),
                grid=value.grid.clone(),
                noise_scale=value.noise_scale.clone(),
            )
            for value in batch.flow_conditioning
        ),
        encode_pixels=tuple(value.clone() for value in batch.encode_pixels),
        encode_grids=tuple(_clone_optional(value) for value in batch.encode_grids),
        decode_latents=tuple(value.clone() for value in batch.decode_latents),
    )


def _attention_tensors(attention: AttentionMetadata) -> Iterator[torch.Tensor]:
    """Enumerate the numerical attention columns in their fixed copy order."""

    yield attention.prefix_lens
    yield attention.query_lens
    yield attention.out_cache_loc
    for value in (
        attention.block_table,
        attention.seq_lens,
        attention.cu_seqlens_q,
        attention.cu_seqlens_k,
        attention.output_indices,
        attention.attention_indexes,
        attention.visible_end,
    ):
        if value is not None:
            yield value


def _batch_tensors(batch: InputBatch) -> Iterator[torch.Tensor]:
    """Enumerate model inputs without traversing arbitrary Python objects."""

    yield from _attention_tensors(batch.attention)
    yield batch.request_pool_indices
    for value in (
        batch.decode_force_finish,
        batch.input_ids,
        batch.input_embeddings,
        batch.embedding_mask,
        batch.positions,
    ):
        if value is not None:
            yield value
    yield from batch.flow_positions
    yield from batch.flow_timesteps
    yield from batch.flow_latents
    for patches in batch.flow_conditioning:
        if patches is not None:
            yield patches.pixels
            yield patches.grid
            yield patches.noise_scale
    yield from batch.encode_pixels
    for grid in batch.encode_grids:
        if grid is not None:
            yield grid
    yield from batch.decode_latents


def _copy_tensors(
    target_tensors: tuple[torch.Tensor, ...],
    source_tensors: tuple[torch.Tensor, ...],
    structure: str,
) -> None:
    """Copy live columns into captured addresses after checking their geometry."""

    if len(target_tensors) != len(source_tensors):
        raise _GraphMiss(f"{structure} tensor structure changed")
    for destination, value in zip(target_tensors, source_tensors, strict=True):
        if (
            destination.shape != value.shape
            or destination.dtype != value.dtype
            or destination.device != value.device
        ):
            raise _GraphMiss(f"{structure} tensor geometry changed")
        destination.copy_(value, non_blocking=True)


def _graph_provider(
    selection: AttentionSelection,
    mode: AttentionMode,
    *,
    head_dim: int,
    block_size: int,
    device: torch.device,
):
    """Resolve the geometry-bound paged-attention backend for a graph mode."""

    provider = selection.select_provider(
        mode, head_dim=head_dim, block_size=block_size, device=device
    )
    if provider is not None and provider.can_bind(
        mode,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
        cuda_graph=True,
    ):
        return provider
    raise _GraphMiss("the selected attention provider is not graph-safe")


def _live_attention(static: AttentionMetadata, live: AttentionMetadata) -> AttentionMetadata:
    """Replan live scalar geometry using the graph's fixed addresses and bounds."""

    return replace(
        live,
        prefix_lens=static.prefix_lens,
        query_lens=static.query_lens,
        out_cache_loc=static.out_cache_loc,
        block_table=static.block_table,
        seq_lens=static.seq_lens,
        cu_seqlens_q=static.cu_seqlens_q,
        cu_seqlens_k=static.cu_seqlens_k,
        output_indices=static.output_indices,
        attention_indexes=static.attention_indexes,
        visible_end=static.visible_end,
        max_seqlen_q=static.max_seqlen_q,
        max_seqlen_k=static.max_seqlen_k,
    )


def _fixed_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Return a bounded tensor view after validating the requested element count."""

    if tensor.ndim != len(shape) or any(value < 0 for value in shape):
        raise _GraphMiss("fixed graph view rank changed")
    strides = tuple(int(value) for value in tensor.stride())
    if any(value < 0 for value in strides):
        raise _GraphMiss("fixed graph view has a negative stride")
    maximum = int(tensor.storage_offset())
    for extent, stride in zip(shape, strides, strict=True):
        if extent:
            maximum += (int(extent) - 1) * stride
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    if maximum >= storage_elements:
        raise _GraphMiss("graph bucket exceeds its fixed input storage")
    return tensor.as_strided(shape, strides, storage_offset=int(tensor.storage_offset()))


def _expand_token_axis(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    """Expand a single-token tensor view to a fixed graph token extent."""

    if tensor.ndim == 1:
        return _fixed_view(tensor, (tokens,))
    if tensor.ndim == 2:
        return _fixed_view(tensor, (int(tensor.shape[0]), tokens))
    raise _GraphMiss("graph token positions have an invalid rank")


def _cuda_batch(batch: InputBatch) -> bool:
    """Check that every model-input tensor belongs to one CUDA device."""

    tensors = tuple(_batch_tensors(batch))
    return bool(
        tensors
        and torch.cuda.is_available()
        and all(value.device.type == "cuda" for value in tensors)
        and len({value.device for value in tensors}) == 1
    )


def _attention_inputs(batch: InputBatch) -> Iterator[torch.Tensor]:
    """Collect stable request and attention addresses retained by a graph."""

    yield batch.request_pool_indices
    yield from _attention_tensors(batch.attention)


def _private_pool_bytes(device: torch.device, pools: set[tuple[int, int]]) -> int:
    """Return allocator residency for the exact pools owned on this device."""

    if device.type != "cuda" or not pools:
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("device") != index:
            continue
        pool_id = segment.get("segment_pool_id")
        if isinstance(pool_id, tuple) and pool_id in pools:
            total += int(segment.get("total_size", 0))
    return total


def _trim_output(output: ExecutionOutput, rows: int) -> ExecutionOutput:
    """Slice every forward-output row tensor to the live batch extent."""

    return ExecutionOutput(
        tuple(output.values[:rows]), output.vocabularies[:rows], layouts=output.layouts[:rows]
    )


def _greedy_decode(
    batch: InputBatch,
    output: ExecutionOutput,
    predicate_state: torch.Tensor | None,
) -> SamplerOutput | None:
    """Return graph-capturable greedy output for eligible decode logits."""

    return _greedy_decode_values(
        batch,
        output,
        predicate_state,
        batch.decode_force_finish,
        clear_force_finish=True,
    )


def _greedy_decode_values(
    batch: InputBatch,
    output: ExecutionOutput,
    predicate_state: torch.Tensor | None,
    force_finish: torch.Tensor | None,
    *,
    clear_force_finish: bool,
) -> SamplerOutput | None:
    """Derive graph-capturable greedy tokens and continuation state from model logits."""

    if (
        batch.attention.attention_mode is not AttentionMode.PAGED_DECODE
        or predicate_state is None
        or force_finish is None
        or len(output.values) != batch.row_count
        or batch.token_row_indices != tuple(range(batch.row_count))
        or any(selection is not TokenSelection.LAST_LOGITS for selection in batch.token_selections)
        or batch.flow_row_indices
    ):
        return None
    rows = tuple(value.reshape(-1) for value in output.values)
    logits = packed_tensor_views(rows)
    if logits is None:
        raise _GraphMiss("decode logits are not one contiguous graph output")
    logits = logits.reshape(batch.row_count, -1)
    from uniserve.nn.logits import greedy_vocabulary

    partitions = output.vocabularies
    if any(partition != partitions[0] for partition in partitions):
        return None
    max_values, tokens = greedy_vocabulary(logits, partitions[0])
    valid = torch.isfinite(max_values)
    active = predicate_state.index_select(0, batch.request_pool_indices.reshape(-1))
    finish = force_finish.reshape(-1) & valid & active
    continuation = valid & active & ~finish
    tags = torch.where(continuation, TOKEN_CONTINUATION_BIT, 0)
    tagged_tokens = tokens.bitwise_or(tags)
    completion = torch.cat(
        (
            valid,
            active,
            tokens,
            torch.zeros_like(tokens),
        )
    )
    if clear_force_finish:
        force_finish.zero_()
    return SamplerOutput(
        tokens=tokens,
        valid=valid,
        active=active,
        finish=finish,
        continuation=continuation,
        tagged_tokens=tagged_tokens,
        completion=completion,
    )


def _trim_greedy(
    output: SamplerOutput | None,
    rows: int,
) -> SamplerOutput | None:
    """Slice padded graph-greedy output tensors back to the live row count."""

    if output is None:
        return None
    total = int(output.tokens.numel())
    if rows < 0 or rows > total or int(output.completion.numel()) != 4 * total:
        raise GraphExecutionError("CUDA graph greedy output has invalid row geometry")
    completion = torch.cat(
        tuple(output.completion[index * total : index * total + rows] for index in range(4))
    )
    return SamplerOutput(
        tokens=output.tokens[:rows],
        valid=output.valid[:rows],
        active=output.active[:rows],
        finish=None if output.finish is None else output.finish[:rows],
        continuation=output.continuation[:rows],
        tagged_tokens=output.tagged_tokens[:rows],
        completion=completion,
    )


def _release_call(method: Callable[[int], object], binding: int) -> Callable[[], None]:
    """Bind a graph resource release method to one capture identity."""

    def release() -> None:
        """Release the captured backend binding identified by this closure."""

        method(binding)

    return release
