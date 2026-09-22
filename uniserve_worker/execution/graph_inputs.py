"""Worker graph shapes, stable numerical inputs and captured call ownership."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from uniserve.model import EmbeddingReplacement, TextInput
from uniserve.nn.attention import (
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime import PrefixCache
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.sampling import greedy
from uniserve.tensors import adjacent_view
from uniserve_worker.protocol.call import ForwardMode

from .batch import ExecutionOutput, InputBatch
from .graphs import Graph
from .sampling import TOKEN_CONTINUATION_BIT, SamplerOutput, TokenSelection


@dataclass(frozen=True, slots=True)
class DiffusionShape:
    rows: int
    height: int
    width: int
    cfg_branches: int


@dataclass(frozen=True, slots=True)
class PrefillShape:
    token_bucket: int
    row_bucket: int
    live_rows: int
    causal: bool = True
    selection: TokenSelection = TokenSelection.LAST_LOGITS


def select_flow_captures(
    shapes: Sequence[tuple[int, int]],
    request_counts: Sequence[int],
    cfg_branches: Sequence[int],
    *,
    max_calls: int,
    max_tokens: int,
    per_image_capacity: int,
    latent_capacity: int,
    physical_tokens: Callable[[int, int], int],
    image_tokens: Callable[[int, int], int],
) -> tuple[DiffusionShape, ...]:
    """Intersect configured shapes with staging.

    per-image and latent capacity.
    """
    return tuple(
        DiffusionShape(rows, height, width, branches)
        for height, width in shapes
        for rows in request_counts
        for branches in cfg_branches
        if 0 < rows <= max_calls
        and rows * physical_tokens(height, width) * branches <= max_tokens
        and image_tokens(height, width) <= per_image_capacity
        and rows * image_tokens(height, width) <= latent_capacity
    )


def select_prefill_captures(
    token_sizes, row_sizes, *, max_rows, max_tokens, visual=False
):
    """Build prefill capture buckets from configured token and row sizes.

    Each row bucket carries the previous bucket's row count as its live-row
    minimum, so a bucket only serves batches larger than the next smaller one.
    """
    buckets = []
    variants = ((True, TokenSelection.LAST_LOGITS),)
    if visual:
        # Feature appends expose the whole image to each query, optionally
        # sampling at its trailing marker after publishing the prefix.
        variants += (
            (False, TokenSelection.LAST_LOGITS),
            (False, TokenSelection.HIDDEN),
        )

    minimum_rows = 1
    for rows in sorted({int(value) for value in row_sizes if value > 1}):
        if minimum_rows > max_rows:
            break
        minimum_tokens = minimum_rows if minimum_rows == 1 else minimum_rows + 1
        for tokens in sorted(
            {
                value
                for value in token_sizes
                if minimum_tokens <= value <= max_tokens
            }
        ):
            buckets.extend(
                PrefillShape(tokens, rows, minimum_rows, causal, selection)
                for causal, selection in variants
            )
        minimum_rows = rows
    return tuple(buckets)


def bind_attention(static, live):
    """Pair captured addresses with the current host sequence metadata."""
    return replace(
        static,
        queries=replace(static.queries, host=live.queries.host),
        prefixes=replace(static.prefixes, host=live.prefixes.host),
    )


def text_shape(batch, *, decode_sizes, prefill_shapes, context_blocks):
    """Choose a resident bucket with the call's attention and output.

    semantics.
    """
    inputs = batch.inputs
    if not isinstance(inputs, TextInput) or not isinstance(
        inputs.attention, PagedInput
    ):
        return None

    attention = inputs.attention
    selection = batch.token_selections[0]
    causal = attention.causal[0]
    if any(value is not selection for value in batch.token_selections):
        return None
    if any(value != causal for value in attention.causal) or any(
        length < 1 for length in attention.queries.host
    ):
        return None

    width = max(context_blocks, attention.block_table.indices.shape[1])
    if (
        batch.forward_mode is ForwardMode.DECODE
        and selection is TokenSelection.LAST_LOGITS
        and causal
        and all(length == 1 for length in attention.queries.host)
    ):
        rows = next(
            (value for value in decode_sizes if value >= batch.row_count), None
        )
        if rows is not None:
            return rows, rows, width, True

    shapes = tuple(
        shape
        for shape in prefill_shapes
        if shape.row_bucket > batch.row_count
        and shape.token_bucket >= inputs.input_ids.numel()
    )
    if not shapes:
        return None
    shape = min(
        shapes, key=lambda value: (value.row_bucket, value.token_bucket)
    )
    return shape.row_bucket, shape.token_bucket, width, False


def _fixed_view(tensor, shape):
    """Borrow a leading view of shape from a bucket tensor's captured.

    storage.
    """
    strides = tensor.stride()
    if len(shape) != tensor.ndim or any(value < 0 for value in shape):
        raise ValueError("graph view rank changed")
    last = tensor.storage_offset() + sum(
        (extent - 1) * stride
        for extent, stride in zip(shape, strides)
        if extent
    )
    if last >= tensor.untyped_storage().nbytes() // tensor.element_size():
        raise ValueError("graph bucket exceeds lane input storage")
    return tensor.as_strided(
        shape, strides, storage_offset=tensor.storage_offset()
    )


def pad_text(batch, rows, tokens, width, decode):
    """Borrow a fixed bucket and make padding inert, including cache writes.

    Physical block zero is valid storage. Padding queries read disposable values
    but never write a block: their write indices are -1 and their results are
    discarded. Prefill padding belongs to one additional numerical sequence.
    """
    inputs, live_rows = batch.inputs, batch.row_count
    attention, live_tokens = inputs.attention, inputs.input_ids.numel()
    if rows < live_rows or tokens < live_tokens:
        raise ValueError("graph shape is smaller than its live inputs")

    padding, extra = tokens - live_tokens, rows - live_rows
    if padding and not extra:
        raise ValueError("token padding requires an additional sequence")
    if not padding and not extra:
        # The staged lengths and offsets already describe the whole bucket.
        # Only page-table capacity can differ from the captured view.
        return widen_prefix(batch, width)

    # Padding tokens belong to one additional inert sequence; any further
    # padding rows are empty sequences.
    dummy = (
        (1,) * extra
        if decode
        else ((padding,) + (0,) * (extra - 1) if extra else ())
    )
    host_queries = attention.queries.host + dummy

    ids = _fixed_view(inputs.input_ids, (tokens,))
    ids[live_tokens:].zero_()
    positions = _fixed_view(
        inputs.positions, (*inputs.positions.shape[:-1], tokens)
    )
    positions[..., live_tokens:].zero_()

    slots = _fixed_view(batch.request_pool_indices, (rows,))
    slots[live_rows:].zero_()
    table = _fixed_view(attention.block_table.indices, (rows, width))
    table[live_rows:].zero_()

    queries = _fixed_view(attention.queries.values, (rows,))
    queries[live_rows:].zero_()
    if decode:
        queries[live_rows:].fill_(1)
    elif padding:
        queries[live_rows : live_rows + 1].fill_(padding)
    query_offsets = _fixed_view(attention.queries.offsets, (rows + 1,))
    torch.cumsum(queries, dim=0, out=query_offsets[1:])

    prefix = _fixed_view(attention.prefixes.values, (rows,))
    prefix[live_rows:].zero_()
    prefix_offsets = _fixed_view(attention.prefixes.offsets, (rows + 1,))
    torch.cumsum(prefix, dim=0, out=prefix_offsets[1:])

    writes = attention.write_indices
    if writes is not None:
        writes = _fixed_view(writes, (tokens,))
        writes[live_tokens:].fill_(-1)

    padded = PagedInput(
        SequenceLengths(
            host=host_queries, values=queries, offsets=query_offsets
        ),
        SequenceLengths(
            host=attention.prefixes.host + (0,) * extra,
            values=prefix,
            offsets=prefix_offsets,
        ),
        BlockTable(table, attention.block_table.block_size),
        writes,
        (attention.causal[0],) * rows,
    )

    embeddings = inputs.embeddings
    if embeddings is not None:
        values = _fixed_view(
            embeddings.values, (tokens, embeddings.values.shape[1])
        )
        mask = _fixed_view(embeddings.mask, (tokens,))
        values[live_tokens:].zero_()
        mask[live_tokens:].zero_()
        embeddings = EmbeddingReplacement(values, mask)

    finish = batch.decode_force_finish
    if finish is not None:
        finish = _fixed_view(finish, (rows,))
        finish[live_rows:].zero_()

    return replace(
        batch,
        inputs=replace(
            inputs,
            input_ids=ids,
            positions=positions,
            attention=padded,
            embeddings=embeddings,
        ),
        request_pool_indices=slots,
        token_selections=(batch.token_selections[0],) * rows,
        decode_force_finish=finish,
    )


def widen_prefix(batch, width):
    """Borrow fixed table capacity while retaining live prefix lengths."""
    attention = getattr(batch.inputs, "attention", None)
    if not isinstance(attention, (PagedInput, SegmentedInput)):
        return batch
    if attention.block_table.indices.shape[1] > width:
        raise ValueError("prefix table exceeds its configured graph width")
    table = _fixed_view(attention.block_table.indices, (batch.row_count, width))
    return replace(
        batch,
        inputs=replace(
            batch.inputs,
            attention=replace(
                attention,
                block_table=BlockTable(table, attention.block_table.block_size),
            ),
        ),
    )


def restore_writes(batch, cache: PrefixCache | None):
    """Snapshot complete touched cache blocks.

    including scales and initialization.

    Capturing or warming a call is observationally neutral to the live prefix.
    Reading addresses here is startup preparation, outside graph capture.
    """
    snapshots = []
    attention = getattr(batch.inputs, "attention", None)
    writes = getattr(attention, "write_indices", None)
    if cache is not None and writes is not None:
        blocks = tuple(
            sorted(
                {
                    int(value) // attention.block_table.block_size
                    for value in writes.cpu().tolist()
                    if value >= 0
                }
            )
        )
        for name in cache.config.layers:
            for tensors in cache.state(name).transfer_views(blocks).values():
                snapshots.extend((tensor, tensor.clone()) for tensor in tensors)
    if batch.decode_force_finish is not None:
        snapshots.append(
            (batch.decode_force_finish, batch.decode_force_finish.clone())
        )

    def restore():
        for target, saved in snapshots:
            target.copy_(saved)

    return restore


def capture_batch(
    context, batch, call, *, pools=None, cache=None, predicates=None
):
    """Capture numerical batch output and greedy decoding on common backing."""
    with context.activate():
        attention = getattr(batch.inputs, "attention", None)
        if attention is not None:
            context.bind_attention(attention)
        restore = restore_writes(batch, cache)

    def compute(static):
        output = call(static)
        return output, greedy_decode(static, output, predicates)

    return Graph.capture(context, batch, compute, pools=pools, restore=restore)


def replay_batch(graph: Graph, batch, *, rows=None, borrow=False):
    """Replay staged inputs and retain the live output rows."""
    context = graph.executable.context
    with context.activate():
        graph.inputs.copy(batch)
        attention = getattr(batch.inputs, "attention", None)
        if attention is not None:
            context.bind_attention(
                bind_attention(graph.inputs.value.inputs.attention, attention)
            )

        output, greedy = graph.executable.replay()
        count = batch.row_count if rows is None else rows
        values = output.values
        if isinstance(batch.inputs, TextInput) and all(
            selection is TokenSelection.HIDDEN
            for selection in batch.token_selections
        ):
            # Capture fixes tensor addresses, not the live sequence cuts.
            # Uniform hidden results share one contiguous,
            # pipeline-published tensor; split its views again using this
            # invocation's lengths.
            view = adjacent_view(values)
            if view is None:
                raise ValueError(
                    "hidden graph results must share contiguous output storage"
                )
            values = view.reshape(-1, values[0].shape[-1]).split(
                attention.queries.host
            )
        result = replace(
            output,
            values=values[:count],
            vocabularies=output.vocabularies[:count],
            layouts=output.layouts[:count],
            greedy=trim_greedy(greedy, count),
        )
        return result if borrow else result.clone()


def greedy_decode(
    batch: InputBatch,
    output: ExecutionOutput,
    predicate_state: torch.Tensor | None,
) -> SamplerOutput | None:
    """Derive graph-capturable greedy tokens and continuation state from model.

    logits.
    """
    force_finish = batch.decode_force_finish
    if (
        not isinstance(batch.inputs, TextInput)
        or not isinstance(batch.inputs.attention, PagedInput)
        or batch.inputs.attention.queries.host != (1,) * batch.row_count
        or predicate_state is None
        or force_finish is None
        or len(output.values) != batch.row_count
        or any(
            selection is not TokenSelection.LAST_LOGITS
            for selection in batch.token_selections
        )
    ):
        return None
    # [rows, vocab] logits, gathered from one contiguous graph output.
    rows = tuple(value.reshape(-1) for value in output.values)
    logits = adjacent_view(rows)
    if logits is None:
        raise ValueError("decode logits are not one contiguous graph output")
    logits = logits.reshape(batch.row_count, -1)
    partitions = output.vocabularies
    if any(partition != partitions[0] for partition in partitions):
        return None

    max_values, tokens = greedy(logits, partitions[0])
    valid = torch.isfinite(max_values)
    active = predicate_state.index_select(
        0, batch.request_pool_indices.reshape(-1)
    )
    finish = force_finish.reshape(-1) & valid & active
    continuation = valid & active & ~finish
    tags = torch.where(continuation, TOKEN_CONTINUATION_BIT, 0)
    tagged_tokens = tokens.bitwise_or(tags)

    # Fixed four-section layout [valid | active | tokens | reserved] per row;
    # trim_greedy depends on these exact sections when slicing padded rows.
    completion = torch.cat(
        (
            valid,
            active,
            tokens,
            torch.zeros_like(tokens),
        )
    )
    return SamplerOutput(
        tokens=tokens,
        valid=valid,
        active=active,
        finish=finish,
        continuation=continuation,
        tagged_tokens=tagged_tokens,
        completion=completion,
    )


def trim_greedy(
    output: SamplerOutput | None,
    rows: int,
) -> SamplerOutput | None:
    """Slice padded graph-greedy output tensors back to the live row count."""
    if output is None:
        return None
    total = int(output.tokens.numel())
    if rows < 0 or rows > total or int(output.completion.numel()) != 4 * total:
        raise CUDAGraphError("CUDA graph greedy output has invalid row count")
    if rows == total:
        return output

    # Slice each of the four completion sections independently; they are
    # concatenated along the row axis, so a plain [:rows] cut would mix them.
    completion = torch.cat(
        tuple(
            output.completion[index * total : index * total + rows]
            for index in range(4)
        )
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
