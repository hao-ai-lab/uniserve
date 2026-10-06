"""Worker graph shapes, stable numerical inputs and captured call ownership.

Text calls replay graphs captured at configured bucket shapes. Selection
(``text_shape``) picks a bucket for a prepared batch, ``pad_text`` widens the
batch's views of the runner's fixed buffers to that bucket and makes the
padding inert. Decode buckets capture the call together with
graph-capturable greedy decoding (``capture_batch``/``replay_batch``);
prefill buckets capture the backbone's hidden states alone
(``capture_hidden``/``replay_hidden``), and the runner selects each row's
logits or hidden rows from them after the replay. The ``select_*`` helpers
turn configured sizes into the capture shapes ``ModelExecutor`` prepares at
startup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import cast

import torch

from uniserve.model import EmbeddingReplacement, TextInput
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime import PrefixCache
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.sampling import greedy
from uniserve.tensors import adjacent_view
from uniserve_worker._uniserve_ipc import PrefillShape as PrefillShape
from uniserve_worker._uniserve_ipc import TextShapes
from uniserve_worker._uniserve_ipc import prefill_units as prefill_units
from uniserve_worker._uniserve_ipc import (
    select_prefill_captures as select_prefill_captures,
)
from uniserve_worker.model_executor.cuda_graph import CUDAGraphRunner
from uniserve_worker.model_executor.input_batch import InputBatch
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.sampling.result import (
    TOKEN_CONTINUATION_BIT,
    SamplerOutput,
)


@dataclass(frozen=True, slots=True)
class DiffusionShape:
    """An image-denoising capture shape.

    ``rows`` images of ``height`` x ``width`` pixels, each evaluated for
    ``cfg_branches`` guidance branches, so one call holds
    ``rows * cfg_branches`` sequences.
    """

    rows: int
    height: int
    width: int
    cfg_branches: int


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
    """Keep the configured shape combinations that fit buffer capacity.

    Every combination of image shape, request count and guidance-branch
    count is kept when the request count is positive and at most
    ``max_calls``, its requests times branches sequences fit ``max_tokens``
    flat tokens as counted by ``physical_tokens``, one image fits
    ``per_image_capacity`` latent units, and all images fit
    ``latent_capacity``.
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


def decode_captures(config, *, max_rows, row_units, num_units):
    """Select the decode batch sizes a CUDA text entry captures at startup.

    ``config`` is the ``WorkerConfig``. A configured size is kept when it is
    positive, at most ``max_rows``, and its rows fit the KV pool's
    allocatable units: capturing ``rows`` decode rows requires one page of
    every cache group per row (``row_units`` units) on the ``num_units``
    pool, whose unit zero is the sentinel. The sizes keep their configured
    order. Empty when the graph policy is off, which leaves decode calls
    eager.
    """
    if config.graph_policy == "off":
        return ()
    return tuple(
        value
        for value in config.decode_graph_batch_sizes
        if 0 < value <= max_rows and value * row_units < num_units
    )


def prefill_captures(
    config,
    *,
    max_rows,
    max_tokens,
    image_builder,
    feature_injection,
    device_causality,
    pool=None,
):
    """Select the prefill buckets a CUDA text entry captures at startup.

    ``config`` is the ``WorkerConfig``: its ``prefill_graph_token_sizes``
    and the fixed ``DEFAULT_PREFILL_GRAPH_ROW_BUCKETS`` size the buckets up to
    ``max_rows`` rows and ``max_tokens`` tokens, the most one buffered call
    holds, keeping those whose batches fit the unit ``pool`` (see
    ``select_prefill_captures``). Causal text is always captured, with
    embedding replacement when the lane has an ``image_builder`` (every
    prefill of such a lane replaces embeddings); a model whose image
    processor declares ``feature_injection`` also appends non-causal image
    feature rows and mixed text/image contexts, which replace embeddings.
    Mixed contexts carry device causal flags through graph replay and are
    captured only when the attention provider accepts that representation
    (its ``device_causality`` capability).
    The graphs evaluate
    hidden states when the deployment's prefill calls select outputs
    (``prefill_outputs``) and only write the K/V cache otherwise. Empty when
    the graph policy is off or prefill graphs are disabled, which leaves
    prefill calls eager.
    """
    from uniserve_worker.config.execution import (
        DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
    )

    if config.graph_policy == "off" or not config.prefill_cuda_graph:
        return ()
    variants: tuple[tuple[bool | None, bool], ...] = (
        (True, bool(image_builder)),
    )
    if feature_injection:
        variants += ((False, True),)
        if device_causality:
            variants += ((None, True),)
    return select_prefill_captures(
        config.prefill_graph_token_sizes,
        DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
        max_rows=max_rows,
        max_tokens=max_tokens,
        variants=variants,
        outputs=config.prefill_outputs,
        pool=pool,
    )


def prefill_rows(shapes, *, max_rows):
    """Return the most rows one prefill call may hold with ``shapes``.

    A bucket serves batches with fewer rows than it pads to, so the bound is
    one less than the widest bucket, at most ``max_rows``. None without
    shapes, when prefill calls run eagerly and only ``max_rows`` applies.
    """
    if not shapes:
        return None
    return min(max_rows, max(shape.row_bucket for shape in shapes) - 1)


def bind_attention(static, live):
    """Pair captured addresses with the current host sequence metadata.

    ``static`` and ``live`` are ``AttentionBatch`` values over the same
    tables and rows. Returns ``static`` with its device tensors unchanged
    and every table's host query and prefix lengths and host start pages
    taken from ``live``, for ``ExecutionContext.bind_attention`` planning
    before a replay.
    """
    queries = replace(static.queries, host=live.queries.host)
    entries = {}
    for table, entry in static.entries.items():
        current = live.entries[table]
        blocks = entry.block_table
        if blocks.start_page is not None:
            blocks = replace(
                blocks, start_page_host=current.block_table.start_page_host
            )
        entries[table] = replace(
            entry,
            queries=queries,
            prefixes=replace(entry.prefixes, host=current.prefixes.host),
            block_table=blocks,
        )
        if isinstance(entry, PagedInput):
            entries[table] = replace(entries[table], causal=current.causal)
    return AttentionBatch(entries, queries)


def text_shape(batch, *, shapes: TextShapes, table_widths):
    """Pair native bucket selection with this input's numerical table widths.

    Returns ``(padding_shape, outputs)`` or None for eager prefill. Native
    selection matches causality, embedding replacement and output kind; the
    tensor backend supplies paged attention and exact host query lengths.
    No device tensor is read back to choose a graph.
    """
    inputs = batch.inputs
    if not isinstance(inputs, TextInput) or not all(
        isinstance(entry, PagedInput)
        for entry in inputs.attention.entries.values()
    ):
        raise CUDAGraphError("text graphs require paged text inputs")
    if set(inputs.attention.entries) != set(
        range(len(inputs.attention.entries))
    ):
        raise CUDAGraphError("text graph tables must use contiguous indices")

    entries = tuple(
        cast(PagedInput, inputs.attention.entries[table])
        for table in range(len(inputs.attention.entries))
    )
    attention = entries[0]
    causal = (
        None if attention.causal_values is not None else attention.causal[0]
    )
    if causal is not None and any(
        value != causal for entry in entries for value in entry.causal
    ):
        raise CUDAGraphError(
            "no prefill graph for mixed causality without device flags"
        )
    queries = attention.queries.host
    if queries is None:
        raise ValueError("graph shape selection requires host query lengths")

    selected = shapes.select(
        queries,
        inputs.input_ids.numel(),
        decode=batch.forward_mode is ForwardMode.DECODE,
        causal=causal,
        embeddings=inputs.embeddings is not None,
        last_logits=all(
            value is TokenSelection.LAST_LOGITS
            for value in batch.token_selections
        ),
        cache_only=all(
            value is TokenSelection.CACHE for value in batch.token_selections
        ),
    )
    if selected is None:
        return None

    rows, tokens, decode, outputs = selected
    widths = tuple(
        max(
            table_widths[table] if table < len(table_widths) else 1,
            entry.block_table.indices.shape[1],
        )
        for table, entry in enumerate(entries)
    )
    return (rows, tokens, widths, decode), outputs


def _fixed_view(tensor, shape):
    """Borrow a view of ``shape`` from where ``tensor`` starts in storage.

    The view keeps ``tensor``'s strides and may extend past its extent into
    the fixed buffer it views, which is how a live batch grows to
    its bucket shape without a copy.

    Raises:
        ValueError: If the rank differs, an extent is negative, or the view
            would end past the underlying storage.
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


def _copy_offsets(offsets: torch.Tensor, lengths: tuple[int, ...]) -> None:
    """Copy the running offsets of host ``lengths``, from zero, into place."""
    offsets.copy_(
        torch.tensor(
            tuple(accumulate(lengths, initial=0)), dtype=offsets.dtype
        ),
        non_blocking=True,
    )


def pad_text(batch, rows, tokens, widths, decode, *, buffers):
    """Borrow a fixed bucket and make padding inert, including cache writes.

    Physical unit zero is valid storage. Padding queries read disposable
    values but never write a unit: their write indices are -1, their start
    pages zero, and their results are discarded. Prefill padding belongs to
    one additional numerical sequence. ``widths[t]`` is the input width of
    numerical table ``t``.

    The batch's tensors must be views of ``buffers``, the runner's fixed
    buffers (``AttentionBuffers``): padding is written in place past the
    live extents, and the returned batch views the same storage at the
    bucket shape. Padding rows use request slot zero, which the block tables
    reserve for padding. Lengths and offsets derive from the host lengths
    and use one copy each; every table's padding clears with one
    launch per column.
    """
    inputs, live_rows = batch.inputs, batch.row_count
    attention, live_tokens = inputs.attention, inputs.input_ids.numel()
    if rows < live_rows or tokens < live_tokens:
        raise ValueError("graph shape is smaller than its live inputs")
    first = next(iter(attention.entries.values()))
    causal = first.causal[0]

    padding, extra = tokens - live_tokens, rows - live_rows
    if padding and not extra:
        raise ValueError("token padding requires an additional sequence")
    if not padding and not extra:
        # The input lengths and offsets already describe the whole bucket.
        # Only page-table capacity can differ from the captured view.
        return widen_prefix(batch, widths)

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

    # The shared query domain pads once, from the host lengths.
    queries = _fixed_view(attention.queries.values, (rows,))
    if extra:
        queries[live_rows:].copy_(
            torch.tensor(dummy, dtype=queries.dtype), non_blocking=True
        )
    query_offsets = _fixed_view(attention.queries.offsets, (rows + 1,))
    _copy_offsets(query_offsets, host_queries)
    shared = SequenceLengths(
        host=host_queries, values=queries, offsets=query_offsets
    )

    # Tables in one batch share one prefix column, padded once; every
    # table's page columns and write addresses pad with one launch each.
    buffers.clear_padding(
        live_rows=live_rows, rows=rows, live_tokens=live_tokens, tokens=tokens
    )
    flags = None
    if first.causal_values is not None:
        flags = _fixed_view(first.causal_values, (rows,))
        flags[live_rows:].fill_(int(causal))
    padded_prefixes = {}
    entries = {}
    for number, entry in attention.entries.items():
        blocks = entry.block_table
        table = _fixed_view(blocks.indices, (rows, widths[number]))
        start = blocks.start_page
        start_host = blocks.start_page_host
        if start is not None:
            start = _fixed_view(start, (rows,))
            if start_host is not None:
                start_host = start_host + (0,) * extra

        prefixes = padded_prefixes.get(id(entry.prefixes.values))
        if prefixes is None:
            prefix = _fixed_view(entry.prefixes.values, (rows,))
            prefix[live_rows:].zero_()
            prefix_offsets = _fixed_view(entry.prefixes.offsets, (rows + 1,))
            host_prefixes = entry.prefixes.host + (0,) * extra
            _copy_offsets(prefix_offsets, host_prefixes)
            prefixes = SequenceLengths(
                host=host_prefixes, values=prefix, offsets=prefix_offsets
            )
            padded_prefixes[id(entry.prefixes.values)] = prefixes

        writes = entry.write_indices
        if writes is not None:
            writes = _fixed_view(writes, (tokens,))

        entries[number] = PagedInput(
            shared,
            prefixes,
            BlockTable(table, blocks.block_size, start, start_host),
            writes,
            entry.causal + (causal,) * extra,
            flags,
        )
    padded = AttentionBatch(entries, shared)

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


def widen_prefix(batch, widths):
    """Borrow fixed table capacity while retaining live prefix lengths.

    ``widths[t]`` is the width numerical table ``t`` widens to; start pages
    and host metadata are kept.
    """
    attention = getattr(batch.inputs, "attention", None)
    if attention is None or not all(
        isinstance(entry, (PagedInput, SegmentedInput))
        for entry in attention.entries.values()
    ):
        return batch
    entries = {}
    for number, entry in attention.entries.items():
        width = widths[number]
        if entry.block_table.indices.shape[1] > width:
            raise ValueError("prefix table exceeds its configured graph width")
        table = _fixed_view(entry.block_table.indices, (batch.row_count, width))
        entries[number] = replace(
            entry, block_table=replace(entry.block_table, indices=table)
        )
    return replace(
        batch,
        inputs=replace(
            batch.inputs,
            attention=AttentionBatch(entries, attention.queries),
        ),
    )


def restore_writes(batch, cache: PrefixCache | None):
    """Snapshot the cache blocks the batch writes and return its restorer.

    Blocks are snapshotted only when ``cache`` is given and the attention
    input has write indices. Each block is saved whole, quantization buffers
    and initialized flags included, along with ``decode_force_finish`` when
    present, so capturing or warming a call leaves the live prefix
    unchanged. Reading write addresses to the host happens here, during
    startup preparation, outside graph capture.
    """
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    attention = getattr(batch.inputs, "attention", None)
    for number, entry in (
        () if cache is None or attention is None else attention.entries.items()
    ):
        if (
            not isinstance(entry, (PagedInput, SegmentedInput))
            or entry.write_indices is None
        ):
            continue
        blocks = tuple(
            sorted(
                {
                    int(value) // entry.block_table.block_size
                    for value in entry.write_indices.cpu().tolist()
                    if value >= 0
                }
            )
        )
        # Only the layers addressed through this table own these pages.
        for name in cache.config.layers:
            if cache.table(name) != number:
                continue
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
    context,
    batch,
    call,
    *,
    pools=None,
    cache=None,
    predicates=None,
    warmup=None,
):
    """Capture numerical batch output and greedy decoding on common backing.

    The graph returns ``(ExecutionOutput, SamplerOutput | None)`` and retains
    ``batch`` as its fixed input. The state ``restore_writes`` snapshots is
    restored after the warm call and after capture.
    """
    restore = _prepare_capture(context, batch, cache)

    def compute(static):
        output = call(static)
        return output, greedy_decode(static, output, predicates)

    return CUDAGraphRunner.capture(
        context,
        batch,
        compute,
        pools=pools,
        restore=restore,
        warmup=None if warmup is None else lambda value: warmup(compute, value),
    )


def capture_hidden(
    context, batch, call, *, pools=None, cache=None, warmup=None
):
    """Capture a prefill bucket's hidden states on the batch's fixed backing.

    ``call(static)`` returns the ``[tokens, hidden]`` backbone output of the
    padded ``batch``, which the graph retains as its fixed input. The state
    ``restore_writes`` snapshots is restored after the warm call and after
    capture.
    """
    restore = _prepare_capture(context, batch, cache)
    return CUDAGraphRunner.capture(
        context, batch, call, pools=pools, restore=restore, warmup=warmup
    )


def _prepare_capture(context, batch, cache):
    """Plan the batch's attention and snapshot the cache blocks it writes."""
    with context.activate():
        attention = getattr(batch.inputs, "attention", None)
        if attention is not None:
            context.bind_attention(attention)
        return restore_writes(batch, cache)


def _replay(graph, batch):
    """Copy ``batch`` into the graph's inputs, rebind its lengths, replay.

    Attention plans take the live host sequence lengths and start pages
    with the captured device addresses. Returns the graph's retained output
    views, which the next replay overwrites.
    """
    context = graph.executable.context
    with context.activate():
        graph.inputs.copy(batch)
        attention = getattr(batch.inputs, "attention", None)
        if attention is not None:
            context.bind_attention(
                bind_attention(graph.inputs.value.inputs.attention, attention),
                replay=True,
            )
        return graph.executable.replay()


def replay_batch(graph: CUDAGraphRunner, batch, *, rows=None, borrow=False):
    """Replay prepared inputs and retain the live output rows.

    ``rows`` is the live row count (default: ``batch.row_count``); outputs of
    padding rows are dropped. With ``borrow``, the result views the graph's
    output storage, which the next replay overwrites; otherwise it is cloned.
    """
    output, greedy = _replay(graph, batch)
    count = batch.row_count if rows is None else rows
    result = replace(
        output,
        values=output.values[:count],
        vocabularies=output.vocabularies[:count],
        layouts=output.layouts[:count],
        greedy=trim_greedy(greedy, count),
    )
    return result if borrow else result.clone()


def replay_hidden(graph: CUDAGraphRunner, batch) -> torch.Tensor:
    """Replay a prefill bucket and return its hidden states.

    The result is the graph's ``[token bucket, hidden]`` output view, whose
    rows past the batch's live tokens hold padding; the next replay of any
    graph sharing that backing overwrites it.
    """
    return _replay(graph, batch)


def greedy_decode(
    batch: InputBatch,
    output: ExecutionOutput,
    predicate_state: torch.Tensor | None,
) -> SamplerOutput | None:
    """Derive graph-capturable greedy tokens and continuation state.

    Returns None unless the batch is paged text input with one query token
    per row, ``predicate_state`` and ``decode_force_finish`` are present,
    every row selects last logits with one output per row, and all rows
    share one vocabulary partition. Otherwise a row is ``valid`` when its
    maximum logit is finite, ``active`` when its request's predicate is set,
    and continues when valid, active and not forced to finish.

    Raises:
        ValueError: If the per-row logits are not one contiguous tensor.
    """
    force_finish = batch.decode_force_finish
    if (
        not isinstance(batch.inputs, TextInput)
        or not all(
            isinstance(entry, PagedInput)
            for entry in batch.inputs.attention.entries.values()
        )
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

    # Four row-length sections [valid | active | tokens | accepted], the
    # layout ``uniserve_worker.sampling.sampler.sampling_columns`` builds;
    # greedy decode accepts no drafts, so the last is zero. trim_greedy
    # slices padded rows per section.
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
