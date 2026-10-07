"""Worker graph shapes, stable numerical inputs and captured call ownership.

Text calls replay graphs captured at configured bucket shapes. Selection
(``text_shape``) picks a bucket for a staged batch, ``pad_text`` widens the
batch's views of the runner's fixed staging to that bucket and makes the
padding inert, and ``capture_batch``/``replay_batch`` capture and replay the
call together with graph-capturable greedy decoding. The ``select_*``
helpers turn configured sizes into the capture shapes ``ModelExecutor``
prepares at startup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

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


@dataclass(frozen=True, slots=True)
class PrefillShape:
    """A prefill capture bucket.

    Attributes:
        token_bucket: Flat token count the bucket pads to.
        row_bucket: Row count the bucket pads to, including one padding
            sequence: ``text_shape`` only selects a bucket with more rows than
            the batch.
        live_rows: Row count of the startup capture batch in
            ``uniserve_worker.model_executor.startup.prepare_prefill``;
            ``select_prefill_captures`` sets it to the next smaller
            configured row size, one for the smallest.
        causal: Attention causality of every row.
        selection: Output selection of every row.
    """

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
    """Keep the configured shape combinations that fit staging capacity.

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


def select_prefill_captures(
    token_sizes, row_sizes, *, max_rows, max_tokens, visual=False
):
    """Build prefill capture buckets from configured token and row sizes.

    ``text_shape`` routes a batch only to a row bucket strictly larger than
    its row count, leaving room for the padding sequence, so row sizes
    of one are dropped and each bucket's ``live_rows`` is the next smaller
    configured row size (one for the smallest). Buckets are built while
    their ``live_rows`` fits ``max_rows``; a bucket's own row count is the
    configured size and may exceed ``max_rows``, since a full batch still
    needs a strictly larger bucket. With ``visual``, each shape is also
    captured for noncausal rows selecting logits or hidden states.
    """
    buckets: list[PrefillShape] = []
    variants: tuple[tuple[bool, TokenSelection], ...] = (
        (True, TokenSelection.LAST_LOGITS),
    )
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
    return AttentionBatch(entries, queries)


def text_shape(batch, *, decode_sizes, prefill_shapes, table_widths):
    """Choose a bucket with the call's attention and output semantics.

    Returns:
        ``(rows, tokens, widths, decode)`` for ``pad_text``, where
        ``widths[t]`` is the width to stage for numerical table ``t``: at
        least its staged width and its ``table_widths`` floor. A
        single-token causal decode batch selecting last logits uses the
        first configured decode size at least its row count, with
        ``tokens == rows``. Any other batch, or a decode batch no decode size
        fits, uses the smallest prefill shape, by rows then tokens, with more
        rows than the batch and at least its token count. None when the
        input is not paged text over tables ``0..n-1``, rows mix causality or
        output selection, a row has no query token, or no configured shape
        fits.

    Raises:
        ValueError: If the attention input has no host query lengths.
    """
    inputs = batch.inputs
    if not isinstance(inputs, TextInput):
        return None
    if set(inputs.attention.entries) != set(
        range(len(inputs.attention.entries))
    ):
        return None

    # Tables share the query domain and causality; they differ only in their
    # pages, and every table is paged, in table order.
    entries = tuple(
        entry
        for entry in (
            inputs.attention.entries[table]
            for table in range(len(inputs.attention.entries))
        )
        if isinstance(entry, PagedInput)
    )
    if len(entries) != len(inputs.attention.entries):
        return None
    attention = entries[0]
    selection = batch.token_selections[0]
    causal = attention.causal[0]
    if any(value is not selection for value in batch.token_selections):
        return None
    if any(value != causal for entry in entries for value in entry.causal):
        return None
    queries = attention.queries.host
    if queries is None:
        raise ValueError("graph shape selection requires host query lengths")
    if any(length < 1 for length in queries):
        return None

    widths = tuple(
        max(
            table_widths[table] if table < len(table_widths) else 1,
            entry.block_table.indices.shape[1],
        )
        for table, entry in enumerate(entries)
    )
    if (
        batch.forward_mode is ForwardMode.DECODE
        and selection is TokenSelection.LAST_LOGITS
        and causal
        and all(length == 1 for length in queries)
    ):
        rows = next(
            (value for value in decode_sizes if value >= batch.row_count), None
        )
        if rows is not None:
            return rows, rows, widths, True

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
    return shape.row_bucket, shape.token_bucket, widths, False


def _fixed_view(tensor, shape):
    """Borrow a view of ``shape`` from where ``tensor`` starts in storage.

    The view keeps ``tensor``'s strides and may extend past its extent into
    the fixed staging buffer it views, which is how a live batch grows to
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


def pad_text(batch, rows, tokens, widths, decode):
    """Borrow a fixed bucket and make padding inert, including cache writes.

    Physical unit zero is valid storage. Padding queries read disposable
    values but never write a unit: their write indices are -1, their start
    pages zero, and their results are discarded. Prefill padding belongs to
    one additional numerical sequence. ``widths[t]`` is the staged width of
    numerical table ``t``.

    The batch's tensors must be views of the runner's fixed staging: padding
    is written in place past the live extents, and the returned batch views
    the same storage at the bucket shape. Padding rows use request slot zero,
    which the block tables reserve for padding.
    """
    inputs, live_rows = batch.inputs, batch.row_count
    attention, live_tokens = inputs.attention, inputs.input_ids.numel()
    if rows < live_rows or tokens < live_tokens:
        raise ValueError("graph shape is smaller than its live inputs")
    causal = next(iter(attention.entries.values())).causal[0]

    padding, extra = tokens - live_tokens, rows - live_rows
    if padding and not extra:
        raise ValueError("token padding requires an additional sequence")
    if not padding and not extra:
        # The staged lengths and offsets already describe the whole bucket.
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

    # The shared query domain pads once; every table then pads its own page
    # columns, prefixes and write addresses against it.
    queries = _fixed_view(attention.queries.values, (rows,))
    queries[live_rows:].zero_()
    if decode:
        queries[live_rows:].fill_(1)
    elif padding:
        queries[live_rows : live_rows + 1].fill_(padding)
    query_offsets = _fixed_view(attention.queries.offsets, (rows + 1,))
    torch.cumsum(queries, dim=0, out=query_offsets[1:])
    shared = SequenceLengths(
        host=host_queries, values=queries, offsets=query_offsets
    )

    entries = {}
    for number, entry in attention.entries.items():
        blocks = entry.block_table
        table = _fixed_view(blocks.indices, (rows, widths[number]))
        table[live_rows:].zero_()
        start = blocks.start_page
        start_host = blocks.start_page_host
        if start is not None:
            start = _fixed_view(start, (rows,))
            start[live_rows:].zero_()
            if start_host is not None:
                start_host = start_host + (0,) * extra

        prefix = _fixed_view(entry.prefixes.values, (rows,))
        prefix[live_rows:].zero_()
        prefix_offsets = _fixed_view(entry.prefixes.offsets, (rows + 1,))
        torch.cumsum(prefix, dim=0, out=prefix_offsets[1:])

        writes = entry.write_indices
        if writes is not None:
            writes = _fixed_view(writes, (tokens,))
            writes[live_tokens:].fill_(-1)

        entries[number] = PagedInput(
            shared,
            SequenceLengths(
                host=entry.prefixes.host + (0,) * extra,
                values=prefix,
                offsets=prefix_offsets,
            ),
            BlockTable(table, blocks.block_size, start, start_host),
            writes,
            (causal,) * rows,
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
    if cache is not None and attention is not None:
        for number, entry in attention.entries.items():
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
                views = cache.state(name).transfer_views(blocks)
                for tensors in views.values():
                    snapshots.extend(
                        (tensor, tensor.clone()) for tensor in tensors
                    )
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
    """Capture numerical batch output and greedy decoding on common backing.

    The graph returns ``(ExecutionOutput, SamplerOutput | None)`` and retains
    ``batch`` as its fixed input. The state ``restore_writes`` snapshots is
    restored after the warm call and after capture.
    """
    with context.activate():
        attention = getattr(batch.inputs, "attention", None)
        if attention is not None:
            context.bind_attention(attention)
        restore = restore_writes(batch, cache)

    def compute(static):
        output = call(static)
        return output, greedy_decode(static, output, predicates)

    return CUDAGraphRunner.capture(
        context, batch, compute, pools=pools, restore=restore
    )


def replay_batch(graph: CUDAGraphRunner, batch, *, rows=None, borrow=False):
    """Replay staged inputs and retain the live output rows.

    ``rows`` is the live row count (default: ``batch.row_count``); outputs of
    padding rows are dropped. With ``borrow``, the result views the graph's
    output storage, which the next replay overwrites; otherwise it is cloned.
    """
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
            if attention is None:
                raise ValueError(
                    "hidden graph results require live attention sequences"
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
        # A paged batch always carries its packed query domain; only a dense
        # singleton has none.
        or batch.inputs.attention.queries is None
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
