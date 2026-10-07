"""Fixed-address input buffers for homogeneous numerical calls.

Each batch runner of ``ModelExecutor`` owns one ``InputBuffers`` instance.
Rust owns backing, host columns, preparation selection and buffer retirement.
The buffers allocate every device column once at its configured capacity,
and ``prepare_inputs`` copies one call's rows into leading slices of
those columns. Because the addresses never change, text graph buckets capture
the buffer columns themselves, and ``graph_inputs.pad_text`` can widen the
input slices in place to a bucket's capacity.

Token buffers hold copied input columns, and canvas buffers hold token canvases
and their answer slots. Diffusion buffers hold attention metadata, positions
and timesteps, while borrowing the rows' latents.
The attention metadata of rows that name their request slots is gathered on
the device from the slots' resident block tables; a caller
may instead pass a prepared host attention batch (``copy_attention``).
Vision, latent-encoding and image-decode buffers own only the
request-slot column and borrow the rows' tensors. Borrowed tensors must
already reside on the runner's device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from uniserve.diffusion.canvas import CanvasState
from uniserve.media import image
from uniserve.model import (
    CanvasInput,
    EmbeddingReplacement,
    TextInput,
    VisionInput,
)
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.tensors import BufferConfig, adjacent_view
from uniserve_worker._uniserve_ipc import InputBuffers as InputBuffers
from uniserve_worker.model_executor._decode_inputs import (
    gather_request_decode_inputs,
)
from uniserve_worker.model_executor._row_inputs import (
    gather_request_rows,
    row_columns_size,
)
from uniserve_worker.model_executor.diffusion_inputs import (
    DecodeInput,
)
from uniserve_worker.model_executor.input_batch import (
    CanvasStepInput,
    ReadoutInput,
)
from uniserve_worker.protocol.call import ForwardMode, MediaCall


@dataclass(frozen=True, slots=True)
class RowBufferConfig:
    """Request-slot capacity of one homogeneous call."""

    max_rows: int

    def __post_init__(self):
        if self.max_rows < 1:
            raise ValueError("input-buffer row capacity must be positive")

    def buffers(self):
        return {
            "request_pool_indices": BufferConfig((self.max_rows,), torch.int64)
        }


@dataclass(frozen=True, slots=True)
class AttentionBufferConfig(RowBufferConfig):
    """Sequence and page-table capacities shared by attention computations.

    ``positions`` has three axes so multimodal positions fit; input preparation
    fills only the leading axis for one-axis positions. ``max_tokens`` bounds
    the positions and write-index columns; ``input_buffer_config`` gives
    text and diffusion input buffers built from one ``TokenBufferConfig`` the
    same bound. ``table_widths`` holds the most pages one call selects per
    row of each numerical block table, in table order.
    """

    max_tokens: int
    table_widths: tuple[int, ...]

    def __post_init__(self):
        RowBufferConfig.__post_init__(self)
        if (
            self.max_tokens < 1
            or not self.table_widths
            or min(self.table_widths) < 1
        ):
            raise ValueError(
                "attention token and block bounds must be positive"
            )

    def buffers(self):
        rows, tokens = self.max_rows, self.max_tokens
        tables = len(self.table_widths)
        return {
            **RowBufferConfig.buffers(self),
            "positions": BufferConfig((3, tokens), torch.int64),
            # [table, row, column]: every table shares the widest capacity so
            # each table's input view is one strided slice.
            "block_tables": BufferConfig(
                (tables, rows, max(self.table_widths)), torch.int32
            ),
            # [table, row]: first selected logical page of windowed tables.
            "start_pages": BufferConfig((tables, rows), torch.int32),
            "cache_lengths": BufferConfig((rows,), torch.int32),
            "query_lengths": BufferConfig((rows,), torch.int32),
            "causal_values": BufferConfig((rows,), torch.int32),
            "cumulative_query_lengths": BufferConfig((rows + 1,), torch.int32),
            "cumulative_prefix_lengths": BufferConfig((rows + 1,), torch.int32),
            # [table, token]: each table addresses its own units.
            "write_indices": BufferConfig((tables, tokens), torch.int64),
            # Host columns of a call's rows for their device buffers
            # (``_gather_attention``).
            "row_columns": BufferConfig(
                (row_columns_size(rows, tables, tokens),), torch.int64
            ),
        }


@dataclass(frozen=True, slots=True)
class TokenBufferConfig(AttentionBufferConfig):
    """Token and optional embedding storage for a language model call.

    ``max_text_tokens`` bounds the token-ID and embedding columns and may be
    smaller than ``max_tokens``, which also covers diffusion calls prepared
    from the same limits. A ``hidden_size`` of zero provisions no embedding
    column.
    """

    max_text_tokens: int
    hidden_size: int
    embedding_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        AttentionBufferConfig.__post_init__(self)
        if (
            self.hidden_size < 0
            or not 1 <= self.max_text_tokens <= self.max_tokens
        ):
            raise ValueError(
                "input-buffer text or embedding capacity is invalid"
            )
        if self.embedding_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise ValueError("input embeddings require a logical compute dtype")

    def buffers(self):
        fields = {
            **AttentionBufferConfig.buffers(self),
            "input_ids": BufferConfig((self.max_text_tokens,), torch.int64),
            "embedding_mask": BufferConfig((self.max_text_tokens,), torch.bool),
            "decode_force_finish": BufferConfig((self.max_rows,), torch.bool),
        }
        if self.hidden_size:
            fields["input_embeddings"] = BufferConfig(
                (self.max_text_tokens, self.hidden_size), self.embedding_dtype
            )
        return fields


# The rows of ``InputBuffers.step_columns``, per canvas step: its
# request slot, seed, block and step index.
STEP_COLUMNS = ("step_slots", "step_seeds", "step_blocks", "step_indices")


@dataclass(frozen=True, slots=True)
class CanvasBufferConfig(AttentionBufferConfig):
    """Canvas token, slot and sampling columns of a token-denoising call.

    ``max_rows`` bounds the canvas rows of one call and ``max_tokens`` their
    tokens, which also bounds the slots they read. A generating canvas step
    copies its row's request slot and sampling coordinates into the rows of
    ``step_columns`` (``STEP_COLUMNS``).
    """

    def buffers(self):
        return {
            **AttentionBufferConfig.buffers(self),
            "input_ids": BufferConfig((self.max_tokens,), torch.int64),
            "slot_tokens": BufferConfig((self.max_tokens,), torch.int64),
            "step_columns": BufferConfig(
                (len(STEP_COLUMNS), self.max_rows), torch.int64
            ),
        }


@dataclass(frozen=True, slots=True)
class DiffusionBufferConfig(AttentionBufferConfig):
    """Attention columns, one solver time per image sequence and a step.

    ``step`` is the batch input's [1] int64 step index. Batched image rows
    carry their own timesteps and are integrated per request, so the batched
    denoisers never read it; it stays zero.
    """

    def buffers(self):
        return {
            **AttentionBufferConfig.buffers(self),
            "timesteps": BufferConfig((self.max_rows,), torch.float32),
            "step": BufferConfig((1,), torch.int64),
        }


def _device_view(buffers, value):
    if value is None or value.device != buffers.device:
        raise ValueError(f"input tensor must already be on {buffers.device}")
    return value


def copy_attention(buffers, attention):
    """Copy every table's paged or segmented columns into fixed buffers.

    ``attention`` is an ``AttentionBatch`` whose entries share one query
    domain and one prefix-length column. Returns a batch over the same
    tables whose device tensors are leading views of this owner's
    columns; host length metadata and host start pages are carried over
    unchanged. A segmented entry is accepted only when its current
    sequences are fully visible, and is rebuilt with each query seeing
    its whole current sequence.

    Raises:
        TypeError: An entry is neither paged nor segmented.
        ValueError: A table or column exceeds capacity, entries do not
            share prefix lengths, or a segmented entry is not fully
            visible or lacks host query lengths.
    """
    entries = attention.entries
    if not all(
        isinstance(entry, (PagedInput, SegmentedInput))
        for entry in entries.values()
    ):
        raise TypeError(
            "worker token buffers require paged or prefix/current attention"
        )
    if any(table >= len(buffers.table_widths) for table in entries):
        raise ValueError("attention tables exceed input-buffer capacity")
    first = next(iter(entries.values()))
    if any(
        entry.prefixes.values is not first.prefixes.values
        for entry in entries.values()
    ):
        raise ValueError("attention input tables must share prefixes")

    # The query domain and the prefix lengths are copied once and shared
    # by every table's entry.
    count = attention.queries.batch_size
    queries = SequenceLengths(
        host=attention.queries.host,
        values=_vector(buffers.query_lengths, attention.queries.values),
        offsets=_vector(
            buffers.cumulative_query_lengths, attention.queries.offsets
        ),
    )
    prefixes = SequenceLengths(
        host=first.prefixes.host,
        values=_vector(buffers.cache_lengths, first.prefixes.values),
        offsets=_vector(
            buffers.cumulative_prefix_lengths, first.prefixes.offsets
        ),
    )

    flags = (
        prepare_causality(buffers, first.causal)
        if isinstance(first, PagedInput)
        else None
    )
    batch_attention = {
        number: _copy_table(buffers, number, entry, queries, prefixes, count)
        for number, entry in entries.items()
    }
    if flags is not None:
        batch_attention = {
            number: replace(entry, causal_values=flags)
            for number, entry in batch_attention.items()
        }
    return AttentionBatch(batch_attention, queries)


def prepare_causality(buffers, causal, *, dynamic=False):
    """Prepare one shared visibility column for a mixed numerical call."""
    if not dynamic and len(set(causal)) < 2:
        return None
    values = buffers.causal_values[: len(causal)]
    values.copy_(torch.tensor(causal, dtype=torch.int32), non_blocking=True)
    return values


def _copy_table(buffers, number, entry, queries, prefixes, count):
    """Copy one table's pages, start pages and write addresses."""
    source = entry.block_table
    if source.indices.shape[1] > buffers.table_widths[number]:
        raise ValueError("block tables exceed input-buffer capacity")

    table = buffers.block_tables[number, :count, : source.indices.shape[1]]
    table.copy_(source.indices, non_blocking=True)
    start = None
    if source.start_page is not None:
        start = buffers.start_pages[number, :count]
        start.copy_(source.start_page, non_blocking=True)
    blocks = BlockTable(table, source.block_size, start, source.start_page_host)

    writes = (
        None
        if entry.write_indices is None
        else _vector(buffers.write_indices[number], entry.write_indices)
    )

    if isinstance(entry, PagedInput):
        return PagedInput(queries, prefixes, blocks, writes, entry.causal)

    if not entry.fully_visible_current:
        raise ValueError(
            "image preparation requires fully visible current sequences"
        )
    maximum = queries.maximum
    if maximum is None:
        raise ValueError(
            "image preparation requires host query lengths for visibility"
        )

    return SegmentedInput(
        queries,
        prefixes,
        blocks,
        writes,
        queries.values[:, None].expand(-1, maximum),
        True,
    )


def _copy_positions(buffers, sources, lengths):
    """Copy the rows' positions into the packed position columns.

    Each source is ``[tokens]`` or ``[axes, tokens]`` with one or three
    axes over its row's tokens; a one-axis row fills axis zero only.
    Rows whose positions share an axis count and a device use one
    copy of their concatenation, the rest row by row.
    """
    shaped = []
    for source, count in zip(sources, lengths, strict=True):
        if source.ndim == 1:
            source = source.unsqueeze(0)
        if (
            source.ndim != 2
            or source.shape[0] not in (1, 3)
            or source.shape[1] != count
        ):
            raise ValueError(
                "positions must have one or three axes over the token span"
            )
        shaped.append(source)

    if (
        len({source.shape[0] for source in shaped}) == 1
        and len({source.device for source in shaped}) == 1
    ):
        packed = shaped[0] if len(shaped) == 1 else torch.cat(shaped, 1)
        buffers.positions[: packed.shape[0], : packed.shape[1]].copy_(
            packed, non_blocking=True
        )
        return

    offset = 0
    for source, count in zip(shaped, lengths, strict=True):
        buffers.positions[: source.shape[0], offset : offset + count].copy_(
            source, non_blocking=True
        )
        offset += count


def clear_padding(buffers, *, live_rows, rows, live_tokens, tokens):
    """Make the padding of every input table inert, one launch per column.

    Rows ``live_rows..rows`` of every table's units and start pages
    become zero (unit zero is valid storage, page zero the first), and
    tokens ``live_tokens..tokens`` of every table's write addresses
    become -1, no cache write. Each column is one ``[table, ...]``
    tensor whose tables are the input views (``_copy_table``), so one
    launch covers every table; columns past a table's graph width are
    never read.
    """
    if rows > live_rows:
        buffers.block_tables[:, live_rows:rows].zero_()
        buffers.start_pages[:, live_rows:rows].zero_()
    if tokens > live_tokens:
        buffers.write_indices[:, live_tokens:tokens].fill_(-1)


def _vector(target, source):
    if source.numel() > target.numel():
        raise ValueError("numerical column exceeds input-buffer capacity")
    result = target[: source.numel()]
    result.copy_(source.reshape(-1), non_blocking=True)
    return result


def _gather_attention(buffers, pages, queries, prefixes, write, causal, tables):
    """Gather resident pages and borrow numerical attention views."""
    count, total = len(queries), sum(queries)
    widths = tuple(table.width for table in pages)
    firsts = tuple(
        table.start_pages if table.windowed else None for table in pages
    )
    block_sizes = tuple(table.block_size for table in pages)
    flags = prepare_causality(buffers, causal)

    gather_request_rows(
        columns=buffers.row_columns,
        rows=count,
        tokens=total,
        width=max(widths),
        request_unit_tables=tables.unit_tables,
        request_start_pages=tables.start_pages,
        table_shapes=tables.table_shapes,
        block_tables=buffers.block_tables,
        start_pages=buffers.start_pages,
        cache_lengths=buffers.cache_lengths,
        query_lengths=buffers.query_lengths,
        query_offsets=buffers.cumulative_query_lengths,
        prefix_offsets=buffers.cumulative_prefix_lengths,
        write_indices=buffers.write_indices,
    )

    shared = SequenceLengths(
        host=queries,
        values=buffers.query_lengths[:count],
        offsets=buffers.cumulative_query_lengths[: count + 1],
    )
    prefix_lengths = SequenceLengths(
        host=prefixes,
        values=buffers.cache_lengths[:count],
        offsets=buffers.cumulative_prefix_lengths[: count + 1],
    )
    entries = {}
    for table_number, (width, first_pages) in enumerate(
        zip(widths, firsts, strict=True)
    ):
        blocks = BlockTable(
            buffers.block_tables[table_number, :count, :width],
            block_sizes[table_number],
            None
            if first_pages is None
            else buffers.start_pages[table_number, :count],
            first_pages,
        )
        if any(write):
            entries[table_number] = PagedInput(
                shared,
                prefix_lengths,
                blocks,
                buffers.write_indices[table_number, :total],
                causal,
                flags,
            )
        else:
            entries[table_number] = SegmentedInput(
                shared,
                prefix_lengths,
                blocks,
                None,
                shared.values[:, None].expand(-1, max(queries)),
                True,
            )
    return AttentionBatch(entries, shared)


def _text(buffers, rows, attention):
    """Copy token IDs, positions and optional embeddings into columns.

    Rows are packed back to back in row order. Positions keep one axis
    unless a row supplies three axes or an image builder is bound, in
    which case all three axes are returned. Embedding inputs are used
    when any row supplies embeddings, and on a lane with an image
    builder whenever a row is not a decode row; tokens without supplied
    embeddings keep a false embedding mask.

    Raises:
        ValueError: A row lacks IDs, positions or a selection or has no
            tokens, the call exceeds text capacity, positions or
            embeddings have the wrong shape, or embeddings are required
            on a lane without an embedding column.
    """
    if any(
        row.token_ids is None or row.positions is None or row.selection is None
        for row in rows
    ):
        raise ValueError(
            "text inputs require IDs, positions and output selection"
        )

    lengths = tuple(row.token_ids.numel() for row in rows)
    total = sum(lengths)
    if min(lengths) < 1 or total > buffers.max_text_tokens:
        raise ValueError("text token count exceeds input-buffer capacity")

    # Clear axes and mask entries that rows below may leave unwritten.
    buffers.positions[:, :total].zero_()
    buffers.embedding_mask[:total].zero_()

    has_embeddings = any(row.token_embeddings is not None for row in rows)
    # Multimodal prefills share one capture representation whether the
    # current prompt inserts features or consists entirely of token IDs.
    use_embeddings = has_embeddings or (
        buffers.image_builder is not None
        and any(row.forward_mode is not ForwardMode.DECODE for row in rows)
    )
    embeddings = None
    if use_embeddings:
        embeddings = buffers.input_embeddings
        if embeddings is None:
            raise ValueError("the lane does not provision embedding inputs")
        embeddings[:total].zero_()

    # When every row's IDs are on one device, copy them with one call:
    # through their shared view when the rows are already adjacent in
    # memory, otherwise after concatenating them. Mixed source devices
    # fall back to per-row copies in the loop below.
    values = tuple(row.token_ids.reshape(-1) for row in rows)
    input_ids = None
    if len({value.device for value in values}) == 1:
        input_ids = adjacent_view(values)
        input_ids = torch.cat(values) if input_ids is None else input_ids
        buffers.input_ids[:total].copy_(input_ids, non_blocking=True)

    # Multimodal text uses one three-axis representation for prefill and
    # decode. Spatial coordinates of ordinary text stay zero; numerical
    # layers can consume the prepared axes directly on every replay.
    _copy_positions(buffers, tuple(row.positions for row in rows), lengths)
    axes = max(
        3 if buffers.image_builder is not None else 1,
        *(
            1 if row.positions.ndim == 1 else row.positions.shape[0]
            for row in rows
        ),
    )
    offset = 0
    for row, length in zip(rows, lengths, strict=True):
        if input_ids is None:
            buffers.input_ids[offset : offset + length].copy_(
                row.token_ids.reshape(-1), non_blocking=True
            )

        if row.token_embeddings is not None:
            values = row.token_embeddings.reshape(length, -1)
            if values.shape[1] != buffers.hidden_size:
                raise ValueError("input embeddings must match the hidden width")
            if embeddings is None:
                raise ValueError("the lane does not provision embedding inputs")

            embeddings[offset : offset + length].copy_(
                values, non_blocking=True
            )
            mask = buffers.embedding_mask[offset : offset + length]
            if row.token_embedding_mask is None:
                mask.fill_(True)
            else:
                mask.copy_(
                    row.token_embedding_mask.reshape(-1), non_blocking=True
                )

        offset += length

    return TextInput(
        buffers.input_ids[:total],
        buffers.positions[0, :total]
        if axes == 1
        else buffers.positions[:axes, :total],
        attention,
        EmbeddingReplacement(embeddings[:total], buffers.embedding_mask[:total])
        if embeddings is not None
        else None,
    )


def _indexed(buffers, rows, pages, tables, states, widths):
    """Gather resident decode rows into paged input buffers on device.

    One kernel reads the rows' request slots from
    ``request_pool_indices`` (already copied by ``prepare_inputs`` on the
    same stream) and fills token, position, length, offset, table, start
    page and write-index columns of every numerical table over the whole
    row capacity. Rows past ``len(rows)`` are initialized as inert
    padding and their request slots reset to 0, so a padded decode graph
    replays consistent inputs. Every row decodes one query token.
    ``pages`` holds the selected pages of every table on the host,
    which bounds each table's width and mirrors its start pages.

    Raises:
        ValueError: A table's selected pages exceed its capacity.
    """
    count = len(rows)
    gather_request_decode_inputs(
        request_pool_indices=buffers.request_pool_indices,
        request_unit_tables=tables.unit_tables,
        request_start_pages=tables.start_pages,
        table_shapes=tables.table_shapes,
        request_cache_lengths=tables.verified_lengths,
        request_tokens=states.future_input_tokens[:, 0],
        request_positions=states.logical_lengths,
        input_ids=buffers.input_ids,
        positions=buffers.positions
        if buffers.image_builder is not None
        else buffers.positions[:1],
        block_tables=buffers.block_tables,
        start_pages=buffers.start_pages,
        cache_lengths=buffers.cache_lengths,
        query_lengths=buffers.query_lengths,
        query_offsets=buffers.cumulative_query_lengths,
        prefix_offsets=buffers.cumulative_prefix_lengths,
        write_indices=buffers.write_indices,
        rows=count,
        columns=max(widths),
    )

    queries = SequenceLengths(
        host=(1,) * count,
        values=buffers.query_lengths[:count],
        offsets=buffers.cumulative_query_lengths[: count + 1],
    )
    prefixes = SequenceLengths(
        host=tuple(row.seq_len for row in rows),
        values=buffers.cache_lengths[:count],
        offsets=buffers.cumulative_prefix_lengths[: count + 1],
    )
    causal = tuple(row.causal for row in rows)

    entries = {}
    for number, (table, width) in enumerate(zip(pages, widths, strict=True)):
        windowed = table.windowed
        entries[number] = PagedInput(
            queries,
            prefixes,
            BlockTable(
                buffers.block_tables[number, :count, :width],
                table.block_size,
                buffers.start_pages[number, :count] if windowed else None,
                table.start_pages if windowed else None,
            ),
            buffers.write_indices[number, :count],
            causal,
        )
    attention = AttentionBatch(entries, queries)

    if buffers.image_builder is not None:
        positions = buffers.positions[:, :count]
    else:
        positions = buffers.positions[0, :count]

    return TextInput(buffers.input_ids[:count], positions, attention)


def _resident_rows(rows, states):
    """Borrow current device continuations without a host readback."""
    return tuple(
        replace(
            row,
            token_ids=states.future_input_tokens[row.request_pool_idx, :1],
            positions=states.logical_lengths[
                row.request_pool_idx : row.request_pool_idx + 1
            ],
            request_indexed_decode=False,
        )
        if row.request_indexed_decode
        else row
        for row in rows
    )


def sampler_buffers(
    *, max_rows, canvas_length, hidden_size, history_depth, dtype
):
    """Contiguous sampler input storage owned by one execution stream."""
    return {
        "canvas": BufferConfig((max_rows, canvas_length), torch.int64),
        "history": BufferConfig(
            (max_rows, history_depth, canvas_length), torch.int64
        ),
        "self_conditioning": BufferConfig(
            (max_rows, canvas_length, hidden_size), dtype
        ),
    }


def _readout(
    buffers, rows, batch_attention, slots, host, width, row_candidates
):
    """Copy packed canvas tokens and their native candidate layout."""
    lengths = tuple(row.query_tokens for row in rows)
    total = sum(lengths)

    # Native slots already index the packed tokens. Tokens and positions
    # from a shared device use one copy each.
    values = tuple(row.token_ids.reshape(-1) for row in rows)
    packed = None
    if len({value.device for value in values}) == 1:
        packed = adjacent_view(values)
        packed = torch.cat(values) if packed is None else packed
        buffers.input_ids[:total].copy_(packed, non_blocking=True)
    _copy_positions(buffers, tuple(row.positions for row in rows), lengths)

    if packed is None:
        offset = 0
        for row, length in zip(rows, lengths, strict=True):
            buffers.input_ids[offset : offset + length].copy_(
                row.token_ids.reshape(-1), non_blocking=True
            )
            offset += length

    count = slots.numel()
    slot_tokens = buffers.slot_tokens[:count]
    slot_tokens.copy_(slots, non_blocking=True)
    backing = buffers.candidate_storage
    backing[: host.numel()].copy_(host, non_blocking=True)
    candidates = backing[: count * width].view(count, width)
    selected = backing[count * width : host.numel()]
    canvas = CanvasInput(
        buffers.input_ids[:total], buffers.positions[0, :total], batch_attention
    )
    return ReadoutInput(
        canvas, slot_tokens, candidates, selected, row_candidates
    )


def _steps(buffers, rows, batch_attention, views, sampling, first):
    """Gather resident sampler tensors into this call's numerical views."""
    state = buffers.canvas_slots
    views = dict(views)
    count, length = len(rows), state.canvas_length
    _copy_positions(
        buffers, tuple(row.positions for row in rows), (length,) * count
    )
    vectors = dict(
        zip(
            STEP_COLUMNS,
            buffers.step_columns[:, :count].unbind(0),
            strict=True,
        )
    )

    state.gather(vectors["step_slots"], views)
    self_conditioning = views["self_conditioning"].view(
        count * length, state.hidden_size
    )
    rows_state = CanvasState(
        seed=vectors["step_seeds"],
        block=vectors["step_blocks"],
        step=vectors["step_indices"],
        canvas=views["canvas"],
        history=views["history"],
        self_conditioning=self_conditioning,
    )
    canvas = CanvasInput(
        views["canvas"].view(-1),
        buffers.positions[0, : count * length],
        batch_attention,
        self_conditioning=self_conditioning,
    )
    return CanvasStepInput(
        canvas,
        rows_state,
        views,
        vectors["step_slots"],
        sampling,
        first=first,
    )


def _images(buffers, rows, attention):
    """Copy denoising positions and timesteps, then bind image inputs.

    Positions and timesteps are copied into the fixed columns; each
    row's latent is borrowed and must already be on this device. The
    image builder binds them into the denoiser's typed input at the
    zero step index.
    """
    if buffers.image_builder is None:
        raise ValueError("image denoising requires its bound input builder")

    sizes = tuple(
        image.Config(row.image_height, row.image_width) for row in rows
    )
    lengths = tuple(
        buffers.image_builder.sequence_length(size) for size in sizes
    )
    if sum(lengths) > buffers.max_tokens:
        raise ValueError("image sequences exceed input-buffer capacity")

    if any(row.timestep is None or row.positions is None for row in rows):
        raise ValueError("image denoising requires positions and a timestep")
    _copy_positions(buffers, tuple(row.positions for row in rows), lengths)

    positions, offset = [], 0
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        positions.append(buffers.positions[:, offset : offset + length])
        buffers.timesteps[index].copy_(row.timestep.reshape(()))
        offset += length

    return buffers.image_builder.bind(
        samples=tuple(_device_view(buffers, row.latent) for row in rows),
        sizes=sizes,
        timesteps=tuple(
            buffers.timesteps[index : index + 1] for index in range(len(rows))
        ),
        positions=tuple(positions),
        attention=attention,
        step=buffers.step,
    )


def _vision(buffers, rows):
    return VisionInput(
        tuple(_device_view(buffers, row.encode_pixels) for row in rows),
        tuple(
            None
            if row.encode_grid is None
            else _device_view(buffers, row.encode_grid)
            for row in rows
        ),
        tuple(row.encode_grid_shape for row in rows),
    )


def _decode(buffers, rows):
    return DecodeInput(
        tuple(_device_view(buffers, row.latent) for row in rows),
        tuple(image.Config(row.image_height, row.image_width) for row in rows),
    )


def input_buffer_config(kind, limits: TokenBufferConfig):
    """Select only the backing consumed by this numerical capability.

    ``limits`` is the entry's text input buffer configuration, derived from
    ``bootstrap.capacity.input_buffer_config``; non-text kinds keep only the
    fields their input buffers need.

    Returns:
        The numerical column configuration for this computation.

    Raises:
        ValueError: ``kind`` has no fixed input buffers.
    """
    if kind is ForwardMode.TOKEN_DENOISING:
        return CanvasBufferConfig(
            limits.max_rows, limits.max_tokens, limits.table_widths
        )
    if isinstance(kind, ForwardMode):
        return limits
    if kind is MediaCall.DENOISING:
        return DiffusionBufferConfig(
            limits.max_rows, limits.max_tokens, limits.table_widths
        )
    if kind in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}:
        return RowBufferConfig(limits.max_rows)
    if kind is MediaCall.IMAGE_DECODING:
        return RowBufferConfig(limits.max_rows)
    raise ValueError(f"unsupported buffered computation {kind}")


def buffered_kinds(*, diffusion: bool):
    """Numerical capabilities served by fixed input buffers.

    Denoising is included only when ``diffusion`` is set, which callers
    derive from whether the model provides an image builder.
    """
    kinds = {
        *ForwardMode,
        MediaCall.VISION_ENCODING,
        MediaCall.LATENT_ENCODING,
        MediaCall.IMAGE_DECODING,
    }
    if diffusion:
        kinds.add(MediaCall.DENOISING)
    return frozenset(kinds)
