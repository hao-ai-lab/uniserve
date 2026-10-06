"""Fixed-address input buffers for homogeneous numerical calls.

Each batch runner of ``ModelExecutor`` owns one ``InputBuffers`` instance.
The runner's call kinds select its subclass through ``input_buffer_config``.
The buffers allocate every device column once at its configured capacity,
and ``prepare_inputs`` copies one call's rows into leading slices of
those columns. Because the addresses never change, text graph buckets capture
the buffer columns themselves, and ``graph_inputs.pad_text`` can widen the
input slices in place to a bucket's capacity.

Token buffers hold copied input columns, and canvas buffers hold token canvases
and their answer slots. Diffusion buffers hold attention metadata, positions
and timesteps, while borrowing the rows' latents.
The attention metadata of rows that name their request slots is gathered on
the device from the slots' resident block tables (``gather_rows``); a caller
may instead pass a prepared host attention batch (``copy_attention``).
Vision, latent-encoding and image-decode buffers own only the
request-slot column and borrow the rows' tensors. Borrowed tensors must
already reside on the runner's device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import accumulate

import numpy as np
import torch

from uniserve.diffusion.canvas import CanvasState
from uniserve.math import bucketed_length
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
from uniserve.runtime.device import fill_cpu_bools, fill_cpu_ints
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig, adjacent_view
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor._decode_inputs import (
    gather_request_decode_inputs,
)
from uniserve_worker.model_executor._row_inputs import (
    ROW_SECTIONS,
    gather_request_rows,
    row_columns_size,
    row_columns_stride,
)
from uniserve_worker.model_executor.attention import (
    prepare_attention,
)
from uniserve_worker.model_executor.diffusion_inputs import (
    DecodeInput,
    DiffusionRow,
    ImageBuilder,
)
from uniserve_worker.model_executor.image_inputs import DecodeRow, VisionRow
from uniserve_worker.model_executor.input_batch import (
    CanvasRow,
    CanvasStepInput,
    CanvasStepRow,
    InputBatch,
    InputRow,
    ReadoutInput,
    TokenRow,
)
from uniserve_worker.protocol.call import ForwardMode, MediaCall
from uniserve_worker.storage.host_buffers import HostBuffers


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
            # (``AttentionBuffers.gather_rows``).
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


# The rows of ``CanvasBuffers.step_columns``, per canvas step: its
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


class InputBuffers:
    """Own request-slot buffers and borrow already placed numerical views.

    ``max_inflight`` sets the depth of the host rings (pinned on CUDA) that
    source asynchronous host-to-device copies; with a depth above one, the
    host can fill the next call's values while earlier copies are still
    pending. The device columns
    themselves are single: successive calls reuse them in the order of the
    stream that copies them.
    """

    # The row types accepted by this input buffer class.
    row_type: type[InputRow] | tuple[type[InputRow], ...]

    # Each field ``config.buffers()`` names becomes a fixed-address column
    # attribute of the same name at construction.
    request_pool_indices: torch.Tensor

    def __init__(self, *, config: RowBufferConfig, device, max_inflight=1):
        self.config = config
        self.device = torch.device(device)
        self.max_rows = config.max_rows
        fields = config.buffers()
        self._backing = TensorBuffers.allocate(fields, device=self.device)
        for name, tensor in self._backing.view(fields).items():
            tensor.zero_()
            setattr(self, name, tensor)
        self._request_host = HostBuffers(
            self.max_rows, dtype=torch.int64, depth=max_inflight, device=device
        )

    def close(self):
        self._request_host.close()
        self._backing.close()

    def validate_rows(self, rows):
        """Check types and capacity before preparing or partitioning rows."""
        if not 0 < len(rows) <= self.max_rows:
            raise ValueError("row count exceeds input-buffer capacity")
        if any(not isinstance(row, self.row_type) for row in rows):
            types = (
                self.row_type
                if isinstance(self.row_type, tuple)
                else (self.row_type,)
            )
            names = " or ".join(kind.__name__ for kind in types)
            raise TypeError(f"these buffers require {names} inputs")

    def prepare_inputs(self, rows, *, forward_mode, **numerical):
        """Prepare one numerical call and retain its output request slots.

        Copies are enqueued on the current CUDA stream, and the host ring's
        reuse fence is recorded there. ``ModelExecutor`` prepares inputs under
        the runner's lane stream; consumers must order after that stream.
        ``numerical`` is forwarded to ``_prepare_inputs``.

        Returns:
            An ``InputBatch`` whose tensors are views of this owner's fixed
            columns or of the rows' borrowed tensors.

        Raises:
            ValueError: The row count is zero or exceeds capacity, the rows
                mix computations, or the input buffer subclass rejects the rows.
            TypeError: A row is not the declared ``row_type``, or token or
                diffusion input buffers receive unsupported attention.
            WorkerError: ``prepare_attention`` rejects the rows while attention
                is built from ``cache`` and ``tables``.
        """
        self.validate_rows(rows)
        # Token rows of any ForwardMode share one buffer layout; a media row
        # must match the call kind exactly.
        if any(
            row.forward_mode != forward_mode
            and not (
                isinstance(row.forward_mode, ForwardMode)
                and isinstance(forward_mode, ForwardMode)
            )
            for row in rows
        ):
            raise ValueError("one input call requires homogeneous computations")

        slot, host = self._request_host.acquire()
        fill_cpu_ints(host, tuple(row.request_pool_idx for row in rows))
        requests = self.request_pool_indices[: len(rows)]
        requests.copy_(host[: len(rows)], non_blocking=True)
        self._request_host.record_copy(slot)

        inputs, selections, finish = self._prepare_inputs(rows, **numerical)
        return InputBatch(forward_mode, inputs, requests, selections, finish)

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        """Copy validated rows into this capability's numerical input.

        ``attention`` supplies prepared attention columns; otherwise this method
        builds them from ``cache`` and ``tables``. ``states`` holds resident
        decode continuations. Returns the input, the rows' output selections
        and the optional decode force-finish column. Input buffers that do not
        use these keywords ignore them.
        """
        raise NotImplementedError

    def _device_view(self, value):
        if value is None or value.device != self.device:
            raise ValueError(f"input tensor must already be on {self.device}")
        return value


class AttentionBuffers(InputBuffers):
    """Copy attention metadata and positions into fixed-address columns."""

    positions: torch.Tensor
    block_tables: torch.Tensor
    start_pages: torch.Tensor
    cache_lengths: torch.Tensor
    query_lengths: torch.Tensor
    cumulative_query_lengths: torch.Tensor
    cumulative_prefix_lengths: torch.Tensor
    write_indices: torch.Tensor

    def __init__(self, *, config: AttentionBufferConfig, **options):
        super().__init__(config=config, **options)
        self.max_tokens = config.max_tokens
        self.table_widths = config.table_widths
        # -1 is the attention write-index sentinel for a token that writes no
        # cache slot; unused capacity starts inert.
        self.write_indices.fill_(-1)
        # Pinned sources of the host columns ``gather_rows`` copies.
        self._row_host = HostBuffers(
            self.row_columns.shape,
            dtype=torch.int64,
            depth=options.get("max_inflight", 1),
            device=self.device,
        )

    def close(self):
        self._row_host.close()
        super().close()

    def validate_rows(self, rows):
        super().validate_rows(rows)
        if sum(row.query_tokens for row in rows) > self.max_tokens:
            raise ValueError("query tokens exceed input-buffer capacity")

    def gather_rows(self, rows, *, tables, cache):
        """Gather the attention columns of token rows from their slots' tables.

        ``rows`` are one call's ``AttentionRow`` values, validated against
        the installed tables (``attention.prepare_attention``). Every numerical
        table selects the pages ``attention.table_pages`` selects: all of a
        full table's installed pages, and a windowed table's pages from the
        first one the row's first query reaches back to, through the row's
        last query. The host supplies per-row lengths, each table's selected
        page count per row and each token's row with one copy; the units,
        first selected pages and write addresses are gathered on the device
        from the request slots' resident tables (``gather_request_rows``).

        Returns an ``AttentionBatch`` over this owner's columns: paged
        entries when any row writes the cache, in which the tokens of rows
        that do not write carry the -1 address, and otherwise segmented
        entries in which every query sees its whole current sequence. Query
        and prefix lengths and windowed tables' first selected pages keep
        exact host mirrors.

        Raises:
            WorkerError: From ``attention.prepare_attention``.
            ValueError: A read-only call has a causal row, or a table or the
                call exceeds this owner's capacity.
        """
        if tables is None:
            raise invalid_descriptor("paged attention requires request tables")

        pages = prepare_attention(rows, cache=cache)
        count = len(rows)
        queries = tuple(int(row.query_tokens) for row in rows)
        prefixes = tuple(int(row.seq_len) for row in rows)
        write = tuple(bool(row.write_kv) for row in rows)
        causal = tuple(bool(row.causal) for row in rows)
        flags = self.prepare_causality(causal)
        if not any(write) and any(causal):
            raise ValueError(
                "read-only prefix/current calls require noncausal current "
                "sequences"
            )
        total = sum(queries)
        if count > self.max_rows or total > self.write_indices.shape[1]:
            raise ValueError("attention rows exceed input-buffer capacity")

        # Host columns, each section ``stride`` long (``_row_inputs``), in a
        # pinned ring slot that keeps its previous call's values: the kernel
        # reads only the live rows of each section, and the offsets' leading
        # zeros are rewritten.
        stride = row_columns_stride(self.max_rows)
        number = len(self.table_widths)
        section = (ROW_SECTIONS + number) * stride
        slot, pinned = self._row_host.acquire()
        host = pinned.numpy()
        host[:count] = [row.request_pool_idx for row in rows]
        host[stride : stride + count] = prefixes
        host[2 * stride : 2 * stride + count] = queries
        host[3 * stride : 3 * stride + count] = write
        host[4 * stride] = host[5 * stride] = 0
        host[4 * stride + 1 : 4 * stride + count + 1] = list(
            accumulate(queries)
        )
        host[5 * stride + 1 : 5 * stride + count + 1] = list(
            accumulate(prefixes)
        )
        host[section : section + total] = np.repeat(np.arange(count), queries)

        widths, firsts, block_sizes = [], [], []
        for table_number, table in enumerate(pages):
            start = (ROW_SECTIONS + table_number) * stride
            host[start : start + count] = table.lengths
            widths.append(table.width)
            block_sizes.append(table.block_size)
            firsts.append(table.start_pages if table.windowed else None)
        if any(
            width > capacity
            for width, capacity in zip(widths, self.table_widths, strict=True)
        ):
            raise ValueError("block tables exceed input-buffer capacity")

        self.row_columns[: section + total].copy_(
            pinned[: section + total], non_blocking=True
        )
        self._row_host.record_copy(slot)
        gather_request_rows(
            columns=self.row_columns,
            rows=count,
            tokens=total,
            width=max(widths),
            request_unit_tables=tables.unit_tables,
            request_start_pages=tables.start_pages,
            table_shapes=tables.table_shapes,
            block_tables=self.block_tables,
            start_pages=self.start_pages,
            cache_lengths=self.cache_lengths,
            query_lengths=self.query_lengths,
            query_offsets=self.cumulative_query_lengths,
            prefix_offsets=self.cumulative_prefix_lengths,
            write_indices=self.write_indices,
        )

        shared = SequenceLengths(
            host=queries,
            values=self.query_lengths[:count],
            offsets=self.cumulative_query_lengths[: count + 1],
        )
        prefix_lengths = SequenceLengths(
            host=prefixes,
            values=self.cache_lengths[:count],
            offsets=self.cumulative_prefix_lengths[: count + 1],
        )
        entries = {}
        for table_number, (width, first_pages) in enumerate(
            zip(widths, firsts, strict=True)
        ):
            blocks = BlockTable(
                self.block_tables[table_number, :count, :width],
                block_sizes[table_number],
                None
                if first_pages is None
                else self.start_pages[table_number, :count],
                first_pages,
            )
            if any(write):
                entries[table_number] = PagedInput(
                    shared,
                    prefix_lengths,
                    blocks,
                    self.write_indices[table_number, :total],
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

    def copy_attention(self, attention):
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
        if any(table >= len(self.table_widths) for table in entries):
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
            values=self._vector(self.query_lengths, attention.queries.values),
            offsets=self._vector(
                self.cumulative_query_lengths, attention.queries.offsets
            ),
        )
        prefixes = SequenceLengths(
            host=first.prefixes.host,
            values=self._vector(self.cache_lengths, first.prefixes.values),
            offsets=self._vector(
                self.cumulative_prefix_lengths, first.prefixes.offsets
            ),
        )

        flags = (
            self.prepare_causality(first.causal)
            if isinstance(first, PagedInput)
            else None
        )
        batch_attention = {
            number: self._copy_table(number, entry, queries, prefixes, count)
            for number, entry in entries.items()
        }
        if flags is not None:
            batch_attention = {
                number: replace(entry, causal_values=flags)
                for number, entry in batch_attention.items()
            }
        return AttentionBatch(batch_attention, queries)

    def prepare_causality(self, causal, *, dynamic=False):
        """Prepare one shared visibility column for a mixed numerical call."""
        if not dynamic and len(set(causal)) < 2:
            return None
        values = self.causal_values[: len(causal)]
        values.copy_(torch.tensor(causal, dtype=torch.int32), non_blocking=True)
        return values

    def _copy_table(self, number, entry, queries, prefixes, count):
        """Copy one table's pages, start pages and write addresses."""
        source = entry.block_table
        if source.indices.shape[1] > self.table_widths[number]:
            raise ValueError("block tables exceed input-buffer capacity")

        table = self.block_tables[number, :count, : source.indices.shape[1]]
        table.copy_(source.indices, non_blocking=True)
        start = None
        if source.start_page is not None:
            start = self.start_pages[number, :count]
            start.copy_(source.start_page, non_blocking=True)
        blocks = BlockTable(
            table, source.block_size, start, source.start_page_host
        )

        writes = (
            None
            if entry.write_indices is None
            else self._vector(self.write_indices[number], entry.write_indices)
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

    def _copy_positions(self, sources, lengths):
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
            self.positions[: packed.shape[0], : packed.shape[1]].copy_(
                packed, non_blocking=True
            )
            return

        offset = 0
        for source, count in zip(shaped, lengths, strict=True):
            self.positions[: source.shape[0], offset : offset + count].copy_(
                source, non_blocking=True
            )
            offset += count

    def clear_padding(self, *, live_rows, rows, live_tokens, tokens):
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
            self.block_tables[:, live_rows:rows].zero_()
            self.start_pages[:, live_rows:rows].zero_()
        if tokens > live_tokens:
            self.write_indices[:, live_tokens:tokens].fill_(-1)

    def _vector(self, target, source):
        if source.numel() > target.numel():
            raise ValueError("numerical column exceeds input-buffer capacity")
        result = target[: source.numel()]
        result.copy_(source.reshape(-1), non_blocking=True)
        return result


class TokenBuffers(AttentionBuffers):
    """Prepare text inputs and resident decode continuations."""

    row_type = TokenRow

    input_ids: torch.Tensor
    embedding_mask: torch.Tensor
    decode_force_finish: torch.Tensor
    # Provisioned only when the configuration declares a hidden width.
    input_embeddings: torch.Tensor | None

    def __init__(
        self, *, config: TokenBufferConfig, image_builder=None, **options
    ):
        super().__init__(config=config, **options)
        self.max_text_tokens = config.max_text_tokens
        self.hidden_size = config.hidden_size
        self.image_builder = image_builder
        # Unused token capacity holds ID 1, the same value request-indexed
        # decode gathers into inactive rows.
        self.input_ids.fill_(1)
        self.input_embeddings = getattr(self, "input_embeddings", None)
        self._finish_host = HostBuffers(
            self.max_rows,
            dtype=torch.bool,
            depth=options.get("max_inflight", 1),
            device=self.device,
        )

    def close(self):
        self._finish_host.close()
        super().close()

    def validate_rows(self, rows):
        super().validate_rows(rows)
        if sum(row.query_tokens for row in rows) > self.max_text_tokens:
            raise ValueError("text token count exceeds input-buffer capacity")

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        count = len(rows)
        # The force-finish column feeds ``graph_inputs.greedy_decode``. It is
        # copied only when every row carries a tagged device decode predicate.
        finish = None
        if all(
            row.decode_predicate is not None and row.decode_predicate_tagged
            for row in rows
        ):
            finish = self.decode_force_finish[:count]
            slot, host = self._finish_host.acquire()
            fill_cpu_bools(host, tuple(row.decode_force_finish for row in rows))
            finish.copy_(host[:count], non_blocking=True)
            self._finish_host.record_copy(slot)

        # Request-indexed decode rows carry no token views; their next token
        # and position live in ``states`` at the row's request slot. Slot 0
        # is the inactive sentinel and never a valid source.
        indexed = any(row.request_indexed_decode for row in rows)
        if indexed:
            if states is None or any(
                row.forward_mode is not ForwardMode.DECODE
                or row.token_ids is not None
                or row.positions is not None
                or row.token_embeddings is not None
                or row.token_embedding_mask is not None
                or row.selection is None
                or not 0 < row.request_pool_idx <= states.request_pool_size
                for row in rows
                if row.request_indexed_decode
            ):
                raise ValueError(
                    "indexed decode requires valid resident request slots"
                )

            # A fully indexed call on one CUDA device with its resident unit
            # tables gathers every input on device. Native preparation checks
            # cache extents and returns page widths and first-page mirrors.
            if (
                attention is None
                and cache is not None
                and tables is not None
                and states.device == tables.unit_tables.device == self.device
                and self.device.type == "cuda"
                and all(row.request_indexed_decode for row in rows)
            ):
                pages = prepare_attention(rows, cache=cache)
                inputs = self._indexed(rows, pages, tables, states)
                return inputs, tuple(row.selection for row in rows), finish

            # Otherwise (prepared attention supplied, not every row indexed,
            # or cache, tables and states not all resident on this CUDA
            # device), borrow each indexed row's token and position views
            # from ``states`` and prepare the ordinary path.
            rows = tuple(
                replace(
                    row,
                    token_ids=states.future_input_tokens[
                        row.request_pool_idx, :1
                    ],
                    positions=states.logical_lengths[
                        row.request_pool_idx : row.request_pool_idx + 1
                    ],
                    request_indexed_decode=False,
                )
                if row.request_indexed_decode
                else row
                for row in rows
            )

        batch_attention = (
            self.gather_rows(rows, tables=tables, cache=cache)
            if attention is None
            else self.copy_attention(attention)
        )
        inputs = self._text(rows, batch_attention)
        return inputs, tuple(row.selection for row in rows), finish

    def _text(self, rows, attention):
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
            row.token_ids is None
            or row.positions is None
            or row.selection is None
            for row in rows
        ):
            raise ValueError(
                "text inputs require IDs, positions and output selection"
            )

        lengths = tuple(row.token_ids.numel() for row in rows)
        total = sum(lengths)
        if min(lengths) < 1 or total > self.max_text_tokens:
            raise ValueError("text token count exceeds input-buffer capacity")

        # Clear axes and mask entries that rows below may leave unwritten.
        self.positions[:, :total].zero_()
        self.embedding_mask[:total].zero_()

        has_embeddings = any(row.token_embeddings is not None for row in rows)
        # Multimodal prefills share one capture representation whether the
        # current prompt inserts features or consists entirely of token IDs.
        use_embeddings = has_embeddings or (
            self.image_builder is not None
            and any(row.forward_mode is not ForwardMode.DECODE for row in rows)
        )
        embeddings = None
        if use_embeddings:
            embeddings = self.input_embeddings
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
            self.input_ids[:total].copy_(input_ids, non_blocking=True)

        # Multimodal text uses one three-axis representation for prefill and
        # decode. Spatial coordinates of ordinary text stay zero; numerical
        # layers can consume the prepared axes directly on every replay.
        self._copy_positions(tuple(row.positions for row in rows), lengths)
        axes = max(
            3 if self.image_builder is not None else 1,
            *(
                1 if row.positions.ndim == 1 else row.positions.shape[0]
                for row in rows
            ),
        )
        offset = 0
        for row, length in zip(rows, lengths, strict=True):
            if input_ids is None:
                self.input_ids[offset : offset + length].copy_(
                    row.token_ids.reshape(-1), non_blocking=True
                )

            if row.token_embeddings is not None:
                values = row.token_embeddings.reshape(length, -1)
                if values.shape[1] != self.hidden_size:
                    raise ValueError(
                        "input embeddings must match the hidden width"
                    )
                if embeddings is None:
                    raise ValueError(
                        "the lane does not provision embedding inputs"
                    )

                embeddings[offset : offset + length].copy_(
                    values, non_blocking=True
                )
                mask = self.embedding_mask[offset : offset + length]
                if row.token_embedding_mask is None:
                    mask.fill_(True)
                else:
                    mask.copy_(
                        row.token_embedding_mask.reshape(-1), non_blocking=True
                    )

            offset += length

        return TextInput(
            self.input_ids[:total],
            self.positions[0, :total]
            if axes == 1
            else self.positions[:axes, :total],
            attention,
            EmbeddingReplacement(
                embeddings[:total], self.embedding_mask[:total]
            )
            if embeddings is not None
            else None,
        )

    def _indexed(self, rows, pages, tables, states):
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
        widths = []
        for number, table in enumerate(pages):
            if table.width > self.table_widths[number]:
                raise ValueError("block tables exceed input-buffer capacity")
            # Power-of-two widths keep the input shapes few.
            widths.append(
                min(bucketed_length(table.width), self.table_widths[number])
            )

        gather_request_decode_inputs(
            request_pool_indices=self.request_pool_indices,
            request_unit_tables=tables.unit_tables,
            request_start_pages=tables.start_pages,
            table_shapes=tables.table_shapes,
            request_cache_lengths=tables.verified_lengths,
            request_tokens=states.future_input_tokens[:, 0],
            request_positions=states.logical_lengths,
            input_ids=self.input_ids,
            positions=self.positions
            if self.image_builder is not None
            else self.positions[:1],
            block_tables=self.block_tables,
            start_pages=self.start_pages,
            cache_lengths=self.cache_lengths,
            query_lengths=self.query_lengths,
            query_offsets=self.cumulative_query_lengths,
            prefix_offsets=self.cumulative_prefix_lengths,
            write_indices=self.write_indices,
            rows=count,
            columns=max(widths),
        )

        queries = SequenceLengths(
            host=(1,) * count,
            values=self.query_lengths[:count],
            offsets=self.cumulative_query_lengths[: count + 1],
        )
        prefixes = SequenceLengths(
            host=tuple(row.seq_len for row in rows),
            values=self.cache_lengths[:count],
            offsets=self.cumulative_prefix_lengths[: count + 1],
        )
        causal = tuple(row.causal for row in rows)

        entries = {}
        for number, (table, width) in enumerate(
            zip(pages, widths, strict=True)
        ):
            windowed = table.windowed
            entries[number] = PagedInput(
                queries,
                prefixes,
                BlockTable(
                    self.block_tables[number, :count, :width],
                    table.block_size,
                    self.start_pages[number, :count] if windowed else None,
                    table.start_pages if windowed else None,
                ),
                self.write_indices[number, :count],
                causal,
            )
        attention = AttentionBatch(entries, queries)

        if self.image_builder is not None:
            positions = self.positions[:, :count]
        else:
            positions = self.positions[0, :count]

        return TextInput(self.input_ids[:count], positions, attention)


class CanvasBuffers(AttentionBuffers):
    """Prepare token canvases, their slots and candidate reads on their lane.

    Canvas tokens, positions, attention metadata and slot indices use fixed
    columns bounded by the configured rows and tokens. The candidate ids a
    call reads have no configured bound, so their backing grows to the
    largest call prepared so far and keeps its address until a larger call
    arrives; the readout head that consumes them runs outside any graph.

    Generating canvas steps (``CanvasStepRow``) gather their rows' resident
    sampler state from the ``CanvasSlots`` bound with ``bind_canvas_slots``;
    their canvas tokens and self-conditioning embeddings are the model's
    inputs. One call contains either readout rows or step rows.
    """

    row_type = (CanvasRow, CanvasStepRow)

    input_ids: torch.Tensor
    slot_tokens: torch.Tensor
    step_columns: torch.Tensor

    def __init__(self, *, config: CanvasBufferConfig, **options):
        super().__init__(config=config, **options)
        # Candidate matrix and selection indices share one int64 backing.
        self._candidates = torch.empty(0, dtype=torch.int64, device=self.device)
        self._canvas_slots = None
        self._canvas_backing = None
        # Pinned sources of the step columns, transferred with one copy.
        self._step_host = HostBuffers(
            self.step_columns.shape,
            dtype=torch.int64,
            depth=options.get("max_inflight", 1),
            device=self.device,
        )

    def close(self):
        self._step_host.close()
        if self._canvas_backing is not None:
            self._canvas_backing.close()
        super().close()

    def bind_canvas_slots(self, slots) -> None:
        """Borrow the resident sampler state that canvas steps gather from."""
        self._canvas_slots = slots
        self._canvas_backing = TensorBuffers.allocate(
            self.sampler_buffers(
                max_rows=min(self.max_rows, slots.request_pool_size),
                canvas_length=slots.canvas_length,
                hidden_size=slots.hidden_size,
                history_depth=slots.history_depth,
                dtype=slots.banks["self_conditioning"].dtype,
            ),
            device=self.device,
        )

    @staticmethod
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

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        batch_attention = (
            self.gather_rows(rows, tables=tables, cache=cache)
            if attention is None
            else self.copy_attention(attention)
        )
        steps = sum(isinstance(row, CanvasStepRow) for row in rows)
        if steps:
            if steps != len(rows):
                raise ValueError(
                    "one canvas call contains readout rows or canvas steps"
                )
            return self._prepare_steps(rows, batch_attention), (), None

        lengths = tuple(row.query_tokens for row in rows)
        total = sum(lengths)
        if total > self.max_tokens:
            raise ValueError("canvas tokens exceed input-buffer capacity")

        # Rows pack back to back: their tokens and positions use one
        # copy each, and each row's slots shift by the tokens of the rows
        # before it.
        values = tuple(row.token_ids.reshape(-1) for row in rows)
        packed = None
        if len({value.device for value in values}) == 1:
            packed = adjacent_view(values)
            packed = torch.cat(values) if packed is None else packed
            self.input_ids[:total].copy_(packed, non_blocking=True)
        self._copy_positions(tuple(row.positions for row in rows), lengths)

        slots, groups, offset = [], [], 0
        for row, length in zip(rows, lengths, strict=True):
            if packed is None:
                self.input_ids[offset : offset + length].copy_(
                    row.token_ids.reshape(-1), non_blocking=True
                )
            offsets = row.candidate_offsets
            for index, token in enumerate(row.slot_tokens):
                slots.append(offset + token)
                groups.append(
                    row.candidate_ids[offsets[index] : offsets[index + 1]]
                )
            offset += length

        count = len(slots)
        slot_tokens = self.slot_tokens[:count]
        slot_tokens.copy_(
            torch.from_numpy(np.asarray(slots, dtype=np.int64)),
            non_blocking=True,
        )

        # A [slots, width] matrix holds every slot's candidates, padded with
        # its first one; the selection names the real entries in order.
        # Both copy from one host array into the shared backing.
        sizes = np.fromiter(map(len, groups), dtype=np.int64, count=count)
        width = int(sizes.max())
        host = np.empty(count * width + int(sizes.sum()), dtype=np.int64)
        matrix = host[: count * width].reshape(count, width)
        for index, group in enumerate(groups):
            matrix[index, : len(group)] = group
            matrix[index, len(group) :] = group[0]
        host[count * width :] = np.flatnonzero(
            np.arange(width)[None, :] < sizes[:, None]
        )
        backing = self._candidate_backing(host.size)
        backing[: host.size].copy_(torch.from_numpy(host), non_blocking=True)
        candidates = backing[: count * width].view(count, width)
        selected = backing[count * width : host.size]

        canvas = CanvasInput(
            self.input_ids[:total], self.positions[0, :total], batch_attention
        )
        inputs = ReadoutInput(
            canvas,
            slot_tokens,
            candidates,
            selected,
            tuple(len(row.candidate_ids) for row in rows),
        )
        return inputs, (), None

    def _prepare_steps(self, rows, batch_attention):
        """Prepare canvas steps: their sampling coordinates and gathered state.

        Every row's canvas follows the previous one in the packed tokens.
        The rows' argmax histories are gathered at the depth of their shared
        stability threshold.
        """
        state = self._canvas_slots
        if state is None:
            raise ValueError("canvas steps require bound sampler state")
        depths = {row.sampling.stability for row in rows}
        lengths = {row.canvas_length for row in rows}
        if len(depths) != 1 or lengths != {state.canvas_length}:
            raise ValueError(
                "canvas steps of one call share the resident canvas length "
                "and one stability threshold"
            )

        count, length = len(rows), state.canvas_length
        if count * length > self.max_tokens:
            raise ValueError("canvas tokens exceed input-buffer capacity")
        self._copy_positions(
            tuple(row.positions for row in rows), (length,) * count
        )
        # The step columns' rows are ``max_rows`` apart, so the live values
        # of all four transfer with one copy, which also carries the stale
        # tails of the first three that no reader sees.
        slot, pinned = self._step_host.acquire()
        host = pinned.numpy()
        host[0, :count] = [row.request_pool_idx for row in rows]
        host[1, :count] = [row.seed for row in rows]
        host[2, :count] = [row.block for row in rows]
        host[3, :count] = [row.step for row in rows]
        size = (len(STEP_COLUMNS) - 1) * self.max_rows + count
        self.step_columns.view(-1)[:size].copy_(
            pinned.view(-1)[:size], non_blocking=True
        )
        self._step_host.record_copy(slot)
        vectors = dict(
            zip(
                STEP_COLUMNS,
                self.step_columns[:, :count].unbind(0),
                strict=True,
            )
        )

        # The gathered canvases are the model's input tokens, and their
        # self-conditioning rows its self-conditioning input, back to back.
        views = dict(
            self._canvas_backing.view(
                self.sampler_buffers(
                    max_rows=count,
                    canvas_length=length,
                    hidden_size=state.hidden_size,
                    history_depth=depths.pop(),
                    dtype=state.banks["self_conditioning"].dtype,
                )
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
            self.positions[0, : count * length],
            batch_attention,
            self_conditioning=self_conditioning,
        )
        return CanvasStepInput(
            canvas,
            rows_state,
            views,
            vectors["step_slots"],
            tuple(row.sampling for row in rows),
            first=all(row.step == 0 for row in rows),
        )

    def _candidate_backing(self, size):
        """Return candidate storage of at least ``size`` int64 elements.

        Replacing the backing is ordered after earlier readers by the
        caching allocator, since every reader runs on the same execution stream.
        Serving prepares inputs outside inference mode, so backing is allocated
        as an ordinary tensor even when startup input preparation, which runs in
        inference mode, grows it first.
        """
        if self._candidates.numel() < size:
            with torch.inference_mode(False):
                self._candidates = torch.empty(
                    size, dtype=torch.int64, device=self.device
                )
        return self._candidates


class DiffusionBuffers(AttentionBuffers):
    """Prepare spatial attention and solver times without text storage."""

    row_type = DiffusionRow

    timesteps: torch.Tensor
    step: torch.Tensor

    def __init__(self, *, image_builder: ImageBuilder, **options):
        super().__init__(**options)
        self.image_builder = image_builder
        self.step.zero_()

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        batch_attention = (
            self.gather_rows(rows, tables=tables, cache=cache)
            if attention is None
            else self.copy_attention(attention)
        )
        return self._images(rows, batch_attention), (), None

    def _images(self, rows, attention):
        """Copy denoising positions and timesteps, then bind image inputs.

        Positions and timesteps are copied into the fixed columns; each
        row's latent is borrowed and must already be on this device. The
        image builder binds them into the denoiser's typed input at the
        zero step index.
        """
        if self.image_builder is None:
            raise ValueError("image denoising requires its bound input builder")

        sizes = tuple(
            image.Config(row.image_height, row.image_width) for row in rows
        )
        lengths = tuple(
            self.image_builder.sequence_length(size) for size in sizes
        )
        if sum(lengths) > self.max_tokens:
            raise ValueError("image sequences exceed input-buffer capacity")

        if any(row.timestep is None or row.positions is None for row in rows):
            raise ValueError(
                "image denoising requires positions and a timestep"
            )
        self._copy_positions(tuple(row.positions for row in rows), lengths)

        positions, offset = [], 0
        for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
            positions.append(self.positions[:, offset : offset + length])
            self.timesteps[index].copy_(row.timestep.reshape(()))
            offset += length

        return self.image_builder.bind(
            samples=tuple(self._device_view(row.latent) for row in rows),
            sizes=sizes,
            timesteps=tuple(
                self.timesteps[index : index + 1] for index in range(len(rows))
            ),
            positions=tuple(positions),
            attention=attention,
            step=self.step,
        )


class VisionBuffers(InputBuffers):
    """Borrow prepared image views; codecs need no token or KV backing."""

    row_type = VisionRow

    def _prepare_inputs(self, rows, **numerical):
        return (
            VisionInput(
                tuple(self._device_view(row.encode_pixels) for row in rows),
                tuple(
                    None
                    if row.encode_grid is None
                    else self._device_view(row.encode_grid)
                    for row in rows
                ),
                tuple(row.encode_grid_shape for row in rows),
            ),
            (),
            None,
        )


class DecodeBuffers(InputBuffers):
    """Borrow image latents together with their requested output extents."""

    row_type = DecodeRow

    def _prepare_inputs(self, rows, **numerical):
        return (
            DecodeInput(
                tuple(self._device_view(row.latent) for row in rows),
                tuple(
                    image.Config(row.image_height, row.image_width)
                    for row in rows
                ),
            ),
            (),
            None,
        )


def input_buffer_config(kind, limits: TokenBufferConfig):
    """Select only the backing consumed by this numerical capability.

    ``limits`` is the entry's text input buffer configuration, derived from
    ``bootstrap.capacity.input_buffer_config``; non-text kinds keep only the
    fields their input buffers need.

    Returns:
        The ``InputBuffers`` subclass and the configuration to build it.

    Raises:
        ValueError: ``kind`` has no fixed input buffers.
    """
    if kind is ForwardMode.TOKEN_DENOISING:
        return CanvasBuffers, CanvasBufferConfig(
            limits.max_rows, limits.max_tokens, limits.table_widths
        )
    if isinstance(kind, ForwardMode):
        return TokenBuffers, limits
    if kind is MediaCall.DENOISING:
        return DiffusionBuffers, DiffusionBufferConfig(
            limits.max_rows, limits.max_tokens, limits.table_widths
        )
    if kind in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}:
        return VisionBuffers, RowBufferConfig(limits.max_rows)
    if kind is MediaCall.IMAGE_DECODING:
        return DecodeBuffers, RowBufferConfig(limits.max_rows)
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
