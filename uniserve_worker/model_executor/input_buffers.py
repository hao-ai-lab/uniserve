"""Fixed-address staging for homogeneous numerical capability calls."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from uniserve.media import image
from uniserve.model import EmbeddingReplacement, TextInput, VisionInput
from uniserve.nn.attention import (
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime.device import fill_cpu_bools, fill_cpu_ints
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig, adjacent_view
from uniserve_worker.model_executor._decode_inputs import (
    gather_request_decode_inputs,
)
from uniserve_worker.model_executor.attention import cache_pages, columns
from uniserve_worker.model_executor.diffusion_inputs import (
    DecodeInput,
    DiffusionRow,
    ImageBuilder,
)
from uniserve_worker.model_executor.image_inputs import DecodeRow, VisionRow
from uniserve_worker.model_executor.input_batch import (
    InputBatch,
    InputRow,
    TokenRow,
)
from uniserve_worker.protocol.call import ForwardMode, MediaCall
from uniserve_worker.storage.host_buffers import HostBuffers


@dataclass(frozen=True, slots=True)
class RowBufferConfig:
    """Request-slot capacity of one homogeneous staged call."""

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
    """Sequence and page-table capacities shared by attention computations."""

    max_tokens: int
    max_blocks_per_row: int

    def __post_init__(self):
        RowBufferConfig.__post_init__(self)
        if min(self.max_tokens, self.max_blocks_per_row) < 1:
            raise ValueError(
                "attention token and block bounds must be positive"
            )

    def buffers(self):
        rows, tokens = self.max_rows, self.max_tokens
        return {
            **RowBufferConfig.buffers(self),
            "positions": BufferConfig((3, tokens), torch.int64),
            "block_tables": BufferConfig(
                (rows, self.max_blocks_per_row), torch.int32
            ),
            "cache_lengths": BufferConfig((rows,), torch.int32),
            "query_lengths": BufferConfig((rows,), torch.int32),
            "cumulative_query_lengths": BufferConfig((rows + 1,), torch.int32),
            "cumulative_prefix_lengths": BufferConfig((rows + 1,), torch.int32),
            "write_indices": BufferConfig((tokens,), torch.int64),
        }


@dataclass(frozen=True, slots=True)
class TokenBufferConfig(AttentionBufferConfig):
    """Token and optional embedding storage for a language model call."""

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


@dataclass(frozen=True, slots=True)
class DiffusionBufferConfig(AttentionBufferConfig):
    """Attention columns and one solver time per image sequence."""

    def buffers(self):
        return {
            **AttentionBufferConfig.buffers(self),
            "timesteps": BufferConfig((self.max_rows,), torch.float32),
        }


class InputBuffers:
    """Own request-slot staging and borrow already placed numerical views."""

    row_type: type[InputRow]

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

    def prepare_inputs(self, rows, *, forward_mode, **numerical):
        """Stage one numerical call and retain its output request slots."""
        if not 0 < len(rows) <= self.max_rows:
            raise ValueError("row count exceeds input-buffer capacity")
        if any(not isinstance(row, self.row_type) for row in rows):
            raise TypeError(
                f"this staging requires {self.row_type.__name__} inputs"
            )
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
        """Stage validated rows into this capability's numerical input.

        ``attention`` supplies prepared attention columns; otherwise staging
        builds them from ``cache`` and ``tables``. ``states`` holds resident
        decode continuations. Returns the input, the rows' output selections
        and the optional decode force-finish column. Stagings that need none
        of these keywords ignore them.
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
    cache_lengths: torch.Tensor
    query_lengths: torch.Tensor
    cumulative_query_lengths: torch.Tensor
    cumulative_prefix_lengths: torch.Tensor
    write_indices: torch.Tensor

    def __init__(self, *, config: AttentionBufferConfig, **options):
        super().__init__(config=config, **options)
        self.max_tokens = config.max_tokens
        self.max_blocks_per_row = config.max_blocks_per_row
        self.write_indices.fill_(-1)

    def stage_attention(self, attention):
        """Copy paged or segmented attention columns into the lane's fixed.

        buffers.
        """
        if not isinstance(attention, (PagedInput, SegmentedInput)):
            raise TypeError(
                "worker token staging requires paged or prefix/current "
                "attention"
            )

        count = attention.queries.batch_size
        queries = SequenceLengths(
            host=attention.queries.host,
            values=self._vector(self.query_lengths, attention.queries.values),
            offsets=self._vector(
                self.cumulative_query_lengths, attention.queries.offsets
            ),
        )
        prefixes = SequenceLengths(
            host=attention.prefixes.host,
            values=self._vector(self.cache_lengths, attention.prefixes.values),
            offsets=self._vector(
                self.cumulative_prefix_lengths, attention.prefixes.offsets
            ),
        )

        source = attention.block_table.indices
        if source.shape[1] > self.max_blocks_per_row:
            raise ValueError("block tables exceed input-buffer capacity")

        table = self.block_tables[:count, : source.shape[1]]
        table.copy_(source, non_blocking=True)
        blocks = BlockTable(table, attention.block_table.block_size)

        writes = (
            None
            if attention.write_indices is None
            else self._vector(self.write_indices, attention.write_indices)
        )

        if isinstance(attention, PagedInput):
            return PagedInput(
                queries, prefixes, blocks, writes, attention.causal
            )

        if not attention.fully_visible_current:
            raise ValueError(
                "image staging requires fully visible current sequences"
            )
        maximum = queries.maximum
        if maximum is None:
            raise ValueError(
                "image staging requires host query lengths for visibility"
            )

        return SegmentedInput(
            queries,
            prefixes,
            blocks,
            writes,
            queries.values[:, None].expand(-1, maximum),
            True,
        )

    def _positions(self, source, offset, count):
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
        self.positions[: source.shape[0], offset : offset + count].copy_(
            source, non_blocking=True
        )

    def _vector(self, target, source):
        if source.numel() > target.numel():
            raise ValueError("numerical column exceeds input-buffer capacity")
        result = target[: source.numel()]
        result.copy_(source.reshape(-1), non_blocking=True)
        return result


class TokenBuffers(AttentionBuffers):
    """Stage text and resident decode continuations on their execution lane."""

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

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        count = len(rows)
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

            if (
                attention is None
                and cache is not None
                and tables is not None
                and states.device == tables.page_tables.device == self.device
                and self.device.type == "cuda"
                and all(row.request_indexed_decode for row in rows)
            ):
                _, width = cache_pages(rows, cache=cache, tables=tables)
                inputs = self._indexed(rows, width, cache, tables, states)
                return inputs, tuple(row.selection for row in rows), finish

            # Without a compatible resident CUDA cache, materialize one token
            # and position per row and continue through the ordinary path.
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

        attention = (
            columns(rows, cache=cache, tables=tables)
            if attention is None
            else attention
        )
        inputs = self._text(rows, self.stage_attention(attention))
        return inputs, tuple(row.selection for row in rows), finish

    def _text(self, rows, attention):
        """Stage token IDs.

        positions and optional embeddings into the text columns.
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
        # Adjacent continuation rows already share backing. Preserve that
        # layout; concatenate only disjoint columns on a common source device.
        values = tuple(row.token_ids.reshape(-1) for row in rows)
        input_ids = None
        if len({value.device for value in values}) == 1:
            input_ids = adjacent_view(values)
            input_ids = torch.cat(values) if input_ids is None else input_ids
            self.input_ids[:total].copy_(input_ids, non_blocking=True)
        # Multimodal text uses one three-axis representation for prefill and
        # decode. Spatial coordinates of ordinary text stay zero; numerical
        # layers can consume the prepared axes directly on every replay.
        offset, axes = 0, 3 if self.image_builder is not None else 1
        for row, length in zip(rows, lengths, strict=True):
            if input_ids is None:
                self.input_ids[offset : offset + length].copy_(
                    row.token_ids.reshape(-1), non_blocking=True
                )

            self._positions(row.positions, offset, length)
            axes = max(
                axes, 1 if row.positions.ndim == 1 else row.positions.shape[0]
            )

            if row.token_embeddings is not None:
                values = row.token_embeddings.reshape(length, -1)
                if values.shape[1] != self.hidden_size:
                    raise ValueError(
                        "input embeddings must match the hidden width"
                    )
                # A row with embeddings makes the call use embedding inputs.
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

    def _indexed(self, rows, width, cache, tables, states):
        """Gather resident decode rows on device directly into paged staging."""
        count = len(rows)
        gather_request_decode_inputs(
            request_pool_indices=self.request_pool_indices,
            request_page_tables=tables.page_tables,
            request_cache_lengths=tables.verified_lengths,
            request_tokens=states.future_input_tokens[:, 0],
            request_positions=states.logical_lengths,
            input_ids=self.input_ids,
            positions=self.positions
            if self.image_builder is not None
            else self.positions[:1],
            block_tables=self.block_tables[:, :width],
            cache_lengths=self.cache_lengths,
            query_lengths=self.query_lengths,
            query_offsets=self.cumulative_query_lengths,
            prefix_offsets=self.cumulative_prefix_lengths,
            write_indices=self.write_indices,
            rows=count,
            group_id=rows[0].group_id,
            page_size=cache.info.block_size,
        )

        writes = self.write_indices[:count]
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

        attention = PagedInput(
            queries,
            prefixes,
            BlockTable(
                self.block_tables[:count, :width], cache.info.block_size
            ),
            writes,
            tuple(row.causal for row in rows),
        )

        if self.image_builder is not None:
            positions = self.positions[:, :count]
        else:
            positions = self.positions[0, :count]

        return TextInput(self.input_ids[:count], positions, attention)


class DiffusionBuffers(AttentionBuffers):
    """Stage spatial attention and solver times without text storage."""

    row_type = DiffusionRow

    timesteps: torch.Tensor

    def __init__(self, *, image_builder: ImageBuilder, **options):
        super().__init__(**options)
        self.image_builder = image_builder

    def _prepare_inputs(
        self, rows, *, attention=None, cache=None, tables=None, states=None
    ):
        attention = (
            columns(rows, cache=cache, tables=tables)
            if attention is None
            else attention
        )
        return self._images(rows, self.stage_attention(attention)), (), None

    def _images(self, rows, attention):
        """Stage denoising positions and timesteps.

        then bind the image inputs.
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

        positions, offset = [], 0
        for index, (row, size, length) in enumerate(
            zip(rows, sizes, lengths, strict=True)
        ):
            if row.timestep is None or row.positions is None:
                raise ValueError(
                    "image denoising requires positions and a timestep"
                )

            self._positions(row.positions, offset, length)
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
            step_index=0,
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
    """Select only the backing consumed by this numerical capability."""
    if isinstance(kind, ForwardMode):
        return TokenBuffers, limits
    if kind is MediaCall.DENOISING:
        return DiffusionBuffers, DiffusionBufferConfig(
            limits.max_rows, limits.max_tokens, limits.max_blocks_per_row
        )
    if kind in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}:
        return VisionBuffers, RowBufferConfig(limits.max_rows)
    if kind is MediaCall.IMAGE_DECODING:
        return DecodeBuffers, RowBufferConfig(limits.max_rows)
    raise ValueError(f"unsupported staged computation {kind}")


def buffered_kinds(*, diffusion: bool):
    """Numerical capabilities served by fixed row staging."""
    kinds = {
        *ForwardMode,
        MediaCall.VISION_ENCODING,
        MediaCall.LATENT_ENCODING,
        MediaCall.IMAGE_DECODING,
    }
    if diffusion:
        kinds.add(MediaCall.DENOISING)
    return frozenset(kinds)
