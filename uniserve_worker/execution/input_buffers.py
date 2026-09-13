"""Fixed-address staging tensors for one physical execution lane."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.modeling.tensors import (
    AttentionMetadata,
    AttentionMode,
    FlowPatches,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.ops.staging import gather_request_decode_inputs
from uniserve_worker.protocol.batch import ForwardMode, PipelineStage
from uniserve_worker.runtime.device import fill_cpu_bools, fill_cpu_ints
from uniserve_worker.runtime.staging_buffers import StagingBuffers

from ..runtime.tensor_buffers import TensorBuffers, TensorSchema
from .batch import InputBatch
from .rows import ForwardRow

if TYPE_CHECKING:
    from ..runtime.block_tables import BlockTables
    from ..runtime.decode_state import DecodeState
    from ..runtime.kv_cache import KVCache


@dataclass(frozen=True, slots=True)
class InputGeometry:
    """Fixed row, token and embedding bounds shared by sizing and allocation."""

    max_rows: int
    max_tokens: int
    max_text_tokens: int
    max_blocks_per_row: int
    hidden_size: int

    def __post_init__(self) -> None:
        if min(self.max_rows, self.max_tokens, self.max_blocks_per_row) < 1:
            raise ValueError("input-buffer row, token, and block bounds must be positive")
        if self.hidden_size < 0:
            raise ValueError("input-buffer hidden bound must not be negative")
        if not 1 <= self.max_text_tokens <= self.max_tokens:
            raise ValueError("input-buffer text-token capacity is invalid")

    def tensor_schema(self) -> dict[str, TensorSchema]:
        """Describe every resident device field, excluding pinned CPU copy sources."""

        rows, tokens, text = self.max_rows, self.max_tokens, self.max_text_tokens
        schema = {
            "input_ids": TensorSchema((text,), torch.int64, fill=1),
            "positions": TensorSchema((3, tokens), torch.int64, fill=0),
            "embedding_mask": TensorSchema((text,), torch.bool, fill=0),
            "request_pool_indices": TensorSchema((rows,), torch.int64, fill=0),
            "decode_force_finish": TensorSchema((rows,), torch.bool, fill=0),
            "block_tables": TensorSchema((rows, self.max_blocks_per_row), torch.int32, fill=0),
            "cache_lengths": TensorSchema((rows,), torch.int32, fill=0),
            "kv_lengths": TensorSchema((rows,), torch.int32, fill=1),
            "query_lengths": TensorSchema((rows,), torch.int32, fill=1),
            "cumulative_query_lengths": TensorSchema((rows + 1,), torch.int32, fill=0),
            "cumulative_kv_lengths": TensorSchema((rows + 1,), torch.int32, fill=0),
            "output_indices": TensorSchema((rows,), torch.int64, fill=0),
            "decode_page_ids": TensorSchema((tokens,), torch.int64, fill=0),
            "decode_page_offsets": TensorSchema((tokens,), torch.int64, fill=0),
            "attention_indexes": TensorSchema((3, tokens), torch.int64, fill=0),
            "visible_end": TensorSchema((rows, tokens), torch.int64, fill=0),
            "seqused_k": TensorSchema((rows,), torch.int32, fill=0),
            "write_page_ids": TensorSchema((tokens,), torch.int64, fill=0),
            "write_page_offsets": TensorSchema((tokens,), torch.int64, fill=0),
            "write_token_indices": TensorSchema((tokens,), torch.int64, fill=0),
            "flow_timesteps": TensorSchema((rows,), torch.float32, fill=0),
        }
        if self.hidden_size:
            schema["input_embeddings"] = TensorSchema(
                (text, self.hidden_size), torch.bfloat16, fill=0
            )
        return schema


class InputBuffers:
    """Own every mutable tensor address used to stage a model invocation."""

    def __init__(
        self,
        *,
        geometry: InputGeometry,
        device: torch.device | str,
        max_inflight: int = 1,
    ) -> None:
        """Allocate fixed-address row, token, attention, and host-staging buffers."""

        self.max_rows = geometry.max_rows
        self.max_tokens = geometry.max_tokens
        self.max_text_tokens = geometry.max_text_tokens
        self.max_blocks_per_row = geometry.max_blocks_per_row
        self.hidden_size = geometry.hidden_size
        self.device = torch.device(device)
        tensors = TensorBuffers.allocate(geometry.tensor_schema(), self.device).capacity
        self.input_ids = tensors["input_ids"]
        self.positions = tensors["positions"]
        self.embedding_mask = tensors["embedding_mask"]
        self.request_pool_indices = tensors["request_pool_indices"]
        self.decode_force_finish = tensors["decode_force_finish"]
        self.block_tables = tensors["block_tables"]
        self.cache_lengths = tensors["cache_lengths"]
        self.kv_lengths = tensors["kv_lengths"]
        self.query_lengths = tensors["query_lengths"]
        self.cumulative_query_lengths = tensors["cumulative_query_lengths"]
        self.cumulative_kv_lengths = tensors["cumulative_kv_lengths"]
        self.output_indices = tensors["output_indices"]
        self.decode_page_ids = tensors["decode_page_ids"]
        self.decode_page_offsets = tensors["decode_page_offsets"]
        self.attention_indexes = tensors["attention_indexes"]
        self.visible_end = tensors["visible_end"]
        self.seqused_k = tensors["seqused_k"]
        self.write_page_ids = tensors["write_page_ids"]
        self.write_page_offsets = tensors["write_page_offsets"]
        self.write_token_indices = tensors["write_token_indices"]
        self.flow_timesteps = tensors["flow_timesteps"]
        self.input_embeddings = tensors.get("input_embeddings")
        self._request_pool_indices_host = StagingBuffers(
            self.max_rows, dtype=torch.int64, depth=max_inflight, device=self.device
        )
        self._decode_force_finish_host = StagingBuffers(
            self.max_rows, dtype=torch.bool, depth=max_inflight, device=self.device
        )

    def close(self) -> None:
        """Release host copy sources before the execution lane destroys its stream."""

        self._request_pool_indices_host.close()
        self._decode_force_finish_host.close()

    def stage(
        self,
        rows: tuple[ForwardRow, ...],
        *,
        forward_mode: ForwardMode | PipelineStage,
        attention: AttentionMetadata | None = None,
        cache: KVCache | None = None,
        tables: BlockTables | None = None,
        states: DecodeState | None = None,
        packed: bool = False,
        binding: int = 0,
    ) -> InputBatch:
        """Stage numerical rows into fixed addresses, including attention and decode gather.

        Startup may supply physical attention geometry for its reserved scratch
        pages. Serving derives it from the actual cache and request tables.
        """

        from .attention import cache_pages, columns, dense_columns

        tasks = rows
        if any(
            task.forward_mode != forward_mode
            and not (
                isinstance(task.forward_mode, ForwardMode) and isinstance(forward_mode, ForwardMode)
            )
            for task in tasks
        ):
            raise ValueError("input staging requires homogeneous computations")
        row_count = len(tasks)
        if not 0 < row_count <= self.max_rows:
            raise ValueError("forward row count exceeds input-buffer capacity")
        indexed = tuple(task for task in tasks if task.request_indexed_decode)
        if indexed:
            if states is None:
                raise ValueError("indexed decode requires resident request state")
            if any(
                task.forward_mode is not ForwardMode.DECODE
                or task.token_ids is not None
                or task.positions is not None
                or task.token_embeddings is not None
                or task.token_embedding_mask is not None
                or task.selection is None
                or not 0 < task.request_pool_idx <= states.request_pool_size
                for task in indexed
            ):
                raise ValueError(
                    "indexed decode requires a valid request slot and no explicit inputs"
                )
        if attention is None:
            textual = all(
                isinstance(task.forward_mode, ForwardMode)
                or task.forward_mode is PipelineStage.DENOISING
                for task in tasks
            )
            if (
                textual
                and states is not None
                and tables is not None
                and cache is not None
                and states.device.type == "cuda"
                and tables.page_tables.device == states.device == self.device
                and all(
                    task.forward_mode is ForwardMode.DECODE and task.request_indexed_decode
                    for task in tasks
                )
            ):
                _, width = cache_pages(tasks, cache=cache, tables=tables)
                return self._stage_request_indexed_decode(
                    tasks,
                    forward_mode=forward_mode,
                    cache=cache,
                    tables=tables,
                    states=states,
                    table_width=width,
                    binding=binding,
                )
        if indexed:
            assert states is not None
            # Resolve indexed rows when the attention representation requires
            # ordinary numerical views instead of the direct decode gather.
            tasks = tuple(
                replace(
                    task,
                    token_ids=states.future_input_tokens[task.request_pool_idx, :1],
                    positions=states.logical_lengths[
                        task.request_pool_idx : task.request_pool_idx + 1
                    ],
                    request_indexed_decode=False,
                )
                if task.request_indexed_decode
                else task
                for task in tasks
            )
        if attention is None:
            attention = (
                columns(tasks, cache=cache, tables=tables, packed=packed)
                if textual
                else dense_columns(len(tasks), tuple(task.query_tokens for task in tasks))
            )
        # Keep row selection local to the owner of the resulting tensor views.
        token_row_indices = tuple(
            index for index, row in enumerate(tasks) if row.token_ids is not None
        )
        flow_row_indices = tuple(
            index
            for index, row in enumerate(tasks)
            if row.latent is not None and row.image_tokens > 0
        )
        token_rows = tuple(tasks[index] for index in token_row_indices)
        flow_rows = tuple(tasks[index] for index in flow_row_indices)
        encode_rows = tuple(row for row in tasks if row.encode_pixels is not None)
        decode_rows = tuple(
            row for row in tasks if row.latent is not None and row.image_tokens == 0
        )
        if any(row.positions is None or row.selection is None for row in token_rows):
            raise ValueError("token rows require positions and output selections")
        if any(row.positions is None or row.timestep is None for row in flow_rows):
            raise ValueError("flow rows require positions and timesteps")
        request_pool_indices = tuple(row.request_pool_idx for row in tasks)
        decode_force_finish = (
            tuple(row.decode_force_finish for row in tasks)
            if all(
                row.decode_predicate is not None and row.decode_predicate_tagged for row in tasks
            )
            else ()
        )
        token_count = len(token_rows)
        token_ids = tuple(cast(torch.Tensor, row.token_ids) for row in token_rows)
        token_embeddings = tuple(row.token_embeddings for row in token_rows)
        token_embedding_masks = tuple(row.token_embedding_mask for row in token_rows)
        token_positions = tuple(cast(torch.Tensor, row.positions) for row in token_rows)
        token_selections = tuple(cast(TokenSelection, row.selection) for row in token_rows)
        flow_positions = tuple(cast(torch.Tensor, row.positions) for row in flow_rows)
        flow_timesteps = tuple(cast(torch.Tensor, row.timestep) for row in flow_rows)
        flow_latents = tuple(cast(torch.Tensor, row.latent) for row in flow_rows)
        flow_conditioning = tuple(row.flow_conditioning for row in flow_rows)
        flow_image_tokens = tuple(row.image_tokens for row in flow_rows)
        flow_heights = tuple(row.image_height for row in flow_rows)
        flow_widths = tuple(row.image_width for row in flow_rows)
        encode_pixels = tuple(cast(torch.Tensor, row.encode_pixels) for row in encode_rows)
        encode_grids = tuple(row.encode_grid for row in encode_rows)
        encode_grid_shapes = tuple(row.encode_grid_shape for row in encode_rows)
        decode_latents = tuple(cast(torch.Tensor, row.latent) for row in decode_rows)
        decode_heights = tuple(row.image_height for row in decode_rows)
        decode_widths = tuple(row.image_width for row in decode_rows)
        self._scrub(
            attention.attention_mode,
            embeddings=any(value is not None for value in token_embeddings),
        )
        request_slot, request_host = self._request_pool_indices_host.acquire()
        fill_cpu_ints(request_host, request_pool_indices)
        self.request_pool_indices[:row_count].copy_(
            request_host[:row_count],
            non_blocking=self.device.type == "cuda",
        )
        self._request_pool_indices_host.record_copy(request_slot)
        staged_decode_force_finish: torch.Tensor | None = None
        if decode_force_finish:
            staged_decode_force_finish = self.decode_force_finish[:row_count]
            if any(decode_force_finish):
                finish_slot, finish_host = self._decode_force_finish_host.acquire()
                fill_cpu_bools(finish_host, decode_force_finish)
                staged_decode_force_finish.copy_(
                    finish_host[:row_count],
                    non_blocking=self.device.type == "cuda",
                )
                self._decode_force_finish_host.record_copy(finish_slot)

        flattened_ids = tuple(value.reshape(-1) for value in token_ids)
        query_lens = [int(value.numel()) for value in flattened_ids]
        total_token_count = sum(query_lens)
        if any(value < 1 for value in query_lens) or total_token_count > self.max_text_tokens:
            raise ValueError("forward token span exceeds input-buffer capacity")
        for position, query in zip(token_positions, query_lens, strict=True):
            if position.ndim == 1:
                if int(position.numel()) != query:
                    raise ValueError("one-axis token positions do not match the token count")
            elif position.ndim != 2 or tuple(position.shape) not in {(1, query), (3, query)}:
                raise ValueError("token positions must have shape [tokens] or [1|3, tokens]")
        packed_ids: torch.Tensor | None = None
        if flattened_ids and len({value.device for value in flattened_ids}) == 1:
            packed_ids = packed_tensor_views(flattened_ids)
            if packed_ids is None:
                packed_ids = torch.cat(flattened_ids, dim=0)
            self.input_ids[:total_token_count].copy_(packed_ids, non_blocking=True)

        position_axes = max(
            (1 if value.ndim == 1 else int(value.shape[0]) for value in token_positions),
            default=1,
        )
        packed_positions: torch.Tensor | None = None
        # Host prefixes and device continuation positions may share one call.
        # Concatenation requires a common source device; heterogeneous columns
        # copy directly into their destination spans below.
        if (
            token_positions
            and all(value.ndim == 1 for value in token_positions)
            and len({value.device for value in token_positions}) == 1
        ):
            position_parts = tuple(value.reshape(-1) for value in token_positions)
            packed_positions = packed_tensor_views(position_parts)
            if packed_positions is None:
                packed_positions = torch.cat(position_parts, dim=0)
            self.positions[0, :total_token_count].copy_(packed_positions, non_blocking=True)

        staged_flow_positions: list[torch.Tensor] = []
        token_offset = 0
        for flat_ids, embeddings, mask, positions in zip(
            flattened_ids,
            token_embeddings,
            token_embedding_masks,
            token_positions,
            strict=True,
        ):
            count = int(flat_ids.numel())
            if packed_ids is None:
                self.input_ids[token_offset : token_offset + count].copy_(
                    flat_ids,
                    non_blocking=True,
                )
            if packed_positions is None:
                self._stage_positions(positions, token_offset, count)
            if embeddings is not None:
                if self.input_embeddings is None:
                    raise ValueError("model input buffers do not provision token embeddings")
                shaped = embeddings.reshape(count, -1)
                if int(shaped.shape[1]) != self.hidden_size:
                    raise ValueError("token embeddings disagree with the fixed hidden width")
                self.input_embeddings[token_offset : token_offset + count].copy_(
                    shaped,
                    non_blocking=True,
                )
                active_mask = self.embedding_mask[token_offset : token_offset + count]
                if mask is None:
                    active_mask.fill_(True)
                else:
                    active_mask.copy_(mask.reshape(-1), non_blocking=True)
            token_offset += count

        flow_offset = token_offset
        for positions, timestep in zip(flow_positions, flow_timesteps, strict=True):
            count = int(positions.shape[-1] if positions.ndim > 1 else positions.numel())
            if count < 1 or flow_offset + count > self.max_tokens:
                raise ValueError("forward flow span exceeds input-buffer capacity")
            self._stage_positions(positions, flow_offset, count)
            axes = 1 if positions.ndim == 1 else int(positions.shape[0])
            staged = (
                self.positions[0, flow_offset : flow_offset + count]
                if axes == 1
                else self.positions[:axes, flow_offset : flow_offset + count]
            )
            staged_flow_positions.append(staged)
            self.flow_timesteps[len(staged_flow_positions) - 1].copy_(timestep.reshape(-1)[0])
            flow_offset += count

        staged_attention = self.stage_attention(attention)
        active_positions = (
            self.positions[0, :token_offset]
            if position_axes == 1
            else self.positions[:position_axes, :token_offset]
        )
        embeddings_view = (
            None
            if self.input_embeddings is None
            or not any(value is not None for value in token_embeddings)
            else self.input_embeddings[:token_offset]
        )
        mask_view = None if embeddings_view is None else self.embedding_mask[:token_offset]
        return InputBatch(
            forward_mode=forward_mode,
            binding=int(binding),
            row_count=row_count,
            attention=staged_attention,
            request_pool_indices=self.request_pool_indices[:row_count],
            decode_force_finish=staged_decode_force_finish,
            token_row_indices=token_row_indices,
            flow_row_indices=flow_row_indices,
            input_ids=self.input_ids[:token_offset] if token_count else None,
            input_embeddings=embeddings_view,
            embedding_mask=mask_view,
            positions=active_positions if token_count else None,
            token_selections=token_selections,
            flow_positions=tuple(staged_flow_positions),
            flow_timesteps=tuple(
                self.flow_timesteps[index : index + 1]
                for index in range(len(staged_flow_positions))
            ),
            flow_latents=tuple(self._device_view(value) for value in flow_latents),
            flow_conditioning=tuple(
                self._stage_flow_conditioning(value) for value in flow_conditioning
            ),
            flow_image_tokens=flow_image_tokens,
            flow_heights=flow_heights,
            flow_widths=flow_widths,
            encode_pixels=tuple(self._device_view(value) for value in encode_pixels),
            encode_grids=tuple(
                None if value is None else self._device_view(value) for value in encode_grids
            ),
            encode_grid_shapes=encode_grid_shapes,
            decode_latents=tuple(self._device_view(value) for value in decode_latents),
            decode_heights=decode_heights,
            decode_widths=decode_widths,
        )

    def _stage_request_indexed_decode(
        self,
        tasks: tuple[ForwardRow, ...],
        *,
        forward_mode: ForwardMode | PipelineStage,
        cache: KVCache,
        tables: BlockTables,
        states: DecodeState,
        table_width: int,
        binding: int,
    ) -> InputBatch:
        """Snapshot mutable request columns directly into graph-stable input addresses."""

        row_count = len(tasks)
        request_pool_indices = tuple(task.request_pool_idx for task in tasks)
        decode_force_finish = (
            tuple(task.decode_force_finish for task in tasks)
            if all(
                task.decode_predicate is not None and task.decode_predicate_tagged for task in tasks
            )
            else ()
        )
        prefix_lens = tuple(task.seq_len for task in tasks)
        causal_rows = tuple(task.causal for task in tasks)
        group_id = tasks[0].group_id
        # Request indices cross through pinned host storage so the copy can be
        # enqueued without synchronizing a CUDA execution stream.
        request_slot, request_host = self._request_pool_indices_host.acquire()
        fill_cpu_ints(request_host, request_pool_indices)
        self.request_pool_indices[:row_count].copy_(
            request_host[:row_count],
            non_blocking=self.device.type == "cuda",
        )
        self._request_pool_indices_host.record_copy(request_slot)
        staged_decode_force_finish: torch.Tensor | None = None
        if decode_force_finish:
            staged_decode_force_finish = self.decode_force_finish[:row_count]
            if any(decode_force_finish):
                finish_slot, finish_host = self._decode_force_finish_host.acquire()
                fill_cpu_bools(finish_host, decode_force_finish)
                staged_decode_force_finish.copy_(
                    finish_host[:row_count],
                    non_blocking=self.device.type == "cuda",
                )
                self._decode_force_finish_host.record_copy(finish_slot)
        width = int(table_width)
        if width < 1 or width > int(self.block_tables.shape[1]):
            raise ValueError("request-indexed decode table width exceeds staging capacity")

        # One device gather snapshots all mutable request-indexed columns into
        # graph-stable staging addresses for this decode launch.
        gather_request_decode_inputs(
            request_pool_indices=self.request_pool_indices,
            request_page_tables=tables.page_tables,
            request_cache_lengths=tables.verified_lengths,
            request_tokens=states.future_input_tokens[:, 0],
            request_positions=states.logical_lengths,
            input_ids=self.input_ids,
            positions=self.positions[0],
            block_tables=self.block_tables[:, :width],
            cache_lengths=self.cache_lengths,
            kv_lengths=self.kv_lengths,
            query_lengths=self.query_lengths,
            decode_page_ids=self.decode_page_ids,
            decode_page_offsets=self.decode_page_offsets,
            rows=row_count,
            group_id=int(group_id),
            page_size=int(cache.block_size),
        )
        output_locations = self.write_page_ids[:row_count]
        output_locations.copy_(self.decode_page_ids[:row_count])
        output_locations.mul_(int(cache.block_size))
        output_locations.add_(self.decode_page_offsets[:row_count])

        # Cache write locations are flattened page-and-offset coordinates; the
        # returned batch retains the two-dimensional table for attention reads.
        return InputBatch(
            attention=AttentionMetadata(
                attention_mode=AttentionMode.PAGED_DECODE,
                prefix_lens=self.cache_lengths[:row_count],
                query_lens=self.query_lengths[:row_count],
                out_cache_loc=output_locations,
                has_cache_writes=True,
                block_table=self.block_tables[:row_count, :width],
                seq_lens=self.kv_lengths[:row_count],
                max_seqlen_k=width * int(cache.block_size),
                causal=bool(len(set(causal_rows)) == 1 and causal_rows[0]),
                causal_rows_cpu=causal_rows,
                prefix_lens_cpu=prefix_lens,
                query_lens_cpu=(1,) * row_count,
                seq_lens_cpu=tuple(value + 1 for value in prefix_lens),
                group_id=int(group_id),
            ),
            forward_mode=forward_mode,
            binding=int(binding),
            row_count=row_count,
            request_pool_indices=self.request_pool_indices[:row_count],
            decode_force_finish=staged_decode_force_finish,
            token_row_indices=tuple(range(row_count)),
            input_ids=self.input_ids[:row_count],
            positions=self.positions[0, :row_count],
            token_selections=tuple(cast(TokenSelection, task.selection) for task in tasks),
        )

    def stage_attention(self, attention: AttentionMetadata) -> AttentionMetadata:
        """Copy attention tensors into fixed storage and return its borrowed views."""

        return replace(
            attention,
            prefix_lens=self._copy_vector(self.cache_lengths, attention.prefix_lens),
            query_lens=self._copy_vector(self.query_lengths, attention.query_lens),
            out_cache_loc=self._copy_vector(self.write_page_ids, attention.out_cache_loc),
            block_table=None
            if attention.block_table is None
            else self._copy_matrix(self.block_tables, attention.block_table),
            seq_lens=None
            if attention.seq_lens is None
            else self._copy_vector(self.kv_lengths, attention.seq_lens),
            cu_seqlens_q=None
            if attention.cu_seqlens_q is None
            else self._copy_vector(self.cumulative_query_lengths, attention.cu_seqlens_q),
            cu_seqlens_k=None
            if attention.cu_seqlens_k is None
            else self._copy_vector(self.cumulative_kv_lengths, attention.cu_seqlens_k),
            output_indices=None
            if attention.output_indices is None
            else self._copy_vector(self.output_indices, attention.output_indices),
            attention_indexes=None
            if attention.attention_indexes is None
            else self._copy_matrix(self.attention_indexes, attention.attention_indexes),
            visible_end=None
            if attention.visible_end is None
            else self._copy_matrix(self.visible_end, attention.visible_end),
        )

    def _scrub(self, mode: AttentionMode, *, embeddings: bool) -> None:
        """Zero reusable fields whose stale values could affect the next staged mode."""

        self.input_ids.fill_(1)
        self.positions.zero_()
        if embeddings and self.input_embeddings is not None:
            self.input_embeddings.zero_()
            self.embedding_mask.zero_()
        self.request_pool_indices.zero_()
        if mode is not AttentionMode.DENSE:
            self.block_tables.zero_()
        self.cache_lengths.zero_()
        self.kv_lengths.fill_(1)
        self.query_lengths.fill_(1)
        self.write_page_ids.zero_()
        if mode is AttentionMode.PAGED_VARLEN:
            self.cumulative_query_lengths.zero_()
            self.cumulative_kv_lengths.zero_()
            self.output_indices.zero_()
        elif mode is AttentionMode.PACKED:
            self.attention_indexes.zero_()
            self.visible_end.zero_()
            self.cumulative_query_lengths.zero_()
        self.flow_timesteps.zero_()

    def _stage_positions(self, source: torch.Tensor, offset: int, count: int) -> None:
        """Copy a bounded slice of position rows into fixed device storage."""

        if source.ndim == 1:
            if int(source.numel()) != count:
                raise ValueError("token positions do not match the token count")
            self.positions[0, offset : offset + count].copy_(source, non_blocking=True)
            return
        if source.ndim != 2 or int(source.shape[0]) not in {1, 3} or int(source.shape[1]) != count:
            raise ValueError("token positions must have shape [tokens] or [1|3, tokens]")
        axes = int(source.shape[0])
        self.positions[:axes, offset : offset + count].copy_(source, non_blocking=True)

    def _copy_vector(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Copy a vector into the leading extent of fixed staging storage."""

        values = source.reshape(-1)
        count = int(values.numel())
        if count > int(target.numel()):
            raise ValueError("attention vector exceeds input-buffer capacity")
        view = target[:count]
        view.copy_(values, non_blocking=True)
        return view

    def _copy_matrix(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Copy a source matrix into the leading rows and columns of bounded storage."""

        if source.ndim != 2:
            raise ValueError("attention table must be a matrix")
        rows, columns = (int(value) for value in source.shape)
        if rows > int(target.shape[0]) or columns > int(target.shape[1]):
            raise ValueError(
                f"attention table {rows}x{columns} exceeds input-buffer capacity "
                f"{int(target.shape[0])}x{int(target.shape[1])}"
            )
        view = target[:rows, :columns]
        view.copy_(source, non_blocking=True)
        return view

    def _stage_flow_conditioning(self, value: FlowPatches | None) -> FlowPatches | None:
        """Stage optional flow patches or clear their active view."""

        if value is None:
            return None
        return FlowPatches(
            self._device_view(value.pixels),
            self._device_view(value.grid),
            self._device_view(value.noise_scale),
        )

    def _device_view(self, value: torch.Tensor) -> torch.Tensor:
        """Return a tensor on the input-buffer device without unnecessary copying."""

        if value.device != self.device:
            raise ValueError(
                f"model input is on {value.device}, expected execution device {self.device}"
            )
        return value


__all__ = ["InputBuffers"]
