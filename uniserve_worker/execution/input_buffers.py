"""Fixed-address staging tensors for one physical execution partition."""

from __future__ import annotations

from dataclasses import replace

import torch

from uniserve_worker.runtime.host_staging import fill_cpu_ints

from .forward_batch import (
    AttnPlan,
    EmptyKvView,
    EmptyMeshView,
    EmptyOutputView,
    FlowPatches,
    ForwardBatch,
    KvView,
    MeshView,
    ModelPhase,
    NoAttention,
    OutputView,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    TokenSelection,
    packed_tensor_views,
)


class InputBuffers:
    """Own every mutable tensor address used to stage a model invocation."""

    def __init__(
        self,
        *,
        max_rows: int,
        max_tokens: int,
        max_text_tokens: int | None = None,
        max_blocks_per_row: int,
        max_latent_pages_per_row: int,
        hidden_size: int,
        device: torch.device | str,
    ) -> None:
        if min(max_rows, max_tokens, max_blocks_per_row) < 1:
            raise ValueError("input-buffer row, token, and block bounds must be positive")
        if max_latent_pages_per_row < 0 or hidden_size < 0:
            raise ValueError("input-buffer latent and hidden bounds must not be negative")
        self.max_rows = int(max_rows)
        self.max_tokens = int(max_tokens)
        self.max_text_tokens = int(max_tokens if max_text_tokens is None else max_text_tokens)
        if self.max_text_tokens < 1 or self.max_text_tokens > self.max_tokens:
            raise ValueError("input-buffer text-token capacity is invalid")
        self.max_blocks_per_row = int(max_blocks_per_row)
        self.max_latent_pages_per_row = int(max_latent_pages_per_row)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)

        self.input_ids = torch.ones(self.max_text_tokens, dtype=torch.int64, device=self.device)
        self.positions = torch.zeros((3, self.max_tokens), dtype=torch.int64, device=self.device)
        self.input_embeddings = (
            None
            if self.hidden_size == 0
            else torch.zeros(
                (self.max_text_tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
        )
        self.embedding_mask = torch.zeros(
            self.max_text_tokens, dtype=torch.bool, device=self.device
        )
        self.request_pool_indices = torch.zeros(
            self.max_rows, dtype=torch.int64, device=self.device
        )
        self.request_pool_indices_host = torch.empty(
            self.max_rows,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self.block_tables = torch.zeros(
            (self.max_rows, self.max_blocks_per_row),
            dtype=torch.int32,
            device=self.device,
        )

        self.cache_lengths = torch.zeros(self.max_rows, dtype=torch.int32, device=self.device)
        self.kv_lengths = torch.ones(self.max_rows, dtype=torch.int32, device=self.device)
        self.query_lengths = torch.ones(self.max_rows, dtype=torch.int32, device=self.device)
        self.cumulative_query_lengths = torch.zeros(
            self.max_rows + 1, dtype=torch.int32, device=self.device
        )
        self.cumulative_kv_lengths = torch.zeros(
            self.max_rows + 1, dtype=torch.int32, device=self.device
        )
        self.output_indices = torch.zeros(self.max_rows, dtype=torch.int64, device=self.device)
        self.decode_page_ids = torch.zeros(self.max_tokens, dtype=torch.int64, device=self.device)
        self.decode_page_offsets = torch.zeros(
            self.max_tokens, dtype=torch.int64, device=self.device
        )
        self.attention_indexes = torch.zeros(
            (3, self.max_tokens), dtype=torch.int64, device=self.device
        )
        self.route_indicators = torch.zeros(self.max_tokens, dtype=torch.bool, device=self.device)
        self.text_indices = torch.zeros(self.max_tokens, dtype=torch.int64, device=self.device)
        self.visible_end = torch.zeros(
            (self.max_rows, self.max_tokens), dtype=torch.int64, device=self.device
        )
        self.seqused_k = torch.zeros(self.max_rows, dtype=torch.int32, device=self.device)
        self.write_page_ids = torch.zeros(self.max_tokens, dtype=torch.int64, device=self.device)
        self.write_page_offsets = torch.zeros(
            self.max_tokens, dtype=torch.int64, device=self.device
        )
        self.write_token_indices = torch.zeros(
            self.max_tokens, dtype=torch.int64, device=self.device
        )
        self.flow_timesteps = torch.zeros(self.max_rows, dtype=torch.float32, device=self.device)

    def stage(
        self,
        *,
        phase: ModelPhase,
        row_count: int,
        request_pool_indices: tuple[int, ...],
        token_row_indices: tuple[int, ...] = (),
        token_ids: tuple[torch.Tensor, ...] = (),
        token_embeddings: tuple[torch.Tensor | None, ...] = (),
        token_embedding_masks: tuple[torch.Tensor | None, ...] = (),
        token_positions: tuple[torch.Tensor, ...] = (),
        token_selections: tuple[TokenSelection, ...] = (),
        flow_row_indices: tuple[int, ...] = (),
        flow_positions: tuple[torch.Tensor, ...] = (),
        flow_timesteps: tuple[torch.Tensor, ...] = (),
        flow_latents: tuple[torch.Tensor, ...] = (),
        flow_conditioning: tuple[FlowPatches | None, ...] = (),
        flow_image_tokens: tuple[int, ...] = (),
        flow_heights: tuple[int, ...] = (),
        flow_widths: tuple[int, ...] = (),
        encode_pixels: tuple[torch.Tensor, ...] = (),
        encode_grids: tuple[torch.Tensor | None, ...] = (),
        encode_grid_shapes: tuple[tuple[int, int] | None, ...] = (),
        decode_latents: tuple[torch.Tensor, ...] = (),
        decode_heights: tuple[int, ...] = (),
        decode_widths: tuple[int, ...] = (),
        kv: KvView | EmptyKvView = EmptyKvView(),
        attention: AttnPlan,
        mesh: MeshView | EmptyMeshView = EmptyMeshView(),
        output: OutputView | EmptyOutputView = EmptyOutputView(),
    ) -> ForwardBatch:
        if row_count < 1 or row_count > self.max_rows:
            raise ValueError("forward row count exceeds input-buffer capacity")
        if len(request_pool_indices) != row_count:
            raise ValueError("forward request-slot column is not row-aligned")
        token_count = len(token_row_indices)
        if any(
            len(values) != token_count
            for values in (
                token_ids,
                token_embeddings,
                token_embedding_masks,
                token_positions,
                token_selections,
            )
        ):
            raise ValueError("forward token columns are not aligned")
        self._scrub(
            attention,
            embeddings=any(value is not None for value in token_embeddings),
        )
        fill_cpu_ints(self.request_pool_indices_host, request_pool_indices)
        self.request_pool_indices[:row_count].copy_(
            self.request_pool_indices_host[:row_count],
            non_blocking=self.device.type == "cuda",
        )

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
        if token_positions and all(value.ndim == 1 for value in token_positions):
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
        return ForwardBatch(
            phase=phase,
            row_count=row_count,
            request_pool_indices=self.request_pool_indices[:row_count],
            token_row_indices=token_row_indices,
            flow_row_indices=flow_row_indices,
            input_ids=self.input_ids[:token_offset] if token_count else None,
            input_embeddings=embeddings_view,
            embedding_mask=mask_view,
            positions=active_positions if token_count else None,
            query_lens=tuple(query_lens),
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
            kv=kv,
            attention=staged_attention,
            mesh=mesh,
            output=output,
        )

    def stage_attention(self, attention: AttnPlan) -> AttnPlan:
        if isinstance(attention, NoAttention):
            return attention
        if isinstance(attention, PagedDecodePlan):
            return replace(
                attention,
                block_table=self._copy_matrix(self.block_tables, attention.block_table),
                cache_seqlens=self._copy_vector(self.cache_lengths, attention.cache_seqlens),
                kv_seqlens=self._copy_vector(self.kv_lengths, attention.kv_seqlens),
                query_lens=self._copy_vector(self.query_lengths, attention.query_lens),
                decode_page_ids=self._copy_vector(self.decode_page_ids, attention.decode_page_ids),
                decode_page_offsets=self._copy_vector(
                    self.decode_page_offsets, attention.decode_page_offsets
                ),
            )
        if isinstance(attention, PagedVarlenPlan):
            return replace(
                attention,
                block_table=self._copy_matrix(self.block_tables, attention.block_table),
                cache_seqlens=self._copy_vector(self.cache_lengths, attention.cache_seqlens),
                query_lens=self._copy_vector(self.query_lengths, attention.query_lens),
                kv_seqlens=self._copy_vector(self.kv_lengths, attention.kv_seqlens),
                cu_seqlens_q=self._copy_vector(
                    self.cumulative_query_lengths, attention.cu_seqlens_q
                ),
                cu_seqlens_k=self._copy_vector(self.cumulative_kv_lengths, attention.cu_seqlens_k),
                output_indices=self._copy_vector(self.output_indices, attention.output_indices),
            )
        return replace(
            attention,
            indexes=self._copy_matrix(self.attention_indexes, attention.indexes),
            route_indicators=self._copy_vector(self.route_indicators, attention.route_indicators),
            text_indices=self._copy_vector(self.text_indices, attention.text_indices),
            visible_end=self._copy_matrix(self.visible_end, attention.visible_end),
            cu_seqlens_q=self._copy_vector(self.cumulative_query_lengths, attention.cu_seqlens_q),
            page_table=self._copy_matrix(self.block_tables, attention.page_table),
            seqused_k=self._copy_vector(self.seqused_k, attention.seqused_k),
            write_page_ids=self._copy_vector(self.write_page_ids, attention.write_page_ids),
            write_page_offsets=self._copy_vector(
                self.write_page_offsets, attention.write_page_offsets
            ),
            write_token_indices=self._copy_vector(
                self.write_token_indices, attention.write_token_indices
            ),
        )

    def _scrub(self, attention: AttnPlan, *, embeddings: bool) -> None:
        self.input_ids.fill_(1)
        self.positions.zero_()
        if embeddings and self.input_embeddings is not None:
            self.input_embeddings.zero_()
            self.embedding_mask.zero_()
        self.request_pool_indices.zero_()
        if not isinstance(attention, NoAttention):
            self.block_tables.zero_()
        if isinstance(attention, (PagedDecodePlan, PagedVarlenPlan)):
            self.cache_lengths.zero_()
            self.kv_lengths.fill_(1)
            self.query_lengths.fill_(1)
        if isinstance(attention, PagedDecodePlan):
            self.decode_page_ids.zero_()
            self.decode_page_offsets.zero_()
        elif isinstance(attention, PagedVarlenPlan):
            self.cumulative_query_lengths.zero_()
            self.cumulative_kv_lengths.zero_()
            self.output_indices.zero_()
        elif isinstance(attention, PackedAttentionPlan):
            self.attention_indexes.zero_()
            self.route_indicators.zero_()
            self.text_indices.zero_()
            self.visible_end.zero_()
            self.cumulative_query_lengths.zero_()
            self.seqused_k.zero_()
            self.write_page_ids.zero_()
            self.write_page_offsets.zero_()
            self.write_token_indices.zero_()
        self.flow_timesteps.zero_()

    def _stage_positions(self, source: torch.Tensor, offset: int, count: int) -> None:
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
        values = source.reshape(-1)
        count = int(values.numel())
        if count > int(target.numel()):
            raise ValueError("attention vector exceeds input-buffer capacity")
        view = target[:count]
        view.copy_(values, non_blocking=True)
        return view

    def _copy_matrix(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
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
        if value is None:
            return None
        return FlowPatches(
            self._device_view(value.pixels),
            self._device_view(value.grid),
            self._device_view(value.noise_scale),
        )

    def _device_view(self, value: torch.Tensor) -> torch.Tensor:
        if value.device != self.device:
            raise ValueError(
                f"model input is on {value.device}, expected execution device {self.device}"
            )
        return value


__all__ = ["InputBuffers"]
