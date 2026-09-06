"""Fixed-address staging tensors for one physical execution lane."""

from __future__ import annotations

import torch

from uniserve_worker.ops.staging import gather_request_decode_inputs
from uniserve_worker.runtime.device import HostStagingRing, fill_cpu_bools, fill_cpu_ints

from .forward_batch import (
    AttentionMode,
    FlowPatches,
    ForwardBatch,
    ModelPhase,
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
        hidden_size: int,
        device: torch.device | str,
        max_inflight: int = 1,
    ) -> None:
        """Allocate fixed-address row, token, attention, and host-staging buffers."""

        if min(max_rows, max_tokens, max_blocks_per_row) < 1:
            raise ValueError("input-buffer row, token, and block bounds must be positive")
        if hidden_size < 0:
            raise ValueError("input-buffer hidden bound must not be negative")
        self.max_rows = int(max_rows)
        self.max_tokens = int(max_tokens)
        self.max_text_tokens = int(max_tokens if max_text_tokens is None else max_text_tokens)
        if self.max_text_tokens < 1 or self.max_text_tokens > self.max_tokens:
            raise ValueError("input-buffer text-token capacity is invalid")
        self.max_blocks_per_row = int(max_blocks_per_row)
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
        self._request_pool_indices_host = HostStagingRing(
            self.max_rows,
            dtype=torch.int64,
            depth=max_inflight,
            device=self.device,
        )
        self.decode_force_finish = torch.zeros(self.max_rows, dtype=torch.bool, device=self.device)
        self._decode_force_finish_host = HostStagingRing(
            self.max_rows,
            dtype=torch.bool,
            depth=max_inflight,
            device=self.device,
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
        decode_force_finish: tuple[bool, ...] = (),
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
        attention: dict[str, object],
    ) -> ForwardBatch:
        """Copy row metadata and model inputs into fixed-address lane buffers and return bounded views."""

        if row_count < 1 or row_count > self.max_rows:
            raise ValueError("forward row count exceeds input-buffer capacity")
        if len(request_pool_indices) != row_count:
            raise ValueError("forward request-slot column is not row-aligned")
        if decode_force_finish and len(decode_force_finish) != row_count:
            raise ValueError("forward decode finish column is not row-aligned")
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
        if attention.get("forward_mode") is AttentionMode.REQUEST_INDEXED_DECODE:
            if any(
                (
                    flow_row_indices,
                    flow_positions,
                    flow_timesteps,
                    flow_latents,
                    flow_conditioning,
                    flow_image_tokens,
                    flow_heights,
                    flow_widths,
                    encode_pixels,
                    encode_grids,
                    encode_grid_shapes,
                    decode_latents,
                    decode_heights,
                    decode_widths,
                )
            ):
                raise ValueError("request-indexed decode cannot mix non-token rows")
            return self._stage_request_indexed_decode(
                phase=phase,
                row_count=row_count,
                request_pool_indices=request_pool_indices,
                decode_force_finish=decode_force_finish,
                token_row_indices=token_row_indices,
                token_ids=token_ids,
                token_embeddings=token_embeddings,
                token_embedding_masks=token_embedding_masks,
                token_positions=token_positions,
                token_selections=token_selections,
                attention=attention,
            )
        self._scrub(
            attention["forward_mode"],
            embeddings=any(value is not None for value in token_embeddings),
        )
        request_slot, request_host = self._request_pool_indices_host.acquire()
        fill_cpu_ints(request_host, request_pool_indices)
        self.request_pool_indices[:row_count].copy_(
            request_host[:row_count],
            non_blocking=self.device.type == "cuda",
        )
        self._request_pool_indices_host.release(request_slot)
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
                self._decode_force_finish_host.release(finish_slot)

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
            forward_mode=staged_attention["forward_mode"],
            req_pool_indices=self.request_pool_indices[:row_count],
            seq_lens=staged_attention["seq_lens"],
            query_lens=staged_attention["query_lens"],
            out_cache_loc=staged_attention["out_cache_loc"],
            has_cache_writes=bool(staged_attention["has_cache_writes"]),
            block_table=staged_attention.get("block_table"),
            kv_lens=staged_attention.get("kv_lens"),
            cu_seqlens_q=staged_attention.get("cu_seqlens_q"),
            cu_seqlens_k=staged_attention.get("cu_seqlens_k"),
            output_indices=staged_attention.get("output_indices"),
            attention_indexes=staged_attention.get("attention_indexes"),
            visible_end=staged_attention.get("visible_end"),
            route_spans=staged_attention.get("route_spans", ()),
            max_seqlen_q=int(staged_attention.get("max_seqlen_q", 0)),
            max_seqlen_k=int(staged_attention.get("max_seqlen_k", 0)),
            causal=bool(staged_attention.get("causal", True)),
            causal_rows_cpu=staged_attention.get("causal_rows_cpu", ()),
            seq_lens_cpu=staged_attention.get("seq_lens_cpu", ()),
            query_lens_cpu=staged_attention.get("query_lens_cpu", ()),
            kv_lens_cpu=staged_attention.get("kv_lens_cpu", ()),
            group_id=int(staged_attention.get("group_id", 0)),
            fully_visible=bool(staged_attention.get("fully_visible", False)),
            binding=int(staged_attention.get("binding", 0)),
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
        *,
        phase: ModelPhase,
        row_count: int,
        request_pool_indices: tuple[int, ...],
        decode_force_finish: tuple[bool, ...],
        token_row_indices: tuple[int, ...],
        token_ids: tuple[torch.Tensor, ...],
        token_embeddings: tuple[torch.Tensor | None, ...],
        token_embedding_masks: tuple[torch.Tensor | None, ...],
        token_positions: tuple[torch.Tensor, ...],
        token_selections: tuple[TokenSelection, ...],
        attention: dict[str, object],
    ) -> ForwardBatch:
        """Gather one-token decode rows from request-indexed state into fixed buffers."""

        # This fast path derives tokens, positions, lengths, and block tables
        # from stable request slots; caller-provided per-token payloads are not
        # permitted because they could disagree with resident state.
        if (
            token_row_indices != tuple(range(row_count))
            or len(token_ids) != row_count
            or len(token_positions) != row_count
            or len(token_selections) != row_count
            or any(int(value.numel()) != 1 for value in token_ids)
            or any(int(value.numel()) != 1 for value in token_positions)
            or any(value is not None for value in token_embeddings)
            or any(value is not None for value in token_embedding_masks)
            or len(attention["seq_lens_cpu"]) != row_count
            or len(attention["kv_lens_cpu"]) != row_count
            or attention["query_lens_cpu"] != (1,) * row_count
        ):
            raise ValueError("request-indexed decode requires one plain token per row")

        # Request indices cross through pinned host storage so the copy can be
        # enqueued without synchronizing a CUDA execution stream.
        request_slot, request_host = self._request_pool_indices_host.acquire()
        fill_cpu_ints(request_host, request_pool_indices)
        self.request_pool_indices[:row_count].copy_(
            request_host[:row_count],
            non_blocking=self.device.type == "cuda",
        )
        self._request_pool_indices_host.release(request_slot)
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
                self._decode_force_finish_host.release(finish_slot)
        width = int(attention["table_width"])
        if width < 1 or width > int(self.block_tables.shape[1]):
            raise ValueError("request-indexed decode table width exceeds staging capacity")

        # One device gather snapshots all mutable request-indexed columns into
        # graph-stable staging addresses for this decode launch.
        gather_request_decode_inputs(
            request_pool_indices=self.request_pool_indices,
            request_page_tables=attention["request_page_tables"],
            request_cache_lengths=attention["request_cache_lengths"],
            request_tokens=attention["request_tokens"],
            request_positions=attention["request_positions"],
            input_ids=self.input_ids,
            positions=self.positions[0],
            block_tables=self.block_tables[:, :width],
            cache_lengths=self.cache_lengths,
            kv_lengths=self.kv_lengths,
            query_lengths=self.query_lengths,
            decode_page_ids=self.decode_page_ids,
            decode_page_offsets=self.decode_page_offsets,
            rows=row_count,
            group_id=int(attention["group_id"]),
            page_size=int(attention["page_size"]),
        )
        output_locations = self.write_page_ids[:row_count]
        output_locations.copy_(self.decode_page_ids[:row_count])
        output_locations.mul_(int(attention["page_size"]))
        output_locations.add_(self.decode_page_offsets[:row_count])

        # Cache write locations are flattened page-and-offset coordinates; the
        # returned batch retains the two-dimensional table for attention reads.
        return ForwardBatch(
            phase=phase,
            row_count=row_count,
            forward_mode=AttentionMode.PAGED_DECODE,
            req_pool_indices=self.request_pool_indices[:row_count],
            seq_lens=self.cache_lengths[:row_count],
            query_lens=self.query_lengths[:row_count],
            out_cache_loc=output_locations,
            has_cache_writes=True,
            block_table=self.block_tables[:row_count, :width],
            kv_lens=self.kv_lengths[:row_count],
            max_seqlen_k=width * int(attention["page_size"]),
            causal=bool(attention["causal"]),
            causal_rows_cpu=attention["causal_rows_cpu"],
            seq_lens_cpu=attention["seq_lens_cpu"],
            query_lens_cpu=attention["query_lens_cpu"],
            kv_lens_cpu=attention["kv_lens_cpu"],
            group_id=int(attention["group_id"]),
            binding=int(attention["binding"]),
            decode_force_finish=staged_decode_force_finish,
            token_row_indices=token_row_indices,
            input_ids=self.input_ids[:row_count],
            positions=self.positions[0, :row_count],
            token_selections=token_selections,
        )

    def stage_attention(self, attention: dict[str, object]) -> dict[str, object]:
        """Copy validated attention side tables into their fixed-address staging buffers."""

        mode = attention["forward_mode"]
        if mode is AttentionMode.REQUEST_INDEXED_DECODE:
            raise ValueError("request-indexed decode must use fused input staging")
        staged = dict(attention)
        staged["seq_lens"] = self._copy_vector(self.cache_lengths, attention["seq_lens"])
        staged["query_lens"] = self._copy_vector(self.query_lengths, attention["query_lens"])
        staged["out_cache_loc"] = self._copy_vector(self.write_page_ids, attention["out_cache_loc"])
        block_table = attention.get("block_table")
        staged["block_table"] = (
            None if block_table is None else self._copy_matrix(self.block_tables, block_table)
        )
        for key, target in (
            ("kv_lens", self.kv_lengths),
            ("cu_seqlens_q", self.cumulative_query_lengths),
            ("cu_seqlens_k", self.cumulative_kv_lengths),
            ("output_indices", self.output_indices),
        ):
            value = attention.get(key)
            staged[key] = None if value is None else self._copy_vector(target, value)
        indexes = attention.get("attention_indexes")
        staged["attention_indexes"] = (
            None if indexes is None else self._copy_matrix(self.attention_indexes, indexes)
        )
        visible = attention.get("visible_end")
        staged["visible_end"] = (
            None if visible is None else self._copy_matrix(self.visible_end, visible)
        )
        return staged

    def _scrub(self, mode: object, *, embeddings: bool) -> None:
        """Zero reusable fields whose stale values could affect the next staged mode."""

        if mode is AttentionMode.REQUEST_INDEXED_DECODE:
            raise ValueError("request-indexed decode must use fused input staging")
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
