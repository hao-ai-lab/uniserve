"""Native media-shaped video sparse attention backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from ...nn.parallel_attention import (
    AttentionContextWorkspace,
    AttentionOutputTargets,
    ParallelAttention,
)
from ...ops import video_sparse as video_sparse_ops
from . import video_sparse_flashinfer, video_sparse_sm100, video_sparse_triton

__all__ = [
    "VideoSparseAttentionBackend",
    "VideoSparseAttentionMetadata",
    "VideoSparseAttentionWorkspace",
    "build_video_sparse_metadata",
    "video_sparse_selected_tiles",
]

TILE = 64
SPARSITY = 0.9


def video_sparse_selected_tiles(video_tiles: int) -> int:
    """Return the ten-percent sparse tile budget, rounded up and bounded to one tile."""

    return max(1, math.ceil((1.0 - SPARSITY) * int(video_tiles)))


@dataclass(frozen=True, slots=True)
class VideoSparseAttentionMetadata:
    """Describes padded tile geometry and per-tile valid row counts for sparse video attention."""

    padded_rows: int
    prefix_tiles: int
    video_tiles: int
    valid_tiles: int
    valid_sizes: torch.Tensor


@dataclass(frozen=True, slots=True)
class VideoSparseAttentionWorkspace:
    """Owns fixed intermediate buffers for pooled scoring, sparse selection, compression, and local compute."""

    attention_output: torch.Tensor
    tile_scores: torch.Tensor
    block_counts: torch.Tensor
    block_indices: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    compressed_tiles: torch.Tensor
    topk_indices_i32: torch.Tensor


def build_video_sparse_metadata(
    *,
    padded_rows: int,
    prefix_tiles: int,
    video_tiles: int,
    valid_sizes: torch.Tensor,
    device: torch.device,
) -> VideoSparseAttentionMetadata:
    """Validate tile geometry and move per-tile valid-row counts onto the execution device."""

    if padded_rows % (TILE * 2):
        raise ValueError("H3 VSA transport requires an even tile-64 count")
    total_tiles = padded_rows // TILE
    if valid_sizes.shape != (total_tiles,):
        raise ValueError("H3 VSA tile-valid metadata does not match padded rows")
    if prefix_tiles + video_tiles > total_tiles:
        raise ValueError("H3 VSA segment tile counts exceed transport geometry")
    valid = valid_sizes.to(device=device, dtype=torch.int32)
    expected = torch.cat(
        (
            torch.ones(prefix_tiles + video_tiles, dtype=torch.bool),
            torch.zeros(total_tiles - prefix_tiles - video_tiles, dtype=torch.bool),
        )
    ).to(device)
    if not bool(torch.equal(valid > 0, expected)):
        raise ValueError("H3 VSA valid sizes do not describe prefix/video/partner tiles")
    return VideoSparseAttentionMetadata(
        padded_rows=padded_rows,
        prefix_tiles=prefix_tiles,
        video_tiles=video_tiles,
        valid_tiles=prefix_tiles + video_tiles,
        valid_sizes=valid,
    )


def _resolve_kernel() -> Callable[..., torch.Tensor]:
    """Resolve the strongest available FastH3 sparse-attention provider."""

    if video_sparse_sm100.available():
        return video_sparse_sm100.block_sparse_attention
    if video_sparse_flashinfer.available():
        return video_sparse_flashinfer.execute_sparse_attention
    if video_sparse_triton.available():
        return video_sparse_triton.execute_sparse_attention
    raise RuntimeError("FastH3 requires a CUDA sparse-attention provider") from (
        video_sparse_sm100.import_error() or video_sparse_triton.import_error()
    )


class VideoSparseAttentionBackend:
    """Checkpoint VSA: sparse top-k attention plus trained dense compression."""

    def __init__(self, metadata: VideoSparseAttentionMetadata) -> None:
        """Bind immutable tile metadata and resolve the sparse attention kernel."""

        self.metadata = metadata
        self.kernel = _resolve_kernel()

    def _compressed_tiles(
        self,
        q_mean: torch.Tensor,
        k_mean: torch.Tensor,
        v_mean: torch.Tensor,
        valid_sizes: torch.Tensor,
        scores: torch.Tensor | None = None,
        compressed: torch.Tensor | None = None,
        query_valid_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool QKV tiles and compute the compressed inter-tile attention representation."""

        if scores is None:
            scores = torch.matmul(q_mean, k_mean.transpose(-1, -2))
            scores.mul_(q_mean.shape[-1] ** -0.5)
        scores.masked_fill_(valid_sizes.view(1, 1, -1) == 0, -torch.inf)
        scores.sub_(scores.amax(dim=-1, keepdim=True)).exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        if compressed is None:
            compressed = torch.matmul(scores, v_mean)
        else:
            torch.matmul(scores, v_mean, out=compressed)
        if query_valid_sizes is None:
            query_valid_sizes = valid_sizes
        compressed.masked_fill_(query_valid_sizes.view(1, -1, 1) == 0, 0)
        return compressed

    def block_map_from_scores(
        self,
        scores: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        counts: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        topk_indices_i32: torch.Tensor | None = None,
        query_tile_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the exempt-prefix map from checkpoint compression scores."""

        heads, tiles, _ = scores.shape
        prefix = prefix_key_indices.shape[0]
        video_tiles = self.metadata.video_tiles
        video_end = prefix + video_tiles
        local_prefix_end = max(0, min(tiles, prefix - query_tile_offset))
        local_video_end = max(0, min(tiles, video_end - query_tile_offset))
        local_video_tiles = local_video_end - local_prefix_end
        width = dense_key_indices.shape[0]
        if indices is None:
            indices = torch.empty((heads, tiles, width), dtype=torch.int32, device=scores.device)
        if counts is None:
            counts = torch.empty((heads, tiles), dtype=torch.int32, device=scores.device)
        indices.zero_()
        counts.fill_(1)
        indices[:, :local_prefix_end, :width] = dense_key_indices
        counts[:, :local_prefix_end] = prefix_count + video_tiles
        if local_video_tiles == 0:
            return counts, indices
        video_scores = scores[:, local_prefix_end:local_video_end, prefix:video_end]
        if topk_indices_i32 is not None:
            video_sparse_ops.threshold_topk_indices(video_scores, topk_indices_i32)
            selected = topk_indices_i32
        else:
            keep_video_tiles = video_sparse_selected_tiles(video_tiles)
            selected = torch.topk(
                video_scores,
                keep_video_tiles,
                dim=-1,
                sorted=True,
            ).indices.to(torch.int32)
        selected.add_(prefix)
        indices[:, local_prefix_end:local_video_end, :prefix] = prefix_key_indices
        topk_positions = torch.arange(
            selected.shape[-1],
            dtype=torch.long,
            device=scores.device,
        ).view(1, 1, -1)
        topk_positions.add_(prefix_count)
        indices[:, local_prefix_end:local_video_end].scatter_(
            2,
            topk_positions.expand(heads, local_video_tiles, -1),
            selected,
        )
        counts[:, local_prefix_end:local_video_end] = prefix_count + selected.shape[-1]
        return counts, indices

    def select_from_pooled(
        self,
        valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        workspace: VideoSparseAttentionWorkspace,
        *,
        query_tile_offset: int = 0,
    ) -> None:
        """Select global fine blocks and compute the separate compression branch.

        Query means cover this owner's contiguous tile interval. Key/value
        means cover the complete ordered sequence, regardless of fine-attention
        transport. Selection is performed once before partitioning key owners.
        """

        q_mean = workspace.pooled_query.permute(1, 0, 2)
        k_mean = workspace.pooled_key.permute(1, 0, 2)
        v_mean = workspace.pooled_value.permute(1, 0, 2)
        torch.matmul(q_mean, k_mean.transpose(-1, -2), out=workspace.tile_scores)
        workspace.tile_scores.mul_(q_mean.shape[-1] ** -0.5)
        query_tiles = q_mean.shape[1]
        self.block_map_from_scores(
            workspace.tile_scores,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            counts=workspace.block_counts,
            indices=workspace.block_indices,
            topk_indices_i32=workspace.topk_indices_i32,
            query_tile_offset=query_tile_offset,
        )
        query_valid = valid_sizes[query_tile_offset : query_tile_offset + query_tiles]
        self._compressed_tiles(
            q_mean,
            k_mean,
            v_mean,
            valid_sizes,
            workspace.tile_scores,
            workspace.compressed_tiles,
            query_valid,
        )

    def forward_parallel(
        self,
        parallel: ParallelAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        workspace: VideoSparseAttentionWorkspace,
        *,
        outputs: tuple[torch.Tensor, ...],
        sync_input: torch.Tensor,
        sync_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
    ) -> torch.Tensor:
        """Compose global sparse selection with shared head and context exchanges."""

        context = parallel.context_group
        group = parallel.ulysses_group
        if len(outputs) != group.world_size:
            raise ValueError("attention output destinations disagree with Ulysses membership")
        if context.world_size > 1 and context_workspace is None:
            raise ValueError("context attention requires transport storage")
        if parallel.mapped:
            transport = context_workspace
            assert transport is not None
            rows = key.shape[0]
            tiles = rows // 64
            start = context.rank_in_group * tiles
            pooled_key = workspace.pooled_key[start : start + tiles]
            pooled_value = workspace.pooled_value[start : start + tiles]
            video_sparse_ops.pool_qkv_means(
                query,
                key,
                value,
                valid_sizes,
                workspace.pooled_query,
                pooled_key,
                pooled_value,
                query_tile_offset=start,
                key_tile_offset=start,
            )
            context.all_gather_into_tensor(workspace.pooled_key, pooled_key.clone())
            context.all_gather_into_tensor(workspace.pooled_value, pooled_value.clone())
            owner_rows = key.shape[0] * (
                parallel.col_group.world_size if parallel.col_group is not None else 1
            )
            context_key, context_value = parallel.distribute_key_value(key, value, transport)
            self.select_from_pooled(
                valid_sizes,
                prefix_key_indices,
                dense_key_indices,
                prefix_count,
                workspace,
                query_tile_offset=start,
            )
            owner_tiles = owner_rows // 64
            capacity_tiles = transport.local_key.shape[0] // 64
            transport.valid_sizes.zero_()
            transport.valid_sizes.view(parallel.key_group.world_size, capacity_tiles)[
                :, :owner_tiles
            ].copy_(valid_sizes.view(parallel.key_group.world_size, owner_tiles))
            # Translate logical block IDs to page-padded peer storage. Padding
            # changes addresses, not the selected key set or validity of its rows.
            indices = workspace.block_indices
            physical_indices = torch.div(
                indices, owner_tiles, rounding_mode="floor"
            ) * capacity_tiles + indices.remainder(owner_tiles)
            self.forward_selected(
                query,
                context_key,
                context_value,
                gate,
                transport.valid_sizes,
                workspace,
                block_indices=physical_indices,
                targets=AttentionOutputTargets(outputs, group.rank_in_group),
            )
            parallel.finish_context(transport)
            return parallel.finish_output(outputs, sync_input, sync_output)

        query_tile_offset = context.rank_in_group * (query.shape[0] // TILE)
        key, value = parallel.distribute_key_value(key, value, context_workspace)
        local_output = (
            outputs[group.rank_in_group].view_as(query)
            if context.world_size == 1 and group.world_size > 1
            else None
        )
        self.forward_local(
            query,
            key,
            value,
            gate,
            valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
            targets=(
                AttentionOutputTargets((local_output,), 0)
                if local_output is not None
                else AttentionOutputTargets(outputs, group.rank_in_group)
            ),
            query_tile_offset=query_tile_offset,
        )
        if local_output is not None:
            # The epilogue has consumed the sparse provider's output; its
            # registered buffer can now receive the head-to-row exchange.
            return parallel.restore_rows(local_output, workspace=workspace.attention_output)
        return parallel.finish_output(outputs, sync_input, sync_output)

    def forward_local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        workspace: VideoSparseAttentionWorkspace,
        *,
        targets: AttentionOutputTargets,
        query_tile_offset: int = 0,
    ) -> torch.Tensor:
        """Evaluate the complete selected key set for this owner's query rows."""

        video_sparse_ops.pool_qkv_means(
            query,
            key,
            value,
            valid_sizes,
            workspace.pooled_query,
            workspace.pooled_key,
            workspace.pooled_value,
            query_tile_offset=query_tile_offset,
        )
        self.select_from_pooled(
            valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
            query_tile_offset=query_tile_offset,
        )
        return self.forward_selected(
            query,
            key,
            value,
            gate,
            valid_sizes,
            workspace,
            block_indices=workspace.block_indices,
            targets=targets,
        )

    def forward_selected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        valid_sizes: torch.Tensor,
        workspace: VideoSparseAttentionWorkspace,
        *,
        block_indices: torch.Tensor,
        targets: AttentionOutputTargets,
    ) -> torch.Tensor:
        """Evaluate one complete sparse loop and apply global compression once.

        Key indices and validity describe physical storage. The immutable
        metadata retains logical geometry independently of page padding.
        """

        return self.kernel(
            query,
            key,
            value,
            mask_block_count=workspace.block_counts,
            mask_block_indices=block_indices,
            valid_sizes=valid_sizes,
            tile_size=TILE,
            prefix_tiles=self.metadata.prefix_tiles,
            gate=gate,
            compressed=workspace.compressed_tiles,
            attention_output=workspace.attention_output,
            targets=targets,
        )
