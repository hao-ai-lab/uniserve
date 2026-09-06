"""Attention communication composed around typed local compute operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .mesh import GroupCoordinator

if TYPE_CHECKING:
    from ..backends.attention.video_sparse import (
        VideoSparseAttentionBackend,
        VideoSparseAttentionWorkspace,
    )


@dataclass(frozen=True)
class AttentionOutputTargets:
    """Borrowed output destinations for a fused head-to-sequence epilogue.

    Input heads belong to ``source_rank``. Each destination owns one contiguous
    row interval with all heads of this sequence group. These tensors describe
    storage only; communicator, allocation, and synchronization ownership remain
    outside the compute backend.
    """

    buffers: tuple[torch.Tensor, ...]
    source_rank: int


@dataclass(frozen=True)
class AttentionContextWorkspace:
    """Fixed-capacity K/V transport storage, separate from sparse compute.

    Mapped storage exposes ordered peer allocations in one virtual key domain.
    Each owner has a page-aligned row capacity, which can exceed its active
    logical rows. Gather storage instead holds a compact replicated key domain.
    The distributed runtime owns mapped allocation lifetime.
    """

    key: torch.Tensor
    value: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    valid_sizes: torch.Tensor
    sync_input: torch.Tensor
    sync_output: torch.Tensor

    @classmethod
    def allocate(
        cls,
        group: GroupCoordinator,
        rows: int,
        heads: int,
        *,
        mapped: bool,
    ) -> AttentionContextWorkspace:
        shape = (rows, heads, 128)
        if mapped:
            keys = group.peer_tensor(
                shape, dtype=torch.bfloat16, name="attention_keys", row_multiple=64
            )
            values = group.peer_tensor(
                shape, dtype=torch.bfloat16, name="attention_values", row_multiple=64
            )
            key, value = keys.global_tensor, values.global_tensor
            local_key, local_value = keys.local, values.local
        else:
            key = torch.empty(
                (rows * group.world_size, heads, 128), dtype=torch.bfloat16, device=group.device
            )
            value = torch.empty_like(key)
            begin = group.rank_in_group * rows
            local_key, local_value = key[begin : begin + rows], value[begin : begin + rows]
        return cls(
            key,
            value,
            local_key,
            local_value,
            torch.empty(key.shape[0] // 64, dtype=torch.int32, device=group.device),
            torch.zeros(1, dtype=torch.int32, device=group.device),
            torch.empty(group.world_size, dtype=torch.int32, device=group.device),
        )


class UlyssesAttention:
    """Exchange projected heads and restore sequence rows around local compute.

    The backend consumes global rows and the current member's head shard. It
    may fuse gate/compression math directly into peer destinations. The bound
    group establishes visibility before the local destination is consumed.
    """

    def __init__(self, backend: VideoSparseAttentionBackend, *, ulysses_group: GroupCoordinator):
        self.backend = backend
        self.ulysses_group = ulysses_group

    def exchange_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """Map [local rows, TP heads, branches, width] to global rows/local heads."""

        group = self.ulysses_group
        if group.world_size == 1:
            return projected
        rows, heads, branches, width = projected.shape
        if heads % group.world_size:
            raise ValueError("projected heads must divide Ulysses membership")
        local_heads = heads // group.world_size
        outgoing = projected.view(rows, group.world_size, local_heads, branches, width)
        outgoing = outgoing.permute(1, 0, 2, 3, 4).contiguous()
        incoming = torch.empty_like(outgoing)
        splits = [1] * group.world_size
        group.all_to_all_single_into(incoming, outgoing, splits, splits)
        return incoming.reshape(rows * group.world_size, local_heads, branches, width)

    def __call__(
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
        outputs: tuple[torch.Tensor, ...],
        sync_input: torch.Tensor,
        sync_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
    ) -> torch.Tensor:
        del context_workspace
        group = self.ulysses_group
        if len(outputs) != group.world_size:
            raise ValueError("attention output destinations disagree with Ulysses membership")
        self.backend.forward_local(
            query,
            key,
            value,
            gate,
            valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
            targets=AttentionOutputTargets(outputs, group.rank_in_group),
        )
        group.all_gather_into_tensor(sync_output, sync_input)
        return outputs[group.rank_in_group]


class GatherAttention(UlyssesAttention):
    """Retain query rows while gathering the complete ordered K/V sequence.

    Ulysses first exchanges projected heads within each context owner. The
    context group then gathers those owners' K/V rows. Sparse selection and
    compression use global key identities, with no partial-softmax merge.
    """

    def __init__(
        self,
        backend: VideoSparseAttentionBackend,
        *,
        ulysses_group: GroupCoordinator,
        context_group: GroupCoordinator,
    ) -> None:
        super().__init__(backend, ulysses_group=ulysses_group)
        self.context_group = context_group

    def __call__(
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
        outputs: tuple[torch.Tensor, ...],
        sync_input: torch.Tensor,
        sync_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
    ) -> torch.Tensor:
        context = self.context_group
        group = self.ulysses_group
        if len(outputs) != group.world_size:
            raise ValueError("attention destinations disagree with Ulysses membership")
        if context_workspace is None:
            raise ValueError("context attention requires transport storage")
        rows = key.shape[0] * context.world_size
        context_key = context_workspace.key[:rows]
        context_value = context_workspace.value[:rows]
        context.all_gather_into_tensor(context_key, key.contiguous())
        context.all_gather_into_tensor(context_value, value.contiguous())
        self.backend.forward_local(
            query,
            context_key,
            context_value,
            gate,
            valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
            targets=AttentionOutputTargets(outputs, group.rank_in_group),
            query_tile_offset=context.rank_in_group * (query.shape[0] // 64),
        )
        group.all_gather_into_tensor(sync_output, sync_input)
        return outputs[group.rank_in_group]


class RingAttention(GatherAttention):
    """Stream selected peer-owned K/V tiles without replicating their storage.

    On a peer-accessible CUDA mesh, mapped owner storage lets the kernel retain
    its complete unnormalized softmax state across every selected key tile.
    There is one physical K/V shard per owner and no owner-wise normalization.
    Publication and reader-completion collectives bound peer access and reuse.
    """

    @property
    def key_group(self) -> GroupCoordinator:
        return self.context_group

    def publish(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace: AttentionContextWorkspace,
    ) -> int:
        """Publish this owner's active rows; return rows per mapped owner."""

        rows = key.shape[0]
        workspace.local_key[:rows].copy_(key)
        workspace.local_value[:rows].copy_(value)
        return rows

    def __call__(
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
        outputs: tuple[torch.Tensor, ...],
        sync_input: torch.Tensor,
        sync_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
    ) -> torch.Tensor:
        from ..ops.video_sparse import pool_qkv_means

        context = self.context_group
        group = self.ulysses_group
        transport = context_workspace
        if transport is None:
            raise ValueError("context attention requires transport storage")
        if len(outputs) != group.world_size:
            raise ValueError("attention destinations disagree with Ulysses membership")
        rows = key.shape[0]
        tiles = rows // 64
        start = context.rank_in_group * tiles
        pooled_key = workspace.pooled_key[start : start + tiles]
        pooled_value = workspace.pooled_value[start : start + tiles]
        pool_qkv_means(
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
        owner_rows = self.publish(key, value, transport)
        self.key_group.all_gather_into_tensor(transport.sync_output, transport.sync_input)
        self.backend.select_from_pooled(
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
        transport.valid_sizes.view(self.key_group.world_size, capacity_tiles)[
            :, :owner_tiles
        ].copy_(valid_sizes.view(self.key_group.world_size, owner_tiles))
        # Translate logical block IDs to page-padded peer storage. Padding
        # changes addresses, not the selected key set or validity of its rows.
        indices = workspace.block_indices
        physical_indices = torch.div(
            indices, owner_tiles, rounding_mode="floor"
        ) * capacity_tiles + indices.remainder(owner_tiles)
        self.backend.forward_selected(
            query,
            transport.key,
            transport.value,
            gate,
            transport.valid_sizes,
            workspace,
            block_indices=physical_indices,
            targets=AttentionOutputTargets(outputs, group.rank_in_group),
        )
        self.key_group.all_gather_into_tensor(transport.sync_output, transport.sync_input)
        group.all_gather_into_tensor(sync_output, sync_input)
        return outputs[group.rank_in_group]


class Attention2D(RingAttention):
    """Gather key columns and stream mapped key rows on orthogonal groups.

    Each query stays with its original owner. Column peers assemble one key-row
    segment; the complete sparse loop reads those segments across the row group.
    This retains the two-dimensional K/V distribution without normalizing and
    rounding independent key-owner outputs.
    """

    def __init__(
        self,
        backend: VideoSparseAttentionBackend,
        *,
        ulysses_group: GroupCoordinator,
        row_group: GroupCoordinator,
        col_group: GroupCoordinator,
        context_group: GroupCoordinator,
    ) -> None:
        super().__init__(backend, ulysses_group=ulysses_group, context_group=context_group)
        self.row_group = row_group
        self.col_group = col_group

    @property
    def key_group(self) -> GroupCoordinator:
        return self.row_group

    def publish(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace: AttentionContextWorkspace,
    ) -> int:
        rows = key.shape[0] * self.col_group.world_size
        self.col_group.all_gather_into_tensor(workspace.local_key[:rows], key.contiguous())
        self.col_group.all_gather_into_tensor(workspace.local_value[:rows], value.contiguous())
        return rows
