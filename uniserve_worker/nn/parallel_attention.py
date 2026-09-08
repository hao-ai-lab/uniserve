"""Attention communication composed around typed local compute operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mesh import Communicator, DeviceMesh


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
class AttentionContextGeometry:
    """Declare the key domain and physical communication used by context attention."""

    group: Communicator
    rows: int
    heads: int
    mapped: bool
    head_dim: int
    dtype: torch.dtype
    block_size: int

    def __post_init__(self) -> None:
        if min(self.rows, self.heads, self.head_dim, self.block_size) < 1:
            raise ValueError("attention context extents must be positive")
        if self.rows % self.block_size:
            raise ValueError("attention context rows must align to its validity blocks")


@dataclass(frozen=True)
class AttentionContextWorkspace:
    """Fixed-capacity K/V transport storage, separate from sparse compute.

    Mapped storage exposes ordered peer allocations in one virtual key domain.
    Each owner has a page-aligned row capacity, which can exceed its active
    logical rows. Gather storage instead holds a compact replicated key domain.
    The distributed runtime owns mapped allocation lifetime.
    ``valid_sizes`` stores valid-row counts for the geometry's explicit block
    size; numerical backends populate it when masking aligned owner capacity.
    """

    key: torch.Tensor
    value: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    valid_sizes: torch.Tensor
    sync_input: torch.Tensor
    sync_output: torch.Tensor


class ParallelAttention:
    """Exchange attention tensors independently of the numerical backend.

    Ulysses partitions heads over the complete sequence. Context bindings
    gather K/V or expose mapped peer storage; two-dimensional bindings gather
    columns before publishing row owners. Compute backends retain their mask,
    selection and softmax semantics. The runtime owns communication storage.
    """

    def __init__(
        self,
        *,
        mesh: DeviceMesh,
    ) -> None:
        self.ulysses_group = mesh.get_group("ulysses")
        self.context_group = mesh.get_group("cp")
        strategy = mesh.parallel_config.sequence_parallel.kind
        self.mapped = self.context_group.world_size > 1 and strategy in {
            "ring",
            "hybrid",
            "attention2d",
        }
        self.col_group = mesh.get_group("cp_col") if strategy == "attention2d" else None
        self.key_group = (
            mesh.get_group("cp_row") if strategy == "attention2d" else self.context_group
        )

    def exchange_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Exchange [local rows, heads, ...] into [global rows, local heads, ...].

        K/V heads fewer than the group size are replicated over adjacent head
        owners, as required by grouped-query attention. Otherwise heads divide
        the group exactly. Trailing dimensions and dtype are preserved.
        """

        if tensor.ndim < 3 or min(tensor.shape[:2]) < 1:
            raise ValueError("attention head exchange requires rows, heads and features")
        group = self.ulysses_group
        if group.world_size == 1:
            return tensor
        rows, heads, *features = tensor.shape
        if heads < group.world_size:
            if group.world_size % heads:
                raise ValueError("K/V head replication must divide Ulysses membership")
            tensor = tensor.repeat_interleave(group.world_size // heads, dim=1)
            heads = group.world_size
        if heads % group.world_size:
            raise ValueError("projected heads must divide Ulysses membership")
        local_heads = heads // group.world_size
        outgoing = tensor.view(rows, group.world_size, local_heads, *features)
        outgoing = outgoing.transpose(0, 1).contiguous()
        incoming = torch.empty_like(outgoing)
        splits = [1] * group.world_size
        group.all_to_all_single_into(incoming, outgoing, splits, splits)
        return incoming.reshape(rows * group.world_size, local_heads, *features)

    def restore_rows(
        self,
        tensor: torch.Tensor,
        *,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange computed head shards back to the owning sequence rows.

        A caller-owned workspace must match the contiguous payload's size,
        dtype and device. Registered storage enables copy-engine transport.
        Backends that write peer output destinations directly can instead call
        ``finish_output`` after their fused head-to-sequence epilogue.
        """

        group = self.ulysses_group
        if tensor.ndim < 3 or tensor.shape[0] % group.world_size:
            raise ValueError("attention output rows must divide Ulysses membership")
        if group.world_size == 1:
            return tensor
        rows = tensor.shape[0] // group.world_size
        heads, *features = tensor.shape[1:]
        outgoing = tensor.reshape(group.world_size, rows, heads, *features).contiguous()
        if workspace is None:
            incoming = torch.empty_like(outgoing)
        else:
            if (
                workspace.numel() != outgoing.numel()
                or workspace.dtype != outgoing.dtype
                or workspace.device != outgoing.device
                or not workspace.is_contiguous()
            ):
                raise ValueError("attention row exchange workspace must match its payload")
            incoming = workspace.view_as(outgoing)
        splits = [1] * group.world_size
        group.all_to_all_single_into(incoming, outgoing, splits, splits)
        return incoming.transpose(0, 1).reshape(rows, heads * group.world_size, *features)

    def distribute_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace: AttentionContextWorkspace | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish context K/V and return stream-consumable physical views.

        Gathered views cover the active rows. Mapped views include each owner's
        aligned capacity; the compute backend applies its validity metadata and
        calls ``finish_context`` before any owner reuses the physical storage.
        """

        context = self.context_group
        if context.world_size == 1:
            return key, value
        if workspace is None:
            raise ValueError("context attention requires transport storage")
        owner_rows = key.shape[0] * (self.col_group.world_size if self.col_group is not None else 1)
        if (
            key.ndim != 3
            or key.shape != value.shape
            or key.shape[1:] != workspace.local_key.shape[1:]
            or not 0 < owner_rows <= workspace.local_key.shape[0]
            or key.dtype != workspace.key.dtype
            or value.dtype != workspace.value.dtype
            or key.device != workspace.key.device
            or value.device != workspace.value.device
        ):
            raise ValueError("context K/V exceeds its declared tensor storage")
        if self.mapped:
            if self.col_group is not None:
                rows = key.shape[0] * self.col_group.world_size
                self.col_group.all_gather_into_tensor(workspace.local_key[:rows], key.contiguous())
                self.col_group.all_gather_into_tensor(
                    workspace.local_value[:rows], value.contiguous()
                )
            else:
                rows = key.shape[0]
                workspace.local_key[:rows].copy_(key)
                workspace.local_value[:rows].copy_(value)
            self.key_group.all_gather_into_tensor(workspace.sync_output, workspace.sync_input)
            return workspace.key, workspace.value
        rows = key.shape[0] * context.world_size
        context_key, context_value = workspace.key[:rows], workspace.value[:rows]
        context.all_gather_into_tensor(context_key, key.contiguous())
        context.all_gather_into_tensor(context_value, value.contiguous())
        return context_key, context_value

    def finish_context(self, workspace: AttentionContextWorkspace | None) -> None:
        """Fence all mapped readers before the next K/V publication reuses storage."""

        if self.mapped:
            if workspace is None:
                raise ValueError("mapped attention requires its reader fence")
            self.key_group.all_gather_into_tensor(workspace.sync_output, workspace.sync_input)

    def finish_output(
        self,
        outputs: tuple[torch.Tensor, ...],
        sync_input: torch.Tensor,
        sync_output: torch.Tensor,
    ) -> torch.Tensor:
        """Fence fused peer output writes and return this sequence owner's result."""

        group = self.ulysses_group
        if len(outputs) != group.world_size:
            raise ValueError("attention outputs disagree with Ulysses membership")
        group.all_gather_into_tensor(sync_output, sync_input)
        return outputs[group.rank_in_group]
