"""Attention communication composed around typed local compute calls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod

import torch

from uniserve.distributed._chunks import (
    _ChunkProducer,
    _produce_chunks,
)
from uniserve.distributed.mesh import DeviceMesh

from .config import AttentionParallelConfig


@dataclass(frozen=True)
class AttentionBuffers:
    """Fixed-capacity K/V transport storage, separate from sparse compute.

    Gather storage holds a compact replicated key domain: every owner's active
    rows, in topology order, in this rank's own memory.
    ``valid_sizes`` stores valid-row counts for the layout's explicit block
    size; numerical backends populate it when masking aligned owner capacity.
    """

    key: torch.Tensor
    value: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor
    valid_sizes: torch.Tensor
    sync_input: torch.Tensor
    sync_output: torch.Tensor


@dataclass(frozen=True)
class OutputBuffers:
    """Borrowed destinations and fences for attention's row restoration.

    A layer whose epilogue composes every sequence owner's rows writes each
    owner's share straight into that owner's storage, and borrows `peers`,
    one destination per member. That storage exists only where the group's
    devices can map one another's memory. A layer that computes its own rows
    and exchanges them afterwards borrows `local` and leaves `peers` empty.
    """

    peers: tuple[torch.Tensor, ...]
    local: torch.Tensor
    receive: torch.Tensor
    sync_input: torch.Tensor
    sync_output: torch.Tensor


@dataclass
class AttentionRowExchange:
    """Borrowed head shards awaiting their sequence-owner exchange.

    A row-local consumer can process completed intervals while later transfers
    remain in flight. A consumer needing complete tensor statistics exhausts
    the numerical iterator before computing those statistics.
    """

    parallel: ParallelAttention
    tensor: torch.Tensor
    producer: _ChunkProducer | None
    receive_workspace: torch.Tensor

    @staticmethod
    def chunk_rows(tensor: torch.Tensor) -> int:
        """Bound each peer payload while preserving tile-aligned row
        intervals.
        """  # noqa: D205
        payload = prod(tensor.shape[1:]) * tensor.element_size()
        return max(128, (32 * 1024 * 1024 // payload // 128) * 128)

    @staticmethod
    def partition_workspace(
        workspace: torch.Tensor, payload: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate incoming head storage from bytes available to a row
        consumer.

        The receive prefix remains borrowed until every exchange completes.
        Its disjoint suffix can serve another transfer while consumers process
        completed rows, without reserving another persistent allocation.
        """  # noqa: D205
        if workspace.device != payload.device or not workspace.is_contiguous():
            raise ValueError(
                "attention exchange requires contiguous scratch on its device"
            )
        storage = workspace.view(torch.uint8).view(-1)
        byte_count = payload.numel() * payload.element_size()
        if storage.numel() < byte_count:
            raise ValueError(
                "attention exchange scratch cannot hold its complete payload"
            )
        return storage[:byte_count].view(payload.dtype).view_as(
            payload
        ), storage[byte_count:]

    def chunks(self) -> Iterator[tuple[slice, torch.Tensor]]:
        """Restore head vectors through borrowed, registered receive storage.

        Consumers exhaust this iterator before the execution owner reuses its
        backing. Projection callbacks retire after their final row publication.
        """
        group = self.parallel.ulysses_group
        rows = self.tensor.shape[0] // group.size
        chunk_rows = self.chunk_rows(self.tensor)
        outgoing = self.tensor.view(group.size, rows, *self.tensor.shape[1:])
        incoming, _ = self.partition_workspace(
            self.receive_workspace, self.tensor
        )

        producer, self.producer = self.producer, None
        if producer is None:
            raise RuntimeError(
                "attention row production has already been consumed"
            )

        intervals = _produce_chunks(
            group, outgoing.shape, self.tensor, incoming, chunk_rows, producer
        )

        def completed() -> Iterator[tuple[slice, torch.Tensor]]:
            for interval, sources in intervals:
                yield interval, torch.cat(sources, dim=1)

        return completed()


class ParallelAttention(torch.nn.Module):
    """Exchange attention tensors independently of the numerical backend.

    Ulysses partitions heads over the complete sequence. A context binding
    gathers every owner's K/V into each rank, which is what an attention
    kernel reads fastest: the copy engines move the rows once, where reading
    a peer's memory inside the kernel pays for every tile it touches. Compute
    backends retain their mask, selection and softmax semantics. The runtime
    owns communication storage.
    """

    def __init__(
        self,
        *,
        mesh: DeviceMesh,
        parallel: AttentionParallelConfig = AttentionParallelConfig(),
    ) -> None:
        torch.nn.Module.__init__(self)

        heads = () if parallel.heads is None else (parallel.heads.axis,)
        context = parallel.context
        context_axes = () if context is None else (context.gather_axis,)
        # Validate before ordering by topology: misspelled axes must not
        # silently disappear, and flattened token order follows the declared
        # mesh.
        mesh.size((*heads, *context_axes))
        context_axes = tuple(axis for axis in mesh.axes if axis in context_axes)

        self.ulysses_group = mesh.get_group(heads)
        self.context_group = mesh.get_group(context_axes)

        self.key_group = self.context_group

    @property
    def context_buffers(self) -> AttentionBuffers | None:
        """Borrow this invocation's context tensors from the caller's scope."""
        if self.context_group.size == 1:
            return None
        buffers = _CONTEXT.get().get(self)
        if buffers is None:
            raise RuntimeError(
                "context attention requires bound numerical buffers"
            )
        return buffers

    def distribute_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish this owner's K/V and return the whole context's rows.

        Every owner contributes its own rows and receives all of them, so the
        returned views cover the active rows of the complete context in
        topology order and live in this rank's own memory for the attention
        call to read.
        """
        context = self.context_group
        if context.size == 1:
            return key, value

        workspace = self.context_buffers
        assert workspace is not None
        if (
            key.ndim != 3
            or key.shape != value.shape
            or key.shape[1:] != workspace.local_key.shape[1:]
            or not 0 < key.shape[0] <= workspace.local_key.shape[0]
            or key.dtype != workspace.key.dtype
            or value.dtype != workspace.value.dtype
            or key.device != workspace.key.device
            or value.device != workspace.value.device
        ):
            raise ValueError("context K/V exceeds its declared tensor storage")

        rows = key.shape[0] * context.size
        context_key, context_value = (
            workspace.key[:rows],
            workspace.value[:rows],
        )
        context._all_gather_into_tensor(context_key, key.contiguous())
        context._all_gather_into_tensor(context_value, value.contiguous())
        return context_key, context_value

    def finish_output(
        self,
        outputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Fence fused peer output writes and return this sequence owner's
        result.
        """  # noqa: D205
        group = self.ulysses_group
        if len(outputs) != group.size:
            raise ValueError(
                "attention outputs disagree with Ulysses membership"
            )
        buffers = self.output_buffers
        group._all_gather_into_tensor(buffers.sync_output, buffers.sync_input)
        return outputs[group.rank]

    @property
    def output_buffers(self) -> OutputBuffers:
        """Borrow fused row-output destinations from the active caller."""
        buffers = _OUTPUT.get().get(self)
        if buffers is None:
            raise RuntimeError(
                "parallel attention requires bound output buffers"
            )
        return buffers

    def _owner_shape(self, query: torch.Tensor) -> tuple[int, int, int]:
        """Return one sequence owner's compact output extents."""
        group = self.ulysses_group
        if query.ndim != 3 or query.shape[0] % group.size:
            raise ValueError(
                "attention output rows must divide Ulysses membership"
            )
        # [tokens / group, heads * group, head_dim].
        return (
            query.shape[0] // group.size,
            query.shape[1] * group.size,
            query.shape[2],
        )

    @staticmethod
    def _owner_view(
        destination: torch.Tensor,
        query: torch.Tensor,
        shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Reshape bound storage into one sequence owner's compact output.

        The rows occupy the destination's leading elements, so reading it as
        one flat run is what lets a single allocation serve every query length
        up to its capacity. That requires contiguous storage.
        """
        if (
            destination.ndim != 3
            or destination.dtype != query.dtype
            or destination.device != query.device
            or not destination.is_contiguous()
            or any(
                size > capacity
                for size, capacity in zip(shape, destination.shape)
            )
        ):
            raise ValueError(
                "attention output exceeds its bound tensor storage"
            )
        return destination.view(-1)[: query.numel()].view(shape)

    def output_destination(self, query: torch.Tensor) -> torch.Tensor:
        """Return this rank's own compact output view.

        A layer that exchanges its rows after computing them writes here and
        never addresses another member's storage.
        """
        return self._owner_view(
            self.output_buffers.local, query, self._owner_shape(query)
        )

    def output_views(self, query: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return compact sequence-owner views for this call's query
        dimensions.
        """  # noqa: D205
        peers = self.output_buffers.peers
        if not peers:
            raise RuntimeError(
                "this attention layer exchanges its rows rather than "
                "composing them into peer storage, so it has no peer "
                "destinations to return"
            )
        if len(peers) != self.ulysses_group.size:
            raise ValueError(
                "attention peer destinations do not match Ulysses membership"
            )
        shape = self._owner_shape(query)
        return tuple(self._owner_view(peer, query, shape) for peer in peers)


# Call-scoped buffer bindings: layers borrow caller-owned storage through these
# context variables instead of holding references between invocations.
_CONTEXT: ContextVar[Mapping[ParallelAttention, AttentionBuffers]] = ContextVar(
    "parallel_attention_context", default={}
)
_OUTPUT: ContextVar[Mapping[ParallelAttention, OutputBuffers]] = ContextVar(
    "parallel_attention_output", default={}
)


@contextmanager
def output_scope(
    bindings: Mapping[ParallelAttention, OutputBuffers],
) -> Iterator[None]:
    """Bind caller-owned output views until its kernels and readers complete.

    Nested scopes restore their enclosing bindings on normal and exceptional
    exit. Physical backing and graph lifetime remain the caller's obligation.
    """
    token = _OUTPUT.set(bindings)
    try:
        yield
    finally:
        _OUTPUT.reset(token)


@contextmanager
def context_scope(
    bindings: Mapping[ParallelAttention, AttentionBuffers],
) -> Iterator[None]:
    """Borrow context buffers for one numerical execution domain.

    The caller retains backing until all kernels, graphs, and readers finish.
    Layers retain no bindings; nested scopes restore the enclosing buffers,
    including when computation raises. Shared weights can use independent
    allocations in separate runtime owners.
    """
    token = _CONTEXT.set(bindings)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
