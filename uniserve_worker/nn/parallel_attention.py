"""Attention communication composed around typed local compute operations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import prod

import torch

from .attention_storage import ExchangeBuffers
from .mesh import Communicator, DeviceMesh, RowChunkProducer
from .parallel_sequence import SequencePartition


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
class AttentionBuffers:
    """Fixed-capacity K/V transport storage, separate from sparse compute.

    Mapped storage exposes ordered peer allocations in one virtual key domain.
    Each owner has a page-aligned row capacity, which can exceed its active
    logical rows. Gather storage instead holds a compact replicated key domain.
    The attention owner retains mapped tensor storage through its final peer read.
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


@dataclass
class AttentionRowExchange:
    """Borrowed head shards awaiting their sequence-owner exchange.

    A row-local consumer can process completed intervals while later transfers
    remain in flight. Consumers with tensor-wide numerical domains materialize
    all rows together. Both modes consume the caller-owned scratch buffers.
    """

    parallel: HeadRowExchange
    tensor: torch.Tensor
    workspace: torch.Tensor
    producer: RowChunkProducer | None = None
    logical_rows: int | None = None
    receive_workspace: torch.Tensor | None = None

    def materialize(self) -> torch.Tensor:
        """Return all sequence-local rows with their complete head vectors."""

        if self.producer is None:
            output = self.parallel.restore_rows(self.tensor, workspace=self.workspace)
            return output if self.logical_rows is None else output[: self.logical_rows]
        return torch.cat([rows for _, rows in self.chunks()], dim=0)

    @staticmethod
    def chunk_rows(tensor: torch.Tensor) -> int:
        """Bound each peer payload while preserving tile-aligned row intervals."""

        payload = prod(tensor.shape[1:]) * tensor.element_size()
        return max(128, (32 * 1024 * 1024 // payload // 128) * 128)

    @staticmethod
    def partition_workspace(
        workspace: torch.Tensor, payload: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate incoming head storage from bytes available to a row consumer.

        The receive prefix remains borrowed until every exchange completes.
        Its disjoint suffix can serve another transfer while consumers process
        completed rows, without reserving another persistent allocation.
        """

        if workspace.device != payload.device or not workspace.is_contiguous():
            raise ValueError("attention exchange requires contiguous scratch on its device")
        storage = workspace.view(torch.uint8).view(-1)
        byte_count = payload.numel() * payload.element_size()
        if storage.numel() < byte_count:
            raise ValueError("attention exchange scratch cannot hold its complete payload")
        return storage[:byte_count].view(payload.dtype).view_as(payload), storage[byte_count:]

    def chunks(
        self, receive_workspace: torch.Tensor | None = None
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Restore head vectors using disjoint caller-owned receive byte capacity.

        Receive storage must accommodate the complete head-shard payload on
        the same device. Registered buffers enable copy-engine transport.
        The iterator must finish before any borrowed buffer is reused.
        """

        group = self.parallel.ulysses_group
        if receive_workspace is None:
            receive_workspace = self.receive_workspace
        if receive_workspace is None:
            receive_workspace = torch.empty_like(self.tensor)
        rows = self.tensor.shape[0] // group.world_size
        chunk_rows = self.chunk_rows(self.tensor)
        outgoing = self.tensor.view(group.world_size, rows, *self.tensor.shape[1:])
        incoming, _ = self.partition_workspace(receive_workspace, self.tensor)
        producer, self.producer = self.producer, None
        if producer is None:
            intervals = group.exchange_row_chunks(outgoing, self.workspace, incoming, chunk_rows)
        else:
            intervals = group.produce_row_chunks(
                outgoing.shape, self.tensor, incoming, chunk_rows, producer
            )

        # Transfer producer ownership to the iterator. Even if the caller keeps
        # this exchange object, QKV can retire before row consumers allocate MLP
        # outputs or begin producing the next layer's projected inputs.
        def completed() -> Iterator[tuple[slice, torch.Tensor]]:
            for interval, sources in intervals:
                start, stop = interval.start, interval.stop
                if self.logical_rows is not None:
                    start, stop = min(start, self.logical_rows), min(stop, self.logical_rows)
                yield slice(start, stop), torch.cat(sources, dim=1)[: stop - start]

        return completed()


@dataclass(frozen=True)
class AttentionHeadRows:
    """Global logical Q/K/V rows with the consuming rank's attention heads."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    partition: SequencePartition


class HeadRowPreparation:
    """Transfer projected row intervals into global head-owned storage.

    Every owner publishes the same physical row intervals. The final interval
    pads only transport capacity; attention receives the original global rows.
    Numerical projections and residual state remain owned by their callers.
    """

    def __init__(
        self,
        exchange: HeadRowExchange,
        partition: SequencePartition,
        storage: ExchangeBuffers | None = None,
    ) -> None:
        if partition.group != exchange.ulysses_group:
            raise ValueError("projected rows require their sequence communicator")
        self.exchange = exchange
        self.partition = partition
        self.storage = storage
        self.position = 0
        self.outputs: list[torch.Tensor] = []
        self.pending: list[Callable[[], None]] = []

    def append(self, interval: slice, values: tuple[torch.Tensor, ...]) -> None:
        partition = self.partition
        if (
            interval.start != self.position
            or not self.position <= interval.stop <= partition.count
            or len(values) != 3
            or any(value.shape[0] != interval.stop - interval.start for value in values)
        ):
            raise ValueError("projected attention requires ordered Q/K/V row intervals")
        group = partition.group
        count = (
            partition.capacity - interval.start
            if interval.stop == partition.count
            else interval.stop - interval.start
        )
        for index, value in enumerate(values):
            role = ("query", "key", "value")[index]
            heads, _ = self.exchange.head_region(value.shape[1])
            if len(self.outputs) <= index:
                shape = (partition.capacity * group.world_size, heads, *value.shape[2:])
                self.outputs.append(
                    value.new_empty(shape)
                    if self.storage is None
                    else self.storage.view(f"{role}_receive", shape, value)
                )
            # AlltoAll uses contiguous peer payloads. Completion places each
            # received interval into its final global row range.
            shape = (group.world_size, count, heads, *value.shape[2:])
            outgoing = (
                value.new_empty(shape)
                if self.storage is None
                else self.storage.view(
                    f"{role}_send",
                    shape,
                    value,
                    offset=interval.start * group.world_size * heads * prod(value.shape[2:]),
                )
            )
            incoming = (
                torch.empty_like(outgoing)
                if self.storage is None
                else self.storage.view(
                    f"{role}_staging",
                    shape,
                    value,
                    offset=interval.start * group.world_size * heads * prod(value.shape[2:]),
                )
            )

            def produce(
                destinations: tuple[torch.Tensor, ...],
                value: torch.Tensor = value,
                heads: int = heads,
            ) -> None:
                for rank, destination in enumerate(destinations):
                    if value.shape[0] < destination.shape[0]:
                        destination[value.shape[0] :].zero_()
                    begin = (
                        rank * heads
                        if value.shape[1] >= group.world_size
                        else rank // (group.world_size // value.shape[1])
                    )
                    destination[: value.shape[0]].copy_(value[:, begin : begin + heads])

            exchange_complete = group.produce_exchange(outgoing, incoming, produce)
            output = self.outputs[index]
            destinations = tuple(
                output.narrow(0, rank * partition.capacity + interval.start, count)
                for rank in range(group.world_size)
            )

            def complete(
                exchange_complete: Callable[[], tuple[torch.Tensor, ...]] = exchange_complete,
                destinations: tuple[torch.Tensor, ...] = destinations,
            ) -> None:
                for destination, source in zip(destinations, exchange_complete(), strict=True):
                    destination.copy_(source)

            self.pending.append(complete)
        self.position = interval.stop

    def finish(self) -> AttentionHeadRows:
        """Hand complete logical head rows to their numerical attention consumer."""

        if self.position != self.partition.count or len(self.outputs) != 3:
            raise ValueError("projected attention did not cover its declared rows")
        for complete in self.pending:
            complete()
        self.pending.clear()
        query, key, value = (output[: self.partition.rows] for output in self.outputs)
        self.outputs.clear()
        return AttentionHeadRows(query, key, value, self.partition)


class HeadRowExchange:
    """Exchange sequence-owned rows and attention-owned head shards."""

    def __init__(self, group: Communicator) -> None:
        self.ulysses_group = group

    def head_region(self, heads: int) -> tuple[int, int]:
        """Return local head count and offset, including adjacent GQA replicas."""

        members = self.ulysses_group.world_size
        rank = self.ulysses_group.rank_in_group
        if heads < members:
            if members % heads:
                raise ValueError("replicated attention heads must divide membership")
            return 1, rank // (members // heads)
        if heads % members:
            raise ValueError("attention heads must divide membership")
        count = heads // members
        return count, rank * count

    def exchange_heads(
        self,
        tensor: torch.Tensor,
        *,
        storage: ExchangeBuffers | None = None,
        role: str = "query",
    ) -> torch.Tensor:
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
        source = tensor.view(rows, group.world_size, local_heads, *features).transpose(0, 1)
        if storage is None:
            outgoing = source.contiguous()
            incoming = torch.empty_like(outgoing)
        else:
            outgoing = storage.view(f"{role}_send", tuple(source.shape), tensor)
            incoming = storage.view(f"{role}_receive", tuple(source.shape), tensor)
            outgoing.copy_(source)
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


class ParallelAttention(HeadRowExchange):
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
        super().__init__(mesh.get_group("ulysses"))
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

    def distribute_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace: AttentionBuffers | None,
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

    def finish_context(self, workspace: AttentionBuffers | None) -> None:
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
