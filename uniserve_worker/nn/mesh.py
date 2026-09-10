"""Component rank geometry and explicitly bound tensor communication groups."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from math import prod
from types import MappingProxyType
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ..profiling import profile_range
from .collective import try_sum_reduction
from .parallel import EntryConfig, ParallelConfig

RowChunkProducer = Callable[[slice, tuple[torch.Tensor, ...]], None]


class RowGather:
    """Exchange ordered row publications through two caller-owned gather slots.

    The consumer receives each logical interval on the current stream. It must
    enqueue all reads before returning; the transport then owns slot reuse and
    peer readiness. Local rows are immediately readable while remote rows are
    delivered after their transfer. Numerical work belongs to the consumer.
    """

    def __init__(
        self,
        group: Communicator,
        rows: int,
        width: int,
        dtype: torch.dtype,
        workspace: torch.Tensor,
        consumer: Callable[[slice, torch.Tensor], None],
    ) -> None:
        if min(rows, width) < 1 or not workspace.is_contiguous():
            raise ValueError("row gathering requires positive geometry and contiguous scratch")
        self.group = group
        self.rows = rows
        self.width = width
        self.dtype = dtype
        self.consumer = consumer
        members = group.world_size
        element_bytes = dtype.itemsize
        elements = workspace.numel() * workspace.element_size() // element_bytes
        capacity_rows = elements // (2 * members * width)
        if capacity_rows < 1:
            raise ValueError("row gathering scratch must hold two complete member rows")
        capacity_rows = min(capacity_rows, (64 * 1024 * 1024) // (width * element_bytes))
        self.chunk_rows = capacity_rows // 128 * 128 if capacity_rows >= 128 else capacity_rows
        byte_count = 2 * members * self.chunk_rows * width * element_bytes
        self.storage = workspace.view(torch.uint8).view(-1)[:byte_count].view(dtype)
        self.storage = self.storage.view(2, members, self.chunk_rows, width)
        self.pending: deque[tuple[int, int, torch.Tensor, Any]] = deque()
        self.published_rows = 0
        self.next_slot = 0

    def _consume(self) -> None:
        start, count, gathered, work = self.pending.popleft()
        if work is not None:
            _finish(work, gathered)
        for backend_rank, logical_rank in enumerate(self.group._backend_order):
            if logical_rank != self.group.rank_in_group:
                begin = logical_rank * self.rows + start
                self.consumer(slice(begin, begin + count), gathered[backend_rank])

    def append(self, start: int, input: torch.Tensor) -> None:
        """Publish the next contiguous local interval without mutating its values."""

        if (
            input.ndim != 2
            or input.shape[1] != self.width
            or input.dtype != self.dtype
            or input.device != self.storage.device
            or start != self.published_rows
            or input.shape[0] < 1
            or start + input.shape[0] > self.rows
        ):
            raise ValueError("row gathering requires ordered matching input intervals")
        backend_rank = self.group._backend_order.index(self.group.rank_in_group)
        for offset in range(0, input.shape[0], self.chunk_rows):
            if len(self.pending) == 2:
                self._consume()
            count = min(self.chunk_rows, input.shape[0] - offset)
            # Tail segments compact the member stride for equal-count gather.
            gathered = (
                self.storage[self.next_slot]
                .view(-1)[: self.group.world_size * count * self.width]
                .view(self.group.world_size, count, self.width)
            )
            local = input[offset : offset + count]
            gathered[backend_rank].copy_(local)
            if self.group.world_size > 1:
                work = dist.all_gather_into_tensor(
                    gathered.flatten(0, 1),
                    gathered[backend_rank],
                    group=self.group._require(),
                    async_op=True,
                )
            else:
                work = None
            begin = start + offset
            logical_begin = self.group.rank_in_group * self.rows + begin
            self.consumer(slice(logical_begin, logical_begin + count), local)
            self.pending.append((begin, count, gathered, work))
            self.next_slot = (self.next_slot + 1) % 2
        self.published_rows += input.shape[0]

    def finish(self) -> None:
        """Consume every published transfer before the caller reuses scratch."""

        if self.published_rows != self.rows:
            raise ValueError("row gathering must publish every row before completion")
        while self.pending:
            self._consume()


def divide(numerator: int, denominator: int) -> int:
    """Return an exact integer quotient after validating divisibility and sign."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator % denominator:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


def _process_group(name: str):
    # A stable backend name is traceable through custom ops; process-group
    # objects cannot cross a torch.library schema or a captured graph boundary.
    return dist.distributed_c10d._resolve_process_group(dist.distributed_c10d.GroupName(name))


def _finish(work: Any, tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        work.block_current_stream()
    else:
        work.wait()


@torch.library.custom_op("uniserve_worker::all_gather_into_tensor", mutates_args=("output",))
def _all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor, group_name: str) -> None:
    with profile_range(
        f"uniserve.collective kind=all_gather group={group_name} rank={dist.get_rank()}"
    ):
        group = _process_group(group_name)
        # In-place AllGather registers one stable allocation for both source
        # and destination. Symmetric workspaces can then use NCCL copy engines.
        local = output.view(-1).narrow(0, group.rank() * input.numel(), input.numel())
        local = local.view_as(input)
        local.copy_(input)
        work = dist.all_gather_into_tensor(output, local, group=group, async_op=True)
        _finish(work, input)


@_all_gather_into_tensor.register_fake
def _all_gather_into_tensor_fake(output, input, group_name):
    pass


@torch.library.custom_op("uniserve_worker::all_to_all_single_into", mutates_args=("output",))
def _all_to_all_single_into(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group_name: str,
) -> None:
    with profile_range(
        f"uniserve.collective kind=all_to_all group={group_name} rank={dist.get_rank()}"
    ):
        # Empty split lists select the native equal-count collective. Explicit
        # lists select variable-count send/recv, including when counts match.
        equal_counts = (
            input.numel() == output.numel() and len(set(output_splits + input_splits)) == 1
        )
        work = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=None if equal_counts else output_splits,
            input_split_sizes=None if equal_counts else input_splits,
            group=_process_group(group_name),
            async_op=True,
        )
        _finish(work, input)


@_all_to_all_single_into.register_fake
def _all_to_all_single_into_fake(output, input, output_splits, input_splits, group_name):
    pass


@torch.library.custom_op("uniserve_worker::all_reduce_max", mutates_args=("value",))
def _all_reduce_max(value: torch.Tensor, group_name: str) -> None:
    with profile_range(
        f"uniserve.collective kind=all_reduce_max group={group_name} rank={dist.get_rank()}"
    ):
        work = dist.all_reduce(
            value, op=dist.ReduceOp.MAX, group=_process_group(group_name), async_op=True
        )
        _finish(work, value)


@_all_reduce_max.register_fake
def _all_reduce_max_fake(value, group_name):
    pass


@torch.library.custom_op("uniserve_worker::group_send_recv", mutates_args=("output",))
def _send_recv(
    value: torch.Tensor,
    output: torch.Tensor,
    dst: int,
    src: int,
    group_name: str,
) -> None:
    group = _process_group(group_name)
    with profile_range(
        f"uniserve.collective kind=send_recv group={group_name} rank={dist.get_rank()}"
    ):
        operations = [
            dist.P2POp(dist.isend, value.reshape(-1).view(torch.uint8), dst, group),
            dist.P2POp(dist.irecv, output.reshape(-1).view(torch.uint8), src, group),
        ]
        for work in dist.batch_isend_irecv(operations):
            _finish(work, value)


@_send_recv.register_fake
def _send_recv_fake(value, output, dst, src, group_name):
    pass


@dataclass(frozen=True)
class SymmetricMemoryWorkspace:
    """Runtime-owned allocation with peer views ordered by logical membership."""

    coordinator: Communicator
    local: torch.Tensor
    peers: tuple[torch.Tensor, ...]
    handle: Any

    @property
    def rank(self) -> int:
        return self.coordinator.rank_in_group

    @property
    def size(self) -> int:
        return self.coordinator.world_size

    def fence(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """Publish arrival on the allocation's group with stream ordering."""

        if tuple(input.shape) != (1,) or tuple(output.shape) != (self.size,):
            raise ValueError("symmetric-memory fence buffers do not match group membership")
        self.coordinator.all_gather_into_tensor(output, input)


@dataclass(frozen=True)
class PeerTensorWorkspace:
    """A logically contiguous tensor whose leading-axis storage lives on peers.

    Each owner writes its local allocation. A group fence must complete before
    kernels read the global view, and again before any owner reuses its local
    storage. The distributed runtime owns both the allocation and its mapping.
    """

    coordinator: Communicator
    local: torch.Tensor
    global_tensor: torch.Tensor

    def fence(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """Order owner publication or reader completion on the current stream."""

        if tuple(input.shape) != (1,) or tuple(output.shape) != (self.coordinator.world_size,):
            raise ValueError("peer-memory fence buffers do not match group membership")
        self.coordinator.all_gather_into_tensor(output, input)


@dataclass(frozen=True)
class Communicator:
    """Tensor communication with group-local roots and ordered logical members.

    The distributed runtime supplies the backend and owns its lifetime. Torch
    sorts process-group ranks; this interface preserves worker_config order even
    when it differs from backend order. Singleton operations need no backend.
    """

    ranks: tuple[int, ...] = (0,)
    rank: int = 0
    name: str = "local"
    device: torch.device = torch.device("cpu")
    _group: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError(f"group {self.name!r} requires unique non-empty membership")
        if any(type(rank) is not int or rank < 0 for rank in self.ranks):
            raise ValueError(f"group {self.name!r} ranks must be nonnegative integers")
        if self.rank not in self.ranks:
            raise ValueError(f"rank {self.rank} is outside group {self.name!r}: {self.ranks}")

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def rank_in_group(self) -> int:
        return self.ranks.index(self.rank)

    def _require(self):
        if self._group is None or not dist.is_initialized():
            raise RuntimeError(
                f"group {self.name!r} requires initialized distributed communication"
            )
        return self._group

    @property
    def backend_name(self) -> str | None:
        """Expose a non-owning collective identity for tensor-layout metadata."""

        return None if self.world_size == 1 else self._require().group_name

    def _peer(self, peer: int) -> int:
        if not 0 <= peer < self.world_size:
            raise ValueError(
                f"peer {peer} is outside group {self.name!r} of size {self.world_size}"
            )
        return self.ranks[peer]

    @property
    def _backend_order(self) -> tuple[int, ...]:
        return tuple(self.ranks.index(rank) for rank in sorted(self.ranks))

    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            group = self._require()
            if not try_sum_reduction(group, value):
                dist.all_reduce(value, group=group)
        return value

    def all_reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            _all_reduce_max(value, self._require().group_name)
        return value

    def all_reduce_min(self, value: torch.Tensor) -> torch.Tensor:
        """Resolve a capacity or bound shared by all members."""

        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self._require())
        return value

    def all_gather(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return value
        chunks = [torch.empty_like(value) for _ in self.ranks]
        dist.all_gather(chunks, value.contiguous(), group=self._require())
        backend_ranks = sorted(self.ranks)
        return torch.cat([chunks[backend_ranks.index(rank)] for rank in self.ranks], dim=dim)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if output.numel() != input.numel() * self.world_size:
            raise ValueError("all-gather output size must equal input size times group size")
        if self.world_size == 1:
            output.reshape(-1).copy_(input.reshape(-1))
            return
        name = self._require().group_name
        if tuple(sorted(self.ranks)) == self.ranks:
            _all_gather_into_tensor(output, input, name)
        else:
            scratch = torch.empty_like(output)
            _all_gather_into_tensor(scratch, input, name)
            sources = scratch.reshape(self.world_size, *input.shape)
            targets = output.reshape(self.world_size, *input.shape)
            for backend_rank, logical_rank in enumerate(self._backend_order):
                targets[logical_rank].copy_(sources[backend_rank])

    def gather_row_chunks(
        self, input: torch.Tensor, workspace: torch.Tensor
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Expose complete local rows, then ready remote intervals in logical order.

        All source staging precedes asynchronous gathers. Consumers may enqueue
        numerical work between yields while subsequent transfers make progress.
        Workspace is caller-owned and remains live until iterator exhaustion.
        """

        if input.ndim != 2 or min(input.shape) < 1:
            raise ValueError("row gathering requires a nonempty matrix")
        rows, width = input.shape
        if self.world_size == 1:
            yield slice(0, rows), input
            return
        byte_count = input.numel() * input.element_size() * self.world_size
        if not workspace.is_contiguous() or workspace.device != input.device:
            raise ValueError("row gathering requires contiguous scratch on the input device")
        if workspace.numel() * workspace.element_size() < byte_count:
            raise ValueError("row gathering scratch cannot hold all input rows")
        storage = workspace.view(torch.uint8).view(-1)[:byte_count].view(input.dtype)
        segment_rows = max(1, ((64 * 1024 * 1024) // (width * input.element_size()) // 128) * 128)
        local_rank = self._backend_order.index(self.rank_in_group)
        segments = []
        for start in range(0, rows, segment_rows):
            count = min(segment_rows, rows - start)
            sources = storage.narrow(
                0, start * self.world_size * width, count * self.world_size * width
            )
            sources = sources.view(self.world_size, count, width)
            sources[local_rank].copy_(input[start : start + count])
            segments.append((start, count, sources))
        pending = [
            dist.all_gather_into_tensor(
                sources.flatten(0, 1), sources[local_rank], group=self._require(), async_op=True
            )
            for _, _, sources in segments
        ]
        begin = self.rank_in_group * rows
        yield slice(begin, begin + rows), input
        for (start, count, sources), work in zip(segments, pending, strict=True):
            _finish(work, input)
            for backend_rank, logical_rank in enumerate(self._backend_order):
                if backend_rank != local_rank:
                    begin = logical_rank * rows + start
                    yield slice(begin, begin + count), sources[backend_rank]

    def all_to_all_single_into(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_splits: tuple[int, ...] | list[int],
        input_splits: tuple[int, ...] | list[int],
    ) -> None:
        for tensor, splits in ((output, output_splits), (input, input_splits)):
            if len(splits) != self.world_size or any(size < 0 for size in splits):
                raise ValueError("all-to-all requires one nonnegative row count per group member")
            if sum(splits) != tensor.shape[0]:
                raise ValueError("all-to-all row counts must cover the tensor")
        if self.world_size == 1:
            output.copy_(input)
            return
        name = self._require().group_name
        order = self._backend_order
        if tuple(sorted(self.ranks)) == self.ranks:
            _all_to_all_single_into(output, input, list(output_splits), list(input_splits), name)
            return
        send = input.split(tuple(input_splits), dim=0)
        packed = torch.cat([send[index] for index in order], dim=0)
        received = torch.empty_like(output)
        backend_output_splits = [output_splits[index] for index in order]
        _all_to_all_single_into(
            received,
            packed,
            backend_output_splits,
            [input_splits[index] for index in order],
            name,
        )
        targets = output.split(tuple(output_splits), dim=0)
        for index, chunk in zip(order, received.split(backend_output_splits, dim=0)):
            targets[index].copy_(chunk)

    def produce_exchange(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        producer: Callable[[tuple[torch.Tensor, ...]], None],
    ) -> Callable[[], tuple[torch.Tensor, ...]]:
        """Publish equal peer payloads and return their deferred completion.

        Physical buffers have a leading member axis. Producer and consumer
        views use logical member order. Contiguous registered buffers permit
        NCCL's zero-CTA AlltoAll; tensor layout and numerical work belong to
        the caller. Both buffers must remain live until completion is consumed.
        """

        if (
            source.ndim < 2
            or source.shape[0] != self.world_size
            or source.shape != destination.shape
            or source.dtype != destination.dtype
            or source.device != destination.device
            or not source.is_contiguous()
            or not destination.is_contiguous()
            or source.data_ptr() == destination.data_ptr()
        ):
            raise ValueError("produced exchange requires distinct matching peer buffers")
        order = self._backend_order
        producer(tuple(source[order.index(rank)] for rank in range(self.world_size)))
        if self.world_size == 1:
            destination.copy_(source)
            work = None
        else:
            work = dist.all_to_all_single(destination, source, group=self._require(), async_op=True)

        def complete() -> tuple[torch.Tensor, ...]:
            if work is not None:
                _finish(work, source)
            return tuple(destination[order.index(rank)] for rank in range(self.world_size))

        return complete

    def exchange_row_chunks(
        self,
        input: torch.Tensor,
        workspace: torch.Tensor,
        output: torch.Tensor,
        chunk_rows: int,
    ) -> Iterator[tuple[slice, tuple[torch.Tensor, ...]]]:
        """Yield equal-count AlltoAll row intervals as their peer writes complete.

        Input axes are logical destination, row, and payload features. Both
        workspace buffers are consumed as scratch. Input storage stays live
        independently of peer writes throughout staging and exchange. Each
        yielded tuple orders source shards by logical membership. Consumers
        must exhaust the iterator before either allocation is reused.
        """

        if input.ndim < 3 or input.shape[0] != self.world_size or chunk_rows < 1:
            raise ValueError("chunked exchange requires a member axis and positive row chunks")
        if (
            not input.is_contiguous()
            or not workspace.is_contiguous()
            or workspace.numel() != input.numel()
            or workspace.device != input.device
            or workspace.dtype != input.dtype
            or not output.is_contiguous()
            or output.numel() != input.numel()
            or output.device != input.device
            or output.dtype != input.dtype
            or len({input.data_ptr(), workspace.data_ptr(), output.data_ptr()}) != 3
        ):
            raise ValueError("chunked exchange requires distinct matching contiguous buffers")

        def produce(interval: slice, destinations: tuple[torch.Tensor, ...]) -> None:
            for rank, destination in enumerate(destinations):
                destination.copy_(input[rank, interval])

        return self.produce_row_chunks(input.shape, workspace, output, chunk_rows, produce)

    def produce_row_chunks(
        self,
        shape: tuple[int, ...],
        workspace: torch.Tensor,
        output: torch.Tensor,
        chunk_rows: int,
        producer: RowChunkProducer,
    ) -> Iterator[tuple[slice, tuple[torch.Tensor, ...]]]:
        """Exchange row intervals as a stream-ordered producer publishes them.

        Shape axes are logical destination, row, and payload. The producer
        writes each supplied logical destination view on the current stream.
        It must finish enqueuing its writes before returning. Transport owns
        ordering and scratch layout, and starts each exchange immediately.
        All production is enqueued before yielding to consumers, allowing its
        temporary inputs to be released before consumer allocations begin.
        Both distinct scratch buffers remain live until iterator exhaustion.
        """

        if len(shape) < 3 or shape[0] != self.world_size or min(shape) < 1 or chunk_rows < 1:
            raise ValueError("row production requires a member axis and positive row chunks")
        if (
            not workspace.is_contiguous()
            or not output.is_contiguous()
            or workspace.numel() != prod(shape)
            or output.numel() != prod(shape)
            or workspace.device != output.device
            or workspace.dtype != output.dtype
            or workspace.data_ptr() == output.data_ptr()
        ):
            raise ValueError("row production requires distinct matching contiguous buffers")
        rows = shape[1]
        row_elements = prod(shape[2:])
        source_flat, target_flat = workspace.view(-1), output.view(-1)
        segments = []
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            offset = start * self.world_size * row_elements
            elements = count * self.world_size * row_elements
            segment_shape = (self.world_size, count, *shape[2:])
            source = source_flat.narrow(0, offset, elements).view(segment_shape)
            target = target_flat.narrow(0, offset, elements).view(segment_shape)
            interval = slice(start, start + count)
            complete = self.produce_exchange(source, target, partial(producer, interval))
            segments.append((interval, complete))

        del producer
        for interval, complete in segments:
            yield interval, complete()

    def gather_into_tensor(
        self, output: torch.Tensor | None, input: torch.Tensor, *, dst: int
    ) -> None:
        global_dst = self._peer(dst)
        if self.rank == global_dst:
            expected = (self.world_size, *input.shape)
            if output is None or tuple(output.shape) != expected:
                raise ValueError(f"gather output must have shape {expected}")
            if output.device != input.device or output.dtype != input.dtype:
                raise ValueError("gather output must match input dtype and device")
            chunks = list(output.unbind(0))
            gather_list = [chunks[index] for index in self._backend_order]
        else:
            if output is not None:
                raise ValueError("only the gather destination may provide output storage")
            gather_list = None
        if self.world_size == 1:
            assert output is not None
            output[0].copy_(input)
            return
        work = dist.gather(
            input, gather_list=gather_list, dst=global_dst, group=self._require(), async_op=True
        )
        _finish(work, input)

    def broadcast(self, value: torch.Tensor, *, src: int = 0) -> torch.Tensor:
        global_src = self._peer(src)
        if self.world_size > 1:
            dist.broadcast(value, src=global_src, group=self._require())
        return value

    def reduce_scatter(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Sum equal partitions and return the local logical member's partition."""

        divide(value.shape[dim], self.world_size)
        if self.world_size == 1:
            return value
        chunks = value.chunk(self.world_size, dim=dim)
        packed = torch.cat([chunks[index].movedim(dim, 0) for index in self._backend_order], dim=0)
        output = torch.empty_like(chunks[0].movedim(dim, 0), memory_format=torch.contiguous_format)
        dist.reduce_scatter_tensor(output, packed.contiguous(), group=self._require())
        return output.movedim(0, dim)

    def send(self, value: torch.Tensor, *, dst: int) -> None:
        """Send a tensor's logical bytes, including dtypes unsupported by NCCL."""

        payload = value.contiguous().reshape(-1).view(torch.uint8)
        dist.send(payload, dst=self._peer(dst), group=self._require())

    def recv(self, value: torch.Tensor, *, src: int) -> torch.Tensor:
        """Receive bytes into caller-owned storage with the agreed shape and dtype."""

        storage = value if value.is_contiguous() else torch.empty_like(value).contiguous()
        dist.recv(storage.reshape(-1).view(torch.uint8), src=self._peer(src), group=self._require())
        if storage is not value:
            value.copy_(storage)
        return value

    def send_recv(
        self,
        value: torch.Tensor,
        output: torch.Tensor,
        *,
        dst: int,
        src: int,
    ) -> None:
        """Exchange tensor bytes with group-local peers in one batched P2P launch.

        Callers keep send storage live until stream completion and supply
        contiguous receive storage with the peer's agreed tensor contract.
        """

        global_dst, global_src = self._peer(dst), self._peer(src)
        if not output.is_contiguous():
            raise ValueError("point-to-point exchange requires contiguous output storage")
        if self.world_size == 1:
            output.copy_(value)
            return
        _send_recv(value.contiguous(), output, global_dst, global_src, self._require().group_name)


@dataclass(frozen=True)
class DeviceMesh:
    """Rank mapping for one component, with named independent and composite groups."""

    ranks: tuple[int, ...] = (0,)
    rank: int = 0
    parallel_config: ParallelConfig = ParallelConfig()
    local_device: torch.device = torch.device("cpu")
    groups: Mapping[str, Communicator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.ranks) != self.parallel_config.world_size:
            raise ValueError(
                f"component has {len(self.ranks)} members but parallel_config requires "
                f"{self.parallel_config.world_size} (TP × sequence × pipeline)"
            )
        if len(set(self.ranks)) != len(self.ranks) or any(
            type(rank) is not int or rank < 0 for rank in self.ranks
        ):
            raise ValueError("component members must be unique nonnegative process ranks")
        if self.rank not in self.ranks:
            raise ValueError(f"rank {self.rank} is outside component membership {self.ranks}")

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        return self.parallel_config.dimensions

    def get_coordinate(self, rank: int | None = None) -> tuple[int, ...]:
        """Resolve a process rank through the component's ordered membership."""

        index = self.ranks.index(self.rank if rank is None else rank)
        coordinates = []
        for _, size in reversed(self.dimensions):
            coordinates.append(index % size)
            index //= size
        return tuple(reversed(coordinates))

    def _selection(self, name: str) -> tuple[str, ...]:
        dimensions = tuple(name for name, _ in self.dimensions)
        if name == "sp":
            return tuple(axis for axis in dimensions if axis.startswith("cp") or axis == "ulysses")
        if name == "cp" and "cp_row" in dimensions:
            return ("cp_row", "cp_col")
        if name not in dimensions:
            raise ValueError(f"unknown mesh dimension {name!r}; declared: {(*dimensions, 'sp')}")
        return (name,)

    def group_members(self, name: str) -> tuple[tuple[int, ...], ...]:
        """Enumerate every fiber, including composites over separated dimensions."""

        selected = self._selection(name)
        names, sizes = zip(*self.dimensions)
        fibers: dict[tuple[int, ...], list[int]] = {}
        for rank, coordinate in zip(self.ranks, product(*(range(size) for size in sizes))):
            fixed = tuple(coord for axis, coord in zip(names, coordinate) if axis not in selected)
            fibers.setdefault(fixed, []).append(rank)
        return tuple(tuple(fiber) for fiber in fibers.values())

    def get_group(self, name: str) -> Communicator:
        self._selection(name)
        if name in self.groups:
            return self.groups[name]
        members = next(members for members in self.group_members(name) if self.rank in members)
        if len(members) != 1:
            raise RuntimeError(f"mesh group {name!r} has not been initialized")
        return Communicator(ranks=members, rank=self.rank, name=name, device=self.local_device)

    def size(self, name: str) -> int:
        selected = self._selection(name)
        return prod(size for axis, size in self.dimensions if axis in selected)

    def coord(self, name: str) -> int:
        selected = self._selection(name)
        result = 0
        for (axis, size), coordinate in zip(self.dimensions, self.get_coordinate()):
            if axis in selected:
                result = result * size + coordinate
        return result

    @property
    def tp_size(self) -> int:
        return self.size("tp")

    @property
    def tp_rank(self) -> int:
        return self.coord("tp")

    @classmethod
    def trivial(cls, device: torch.device | str = "cpu") -> DeviceMesh:
        return cls(local_device=torch.device(device))


@dataclass(frozen=True)
class EntryBindings:
    """Bind configured entries to this rank's meshes and process communicator.

    The configured member order defines pipeline coordinates. Tensor replicas
    share one logical output; sequence and temporal members retain their rows.
    """

    entries: Mapping[str, EntryConfig]
    meshes: Mapping[str, DeviceMesh]
    process_group: Communicator

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "meshes", MappingProxyType(dict(self.meshes)))
        if not self.entries or not any(self.owns(name) for name in self.entries):
            raise ValueError("rank has no configured computation entry")
        for name, entry in self.entries.items():
            if any(rank not in self.process_group.ranks for rank in entry.ranks):
                raise ValueError(f"entry {name} members lie outside its Worker")
            if entry.distribution is None:
                mesh = self.meshes.get(name)
                if (mesh is not None) != self.owns(name):
                    raise ValueError(f"entry {name} requires its local mesh")
                if mesh is not None and (
                    mesh.ranks != entry.ranks or mesh.parallel_config != entry.parallel_config
                ):
                    raise ValueError(f"entry {name} mesh disagrees with configuration")

    def owns(self, entry: str) -> bool:
        """Whether this rank executes the configured entry."""
        configured = self.entries.get(entry)
        return configured is not None and self.process_group.rank in configured.ranks

    def input_ranks(self, entry: str) -> tuple[int, ...]:
        """First pipeline-stage input members in configured order."""
        config = self.entries[entry]
        if config.distribution is not None:
            return config.ranks
        width = config.parallel_config.world_size // config.parallel_config.pipeline_parallel_size
        return config.ranks[:width]

    def output_ranks(self, entry: str) -> tuple[int, ...]:
        """Final pipeline-stage members with tensor replicas counted once."""
        config = self.entries[entry]
        if config.distribution is not None:
            return config.ranks
        geometry = DeviceMesh(
            config.ranks, config.ranks[0], config.parallel_config, self.process_group.device
        )
        axes = tuple(name for name, _ in geometry.dimensions)
        return tuple(
            rank
            for rank in config.ranks
            if geometry.get_coordinate(rank)[axes.index("tp")] == 0
            and geometry.get_coordinate(rank)[axes.index("pp")]
            == config.parallel_config.pipeline_parallel_size - 1
        )
