"""Stream-bound NCCL collectives for local tensor parallelism.

Each computation stream owns the communicators that enqueue its collectives,
so captured graphs replay without host dispatch. The stream's
:class:`StreamCommunication` also owns every window registered on those
communicators and the storage behind it; execution contexts borrow them.

Peers address a rank's storage directly only through those windows. Every
other buffer a collective receives is borrowed, typically caching-allocator
storage, and NCCL moves it through its own buffers. NCCL would otherwise
register such a buffer for direct peer access when a collective is captured
in a CUDA graph; see :func:`_forbid_implicit_registration`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import contextmanager
from ctypes import addressof, c_void_p
from types import MappingProxyType
from typing import Protocol

import torch
import torch.distributed as dist

from uniserve.runtime.cuda import (
    create_sibling_stream,
    cuda_value,
    destroy_stream,
    driver,
)
from uniserve.runtime.resources import close_resources


def _forbid_implicit_registration() -> None:
    """Keep NCCL from registering borrowed buffers of captured collectives.

    By default NCCL registers the send and receive buffers of a collective
    captured in a CUDA graph and maps the physical allocations behind them
    into its peers, which then write to them directly on every replay, and
    it reuses a registration for later captures by address. Borrowed buffers
    are caching-allocator storage: with expandable segments a tensor spans
    several 20 MiB physical chunks that the allocator maps, unmaps and hands
    to other tensors, so no registration of it stays valid for the graphs
    that replay it. Where NCCL can import those chunks (fabric handles across
    GB200 hosts, POSIX handles on PCIe hosts), captured collectives on such
    registrations write outside mapped memory. Only ``register_buffers``
    windows, whose storage the stream owns, are addressed by peers.

    NCCL reads the setting when it first enqueues a captured collective, so
    setting it before any capture governs the whole process. The setting is
    process-wide and overrides the environment: no UniServe collective may
    run with it enabled.
    """
    os.environ["NCCL_GRAPH_REGISTER"] = "0"


class _CollectiveWork:
    """A published transfer with a consumer-supplied dependency.

    A published transfer whose consumer supplies the final stream dependency.
    """

    def __init__(self, event: torch.cuda.Event, owner: NcclCommunicator):
        self.event, self.owner = event, owner

    def block_current_stream(self) -> None:
        stream = torch.cuda.current_stream(self.owner._stream.device)
        stream.wait_event(self.event)
        if stream == self.owner._stream and self.owner._pending is self:
            # Once the origin has joined, this event must not introduce an
            # external dependency into a later, independent graph capture.
            self.owner._pending = None


class NcclCommunicator:
    """Own ordered NCCL transport for a borrowed computation stream.

    Synchronous numerical calls execute on the computation stream. Streamed
    publications use an owned stream in the same CUDA/Green Context and join
    when their remote outputs are consumed. Rank ordering matches the process
    group; the model Communicator handles logical membership ordering.
    """

    def __init__(self, group, stream: torch.cuda.Stream) -> None:
        import nccl.bindings.nccl as nccl

        # Before this communicator can take part in any captured collective.
        _forbid_implicit_registration()
        self._nccl = nccl
        self._stream = stream
        # The owned publication stream exists from construction until close.
        self._transfer: torch.cuda.ExternalStream | None = None
        self._raw_transfer = None
        self._pending: _CollectiveWork | None = None
        self._comm = c_void_p()
        self._windows: dict[tuple[int, int], tuple[c_void_p, torch.Tensor]] = {}
        self._rank = dist.get_rank(group)
        self._size = dist.get_world_size(group)
        self._ranks = tuple(dist.get_process_group_ranks(group))

        identity = [bytes(nccl.get_unique_id()) if self._rank == 0 else None]
        dist.broadcast_object_list(identity, src=self._ranks[0], group=group)
        unique_id = identity[0]
        if not isinstance(unique_id, bytes):
            raise RuntimeError(
                "NCCL initialization did not receive a unique identifier"
            )

        try:
            with torch.cuda.device(stream.device):
                cu = driver()
                origin = cu.CUstream(stream.cuda_stream)
                green = cuda_value(
                    cu.cuStreamGetGreenCtx(origin),
                    "query communication context",
                )

                config = nccl.Config()
                # The deployed Green Context driver cannot batch-copy mapped
                # peer windows. Keep NCCL's native CTA algorithms on that
                # partition's SMs; ordinary contexts can use copy engines.
                config.cta_policy = 0 if int(green) else 0x02  # DEFAULT or ZERO
                nccl.comm_init_rank_config(
                    addressof(self._comm),
                    self._size,
                    bytearray(unique_id),
                    self._rank,
                    config.ptr,
                )

                # Reuse the computation's actual context, including its SM
                # partition. Communication owns a stream, not another resource
                # partition, and every transfer rejoins its numerical consumer.
                self._raw_transfer, self._transfer = create_sibling_stream(
                    stream, "communication"
                )
        except BaseException as error:
            # Partial construction still attempts every acquired resource's
            # release, recording cleanup failures on the original error.
            if self._raw_transfer is not None:
                try:
                    destroy_stream(self._raw_transfer, "communication")
                except BaseException as cleanup_error:
                    error.add_note(
                        f"Communication stream cleanup failed: "
                        f"{cleanup_error!r}"
                    )
                self._raw_transfer = None
            if self._comm.value:
                try:
                    nccl.comm_abort(self._comm.value)
                except BaseException as cleanup_error:
                    error.add_note(
                        f"NCCL initialization cleanup failed: {cleanup_error!r}"
                    )
                self._comm = c_void_p()
            raise

    @property
    def transfer_stream(self) -> torch.cuda.Stream | None:
        """Return the owned stream that carries streamed publications."""
        return self._transfer

    def _arguments(
        self,
        value: torch.Tensor,
        output: torch.Tensor | None = None,
        *,
        asynchronous: bool = False,
    ) -> tuple[int, int]:
        """Validate operands for a launch.

        Validate operands and return the communicator and stream for a
        launch.
        """
        if not self._comm.value:
            raise RuntimeError("computation collective is closed")
        if value.device != self._stream.device or not value.is_contiguous():
            raise ValueError(
                "computation collectives require contiguous tensors on "
                "their device"
            )
        if output is not None and (
            output.device != value.device
            or output.dtype != value.dtype
            or not output.is_contiguous()
        ):
            raise ValueError(
                "collective output must match input dtype, device, and layout"
            )

        if asynchronous:
            if self._transfer is None:
                raise RuntimeError("computation collective is closed")
            return self._comm.value, self._transfer.cuda_stream

        if self._pending is not None:
            # Synchronous collectives keep the same communicator order even
            # when a projection consumer invokes one before exhausting a gather.
            self._stream.wait_event(self._pending.event)
            self._pending = None
        return self._comm.value, self._stream.cuda_stream

    def _start(self, call, *args) -> _CollectiveWork:
        """Launch on the transfer stream.

        Launch on the transfer stream after the computation stream's inputs.
        """
        transfer = self._transfer
        if transfer is None:
            raise RuntimeError("computation collective is closed")
        transfer.wait_stream(self._stream)
        completed = torch.cuda.Event()
        try:
            call(*args)
        except BaseException:
            # Join the partial launch back so the computation stream never
            # overtakes a failed transfer's still-running kernel.
            completed.record(transfer)
            self._stream.wait_event(completed)
            raise

        completed.record(transfer)
        work = _CollectiveWork(completed, self)
        self._pending = work
        return work

    def start_all_gather(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> _CollectiveWork:
        if output.numel() != value.numel() * self._size:
            raise ValueError(
                "collective gather output must hold every rank's contribution"
            )
        return self._start(
            self._nccl.all_gather,
            value.data_ptr(),
            output.data_ptr(),
            value.numel(),
            self._dtype(value),
            *self._arguments(value, output, asynchronous=True),
        )

    def start_all_to_all(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> _CollectiveWork:
        if output.numel() != value.numel() or value.numel() % self._size:
            raise ValueError(
                "asynchronous exchange requires equal peer payloads"
            )
        return self._start(
            self._nccl.allto_all,
            value.data_ptr(),
            output.data_ptr(),
            value.numel() // self._size,
            self._dtype(value),
            *self._arguments(value, output, asynchronous=True),
        )

    def _dtype(self, value: torch.Tensor) -> int:
        types = self._nccl.DataType
        return {
            torch.bool: types.Uint8,
            torch.uint8: types.Uint8,
            torch.int8: types.Int8,
            torch.int32: types.Int32,
            torch.int64: types.Int64,
            torch.float16: types.Float16,
            torch.bfloat16: types.Bfloat16,
            torch.float32: types.Float32,
            torch.float64: types.Float64,
        }[value.dtype]

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None:
        reduction = {
            "sum": self._nccl.RedOp.Sum,
            "max": self._nccl.RedOp.Max,
            "min": self._nccl.RedOp.Min,
        }[op]
        self._nccl.all_reduce(
            value.data_ptr(),
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            reduction,
            *self._arguments(value),
        )

    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None:
        if output.numel() != value.numel() * self._size:
            raise ValueError(
                "collective gather output must hold every rank's contribution"
            )
        self._nccl.all_gather(
            value.data_ptr(),
            output.data_ptr(),
            value.numel(),
            self._dtype(value),
            *self._arguments(value, output),
        )

    def all_to_all(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> None:
        self._arguments(value, output)
        if len(input_splits) != self._size or len(output_splits) != self._size:
            raise ValueError("collective exchange requires one split per rank")

        if len(set(input_splits + output_splits)) == 1:
            self._nccl.allto_all(
                value.data_ptr(),
                output.data_ptr(),
                value.numel() // self._size,
                self._dtype(value),
                *self._arguments(value, output),
            )
            return

        # Uneven exchanges decompose into one grouped send/recv pair per peer.
        send_rows, receive_rows = (
            value.split(input_splits),
            output.split(output_splits),
        )
        self._nccl.group_start()
        try:
            for peer, (send, receive) in enumerate(
                zip(send_rows, receive_rows, strict=True)
            ):
                if send.numel():
                    self.send(send, self._ranks[peer])
                if receive.numel():
                    self.recv(receive, self._ranks[peer])
        finally:
            self._nccl.group_end()

    def gather(
        self, outputs: list[torch.Tensor] | None, value: torch.Tensor, root: int
    ) -> None:
        self._arguments(value)
        self._ranks.index(root)
        if self._ranks[self._rank] == root:
            if outputs is None or len(outputs) != self._size:
                raise ValueError(
                    "collective gather requires one destination per rank"
                )
            for output in outputs:
                self._arguments(value, output)
                if output.numel() != value.numel():
                    raise ValueError(
                        "gather destination must match the contribution size"
                    )

        self._nccl.group_start()
        try:
            self.send(value, root)
            if self._ranks[self._rank] == root:
                assert outputs is not None
                for peer, output in zip(self._ranks, outputs, strict=True):
                    self.recv(output, peer)
        finally:
            self._nccl.group_end()

    def broadcast(self, value: torch.Tensor, root: int) -> None:
        self._nccl.broadcast(
            value.data_ptr(),
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(root),
            *self._arguments(value),
        )

    def reduce_scatter(self, output: torch.Tensor, value: torch.Tensor) -> None:
        if value.numel() != output.numel() * self._size:
            raise ValueError(
                "collective reduction requires one output-sized partition "
                "per rank"
            )
        self._nccl.reduce_scatter(
            value.data_ptr(),
            output.data_ptr(),
            output.numel(),
            self._dtype(value),
            self._nccl.RedOp.Sum,
            *self._arguments(value, output),
        )

    def send(self, value: torch.Tensor, peer: int) -> None:
        self._nccl.send(
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(peer),
            *self._arguments(value),
        )

    def recv(self, value: torch.Tensor, peer: int) -> None:
        self._nccl.recv(
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(peer),
            *self._arguments(value),
        )

    def send_recv(
        self, output: torch.Tensor, value: torch.Tensor, dst: int, src: int
    ) -> None:
        self._nccl.group_start()
        try:
            self.send(value.reshape(-1).view(torch.uint8), dst)
            self.recv(output.reshape(-1).view(torch.uint8), src)
        finally:
            self._nccl.group_end()

    def register_buffers(self, *buffers: torch.Tensor) -> None:
        """Register matching VMM allocations.

        Register matching VMM allocations collectively before their first use.

        Every rank supplies the same buffer sequence and byte capacities. Both
        sides of a symmetric exchange must use registered allocations; mixing
        these with ordinary CUDA allocations is unsafe under graph replay.
        Registrations retain their backing until the communicator is closed.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "register communication buffers before CUDA graph capture"
            )

        for value in buffers:
            self._arguments(value)
            key = (value.data_ptr(), value.numel() * value.element_size())
            if key in self._windows:
                continue
            window = c_void_p()
            self._nccl.comm_window_register(
                self._comm.value, key[0], key[1], addressof(window), 0x01
            )  # NCCL_WIN_COLL_SYMMETRIC
            self._windows[key] = window, value

    def close(self) -> None:
        """Destroy the communicator.

        Destroy the communicator after its runner has retired all graph use.
        Destruction is collective: it completes only once every rank of the
        communicator calls it, so this belongs on a path all of them reach.
        A rank releasing after a failure calls ``abort`` instead.
        """
        if self._transfer is not None:
            self._transfer.synchronize()

        if self._comm.value:
            for window, _ in self._windows.values():
                self._nccl.comm_window_deregister(
                    self._comm.value, window.value
                )
            self._windows.clear()
            communicator, self._comm = self._comm.value, c_void_p()
            self._nccl.comm_destroy(communicator)

        self._release_stream()

    def abort(self) -> None:
        """Invalidate local use and retain native resources until process exit.

        NCCL abort can wait on other communicators or captured graph users.
        Failed execution cannot prove those users have retired, so the owning
        process must exit without entering native communicator teardown.
        """
        from .resources import retain_until_exit

        if self._comm.value:
            # Retain the handle, registered tensors and streams together. No
            # failed access is acknowledged as completed or safe to reuse.
            retain_until_exit((self, self._comm))
            self._comm = c_void_p()

    def _release_stream(self) -> None:
        """Destroy the communication stream this communicator owns."""
        if self._raw_transfer is not None:
            destroy_stream(self._raw_transfer, "communication")
            self._raw_transfer = None
            self._transfer = self._pending = None


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


class GatherPool:
    """Lend transport buffers until their readers are enqueued.

    Iterators may nest while upstream projections still have unread payloads.
    Each active borrower receives distinct backing; completed borrowers reuse
    storage on the owner's serialized stream. Captured graphs keep addressing
    every buffer, so buffers live as long as the pool. With a communicator,
    each buffer is registered as one of its windows when it is created.
    """

    def __init__(self, group, communicator: NcclCommunicator | None):
        self.group = group
        self.communicator = communicator
        self.buffers: list[torch.Tensor] = []
        self.borrowed: set[int] = set()
        self.symmetric = dist.get_backend(group._require()) == "nccl"

    @contextmanager
    def borrow(self, size, device, *, capacity=None):
        if device != self.group.device:
            raise ValueError(
                "projection exchange must use its communicator's device"
            )
        if capacity is not None and size > capacity:
            raise ValueError("projection exchange exceeds the bound workspace")
        amount = size if capacity is None else capacity

        buffer = next(
            (
                value
                for value in self.buffers
                if id(value) not in self.borrowed and value.numel() >= amount
            ),
            None,
        )
        if buffer is None:
            if _capturing(device):
                raise RuntimeError(
                    "prepare projection exchange backing before capture"
                )
            if self.symmetric:
                from ._peer_storage import allocate_collective_buffer

                buffer = allocate_collective_buffer(
                    (amount,), dtype=torch.uint8, device=device
                )
            else:
                buffer = torch.empty(amount, dtype=torch.uint8, device=device)
            if self.communicator is not None:
                self.communicator.register_buffers(buffer)
            self.buffers.append(buffer)

        self.borrowed.add(id(buffer))
        try:
            yield buffer
        finally:
            self.borrowed.remove(id(buffer))


class _Allocation(Protocol):
    """Backing storage that its owner retires exactly once."""

    def close(self) -> None: ...


class StreamCommunication:
    """Own the communicators, windows and window storage of one stream.

    Every execution context running on the stream borrows these resources.
    Binding a communicator, registering a window and retiring either are
    collective over the group, so they live as long as the stream rather than
    any one context: ranks may prepare different numbers of contexts, in
    different orders, but each rank binds a stream's groups in one order and
    closes the stream on a path all members reach.
    """

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self._communicators: dict[str, NcclCommunicator] = {}
        self._gather_pools: dict[object, GatherPool] = {}
        self._windows: dict[
            Hashable, tuple[object, tuple[_Allocation, ...]]
        ] = {}
        self._closed = False

    @property
    def communicators(self) -> Mapping[str, NcclCommunicator]:
        """Borrow the bound communicators by process-group name.

        The view is live: groups bound later appear in scopes already entered.
        """
        return MappingProxyType(self._communicators)

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("stream communication is closed")

    def bind(self, groups: Iterable) -> None:
        """Create communicators for multi-rank NCCL groups not yet bound.

        Creation is collective over each group: every member binds the same
        groups on its corresponding stream in the same order. Groups already
        bound are borrowed again without communication.
        """
        self._open()
        created: dict[str, NcclCommunicator] = {}
        try:
            for communicator in groups:
                if communicator.size == 1:
                    continue
                group = communicator._require()
                name = group.group_name
                if (
                    dist.get_backend(group) == "nccl"
                    and name not in self._communicators
                    and name not in created
                ):
                    created[name] = NcclCommunicator(group, self.stream)
        except BaseException as error:
            for binding in reversed(tuple(created.values())):
                try:
                    binding.close()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"collective binding cleanup failed: {cleanup_error!r}"
                    )
            raise
        self._communicators.update(created)

    def gather_pool(self, group) -> GatherPool:
        """Borrow the stream's transport pool for one gather group."""
        self._open()
        pool = self._gather_pools.get(group)
        if pool is None:
            pool = GatherPool(
                group, self._communicators.get(group._require().group_name)
            )
            self._gather_pools[group] = pool
        return pool

    def windows(
        self,
        key: Hashable,
        group,
        allocate: Callable[
            [], tuple[object, tuple[_Allocation, ...], tuple[torch.Tensor, ...]]
        ],
    ) -> object:
        """Borrow storage whose tensors are registered on ``group``'s windows.

        ``allocate`` returns the borrowed value, the allocations that own its
        backing, and the tensors to register. It runs once per ``key``; later
        contexts borrow the same registered storage, so re-preparation neither
        registers new windows nor retains retired backing.
        """
        self._open()
        if key not in self._windows:
            if _capturing(self.stream.device):
                raise RuntimeError(
                    "prepare registered communication storage before capture"
                )
            communicator = self._communicators.get(group._require().group_name)
            if communicator is None:
                raise RuntimeError(
                    "bind a group's communicator before registering windows"
                )
            value, allocations, tensors = allocate()
            try:
                communicator.register_buffers(*tensors)
            except BaseException:
                close_resources(
                    *(allocation.close for allocation in allocations)
                )
                raise
            self._windows[key] = value, allocations
        return self._windows[key][0]

    def close(self, *, aborted: bool = False) -> None:
        """Retire communicators, then the storage their windows registered.

        Normal retirement is collective over every bound group; call it on a
        path all members reach after every context and graph on the stream
        has retired. ``aborted`` retains every native resource until process
        exit without waiting for peers or the device.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            for communicator in self._communicators.values():
                communicator.abort()
            return

        try:
            close_resources(
                *(
                    communicator.close
                    for communicator in self._communicators.values()
                ),
                *(
                    allocation.close
                    for _, allocations in self._windows.values()
                    for allocation in allocations
                ),
            )
        finally:
            self._communicators.clear()
            self._gather_pools.clear()
            self._windows.clear()
