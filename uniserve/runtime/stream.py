"""CUDA streams and optional SM partitions."""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from typing import Any

import torch

from uniserve.runtime.cuda import CUDAError, cuda_status, cuda_value, driver
from uniserve.runtime.resources import close_resources


class CUDAStream:
    """Own one CUDA execution stream and the resources bound to it.

    CPU execution has no CUDAStream. Ordinary bindings retain a PyTorch stream;
    partitioned bindings own a native stream and may borrow their parent's
    Green Context. A parent outlives every fork that uses its SM allocation.
    Ingress/output event slots are reused in submission order on the worker.

    The stream also owns its :attr:`communication`: the communicators that
    enqueue collectives on it, their registered windows and the storage behind
    them. Execution contexts on the stream borrow these resources, and
    :meth:`close` retires them collectively with the stream.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        stream: torch.cuda.Stream,
        sm_count: int,
        green: Any = None,
        context: Any = None,
        raw_stream: Any = None,
        event_slots: int = 2,
        parent: CUDAStream | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA stream requires a CUDA device")
        if event_slots < 1:
            raise ValueError("CUDA stream requires a positive event bound")

        self.device = device
        self.stream = stream
        self.sm_count = int(sm_count)
        self.green = green
        self.context = context
        self.raw_stream = raw_stream
        self._parent = parent
        self._closed = False
        self._event_cursor = 0
        self._ingress_events = tuple(
            torch.cuda.Event(blocking=False) for _ in range(event_slots)
        )
        self._output_events = tuple(
            torch.cuda.Event(blocking=False) for _ in range(event_slots)
        )

        from ._collectives import StreamCommunication

        self.communication = StreamCommunication(stream)

    @classmethod
    def external(
        cls, stream: torch.cuda.Stream, *, event_slots: int = 2
    ) -> CUDAStream:
        """Own the resources bound to an existing full-device PyTorch stream.

        The PyTorch stream remains its creator's; closing this owner retires
        the communicators, windows and storage bound to the stream.
        """
        return cls(
            device=stream.device,
            stream=stream,
            sm_count=torch.cuda.get_device_properties(
                stream.device
            ).multi_processor_count,
            event_slots=event_slots,
        )

    @property
    def full_device(self) -> bool:
        return self.green is None

    def verify(self) -> None:
        """Verify that the native stream retains its configured SM partition."""
        if self.green is None:
            return

        cu = driver()
        associated = cuda_value(
            cu.cuStreamGetGreenCtx(self.raw_stream), "query stream context"
        )
        if int(associated) != int(self.green):
            raise CUDAError("stream is detached from its Green Context")

        resource = cuda_value(
            cu.cuGreenCtxGetDevResource(
                self.green, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            ),
            "query Green Context SM resource",
        )
        if int(resource.sm.smCount) != self.sm_count:
            raise CUDAError("Green Context SM resource changed after startup")

    def wait(self, producer: torch.cuda.Stream) -> None:
        """Order this stream after a producer.

        Order this stream after a producer without allocating per-call
        events.
        """
        if int(producer.cuda_stream) == int(self.stream.cuda_stream):
            return
        event = self._ingress_events[
            self._event_cursor % len(self._ingress_events)
        ]
        event.record(producer)
        self.stream.wait_event(event)

    def record(self) -> torch.cuda.Event | None:
        """Return a producer fence when a cross-stream join is needed.

        Return a producer fence only when the caller needs a cross-stream
        join.
        """
        current = torch.cuda.current_stream(self.device)
        if int(current.cuda_stream) == int(self.stream.cuda_stream):
            return None

        event = self._output_events[
            self._event_cursor % len(self._output_events)
        ]
        self._event_cursor += 1
        event.record(self.stream)
        return event

    def synchronize(self) -> None:
        self.stream.synchronize()

    def fork(self) -> CUDAStream:
        """Own another ordered stream within this same device resource grant.

        The parent must outlive its forks, including their captured graphs.
        Forking a partition borrows its context and does not acquire any SMs.
        """
        raw = None
        if self.green is None:
            stream = torch.cuda.Stream(device=self.device)
        else:
            cu = driver()
            raw = cuda_value(
                cu.cuGreenCtxStreamCreate(
                    self.green,
                    int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING),
                    self.stream.priority,
                ),
                "create computation stream in existing partition",
            )
            # Wrap the driver handle so PyTorch dispatches onto the same stream.
            stream = torch.cuda.ExternalStream(int(raw), device=self.device)

        try:
            return CUDAStream(
                device=self.device,
                stream=stream,
                sm_count=self.sm_count,
                green=self.green,
                context=self.context,
                raw_stream=raw,
                event_slots=len(self._ingress_events),
                parent=self,
            )
        except BaseException:
            if raw is not None:
                _destroy_stream(raw)
            raise

    def __enter__(self) -> CUDAStream:
        if self._closed:
            raise CUDAError("CUDA stream is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Communicator retirement waits on peers this rank cannot observe, so
        # an exception retains bound communication; an unbound stream closes
        # normally once its submitted work has finished.
        from .resources import streams_idle

        try:
            self.close(
                aborted=exc is not None
                and (
                    bool(self.communication.communicators)
                    or not streams_idle((self.stream,))
                )
            )
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"CUDA stream cleanup failed: {error!r}")

    def close(self, *, aborted: bool = False) -> None:
        """Retire bound communication, drain accesses and release the stream.

        Communicator retirement is collective, so every member rank closes its
        corresponding stream after the contexts and graphs that borrow it.
        Aborted close retains every resource without waiting for peers or the
        device; the owning process must exit before reclaiming them.
        """
        if self._closed:
            return
        self._closed = True

        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            self.communication.close(aborted=True)
            return

        actions = [self.communication.close, self.stream.synchronize]
        if self.raw_stream is not None:
            actions.append(partial(_destroy_stream, self.raw_stream))
        # Only the partition owner destroys the Green Context; forks borrow it.
        if self.green is not None and self._parent is None:
            actions.append(partial(_destroy_context, self.green))

        try:
            close_resources(*actions)
        finally:
            self._ingress_events = ()
            self._output_events = ()
            self.raw_stream = None
            self.green = None
            self.context = None
            self._parent = None


def _destroy_stream(stream: Any) -> None:
    cuda_status(driver().cuStreamDestroy(stream), "destroy CUDA stream")


def _destroy_context(context: Any) -> None:
    cuda_status(driver().cuGreenCtxDestroy(context), "destroy Green Context")


def partition_streams(
    device: torch.device,
    sm_counts: tuple[int, ...],
    *,
    event_slots: int | tuple[int, ...] = 2,
) -> tuple[CUDAStream, ...]:
    """Realize exact, disjoint Green Context resources from one split tree."""
    if not sm_counts:
        return ()

    if device.type != "cuda":
        raise CUDAError("execution lanes require a CUDA device")
    if any(type(count) is not int or count < 1 for count in sm_counts):
        raise CUDAError("stream SM counts must be positive integers")

    if isinstance(event_slots, int):
        slot_counts = (event_slots,) * len(sm_counts)
    else:
        slot_counts = event_slots
    if len(slot_counts) != len(sm_counts) or any(
        count < 1 for count in slot_counts
    ):
        raise CUDAError("each CUDA stream requires a positive event-slot count")

    torch.cuda.init()
    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    cu = driver()
    cuda_status(cu.cuInit(0), "initialize CUDA driver")
    cuda_device = cuda_value(cu.cuDeviceGet(index), "resolve CUDA device")
    full = cuda_value(
        cu.cuDeviceGetDevResource(
            cuda_device, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ),
        "query device SM resource",
    )

    requested = sum(sm_counts)
    available = int(full.sm.smCount)
    if requested > available:
        raise CUDAError(
            f"lane SM budgets require {requested} SMs but the device "
            f"exposes {available}"
        )

    acquisition = ExitStack()
    partitions = ExitStack()
    realized: list[CUDAStream] = []
    try:
        # Realize the requested total as one Green Context first; lane
        # partitions are carved out of its SM resource below. The aggregate
        # and intermediate remainder contexts are destroyed before returning.
        aggregate, _unassigned = _split_one(cu, full, requested)
        aggregate_green = _green_from_resources(cu, cuda_device, (aggregate,))
        partitions.callback(_destroy_context, aggregate_green)
        current_resource = _green_resource(cu, aggregate_green)

        for ordinal, (requested_count, slots) in enumerate(
            zip(sm_counts, slot_counts, strict=True)
        ):
            # The last lane takes the whole remaining resource; earlier lanes
            # split their exact SM count off and pass the remainder along.
            if ordinal + 1 == len(sm_counts):
                lane_resource = current_resource
                remainder = None
            else:
                lane_resource, remainder = _split_one(
                    cu, current_resource, requested_count
                )

            lane_green = _green_from_resources(
                cu, cuda_device, (lane_resource,)
            )
            acquisition.callback(_destroy_context, lane_green)
            context = cuda_value(
                cu.cuCtxFromGreenCtx(lane_green), "resolve lane context"
            )
            raw_stream = cuda_value(
                cu.cuGreenCtxStreamCreate(
                    lane_green, int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING), 0
                ),
                "create lane origin stream",
            )
            acquisition.callback(_destroy_stream, raw_stream)

            resolved = _green_resource(cu, lane_green)
            sm_count = int(resolved.sm.smCount)
            if sm_count != requested_count:
                raise CUDAError(
                    f"CUDA stream received {sm_count} SMs, "
                    f"expected {requested_count}"
                )
            associated = cuda_value(
                cu.cuStreamGetGreenCtx(raw_stream), "query lane origin stream"
            )
            if int(associated) != int(lane_green):
                raise CUDAError(
                    "lane origin stream has the wrong Green Context"
                )

            realized.append(
                CUDAStream(
                    device=device,
                    sm_count=sm_count,
                    green=lane_green,
                    context=context,
                    raw_stream=raw_stream,
                    stream=torch.cuda.ExternalStream(
                        int(raw_stream), device=device
                    ),
                    event_slots=slots,
                )
            )

            if remainder is not None:
                remainder_green = _green_from_resources(
                    cu, cuda_device, (remainder,)
                )
                partitions.callback(_destroy_context, remainder_green)
                current_resource = _green_resource(cu, remainder_green)
        if sum(item.sm_count for item in realized) != requested:
            raise CUDAError(
                "lane SM resources do not form the configured disjoint total"
            )

        partitions.close()
    except BaseException as error:
        try:
            close_resources(acquisition.close, partitions.close)
        except BaseException as cleanup_error:
            error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
        raise
    # Each returned context owns its stream and exact SM partition.
    acquisition.pop_all()
    return tuple(realized)


def _split_one(cu: Any, resource: Any, count: int) -> tuple[Any, Any]:
    """Split one resource between green and default execution lanes."""
    result = cu.cuDevSmResourceSplitByCount(1, resource, 0, int(count))
    cuda_status(result, "split SM resource")
    groups, group_count, remainder = result[1], int(result[2]), result[3]
    if group_count != 1 or not groups:
        raise CUDAError("CUDA could not realize the requested SM resource")
    group = groups[0]
    if int(group.sm.smCount) != int(count):
        raise CUDAError(
            f"CUDA rounded an exact SM request from {count} to "
            f"{int(group.sm.smCount)}"
        )
    return group, remainder


def _green_from_resources(
    cu: Any, device: Any, resources: tuple[Any, ...]
) -> Any:
    """Construct a green context from split device resources when supported."""
    descriptor = cuda_value(
        cu.cuDevResourceGenerateDesc(list(resources), len(resources)),
        "generate Green Context resource descriptor",
    )
    return cuda_value(
        cu.cuGreenCtxCreate(
            descriptor,
            device,
            int(cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "create Green Context",
    )


def _green_resource(cu: Any, green: Any) -> Any:
    """Extract the green-partition handle returned by a CUDA split call."""
    return cuda_value(
        cu.cuGreenCtxGetDevResource(
            green, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ),
        "query Green Context SM resource",
    )


__all__ = [
    "CUDAStream",
    "partition_streams",
]
