"""PyTorch views of native execution streams and bound communication."""

from __future__ import annotations

import torch

from uniserve_worker._uniserve_ipc import CUDAEvent
from uniserve_worker._uniserve_ipc import CUDAStream as NativeStream

from .cuda import CUDAError
from .resources import close_resources


class CUDAStream:
    """Own an execution stream and the communication bound to it.

    Rust owns the stream, its optional SM partition and reusable fences.
    Python retains the PyTorch view and numerical communication resources.
    Close contexts and graphs before closing the stream they borrow.
    """

    def __init__(self, native: NativeStream, stream: torch.cuda.Stream):
        from ._collectives import StreamCommunication

        self._native = native
        self.stream = stream
        self.device = stream.device
        self.communication = StreamCommunication(stream, native)

    @classmethod
    def external(
        cls, stream: torch.cuda.Stream, *, event_slots: int = 2
    ) -> CUDAStream:
        """Bind execution resources to a caller-owned PyTorch stream."""
        native = NativeStream(
            stream.device.index, stream.cuda_stream, event_slots
        )
        return cls(native, stream)

    @property
    def sm_count(self) -> int:
        return self._native.sm_count

    @property
    def full_device(self) -> bool:
        return self._native.full_device

    def wait(self, producer: torch.cuda.Stream) -> None:
        """Order this stream after a producer using a reusable device event."""
        self._native.wait(producer.cuda_stream)

    def record(self) -> CUDAEvent | None:
        """Fence this stream for a later cross-stream join.

        The consumer enqueues its wait before the configured event ring wraps.
        Independent lanes may submit work before their consumers join them.
        """
        current = torch.cuda.current_stream(self.device)
        return self._native.record(current.cuda_stream)

    def synchronize(self) -> None:
        self._native.synchronize()

    def fork(self) -> CUDAStream:
        """Own an independent stream retaining this stream's SM partition."""
        native = self._native.fork()
        stream = torch.cuda.ExternalStream(native.handle, device=self.device)
        return CUDAStream(native, stream)

    def __enter__(self) -> CUDAStream:
        if self._native.closed:
            raise CUDAError("CUDA stream is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
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
        """Retire communication, then drain and release the native stream.

        Communicator retirement is collective. Aborted execution retains its
        communication and device resources until the owning process exits.
        """
        if self._native.closed:
            return

        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            self.communication.close(aborted=True)
            self._native.close(aborted=True)
            return

        close_resources(self.communication.close, self._native.close)


def partition_streams(
    device: torch.device,
    sm_counts: tuple[int, ...],
    *,
    event_slots: int | tuple[int, ...] = 2,
) -> tuple[CUDAStream, ...]:
    """Allocate disjoint SM partitions and expose their PyTorch stream views."""
    if not sm_counts:
        return ()
    if device.type != "cuda":
        raise CUDAError("execution lanes require a CUDA device")

    torch.cuda.init()
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    slots = (
        (event_slots,) * len(sm_counts)
        if isinstance(event_slots, int)
        else event_slots
    )
    native = NativeStream.partition(index, sm_counts, slots)
    return tuple(
        CUDAStream(
            stream, torch.cuda.ExternalStream(stream.handle, device=index)
        )
        for stream in native
    )


__all__ = ["CUDAStream", "partition_streams"]
