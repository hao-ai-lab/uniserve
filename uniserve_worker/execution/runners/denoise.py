"""Request-bound denoising with exact geometry and schedule variants."""

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import torch

from ...nn.diffusion.schedule import DiffusionSchedule
from ...nn.mesh import Communicator
from ..denoising import DenoisingStep
from ..graph.backend import CudaGraphBackend, GraphExecutionError
from .module import TensorOutput, capture_required


def restore_samples(operation: DenoisingStep) -> Callable[[], None]:
    """Snapshot only solver state; learned predictions are disposable scratch."""

    snapshots = tuple(value.clone() for value in operation.samples)

    def restore() -> None:
        for value, saved in zip(operation.samples, snapshots, strict=True):
            value.copy_(saved)

    return restore


@dataclass(slots=True)
class _Slot:
    signature: Hashable
    geometry: Hashable
    operations: dict[Hashable, DenoisingStep] = field(default_factory=dict)


class DenoiseRunner:
    """Own graph-bound request slots while sharing the model's step binding.

    Physical slot identity and backing addresses participate in residency. Pooled
    storage can be reused by subsequent requests; graphs retain only numerical
    views, never request identities. Replacing backing or geometry retires the
    dependent graphs before their metadata is freed.
    """

    def __init__(
        self,
        bind_step: Callable[..., DenoisingStep],
        signature: Callable[..., Hashable],
        *,
        device: torch.device,
        backend: CudaGraphBackend[TensorOutput] | None,
        groups: tuple[Communicator, ...],
        capacity: int,
    ) -> None:
        self.bind_step = bind_step
        self.signature = signature
        self.device = device
        self.backend = backend
        self.groups = groups
        self.capacity = max(2, capacity)
        self._slots: OrderedDict[Hashable, _Slot] = OrderedDict()
        self._closed = False

    @torch.inference_mode()
    def warmup(self, tensors: object, metadata: object, schedule: DiffusionSchedule) -> None:
        operation = self.bind_step(tensors, metadata, 0, schedule)
        restore = restore_samples(operation)
        try:
            operation()
        finally:
            restore()
            torch.cuda.current_stream(self.device).synchronize()

    @torch.inference_mode()
    def run(
        self,
        tensors: object,
        metadata: object,
        step: int,
        schedule: DiffusionSchedule,
        *,
        slot: Hashable,
        geometry: Hashable,
    ) -> tuple[TensorOutput, str]:
        if self._closed:
            raise GraphExecutionError("denoising runner is closed")
        operation = self.bind_step(tensors, metadata, step, schedule)
        if self.backend is None:
            return operation(), "eager"
        backing = tuple(
            (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
            for value in operation.samples
        )
        signature = (self.signature(tensors, metadata), backing, id(metadata))
        variant = (step, id(schedule))
        key = (slot, signature, variant)
        missing = capture_required(not self.backend.contains(key), self.groups, self.device)
        if missing:
            torch.cuda.current_stream(self.device).synchronize()
            resident = self._slots.get(slot)
            if resident is not None and resident.signature != signature:
                self.discard_slot(slot)
                resident = None
            if resident is None and len(self._slots) >= self.capacity:
                self.discard_slot(next(iter(self._slots)))
            self.backend.discard(key)
            restore = restore_samples(operation)
            try:
                self.backend.capture_one(
                    key, operation, keepalive=(tensors, metadata, schedule), restore=restore
                )
            finally:
                del restore
            if resident is None:
                resident = _Slot(signature, geometry)
                self._slots[slot] = resident
            resident.operations[variant] = operation
        self._slots.move_to_end(slot)
        return self.backend.replay(key), "graph_capture" if missing else "graph_replay"

    def discard_slot(self, slot: Hashable) -> None:
        """Release an already-drained request's graphs before its tensor backing."""

        resident = self._slots.pop(slot, None)
        if resident is not None and self.backend is not None:
            for variant in resident.operations:
                self.backend.discard((slot, resident.signature, variant))

    def discard_geometry(self, geometry: Hashable) -> None:
        for slot, resident in tuple(self._slots.items()):
            if resident.geometry == geometry:
                self.discard_slot(slot)

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
        self._slots.clear()
        self._closed = True
