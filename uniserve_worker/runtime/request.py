"""Request admission, accepted progress, and references to in-flight outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig

from ..foundation.errors import invalid_descriptor
from ..protocol.batch import (
    BatchCommand,
    ComputationId,
    Finish,
    ImageParams,
    NewRequest,
    OpStatus,
    RequestKey,
    SamplingParams,
    ScheduledRequest,
    Start,
    TensorRef,
)

if TYPE_CHECKING:
    from ..execution.diffusion_state import ImageState, VideoState
    from ..execution.output import OutputBuffer, PendingOutput


@dataclass(frozen=True, slots=True)
class RequestProgress:
    """Immutable accepted or projected coordinates for a state-consuming operation."""

    logical_position: int = 0
    rng_counter: int = 0
    latent_product: TensorRef | None = None
    flow_step: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0
    prompt_logits_ready: bool = False

    def __post_init__(self) -> None:
        if self.logical_position < 0 or self.rng_counter < 0 or self.flow_step < 0:
            raise invalid_descriptor("request execution coordinates are negative")
        if not 0 <= self.kv_visible_len <= self.kv_computed_len:
            raise invalid_descriptor("request KV extents are not contained")


@dataclass(slots=True)
class RequestState:
    """One scheduler-assigned slot and the live state of an admitted request epoch."""

    request_key: RequestKey
    request_pool_idx: int
    admission: NewRequest
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    accepted_progress: RequestProgress
    accepted_op_id: ComputationId = ComputationId(0, 0)
    diffusion: ImageState | VideoState | None = None
    tail: PendingOutput | None = None
    pending_operations: dict[ComputationId, PendingOutput] = field(default_factory=dict)
    closed: bool = False
    retired: bool = False

    @property
    def request_id(self) -> int:
        return self.request_key.request_id


class RequestPool:
    """Own request slots and publish accepted progress in predecessor order."""

    def __init__(
        self,
        max_request_pool_size: int,
        *,
        state_buffers: Mapping[str, BufferConfig] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        size = int(max_request_pool_size)
        if size < 1:
            raise ValueError("request-pool capacity must be positive")
        self.max_request_pool_size = size
        self._closed = False
        self._rows: list[RequestState | None] = [None] * (size + 1)
        self._slots_by_request: dict[int, int] = {}
        self.tensor_slots = (
            tuple(
                TensorBuffers.allocate(
                    state_buffers, device=device, pin_memory=torch.device(device).type == "cuda"
                )
                for _ in range(size)
            )
            if state_buffers
            else ()
        )

    def tensors(self, request_pool_idx: int) -> TensorBuffers:
        """Borrow storage while holding the execution lease through device completion."""

        slot = self._validate_slot(request_pool_idx)
        if not self.tensor_slots:
            raise invalid_descriptor("request has no declared persistent tensor storage")
        return self.tensor_slots[slot - 1]

    def close(self) -> None:
        """Release drained request views before their communication owners retire.

        The caller must first stop admissions and drain outputs and graphs.
        Request records may remain borrowed by completion observers, so clear
        their numerical references as well as this pool's backing allocations.
        """

        if self._closed:
            return
        self._closed = True
        for row in self._rows:
            if row is not None:
                row.diffusion = None
                row.tail = None
                row.pending_operations.clear()
        self._rows.clear()
        self._slots_by_request.clear()
        self.tensor_slots = ()

    def get(self, request_id: int) -> RequestState:
        row = self.peek(request_id)
        if row is None:
            raise invalid_descriptor(f"unknown request {request_id}")
        return row

    def peek(self, request_id: int) -> RequestState | None:
        slot = self._slots_by_request.get(int(request_id))
        return None if slot is None else self._rows[slot]

    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._slots_by_request))

    def create_outputs(
        self,
        operations: Sequence[ScheduledRequest],
        request_pool_indices: Sequence[int],
        buffer: OutputBuffer,
    ) -> tuple[PendingOutput, ...]:
        """Validate scheduler ownership and capture each operation's stable predecessor."""

        from ..execution.output import PendingOutput

        if len(operations) != len(request_pool_indices):
            raise invalid_descriptor("request-pool indices are not aligned with operations")
        if len({operation.request_key for operation in operations}) != len(operations):
            raise invalid_descriptor("a completion group repeats a request")
        slots = tuple(self._validate_slot(value) for value in request_pool_indices)
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("a completion group repeats a request-pool index")
        outputs = []
        for index, (operation, slot) in enumerate(zip(operations, slots, strict=True)):
            request = self.get(operation.request_key.request_id)
            if request.request_key != operation.request_key:
                raise invalid_descriptor(f"operation {operation.op_id} has a stale request key")
            if self._rows[slot] is not request or request.request_pool_idx != slot:
                raise invalid_descriptor(f"operation {operation.op_id} has a stale request slot")
            if request.closed:
                raise invalid_descriptor(f"operation {operation.op_id} targets a closed request")
            if operation.op_id in request.pending_operations:
                raise invalid_descriptor("request operation is already executing")
            predecessor: PendingOutput | RequestProgress | None = None
            if operation.predecessor is not None:
                predecessor = request.tail or request.accepted_progress
                if isinstance(predecessor, PendingOutput):
                    if not predecessor.successors_ready and predecessor.accepted_progress is None:
                        raise invalid_descriptor("request predecessor acceptance is unresolved")
                # Runtime can receive an operation after its predecessor was retired;
                # the admitted accepted snapshot is the stable base in that case.
            outputs.append(PendingOutput(operation, request, buffer, index, predecessor))
        return tuple(outputs)

    def validate_pending(self, outputs: Sequence[PendingOutput]) -> None:
        """Preflight request references before any resource publication becomes visible."""

        for output in outputs:
            request = output.request
            if (
                self.peek(request.request_id) is not request
                or request.request_key != output.request_key
            ):
                raise RuntimeError("request publication lost its admitted slot")
            if output.op_id in request.pending_operations:
                raise RuntimeError("request publication repeats an executing operation")

    def add_pending(self, outputs: Sequence[PendingOutput]) -> None:
        """Install the same output objects used by execution and dependent operations."""

        self.validate_pending(outputs)
        # Keep the verification acceptance boundary for the entire completion group.
        continuation = all(output.draft_tokens is None for output in outputs)
        for output in outputs:
            request = output.request
            request.pending_operations[output.op_id] = output
            output.successors_ready = continuation
            if output.operation.predecessor is not None:
                request.tail = output

    def apply_outputs(self, outputs: Sequence[PendingOutput]) -> None:
        """Apply actual acceptance in causal order; late outputs never replace newer state."""

        from ..execution.output import PendingOutput

        for output in outputs:
            predecessor = output.predecessor
            if isinstance(predecessor, PendingOutput):
                self.apply_outputs((predecessor,))
            request = output.request
            if request.pending_operations.get(output.op_id) is not output:
                continue
            if output.value is None:
                raise RuntimeError("request output has not been materialized")
            del request.pending_operations[output.op_id]
            if self.peek(request.request_id) is request:
                if output.accepted_progress is not None and output.op_id > request.accepted_op_id:
                    request.accepted_progress = output.accepted_progress
                    request.accepted_op_id = output.op_id
                if output.value.status is OpStatus.ERROR:
                    request.closed = True
                if request.tail is output:
                    request.tail = None
            output.predecessor = None
            output.successors_ready = True

    def cancel_outputs(self, outputs: Sequence[PendingOutput]) -> None:
        """Close requests whose submitted numerical acceptance can no longer be determined."""

        for output in outputs:
            request = output.request
            request.pending_operations.pop(output.op_id, None)
            request.closed = True
            if request.tail is output:
                request.tail = None
            output.predecessor = None

    def start(self, admission: NewRequest) -> int | None:
        """Bind an immutable admission to the exact scheduler-assigned slot."""

        return self._apply_start(admission)

    def finish(self, request_key: RequestKey) -> None:
        request = self.peek(request_key.request_id)
        if request is not None and request.request_key == request_key:
            request.closed = True

    def apply_commands(self, commands: Sequence[BatchCommand]) -> tuple[int, ...]:
        started = []
        for command in commands:
            if isinstance(command, Start):
                slot = self.start(command.request)
                if slot is not None:
                    started.append(slot)
            elif isinstance(command, Finish):
                self.finish(command.request_key)
        return tuple(started)

    def retirement_ready(self, request_key: RequestKey) -> bool:
        request = self.peek(request_key.request_id)
        return (
            request is None
            or request.request_key != request_key
            or all(output.ready() for output in request.pending_operations.values())
        )

    def drop(self, request_id: int) -> None:
        """Remove a row only after its execution leases have been released."""

        slot = self._slots_by_request.pop(int(request_id), None)
        if slot is not None:
            self._rows[slot] = None

    def retire(self, request_id: int) -> None:
        row = self.get(request_id)
        if row.retired:
            return
        if not row.closed or not self.retirement_ready(row.request_key):
            raise RuntimeError("request retirement requires closed, completed execution")
        row.pending_operations.clear()
        row.tail = None
        row.diffusion = None
        row.retired = True

    def _validate_slot(self, request_pool_idx: int) -> int:
        if self._closed:
            raise RuntimeError("request pool is closed")
        slot = int(request_pool_idx)
        if not 1 <= slot <= self.max_request_pool_size:
            raise invalid_descriptor(f"request-pool index {slot} exceeds capacity")
        return slot

    def _apply_start(self, admission: NewRequest) -> int | None:
        slot = self._validate_slot(admission.request_pool_idx)
        base = self.peek(admission.request_key.request_id)
        if base is not None and base.retired and base.request_key != admission.request_key:
            self.drop(base.request_id)
            base = None
        occupant = self._rows[slot]
        if (
            occupant is not None
            and occupant.retired
            and occupant.request_key != admission.request_key
        ):
            self.drop(occupant.request_id)
            occupant = None
        if base is not None:
            if base.admission != admission or occupant is not base:
                raise invalid_descriptor("request admission conflicts with resident state")
            return None
        if occupant is not None:
            raise invalid_descriptor(f"request-pool index {slot} is occupied")
        prefix = 0 if admission.ar is None else int(admission.ar.initial_position)
        self._rows[slot] = RequestState(
            request_key=admission.request_key,
            request_pool_idx=slot,
            admission=admission,
            sampling=None if admission.ar is None else admission.ar.sampling,
            image=None if admission.umm is None else admission.umm.image,
            negative_token_ids=() if admission.ar is None else admission.ar.negative_token_ids,
            finish_token_ids=() if admission.ar is None else admission.ar.finish_token_ids,
            accepted_progress=RequestProgress(
                logical_position=prefix, kv_visible_len=prefix, kv_computed_len=prefix
            ),
        )
        self._slots_by_request[admission.request_key.request_id] = slot
        return slot
