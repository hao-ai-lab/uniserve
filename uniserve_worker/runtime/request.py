"""Request admission, accepted progress, and references to in-flight outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.sampling import SamplingParams
from uniserve.tensors import BufferConfig

from ..foundation.errors import invalid_descriptor
from ..protocol.batch import (
    BatchCommand,
    Finish,
    NewRequest,
    Start,
)
from ..protocol.call import Call, CallStatus, ImageParams
from ..protocol.identity import CallId, RequestKey

if TYPE_CHECKING:
    from ..execution.diffusion_state import ImageState, VideoState
    from ..execution.output import OutputBuffer, PendingOutput


@dataclass(frozen=True, slots=True)
class RequestProgress:
    """Immutable accepted or projected coordinates for a state-consuming.

    call.
    """

    logical_position: int = 0
    rng_counter: int = 0
    flow_step: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0
    prompt_logits_ready: bool = False

    def __post_init__(self) -> None:
        if (
            self.logical_position < 0
            or self.rng_counter < 0
            or self.flow_step < 0
        ):
            raise invalid_descriptor(
                "request execution coordinates are negative"
            )
        if not 0 <= self.kv_visible_len <= self.kv_computed_len:
            raise invalid_descriptor("request KV extents are not contained")


@dataclass(slots=True)
class RequestState:
    """One scheduler-assigned slot and the live state of an admitted request.

    epoch.
    """

    request_key: RequestKey
    request_pool_idx: int
    admission: NewRequest
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    accepted_progress: RequestProgress
    accepted_call_id: CallId = CallId(0, 0)
    # The latest state-advancing call whose result the request accepted with
    # status OK; the admission root before any. A later call of the request
    # follows this one when none of its state-advancing calls is in flight.
    state_call_id: CallId = CallId(0, 0)
    # State one call produces and a later call reads, which the engine does not
    # track: it lives with the request rather than in a call's stated
    # coordinates. rng_counter is the request's sampling position and is not
    # the wire's Rng.semantic_index_base, which the engine derives per call
    # from the logical position, the scheduled span or the image identity.
    prompt_logits_ready: bool = False
    rng_counter: int = 0
    diffusion: ImageState | VideoState | None = None
    pending_calls: dict[CallId, PendingOutput] = field(default_factory=dict)
    closed: bool = False
    retired: bool = False

    @property
    def request_id(self) -> int:
        return self.request_key.request_id


class RequestPool:
    """Own request slots and publish accepted progress as calls complete."""

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
        # Rows are one-based to match scheduler slot ids; row 0 stays empty.
        self._rows: list[RequestState | None] = [None] * (size + 1)
        self._slots_by_request: dict[int, int] = {}

        self.tensor_slots = (
            tuple(
                TensorBuffers.allocate(
                    state_buffers,
                    device=device,
                    pin_memory=torch.device(device).type == "cuda",
                )
                for _ in range(size)
            )
            if state_buffers
            else ()
        )

    def tensors(self, request_pool_idx: int) -> TensorBuffers:
        """Borrow storage while holding the execution lease through device.

        completion.
        """
        slot = self._validate_slot(request_pool_idx)
        if not self.tensor_slots:
            raise invalid_descriptor(
                "request has no declared persistent tensor storage"
            )
        return self.tensor_slots[slot - 1]

    def close(self) -> None:
        """Release drained request views before their communication owners.

        retire. The caller must first stop admissions and drain outputs
        and graphs. Request records may remain borrowed by completion
        observers, so clear their numerical references as well as this
        pool's backing allocations.
        """
        if self._closed:
            return
        self._closed = True
        for row in self._rows:
            if row is not None:
                row.diffusion = None
                row.pending_calls.clear()
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
        calls: Sequence[Call],
        request_pool_indices: Sequence[int],
        buffer: OutputBuffer,
    ) -> tuple[PendingOutput, ...]:
        """Validate scheduler ownership and capture each call's stable.

        predecessor.
        """
        from ..execution.output import PendingOutput

        if len(calls) != len(request_pool_indices):
            raise invalid_descriptor(
                "request-pool indices are not aligned with calls"
            )
        if len({call.request_key for call in calls}) != len(calls):
            raise invalid_descriptor("a completion group repeats a request")

        slots = tuple(
            self._validate_slot(value) for value in request_pool_indices
        )
        if len(set(slots)) != len(slots):
            raise invalid_descriptor(
                "a completion group repeats a request-pool index"
            )

        outputs = []
        for index, (call, slot) in enumerate(zip(calls, slots, strict=True)):
            request = self.get(call.request_key.request_id)
            if request.request_key != call.request_key:
                raise invalid_descriptor(
                    f"call {call.call_id} has a stale request key"
                )
            if (
                self._rows[slot] is not request
                or request.request_pool_idx != slot
            ):
                raise invalid_descriptor(
                    f"call {call.call_id} has a stale request slot"
                )
            if request.closed:
                raise invalid_descriptor(
                    f"call {call.call_id} targets a closed request"
                )
            if call.call_id in request.pending_calls:
                raise invalid_descriptor("request call is already executing")
            outputs.append(PendingOutput(call, request, buffer, index))
        return tuple(outputs)

    def validate_pending(self, outputs: Sequence[PendingOutput]) -> None:
        """Preflight request references before any resource publication becomes.

        visible.
        """
        for output in outputs:
            request = output.request
            if (
                self.peek(request.request_id) is not request
                or request.request_key != output.request_key
            ):
                raise RuntimeError("request publication lost its admitted slot")
            if output.call_id in request.pending_calls:
                raise RuntimeError(
                    "request publication repeats an executing call"
                )

    def add_pending(self, outputs: Sequence[PendingOutput]) -> None:
        """Install the same output objects used by execution.

        and dependent calls.
        """
        self.validate_pending(outputs)
        for output in outputs:
            output.request.pending_calls[output.call_id] = output

    def predecessors(
        self, calls: Sequence[Call]
    ) -> dict[CallId, CallId | None]:
        """Name the call each of these calls follows in its request.

        A call follows the request's latest state-advancing call: the last one
        in flight on this rank, otherwise the last one the request accepted,
        otherwise the admission root. On the media path only a state-advancing
        call follows the chain; the media branches around it complete
        independently, so their products are released by the engine's commands
        rather than by a successor's arrival. A call for a request this rank
        does not hold follows nothing, and validation names it later.
        """
        predecessors: dict[CallId, CallId | None] = {}
        for call in calls:
            request = self.peek(call.request_key.request_id)
            if request is None or request.request_key != call.request_key:
                predecessors[call.call_id] = None
                continue
            media = request.admission.diffusion is not None
            if media and not call.advances_state:
                predecessors[call.call_id] = None
                continue
            in_flight = [
                pending.call_id
                for pending in request.pending_calls.values()
                if pending.call.advances_state
            ]
            predecessors[call.call_id] = (
                in_flight[-1] if in_flight else request.state_call_id
            )
        return predecessors

    def apply_outputs(self, outputs: Sequence[PendingOutput]) -> None:
        """Apply actual acceptance; a late output never replaces newer state.

        Calls reach a rank in the order the engine dispatched them and execute
        in that order, so acceptance applies in the order it arrives. A call
        identity that does not advance the request's is ignored.
        """
        for output in outputs:
            request = output.request
            if request.pending_calls.get(output.call_id) is not output:
                continue
            if output.value is None:
                raise RuntimeError("request output has not been materialized")
            del request.pending_calls[output.call_id]

            if self.peek(request.request_id) is request:
                if (
                    output.accepted_progress is not None
                    and output.call_id > request.accepted_call_id
                ):
                    request.accepted_progress = output.accepted_progress
                    request.accepted_call_id = output.call_id
                    # The device state a call produced becomes visible to the
                    # request's later calls only here, so a completion group
                    # that fails never exposes a partial trajectory.
                    request.prompt_logits_ready = (
                        output.accepted_progress.prompt_logits_ready
                    )
                    request.rng_counter = output.accepted_progress.rng_counter
                if (
                    output.value.status is CallStatus.OK
                    and output.call.advances_state
                    and output.call_id > request.state_call_id
                ):
                    request.state_call_id = output.call_id
                if output.value.status is CallStatus.ERROR:
                    request.closed = True

    def cancel_outputs(self, outputs: Sequence[PendingOutput]) -> None:
        """Close requests whose submitted numerical acceptance can no longer be.

        determined.
        """
        for output in outputs:
            request = output.request
            request.pending_calls.pop(output.call_id, None)
            request.closed = True

    def start(self, admission: NewRequest) -> int | None:
        """Bind an immutable admission to the exact scheduler-assigned slot."""
        return self._apply_start(admission)

    def finish(self, request_key: RequestKey) -> None:
        request = self.peek(request_key.request_id)
        if request is not None and request.request_key == request_key:
            request.closed = True

    def apply_commands(
        self, commands: Sequence[BatchCommand]
    ) -> tuple[int, ...]:
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
            or all(output.ready() for output in request.pending_calls.values())
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
            raise RuntimeError(
                "request retirement requires closed, completed execution"
            )
        row.pending_calls.clear()
        row.diffusion = None
        row.retired = True

    def _validate_slot(self, request_pool_idx: int) -> int:
        if self._closed:
            raise RuntimeError("request pool is closed")
        slot = int(request_pool_idx)
        if not 1 <= slot <= self.max_request_pool_size:
            raise invalid_descriptor(
                f"request-pool index {slot} exceeds capacity"
            )
        return slot

    def _apply_start(self, admission: NewRequest) -> int | None:
        slot = self._validate_slot(admission.request_pool_idx)

        # Evict a retired epoch still occupying the request id or the slot.
        base = self.peek(admission.request_key.request_id)
        if (
            base is not None
            and base.retired
            and base.request_key != admission.request_key
        ):
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
                raise invalid_descriptor(
                    "request admission conflicts with resident state"
                )
            return None
        if occupant is not None:
            raise invalid_descriptor(f"request-pool index {slot} is occupied")

        prefix = (
            0
            if admission.generation is None
            else int(admission.generation.initial_position)
        )
        self._rows[slot] = RequestState(
            request_key=admission.request_key,
            request_pool_idx=slot,
            admission=admission,
            sampling=None
            if admission.generation is None
            else admission.generation.sampling,
            image=admission.image,
            negative_token_ids=()
            if admission.generation is None
            else admission.generation.negative_token_ids,
            finish_token_ids=()
            if admission.generation is None
            else admission.generation.finish_token_ids,
            accepted_progress=RequestProgress(
                logical_position=prefix,
                kv_visible_len=prefix,
                kv_computed_len=prefix,
            ),
        )
        self._slots_by_request[admission.request_key.request_id] = slot
        return slot
