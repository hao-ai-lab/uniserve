"""Request admission, pending call identities, and accepted progress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from uniserve.sampling import SamplingParams
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.batch import (
    BatchCommand,
    Finish,
    NewRequest,
    Start,
)
from uniserve_worker.protocol.call import Call, CallStatus, ImageParams
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.storage.request_slots import RequestSlots

if TYPE_CHECKING:
    from uniserve_worker.execution.diffusion_state import ImageState, VideoState


@dataclass(frozen=True, slots=True)
class RequestProgress:
    """Accepted or projected coordinates for a state-consuming call."""

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


@dataclass(frozen=True, slots=True)
class RequestResult:
    """An observed call status and the progress accepted by its output owner."""

    request_key: RequestKey
    call_id: CallId
    status: CallStatus
    progress: RequestProgress | None


@dataclass(slots=True)
class RequestState:
    """A scheduler-assigned slot and the state of an admitted request epoch."""

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
    pending_calls: dict[CallId, Call] = field(default_factory=dict)
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

        self.storage = RequestSlots(
            size, state_buffers=state_buffers, device=device
        )

    def close(self) -> None:
        """Release drained request views before communication owners retire.

        The caller must first stop admissions and drain outputs
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
        self.storage.close()

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

    def bind_calls(
        self,
        calls: Sequence[Call],
        request_pool_indices: Sequence[int],
    ) -> tuple[RequestState, ...]:
        """Resolve admitted requests after validating their slot ownership."""
        if len(calls) != len(request_pool_indices):
            raise invalid_descriptor(
                "request-pool indices are not aligned with calls"
            )
        if len({call.request_key for call in calls}) != len(calls):
            raise invalid_descriptor("a batch repeats a request")

        slots = tuple(
            self._validate_slot(value) for value in request_pool_indices
        )
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("a batch repeats a request-pool index")

        requests = []
        for call, slot in zip(calls, slots, strict=True):
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
            requests.append(request)
        return tuple(requests)

    def validate_pending(self, calls: Sequence[Call]) -> None:
        """Validate pending identities before exposing resource publications."""
        seen: set[tuple[RequestKey, CallId]] = set()
        for call in calls:
            request = self.get(call.request_key.request_id)
            if request.request_key != call.request_key:
                raise RuntimeError("request publication lost its admitted slot")
            identity = (call.request_key, call.call_id)
            if call.call_id in request.pending_calls or identity in seen:
                raise RuntimeError(
                    "request publication repeats an executing call"
                )
            seen.add(identity)

    def add_pending(self, calls: Sequence[Call]) -> None:
        """Record submitted call identities for ordering and retirement."""
        self.validate_pending(calls)
        for call in calls:
            self.get(call.request_key.request_id).pending_calls[
                call.call_id
            ] = call

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
                if pending.advances_state
            ]
            predecessors[call.call_id] = (
                in_flight[-1] if in_flight else request.state_call_id
            )
        return predecessors

    def apply_result(self, result: RequestResult) -> None:
        """Accept one observed result without replacing newer request progress.

        Pending calls carry identity and ordering only. Output readiness and
        materialization are established by execution before this update arrives.
        """
        request = self.peek(result.request_key.request_id)
        if request is None or request.request_key != result.request_key:
            return
        call = request.pending_calls.pop(result.call_id, None)
        if call is None:
            return
        if (
            result.progress is not None
            and result.call_id > request.accepted_call_id
        ):
            request.accepted_progress = result.progress
            request.accepted_call_id = result.call_id
            request.prompt_logits_ready = result.progress.prompt_logits_ready
            request.rng_counter = result.progress.rng_counter
        if (
            result.status is CallStatus.OK
            and call.advances_state
            and result.call_id > request.state_call_id
        ):
            request.state_call_id = result.call_id
        if result.status is CallStatus.ERROR:
            request.closed = True

    def cancel_calls(self, calls: Sequence[Call]) -> None:
        """Close requests whose submitted acceptance cannot be determined."""
        for call in calls:
            request = self.peek(call.request_key.request_id)
            if request is None or request.request_key != call.request_key:
                continue
            request.pending_calls.pop(call.call_id, None)
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
            or not request.pending_calls
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
