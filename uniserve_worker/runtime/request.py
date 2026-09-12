"""Stable request slots and execution progress retained by actual consumers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from uniserve_worker.execution.batch import ComputationId

from ..execution.batch import (
    BatchCommand,
    CompletionState,
    Finish,
    ImageParams,
    ModelOutput,
    NewRequest,
    OpStatus,
    RequestKey,
    SamplingParams,
    ScheduledRequest,
    Start,
    TensorRef,
)
from ..execution.bounded_storage import BoundedTensorStorage, TensorSchema
from ..foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class RequestRuntime:
    """Execution coordinates; projected scalars become concrete after output readiness."""

    logical_position: int = 0
    rng_counter: int = 0
    latent_product: TensorRef | None = None
    flow_step: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0

    def __post_init__(self) -> None:
        if self.logical_position < 0 or self.rng_counter < 0 or self.flow_step < 0:
            raise invalid_descriptor("request execution coordinates are negative")
        if not 0 <= self.kv_visible_len <= self.kv_computed_len:
            raise invalid_descriptor("request KV extents are not contained")


@dataclass(frozen=True, slots=True)
class SpeculativeSelection:
    """Acceptance geometry required to reconcile one verification result."""

    draft_tokens: tuple[int, ...]
    terminal_prefix: int | None
    base_logical_position: int
    base_rng_counter: int
    base_kv_visible: int
    initialized_kv: int


@dataclass(slots=True)
class ExecutionProgress:
    """One current or in-flight result, retained directly by its dependent draft.

    The predecessor is released when acceptance is known. This is not a history
    index: completed results without a current-state or execution consumer have
    no owner and are reclaimed, independently of Runtime output observation.
    """

    op_id: ComputationId
    runtime: RequestRuntime
    prompt_logits_ready: bool = False
    completion: CompletionState | None = None
    predecessor: ExecutionProgress | None = None
    speculative: SpeculativeSelection | None = None
    resolved: bool = False

    def accepted_runtime(self) -> RequestRuntime:
        """Resolve a query-ready input at the output boundary, never during scheduling."""

        if not self.resolved and self.completion is not None:
            self.resolve(self.completion.finalize())
        return self.runtime

    def resolve(self, record: ModelOutput) -> None:
        """Apply numerical acceptance once and release the predecessor dependency."""

        if self.resolved:
            return
        if record.status in (OpStatus.PREDICATED, OpStatus.ERROR):
            if self.predecessor is not None:
                self.runtime = self.predecessor.accepted_runtime()
                self.prompt_logits_ready = self.predecessor.prompt_logits_ready
        else:
            runtime = self.runtime
            rng_counter = int(runtime.rng_counter)
            selection = self.speculative
            if selection is not None:
                accepted = len(record.committed_tokens)
                if accepted > len(selection.draft_tokens) + 1:
                    raise RuntimeError("speculative completion acceptance is inconsistent")
                if (
                    record.kv_computed_len != selection.initialized_kv
                    or record.kv_visible_len != selection.base_kv_visible + accepted
                    or record.kv_visible_len > record.kv_computed_len
                    or record.position != selection.base_logical_position + accepted
                ):
                    raise RuntimeError("speculative acceptance exceeds initialized KV state")
                rng_counter = selection.base_rng_counter + accepted
            self.runtime = RequestRuntime(
                logical_position=int(record.position),
                rng_counter=rng_counter,
                latent_product=runtime.latent_product,
                flow_step=int(runtime.flow_step),
                kv_visible_len=int(record.kv_visible_len),
                kv_computed_len=int(record.kv_computed_len),
            )
        self.resolved = True
        self.predecessor = None
        self.speculative = None
        self.completion = None


@dataclass(slots=True)
class Request:
    """Current execution state and live operations for one admitted request epoch."""

    request_key: RequestKey
    request_pool_idx: int
    admission: NewRequest
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    current: ExecutionProgress
    pending_operations: dict[ComputationId, ExecutionProgress] = field(default_factory=dict)
    closed: bool = False
    retired: bool = False
    media_video_units: int = 0
    media_audio_written: bool = False
    media_finalized: bool = False
    last_op_id: ComputationId | None = None
    last_run_id: int | None = None

    @property
    def request_id(self) -> int:
        return self.request_key.request_id

    @property
    def request_epoch(self) -> int:
        return self.request_key.request_epoch

    @property
    def logical_position(self) -> int:
        return self.current.runtime.logical_position

    @property
    def rng_counter(self) -> int:
        return self.current.runtime.rng_counter

    @property
    def latent_product(self) -> TensorRef | None:
        return self.current.runtime.latent_product

    @property
    def flow_step(self) -> int:
        return self.current.runtime.flow_step

    @property
    def prompt_logits_ready(self) -> bool:
        return self.current.prompt_logits_ready


class RequestDraft:
    """Prepared scalar changes holding the actual predecessor result they consume."""

    __slots__ = (
        "request",
        "predecessor",
        "logical_position",
        "rng_counter",
        "latent_product",
        "flow_step",
        "prompt_logits_ready",
    )

    def __init__(self, request: Request, *, consumes_state: bool = True) -> None:
        self.request = request
        self.predecessor = (
            request.current
            if consumes_state
            else ExecutionProgress(ComputationId(0, 0), RequestRuntime(), resolved=True)
        )
        self.install_runtime(self.predecessor.runtime)
        self.prompt_logits_ready = self.predecessor.prompt_logits_ready

    def install_runtime(self, runtime: RequestRuntime) -> None:
        if runtime.latent_product is not None and (
            runtime.latent_product.request_key != self.request.request_key
        ):
            raise invalid_descriptor("request latent belongs to another request")
        self.logical_position = int(runtime.logical_position)
        self.rng_counter = int(runtime.rng_counter)
        self.latent_product = runtime.latent_product
        self.flow_step = int(runtime.flow_step)


@dataclass(frozen=True, slots=True)
class _ExecutionAssignment:
    operation: ScheduledRequest
    candidate: RequestDraft
    progress: ExecutionProgress


@dataclass(slots=True)
class RequestPublication:
    """Atomic lane publication; reservation exposes device dependencies to successors."""

    pool: RequestPool
    run_id: int
    assignments: tuple[_ExecutionAssignment, ...]
    _finished: bool = False
    _reserved: bool = False

    @property
    def successors_ready(self) -> bool:
        return self._reserved

    def reserve(self) -> None:
        self.pool.reserve(self)

    def finish(self, completions: tuple[ModelOutput, ...]) -> None:
        if not self._finished:
            self.pool.publish(self, completions)

    def cancel(self) -> None:
        if not self._finished:
            self.pool.cancel(self)


class RequestPool:
    """Own stable slots; run admission bounds the records for unfinished operations."""

    def __init__(
        self,
        max_request_pool_size: int,
        *,
        tensor_schema: Mapping[str, TensorSchema] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        size = int(max_request_pool_size)
        if size < 1:
            raise ValueError("request-pool capacity must be positive")
        self.max_request_pool_size = size
        self._rows: list[Request | None] = [None] * (size + 1)
        self._slots_by_request: dict[int, int] = {}
        self.tensor_slots = (
            tuple(BoundedTensorStorage.allocate(tensor_schema, device) for _ in range(size))
            if tensor_schema
            else ()
        )

    def tensors(self, request_pool_idx: int) -> BoundedTensorStorage:
        """Borrow storage while holding the execution lease through device completion."""

        slot = self._validate_slot(request_pool_idx)
        if not self.tensor_slots:
            raise invalid_descriptor("request has no declared persistent tensor storage")
        return self.tensor_slots[slot - 1]

    def get(self, request_id: int) -> Request:
        row = self.peek(request_id)
        if row is None:
            raise invalid_descriptor(f"unknown request {request_id}")
        return row

    def peek(self, request_id: int) -> Request | None:
        slot = self._slots_by_request.get(int(request_id))
        return None if slot is None else self._rows[slot]

    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._slots_by_request))

    def __contains__(self, request_id: object) -> bool:
        return isinstance(request_id, int) and request_id in self._slots_by_request

    def stage_lane(
        self, operations: Sequence[ScheduledRequest], request_pool_indices: Sequence[int]
    ) -> tuple[RequestDraft, ...]:
        if len(operations) != len(request_pool_indices):
            raise invalid_descriptor("request-pool indices are not aligned with operations")
        if len({operation.request_key.request_id for operation in operations}) != len(operations):
            raise invalid_descriptor("a lane repeats a request")
        slots = tuple(self._validate_slot(value) for value in request_pool_indices)
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("a lane repeats a request-pool index")
        candidates = []
        for operation, slot in zip(operations, slots, strict=True):
            row = self.get(operation.request_key.request_id)
            if row.request_key != operation.request_key:
                raise invalid_descriptor(f"operation {operation.op_id} has a stale request key")
            if self._rows[slot] is not row or row.request_pool_idx != slot:
                raise invalid_descriptor(f"operation {operation.op_id} has a stale request slot")
            if row.closed:
                raise invalid_descriptor(f"operation {operation.op_id} targets a closed request")
            if operation.op_id in row.pending_operations:
                raise invalid_descriptor("request operation is already executing")
            candidates.append(RequestDraft(row, consumes_state=operation.predecessor is not None))
        return tuple(candidates)

    def prepare_publication(
        self,
        *,
        run_id: int,
        operations: Sequence[ScheduledRequest],
        candidates: Sequence[RequestDraft],
        runtimes: Mapping[int, RequestRuntime],
        completions: Mapping[int, CompletionState],
        speculative: Mapping[int, SpeculativeSelection],
    ) -> RequestPublication:
        if len(operations) != len(candidates):
            raise RuntimeError("request publication columns are not aligned")
        assignments = []
        for operation, draft in zip(operations, candidates, strict=True):
            row = draft.request
            if self.peek(row.request_id) is not row or row.request_key != operation.request_key:
                raise RuntimeError("request publication lost its admitted slot")
            request_id = row.request_id
            progress = ExecutionProgress(
                op_id=operation.op_id,
                runtime=runtimes[request_id],
                prompt_logits_ready=draft.prompt_logits_ready,
                completion=completions[request_id],
                predecessor=draft.predecessor if operation.predecessor is not None else None,
                speculative=speculative.get(request_id),
            )
            assignments.append(_ExecutionAssignment(operation, draft, progress))
        return RequestPublication(self, int(run_id), tuple(assignments))

    def reserve(self, publication: RequestPublication) -> None:
        if publication.pool is not self:
            raise RuntimeError("request publication belongs to another pool")
        if publication._reserved:
            return
        # Verification retains its existing acceptance boundary. Ordinary AR
        # exposes device tokens before host observation, without an acknowledgement.
        if any(item.progress.speculative is not None for item in publication.assignments):
            return
        for item in publication.assignments:
            row = item.candidate.request
            if self.peek(row.request_id) is not row:
                raise RuntimeError("request reservation lost its stable slot")
            row.pending_operations[item.operation.op_id] = item.progress
            if item.operation.predecessor is not None:
                row.current = item.progress
        publication._reserved = True

    def publish(
        self, publication: RequestPublication, completions: tuple[ModelOutput, ...]
    ) -> None:
        if publication.pool is not self:
            raise RuntimeError("request publication belongs to another pool")
        records = {(record.request_key, record.op_id): record for record in completions}
        if len(records) != len(completions):
            raise RuntimeError("lane completion repeats an operation identity")
        for item in publication.assignments:
            record = records.get((item.operation.request_key, item.operation.op_id))
            if record is None:
                raise RuntimeError("request publication lost its operation completion")
            item.progress.resolve(record)
        for item in publication.assignments:
            row = item.candidate.request
            row.pending_operations.pop(item.operation.op_id, None)
            if row.closed or self.peek(row.request_id) is not row:
                continue
            if not publication._reserved and item.operation.predecessor is not None:
                row.current = item.progress
            row.last_op_id = item.operation.op_id
            row.last_run_id = publication.run_id
        publication.assignments = ()
        publication._finished = True

    def cancel(self, publication: RequestPublication) -> None:
        if publication.pool is not self:
            raise RuntimeError("request publication belongs to another pool")
        for item in publication.assignments:
            row = item.candidate.request
            row.pending_operations.pop(item.operation.op_id, None)
            # A failed execution closes its request; it cannot resume from a
            # speculative projection whose numerical acceptance is unknown.
            row.closed = True
        publication.assignments = ()
        publication._finished = True

    def apply_commands(self, commands: Sequence[BatchCommand]) -> tuple[int, ...]:
        started = []
        for command in commands:
            if isinstance(command, Start):
                slot = self._apply_start(command.request)
                if slot is not None:
                    started.append(slot)
            elif isinstance(command, Finish):
                row = self.peek(command.request_key.request_id)
                if row is not None and row.request_key == command.request_key:
                    row.closed = True
        return tuple(started)

    def drop(self, request_id: int) -> None:
        """Remove a row only after its execution leases have been released."""

        slot = self._slots_by_request.pop(int(request_id), None)
        if slot is not None:
            self._rows[slot] = None

    def retirement_ready(self, request_key: RequestKey) -> bool:
        row = self.peek(request_key.request_id)
        if row is None or row.request_key != request_key:
            return True
        return all(
            progress.completion is None or progress.completion.ready()
            for progress in row.pending_operations.values()
        )

    def retire(self, request_id: int) -> None:
        row = self.get(request_id)
        if row.retired:
            return
        if not row.closed or not self.retirement_ready(row.request_key):
            raise RuntimeError("request retirement requires closed, completed execution")
        row.pending_operations.clear()
        row.retired = True

    def _validate_slot(self, request_pool_idx: int) -> int:
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
        self._rows[slot] = Request(
            request_key=admission.request_key,
            request_pool_idx=slot,
            admission=admission,
            sampling=None if admission.ar is None else admission.ar.sampling,
            image=None if admission.umm is None else admission.umm.image,
            negative_token_ids=() if admission.ar is None else admission.ar.negative_token_ids,
            finish_token_ids=() if admission.ar is None else admission.ar.finish_token_ids,
            current=ExecutionProgress(
                op_id=ComputationId(0, 0),
                runtime=RequestRuntime(
                    logical_position=prefix, kv_visible_len=prefix, kv_computed_len=prefix
                ),
                resolved=True,
            ),
        )
        self._slots_by_request[admission.request_key.request_id] = slot
        return slot
