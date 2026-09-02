"""Stable request slots, unresolved producers, and atomic request commits."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from ..execution.batch import (
    Close,
    Commit,
    Control,
    CompletionState,
    DevicePoint,
    FixedPoint,
    ImageParams,
    ModelOutput,
    NewRequest,
    Operation,
    OpStatus,
    ProductRef,
    RequestKey,
    SamplingParams,
    VersionRef,
)
from ..foundation.errors import invalid_descriptor

MAX_REQUEST_HISTORY_POINTS = 262_144


@dataclass(frozen=True, slots=True)
class RequestRuntime:
    """Host-visible logical coordinates associated with one resolved point."""

    logical_position: int
    rng_counter: int
    latent_product: ProductRef | None
    flow_step: int
    kv_visible_len: int
    kv_computed_len: int

    def __post_init__(self) -> None:
        if self.logical_position < 0 or self.rng_counter < 0 or self.flow_step < 0:
            raise invalid_descriptor("resolved request coordinates are negative")
        if self.latent_product is not None and (
            self.latent_product.kind.value != "latent"
            or self.latent_product.storage_class.value != "latent_arena"
        ):
            raise invalid_descriptor("resolved request latent identity is invalid")
        if not isinstance(self.kv_visible_len, int):
            if (
                not callable(getattr(self.kv_visible_len, "ready", None))
                or self.kv_computed_len < 0
            ):
                raise invalid_descriptor("resolved request pending KV extent is invalid")
            return
        if not (0 <= self.kv_visible_len <= self.kv_computed_len):
            raise invalid_descriptor("resolved request KV extents are not contained")


@dataclass(frozen=True, slots=True)
class _RequestCommit:
    operation: Operation
    candidate: RequestDraft
    base: Request | None
    selected: VersionRef
    runtime: RequestRuntime
    completion: CompletionState | None
    speculative: SpeculativeCommit | None


@dataclass(frozen=True, slots=True)
class _ResolvedCommit:
    assignment: _RequestCommit
    record: ModelOutput
    selected: VersionRef
    runtime: RequestRuntime
    prefixes: tuple[tuple[VersionRef, RequestRuntime], ...]


@dataclass(frozen=True, slots=True)
class RequestPublication:
    table_token: object
    step_id: int
    assignments: tuple[_RequestCommit, ...]
    _finished: bool = field(default=False, init=False, repr=False, compare=False)
    _reserved: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def successors_ready(self) -> bool:
        return self._reserved

    def reserve(self) -> None:
        table = self.table_token
        if not isinstance(table, RequestPool):
            raise RuntimeError("request reservation lost its request-pool owner")
        table.reserve(self)

    def finish(self, completions: tuple[ModelOutput, ...]) -> None:
        if self._finished:
            return
        table = self.table_token
        if not isinstance(table, RequestPool):
            raise RuntimeError("request publication lost its request-pool owner")
        table.publish(self, completions)

    def cancel(self) -> None:
        if self._finished:
            return
        table = self.table_token
        if not isinstance(table, RequestPool):
            raise RuntimeError("request publication lost its request-pool owner")
        table.cancel(self)


@dataclass(frozen=True, slots=True)
class SpeculativeCommit:
    """Compact verify-prefix data applied only after output readiness."""

    draft_tokens: tuple[int, ...]
    terminal_prefix: int | None
    base_logical_position: int
    base_rng_counter: int
    base_kv_visible: int
    initialized_kv: int


@dataclass(slots=True)
class Request:
    """Bounded host state for one scheduler-assigned request slot."""

    request_key: RequestKey
    request_pool_idx: int
    admission: NewRequest
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    version: int = 0
    resolved_op_id: int = 0
    committed_point: int = 0
    committed_op_id: int = 0
    public_event_limit: int = 0
    applied_control_seq: int = 0
    control_history: dict[tuple[int, str], Commit | Close] = field(default_factory=dict)
    resolved_versions: dict[tuple[int, int], VersionRef] = field(default_factory=dict)
    resolved_runtime: dict[tuple[int, int], RequestRuntime] = field(default_factory=dict)
    resolved_operations: dict[int, VersionRef] = field(default_factory=dict)
    projected_runtime: dict[int, RequestRuntime] = field(default_factory=dict)
    pending_operations: dict[int, CompletionState] = field(default_factory=dict)
    unresolved_operations: dict[int, _RequestCommit] = field(default_factory=dict)
    declared_parents: dict[int, VersionRef] = field(default_factory=dict)
    terminal_cutoff: VersionRef | None = None
    latent_product: ProductRef | None = None
    prompt_logits_ready: bool = False
    logical_position: int = 0
    flow_step: int = 0
    rng_counter: int = 0
    last_op_id: int | None = None
    last_step_id: int | None = None

    def __post_init__(self) -> None:
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request row has an invalid scheduler slot")

    @property
    def session_id(self) -> int:
        return self.request_key.session_id

    @property
    def epoch(self) -> int:
        return self.request_key.epoch

    def committed_version(self) -> VersionRef:
        return VersionRef(
            request_key=self.request_key,
            producer_op_id=self.committed_op_id,
            point=FixedPoint(self.committed_point),
        )

    def resolved_version(self) -> VersionRef:
        return VersionRef(
            request_key=self.request_key,
            producer_op_id=self.resolved_op_id,
            point=FixedPoint(self.version),
        )

    def install_runtime(self, runtime: RequestRuntime) -> None:
        if (
            runtime.latent_product is not None
            and runtime.latent_product.request_key != self.request_key
        ):
            raise invalid_descriptor("resolved request latent belongs to another request")
        self.logical_position = runtime.logical_position
        self.rng_counter = runtime.rng_counter
        self.latent_product = runtime.latent_product
        self.flow_step = runtime.flow_step

    @staticmethod
    def point_key(version: VersionRef) -> tuple[int, int]:
        point = version.point
        if not isinstance(point, FixedPoint):
            raise invalid_descriptor("request runtime requires a fixed version")
        return int(version.producer_op_id), int(point.point_index)

    def runtime_for(self, version: VersionRef) -> RequestRuntime | None:
        runtime = self.resolved_runtime.get(self.point_key(version))
        if runtime is not None:
            return runtime
        pending = self.unresolved_operations.get(int(version.producer_op_id))
        return pending.runtime if pending is not None and pending.selected == version else None

    def selected_for_operation(self, op_id: int) -> VersionRef | None:
        selected = self.resolved_operations.get(int(op_id))
        if selected is not None:
            return selected
        pending = self.unresolved_operations.get(int(op_id))
        return None if pending is None else pending.selected

    def resolve_version(self, version: VersionRef) -> VersionRef | None:
        if isinstance(version.point, FixedPoint):
            return version
        return self.selected_for_operation(version.producer_op_id)

    def execution_runtime_for_operation(
        self,
        op_id: int,
        point_index: int,
    ) -> RequestRuntime | None:
        runtime = self.resolved_runtime.get((int(op_id), int(point_index)))
        if runtime is not None:
            return runtime
        projected = self.projected_runtime.get(int(op_id))
        if projected is not None and int(point_index) == 1:
            return projected
        pending = self.unresolved_operations.get(int(op_id))
        if pending is None:
            return None
        point = pending.selected.point
        return (
            pending.runtime
            if isinstance(point, FixedPoint) and int(point.point_index) == int(point_index)
            else None
        )

    def semantic_parent_for_operation(self, op_id: int) -> VersionRef | None:
        parent = self.declared_parents.get(int(op_id))
        if parent is None:
            pending = self.unresolved_operations.get(int(op_id))
            parent = None if pending is None else pending.operation.parent
        return None if parent is None else self.resolve_version(parent)

    def finish(
        self,
        draft: RequestDraft,
        *,
        operation: Operation,
        selected: VersionRef,
        runtime: RequestRuntime,
        completion: CompletionState | None,
        step_id: int,
    ) -> None:
        """Commit one prepared output row to this stable request slot."""

        if draft.request is not self or operation.request_key != self.request_key:
            raise RuntimeError("request finish crossed stable slot ownership")
        point = selected.point
        if not isinstance(point, FixedPoint):
            raise RuntimeError("request finish requires a fixed selected point")
        self.logical_position = int(runtime.logical_position)
        self.rng_counter = int(runtime.rng_counter)
        self.latent_product = runtime.latent_product
        self.flow_step = int(runtime.flow_step)
        self.prompt_logits_ready = bool(draft.prompt_logits_ready)
        self.version = int(point.point_index)
        self.resolved_op_id = int(selected.producer_op_id)
        if operation.advances_state:
            key = self.point_key(selected)
            self.resolved_versions[key] = selected
            self.resolved_runtime[key] = runtime
        else:
            self.resolved_runtime[self.point_key(selected)] = runtime
            self.install_runtime(runtime)
        self.resolved_operations[int(operation.op_id)] = selected
        if completion is not None:
            self.pending_operations[int(operation.op_id)] = completion
        self.declared_parents[int(operation.op_id)] = operation.parent
        self.last_op_id = int(operation.op_id)
        self.last_step_id = int(step_id)
        self.unresolved_operations.pop(int(operation.op_id), None)


class RequestDraft:
    """Prepared scalar changes for one stable request, with shared immutable history."""

    __slots__ = (
        "request",
        "logical_position",
        "rng_counter",
        "latent_product",
        "flow_step",
        "prompt_logits_ready",
    )

    def __init__(self, request: Request, source: RequestDraft | None = None) -> None:
        self.request = request
        state = request if source is None else source
        self.logical_position = int(state.logical_position)
        self.rng_counter = int(state.rng_counter)
        self.latent_product = state.latent_product
        self.flow_step = int(state.flow_step)
        self.prompt_logits_ready = bool(state.prompt_logits_ready)

    def __getattr__(self, name: str) -> object:
        return getattr(self.request, name)

    def install_runtime(self, runtime: RequestRuntime) -> None:
        if runtime.latent_product is not None and (
            runtime.latent_product.request_key != self.request.request_key
        ):
            raise invalid_descriptor("resolved request latent belongs to another request")
        self.logical_position = int(runtime.logical_position)
        self.rng_counter = int(runtime.rng_counter)
        self.latent_product = runtime.latent_product
        self.flow_step = int(runtime.flow_step)


class RequestPool:
    """Own the fixed host row indexed by every live scheduler request slot."""

    def __init__(
        self,
        max_request_pool_size: int,
        *,
        history_capacity: int = MAX_REQUEST_HISTORY_POINTS,
    ) -> None:
        size = int(max_request_pool_size)
        history = int(history_capacity)
        if size < 1 or history < 1:
            raise ValueError("request-table capacities must be positive")
        self.max_request_pool_size = size
        self.history_capacity = history
        self._rows: list[Request | None] = [None] * (size + 1)
        self._slots_by_session: dict[int, int] = {}

    def get(self, session_id: int) -> Request:
        row = self.peek(session_id)
        if row is None:
            raise invalid_descriptor(f"unknown session {session_id}")
        return row

    def peek(self, session_id: int) -> Request | None:
        slot = self._slots_by_session.get(int(session_id))
        return None if slot is None else self._rows[slot]

    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._slots_by_session))

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, int) and session_id in self._slots_by_session

    def stage_partition(
        self,
        operations: Sequence[Operation],
        admissions: Sequence[NewRequest],
        request_pool_indices: Sequence[int],
    ) -> tuple[tuple[RequestDraft, ...], tuple[Request | None, ...]]:
        """Build scalar drafts without changing stable request slots."""

        if len(operations) != len(request_pool_indices):
            raise invalid_descriptor("request-pool indices are not aligned with operations")
        if len({operation.request_key.session_id for operation in operations}) != len(operations):
            raise invalid_descriptor("a partition repeats a request")
        slots = tuple(self._validate_slot(value) for value in request_pool_indices)
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("a partition repeats a request-pool index")
        admitted = {value.request_key.session_id: value for value in admissions}
        if len(admitted) != len(admissions):
            raise invalid_descriptor("a partition repeats an admission")
        operation_sessions = {operation.request_key.session_id for operation in operations}
        if set(admitted) - operation_sessions:
            raise invalid_descriptor("a partition admission has no operation")

        candidates: list[RequestDraft] = []
        bases: list[Request | None] = []
        for operation, slot in zip(operations, slots, strict=True):
            session_id = int(operation.request_key.session_id)
            admission = admitted.get(session_id)
            base = self.peek(session_id)
            occupant = self._rows[slot]
            if base is None:
                if admission is None:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} references an unknown session"
                    )
                if occupant is not None:
                    raise invalid_descriptor(
                        f"request-pool index {slot} is occupied by session {occupant.session_id}"
                    )
                request = self._admission_row(admission)
            else:
                if occupant is not base or int(base.request_pool_idx) != slot:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} names request-pool index {slot}; "
                        f"session index is {base.request_pool_idx}"
                    )
                if admission is not None and admission != base.admission:
                    raise invalid_descriptor(
                        f"session {session_id} admission conflicts with committed state"
                    )
                request = base
            source: RequestDraft | None = None
            if base is not None:
                pending_parent = base.unresolved_operations.get(
                    int(operation.parent.producer_op_id)
                )
                if pending_parent is not None:
                    selected_parent = base.resolve_version(operation.parent)
                    if selected_parent == pending_parent.selected:
                        source = pending_parent.candidate
            candidate = RequestDraft(request, source)
            self._validate_operation(candidate, operation, slot)
            candidates.append(candidate)
            bases.append(base)
        return tuple(candidates), tuple(bases)

    def prepare_publication(
        self,
        *,
        step_id: int,
        operations: Sequence[Operation],
        candidates: Sequence[RequestDraft],
        bases: Sequence[Request | None],
        selected_versions: Mapping[int, VersionRef],
        runtimes: Mapping[int, RequestRuntime],
        completions: Mapping[int, CompletionState],
        speculative: Mapping[int, SpeculativeCommit],
    ) -> RequestPublication:
        """Validate and freeze one stable-slot request commit."""

        if len(operations) != len(candidates) or len(operations) != len(bases):
            raise RuntimeError("request-row publication columns are not aligned")
        assignments: list[_RequestCommit] = []
        for operation, row, base in zip(operations, candidates, bases, strict=True):
            session_id = int(operation.request_key.session_id)
            slot = int(row.request_pool_idx)
            current = self._rows[slot]
            if current is not base or (base is not None and self.peek(session_id) is not base):
                raise RuntimeError(f"request {session_id} changed before row publication")
            selected = selected_versions.get(session_id)
            runtime = runtimes.get(session_id)
            if selected is None or runtime is None:
                raise RuntimeError("request-row publication is missing its resolved outcome")
            completion = completions.get(session_id)
            if operation.advances_state and completion is None:
                raise RuntimeError("request-row publication is missing its pending completion")
            point = selected.point
            if not isinstance(point, FixedPoint):
                raise RuntimeError("published request version must be fixed")
            additions = 1 if operation.advances_state and isinstance(point.point_index, int) else 0
            if (
                len(row.resolved_versions)
                + len(row.unresolved_operations)
                + additions
                > self.history_capacity
            ):
                raise invalid_descriptor("request history capacity is exhausted")
            parent = operation.parent
            resolved_parent = row.resolve_version(parent)
            parent_matches = (
                row.committed_version() == parent
                if parent.is_fixed()
                else resolved_parent is not None
            )
            if row.request_key != operation.request_key or not parent_matches:
                raise RuntimeError(f"request {session_id} candidate has a stale parent")
            if not operation.advances_state and selected != resolved_parent:
                raise RuntimeError("non-state request publication changed its resolved parent")
            assignments.append(
                _RequestCommit(
                    operation=operation,
                    candidate=row,
                    base=base,
                    selected=selected,
                    runtime=runtime,
                    completion=completion,
                    speculative=speculative.get(session_id),
                )
            )
        return RequestPublication(
            table_token=self,
            step_id=int(step_id),
            assignments=tuple(assignments),
        )

    def reserve(self, publication: RequestPublication) -> None:
        """Install bounded unresolved lineage without changing committed request state."""

        if publication.table_token is not self:
            raise RuntimeError("request reservation belongs to another request pool")
        if publication._reserved:
            return
        if any(assignment.speculative is not None for assignment in publication.assignments):
            return
        for assignment in publication.assignments:
            row = assignment.candidate.request
            operation_id = int(assignment.operation.op_id)
            existing = row.unresolved_operations.get(operation_id)
            if existing is not None and existing is not assignment:
                raise invalid_descriptor("request operation already has unresolved state")
            current = self._rows[int(row.request_pool_idx)]
            if assignment.base is None:
                if current is not None and current is not row:
                    raise RuntimeError("request reservation lost its admission slot")
            elif current is not assignment.base:
                raise RuntimeError("request reservation lost its stable request slot")
        for assignment in publication.assignments:
            row = assignment.candidate.request
            slot = int(row.request_pool_idx)
            if assignment.base is None and self._rows[slot] is None:
                self._rows[slot] = row
                self._slots_by_session[int(row.session_id)] = slot
            operation_id = int(assignment.operation.op_id)
            row.unresolved_operations[operation_id] = assignment
            row.projected_runtime[operation_id] = assignment.runtime
            if assignment.completion is not None:
                row.pending_operations[operation_id] = assignment.completion
        object.__setattr__(publication, "_reserved", True)

    def publish(
        self,
        publication: RequestPublication,
        completions: tuple[ModelOutput, ...],
    ) -> None:
        """Finish every prepared draft against its stable request slot."""

        if publication.table_token is not self:
            raise RuntimeError("request-row publication belongs to another table")
        if publication._finished:
            return
        by_operation = {
            (record.request_key, int(record.op_id)): record for record in completions
        }
        if len(by_operation) != len(completions):
            raise RuntimeError("partition completion repeats an operation identity")
        resolved: list[_ResolvedCommit] = []
        for assignment in publication.assignments:
            record = by_operation.get(
                (assignment.operation.request_key, int(assignment.operation.op_id))
            )
            if record is None:
                raise RuntimeError("request-row publication lost its concrete completion")
            if record.status is OpStatus.ERROR:
                self._cancel_assignment(assignment)
                continue
            draft = assignment.candidate
            row = draft.request
            base = assignment.base
            slot = int(row.request_pool_idx)
            expected = row if publication._reserved else base
            if self._rows[slot] is not expected:
                raise RuntimeError(f"request {row.session_id} changed before row assignment")
            selected = assignment.selected
            unresolved_runtime = assignment.runtime
            if assignment.speculative is not None:
                selected = VersionRef(
                    request_key=assignment.operation.request_key,
                    producer_op_id=int(assignment.operation.op_id),
                    point=FixedPoint(int(record.selected_point)),
                )
                unresolved_runtime = RequestRuntime(
                    logical_position=int(record.logical_lengths.token_len),
                    rng_counter=(
                        int(assignment.speculative.base_rng_counter)
                        + int(record.selected_point)
                    ),
                    latent_product=draft.latent_product,
                    flow_step=int(draft.flow_step),
                    kv_visible_len=int(record.logical_lengths.kv_visible_len),
                    kv_computed_len=int(record.logical_lengths.kv_computed_len),
                )
            if record.status is OpStatus.PREDICATED:
                selected_parent = row.resolve_version(assignment.operation.parent)
                parent_runtime = (
                    None if selected_parent is None else row.runtime_for(selected_parent)
                )
                if selected_parent is None or parent_runtime is None:
                    raise RuntimeError("predicated request publication lost its parent state")
                selected = selected_parent
                unresolved_runtime = parent_runtime
            point = selected.point
            if not isinstance(point, FixedPoint):
                raise RuntimeError("request publication selected a non-fixed version")
            selected = VersionRef(
                request_key=selected.request_key,
                producer_op_id=int(selected.producer_op_id),
                point=FixedPoint(int(point.point_index)),
            )
            resolved_runtime = RequestRuntime(
                logical_position=int(unresolved_runtime.logical_position),
                rng_counter=int(unresolved_runtime.rng_counter),
                latent_product=unresolved_runtime.latent_product,
                flow_step=int(unresolved_runtime.flow_step),
                kv_visible_len=int(unresolved_runtime.kv_visible_len),
                kv_computed_len=int(unresolved_runtime.kv_computed_len),
            )
            prefixes = self._speculative_prefixes(row, assignment, record)
            final_keys = set(row.resolved_versions)
            if assignment.operation.advances_state:
                final_keys.add(row.point_key(selected))
            final_keys.update(row.point_key(version) for version, _runtime in prefixes)
            if len(final_keys) > self.history_capacity:
                raise invalid_descriptor("request history capacity is exhausted")
            resolved.append(
                _ResolvedCommit(
                    assignment=assignment,
                    record=record,
                    selected=selected,
                    runtime=resolved_runtime,
                    prefixes=prefixes,
                )
            )

        for commit in resolved:
            assignment = commit.assignment
            draft = assignment.candidate
            row = draft.request
            if assignment.base is None and not publication._reserved:
                slot = int(row.request_pool_idx)
                self._rows[slot] = row
                self._slots_by_session[int(row.session_id)] = slot
            row.finish(
                draft,
                operation=assignment.operation,
                selected=commit.selected,
                runtime=commit.runtime,
                completion=assignment.completion,
                step_id=publication.step_id,
            )
            if commit.prefixes:
                self._install_prefixes(
                    row,
                    int(assignment.operation.op_id),
                    commit.prefixes,
                )
        object.__setattr__(publication, "_finished", True)

    def cancel(self, publication: RequestPublication) -> None:
        if publication.table_token is not self:
            raise RuntimeError("request cancellation belongs to another request pool")
        if publication._finished:
            return
        for assignment in publication.assignments:
            self._cancel_assignment(assignment)
        object.__setattr__(publication, "_finished", True)

    def _cancel_assignment(self, assignment: _RequestCommit) -> None:
        row = assignment.candidate.request
        operation_id = int(assignment.operation.op_id)
        if row.unresolved_operations.get(operation_id) is assignment:
            row.unresolved_operations.pop(operation_id)
            row.projected_runtime.pop(operation_id, None)
            row.pending_operations.pop(operation_id, None)
        if (
            assignment.base is None
            and not row.unresolved_operations
            and row.resolved_op_id == 0
            and self._rows[int(row.request_pool_idx)] is row
        ):
            self._rows[int(row.request_pool_idx)] = None
            self._slots_by_session.pop(int(row.session_id), None)

    def _speculative_prefixes(
        self,
        row: Request,
        assignment: _RequestCommit,
        record: ModelOutput,
    ) -> tuple[tuple[VersionRef, RequestRuntime], ...]:
        plan = assignment.speculative
        if plan is None:
            return ()
        tokens = tuple(int(value) for value in record.committed_tokens)
        selected_point = len(tokens)
        if selected_point != int(record.selected_point) or selected_point > len(plan.draft_tokens) + 1:
            raise RuntimeError("speculative completion selection is inconsistent")
        selected_kv = int(plan.base_kv_visible) + selected_point
        if (
            int(record.logical_lengths.kv_computed_len) != int(plan.initialized_kv)
            or selected_kv > int(record.logical_lengths.kv_computed_len)
        ):
            raise RuntimeError("speculative KV selection is outside initialized state")
        return tuple(
            (
                VersionRef(
                    request_key=assignment.operation.request_key,
                    producer_op_id=int(assignment.operation.op_id),
                    point=FixedPoint(point_index),
                ),
                RequestRuntime(
                    logical_position=int(plan.base_logical_position) + point_index,
                    rng_counter=int(plan.base_rng_counter) + point_index,
                    latent_product=row.latent_product,
                    flow_step=row.flow_step,
                    kv_visible_len=int(plan.base_kv_visible) + point_index,
                    kv_computed_len=int(record.logical_lengths.kv_computed_len),
                ),
            )
            for point_index in range(1, selected_point + 1)
        )

    def apply_controls(self, controls: Sequence[Control]) -> None:
        for control in controls:
            if isinstance(control, Commit):
                self._apply_commit(control)
            elif isinstance(control, Close):
                self._apply_close(control)

    def resolve_predicated(
        self,
        session_id: int,
        op_id: int,
        parent: VersionRef,
    ) -> tuple[VersionRef, RequestRuntime]:
        row = self.get(session_id)
        selected = row.resolve_version(parent)
        runtime = None if selected is None else row.runtime_for(selected)
        if selected is None or runtime is None:
            raise invalid_descriptor(f"predicated operation {op_id} lost its resolved parent")
        point = selected.point
        if not isinstance(point, FixedPoint):
            raise invalid_descriptor(f"predicated operation {op_id} resolved to a device point")
        selected = VersionRef(
            request_key=selected.request_key,
            producer_op_id=selected.producer_op_id,
            point=FixedPoint(point.point_index),
        )
        return selected, runtime

    def finalize_prefixes(
        self,
        session_id: int,
        op_id: int,
        prefixes: Sequence[tuple[VersionRef, RequestRuntime]],
    ) -> tuple[VersionRef, RequestRuntime]:
        if not prefixes:
            raise invalid_descriptor("resolved token operation has no prefix states")
        row = self.get(session_id)
        return self._install_prefixes(row, op_id, prefixes)

    def _install_prefixes(
        self,
        row: Request,
        op_id: int,
        prefixes: Sequence[tuple[VersionRef, RequestRuntime]],
    ) -> tuple[VersionRef, RequestRuntime]:
        if not prefixes:
            raise invalid_descriptor("resolved token operation has no prefix states")
        new_keys = {
            row.point_key(version)
            for version, _runtime in prefixes
            if row.point_key(version) not in row.resolved_versions
        }
        if len(row.resolved_versions) + len(new_keys) > self.history_capacity:
            raise invalid_descriptor("request history capacity is exhausted")
        previous_point = 0
        for version, runtime in prefixes:
            if version.request_key != row.request_key or version.producer_op_id != int(op_id):
                raise invalid_descriptor("resolved token prefix has the wrong lineage")
            point = version.point
            if not isinstance(point, FixedPoint) or int(point.point_index) != previous_point + 1:
                raise invalid_descriptor("resolved token prefixes are not contiguous")
            key = row.point_key(version)
            row.resolved_versions[key] = version
            row.resolved_runtime[key] = runtime
            previous_point = int(point.point_index)
        selected, runtime = prefixes[-1]
        point = cast(FixedPoint, selected.point)
        row.resolved_operations[int(op_id)] = selected
        row.version = int(point.point_index)
        row.resolved_op_id = int(op_id)
        row.install_runtime(runtime)
        return selected, runtime

    def drop(self, session_id: int) -> None:
        selected = int(session_id)
        slot = self._slots_by_session.pop(selected, None)
        if slot is not None:
            self._rows[slot] = None

    def snapshot_committed(self, session_ids: set[int]) -> tuple[Request, ...]:
        snapshots: list[Request] = []
        for session_id in sorted(int(value) for value in session_ids):
            snapshot = copy.deepcopy(self.get(session_id))
            snapshot.version = snapshot.committed_point
            snapshot.resolved_op_id = snapshot.committed_op_id
            committed = snapshot.committed_version()
            runtime = snapshot.runtime_for(committed)
            if runtime is None:
                raise invalid_descriptor("committed request runtime state is missing")
            snapshot.install_runtime(runtime)
            key = snapshot.point_key(committed)
            snapshot.resolved_versions = {key: committed}
            snapshot.resolved_runtime = {key: runtime}
            snapshot.resolved_operations = {snapshot.committed_op_id: committed}
            snapshot.projected_runtime = {}
            snapshot.pending_operations = {}
            snapshot.unresolved_operations = {}
            snapshot.declared_parents = {snapshot.committed_op_id: committed}
            snapshots.append(snapshot)
        return tuple(snapshots)

    def restore_rows(
        self,
        rows: Sequence[Request],
        session_ids: set[int] | None = None,
    ) -> None:
        staged = {int(row.session_id): copy.deepcopy(row) for row in rows}
        if len(staged) != len(rows):
            raise invalid_descriptor("request snapshot repeats a session identity")
        selected = set(staged) if session_ids is None else {int(value) for value in session_ids}
        if not set(staged) <= selected:
            raise invalid_descriptor("request snapshot contains an undeclared session")

        retained_slots = {
            slot: row.session_id
            for slot, row in enumerate(self._rows)
            if row is not None and row.session_id not in selected
        }
        for session_id, row in staged.items():
            slot = self._validate_slot(row.request_pool_idx)
            occupant = retained_slots.get(slot)
            if occupant is not None:
                raise invalid_descriptor(
                    f"request snapshot slot {slot} is occupied by session {occupant}"
                )
            retained_slots[slot] = session_id
            if row.epoch < 0 or row.version < 0 or row.committed_point < 0:
                raise invalid_descriptor("request snapshot version is invalid")
            if row.admission.request_key != row.request_key:
                raise invalid_descriptor("request snapshot admission belongs to another request")
            if (
                row.resolved_versions.get(row.point_key(row.resolved_version()))
                != row.resolved_version()
                or row.runtime_for(row.resolved_version()) is None
                or row.resolved_versions.get(row.point_key(row.committed_version()))
                != row.committed_version()
                or row.runtime_for(row.committed_version()) is None
            ):
                raise invalid_descriptor("request snapshot lineage state is incomplete")

        for session_id in selected:
            self.drop(session_id)
        for session_id, row in staged.items():
            slot = int(row.request_pool_idx)
            self._rows[slot] = row
            self._slots_by_session[session_id] = slot

    def _validate_slot(self, request_pool_idx: int) -> int:
        slot = int(request_pool_idx)
        if slot < 1 or slot > self.max_request_pool_size:
            raise invalid_descriptor(
                f"request-pool index {slot} exceeds capacity {self.max_request_pool_size}"
            )
        return slot

    def _admission_row(self, admission: NewRequest) -> Request:
        slot = self._validate_slot(admission.request_pool_idx)
        prefix_len = 0 if admission.und is None else int(admission.und.initial_position)
        row = Request(
            request_key=admission.request_key,
            request_pool_idx=slot,
            admission=admission,
            sampling=None if admission.und is None else admission.und.sampling,
            image=None if admission.gen_admission is None else admission.gen_admission.image,
            negative_token_ids=() if admission.und is None else admission.und.negative_token_ids,
            finish_token_ids=() if admission.und is None else admission.und.finish_token_ids,
            logical_position=prefix_len,
        )
        root = row.committed_version()
        key = row.point_key(root)
        row.resolved_versions[key] = root
        row.resolved_operations[0] = root
        row.declared_parents[0] = root
        row.resolved_runtime[key] = RequestRuntime(
            logical_position=prefix_len,
            rng_counter=0,
            latent_product=None,
            flow_step=0,
            kv_visible_len=prefix_len,
            kv_computed_len=prefix_len,
        )
        return row

    @staticmethod
    def _validate_operation(row: Request, operation: Operation, slot: int) -> None:
        if row.request_key != operation.request_key:
            raise invalid_descriptor(f"operation {operation.op_id} has a stale request key")
        if int(row.request_pool_idx) != int(slot):
            raise invalid_descriptor(f"operation {operation.op_id} has a stale request slot")
        if row.terminal_cutoff is not None:
            raise invalid_descriptor(f"operation {operation.op_id} targets a closed request")
        if operation.control_seq != row.applied_control_seq:
            raise invalid_descriptor(
                f"operation {operation.op_id} requires control sequence "
                f"{operation.control_seq}; worker applied {row.applied_control_seq}"
            )
        parent = operation.parent
        if parent.is_fixed():
            if parent != row.committed_version():
                raise invalid_descriptor(
                    f"operation {operation.op_id} parent does not match committed state"
                )
        elif row.resolve_version(parent) is None:
            raise invalid_descriptor(
                f"operation {operation.op_id} names an unresolved device parent"
            )

    def _control_identity(
        self,
        row: Request,
        control: Commit | Close,
    ) -> tuple[bool, tuple[int, str]]:
        kind = "commit" if isinstance(control, Commit) else "close"
        identity = (int(control.control_seq), kind)
        existing = row.control_history.get(identity)
        if existing is not None:
            if existing != control:
                raise invalid_descriptor(
                    f"control identity {identity} conflicts with its committed content"
                )
            return True, identity
        if len(row.control_history) >= self.history_capacity:
            raise invalid_descriptor("request control history capacity is exhausted")
        if int(control.control_seq) != row.applied_control_seq + 1:
            raise invalid_descriptor(
                f"control sequence {control.control_seq} for request {row.session_id} "
                f"does not follow {row.applied_control_seq}"
            )
        return False, identity

    def _apply_commit(self, control: Commit) -> None:
        row = self.get(control.request_key.session_id)
        if control.request_key != row.request_key:
            raise invalid_descriptor("commit control has a stale request key")
        duplicate, identity = self._control_identity(row, control)
        if duplicate:
            return
        if row.terminal_cutoff is not None:
            raise invalid_descriptor("commit control targets a closed request")
        if control.expected_parent != row.committed_version():
            raise invalid_descriptor("commit control expected parent is not current")
        selected = control.selected
        if not selected.is_fixed():
            raise invalid_descriptor("commit control selected point is not fixed")
        if (
            row.resolved_versions.get(row.point_key(selected)) != selected
            and row.selected_for_operation(selected.producer_op_id) != selected
        ):
            raise invalid_descriptor("commit control selected point was not resolved")
        if row.semantic_parent_for_operation(selected.producer_op_id) != control.expected_parent:
            raise invalid_descriptor("commit control selected point has a different parent")
        if row.runtime_for(selected) is None:
            raise invalid_descriptor("commit control selected point lost its runtime")
        point = cast(FixedPoint, selected.point)
        if int(control.public_event_limit) < row.public_event_limit:
            raise invalid_descriptor("commit control regresses the public event limit")
        row.committed_point = int(point.point_index)
        row.committed_op_id = int(selected.producer_op_id)
        row.public_event_limit = int(control.public_event_limit)
        row.applied_control_seq = int(control.control_seq)
        row.control_history[identity] = control
        self._prune_committed(row)

    @staticmethod
    def _prune_committed(row: Request) -> None:
        frontier = int(row.committed_op_id)
        referenced = {
            int(parent.producer_op_id)
            for op_id, parent in row.declared_parents.items()
            if int(op_id) > frontier
        }
        retained_ops = referenced | {frontier} | {
            int(op_id) for op_id in row.resolved_operations if int(op_id) > frontier
        }
        row.resolved_operations = {
            op_id: version
            for op_id, version in row.resolved_operations.items()
            if int(op_id) in retained_ops
        }
        row.declared_parents = {
            op_id: parent
            for op_id, parent in row.declared_parents.items()
            if int(op_id) in retained_ops
        }
        row.projected_runtime = {
            op_id: runtime
            for op_id, runtime in row.projected_runtime.items()
            if int(op_id) in retained_ops or int(op_id) in row.unresolved_operations
        }
        row.pending_operations = {
            op_id: pending
            for op_id, pending in row.pending_operations.items()
            if int(op_id) > frontier
        }

    def _apply_close(self, control: Close) -> None:
        row = self.get(control.request_key.session_id)
        if control.request_key != row.request_key:
            raise invalid_descriptor("close control has a stale request key")
        duplicate, identity = self._control_identity(row, control)
        if duplicate:
            return
        cutoff = control.cutoff
        if not cutoff.is_fixed():
            raise invalid_descriptor("close control cutoff is not fixed")
        point = cast(FixedPoint, cutoff.point)
        reachable = cutoff == row.committed_version() or (
            row.resolved_versions.get(row.point_key(cutoff)) == cutoff
            or row.selected_for_operation(cutoff.producer_op_id) == cutoff
        )
        if not reachable:
            raise invalid_descriptor("close control cutoff is not on the resolved lineage")
        runtime = row.runtime_for(cutoff)
        if runtime is None:
            raise invalid_descriptor("close control cutoff lost its runtime")
        row.committed_point = int(point.point_index)
        row.committed_op_id = int(cutoff.producer_op_id)
        row.version = int(point.point_index)
        row.resolved_op_id = int(cutoff.producer_op_id)
        row.install_runtime(runtime)
        row.terminal_cutoff = cutoff
        row.applied_control_seq = int(control.control_seq)
        row.control_history[identity] = control
        key = row.point_key(cutoff)
        row.resolved_versions = {key: cutoff}
        row.resolved_runtime = {key: runtime}
        row.resolved_operations = {int(cutoff.producer_op_id): cutoff}
        row.projected_runtime = {}
        row.pending_operations = {}
        row.unresolved_operations = {}
        row.declared_parents = {int(cutoff.producer_op_id): cutoff}


__all__ = [
    "MAX_REQUEST_HISTORY_POINTS",
    "Request",
    "RequestDraft",
    "RequestRuntime",
    "RequestPool",
    "RequestPublication",
    "SpeculativeCommit",
]
