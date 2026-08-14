"""Fixed scheduler-slot request rows and direct candidate publication."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from ..batch import (
    Admission,
    Close,
    Commit,
    Control,
    FixedPoint,
    ImageParams,
    Operation,
    ProductRef,
    RequestKey,
    SamplingParams,
    VersionRef,
    control_content_digest,
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
    kv_reserved_len: int
    kv_initialized_len: int
    kv_visible_len: int
    kv_committed_len: int
    kv_published_len: int

    def __post_init__(self) -> None:
        if self.logical_position < 0 or self.rng_counter < 0 or self.flow_step < 0:
            raise invalid_descriptor("resolved request coordinates are negative")
        if self.latent_product is not None and (
            self.latent_product.kind.value != "latent"
            or self.latent_product.storage_class.value != "latent_arena"
        ):
            raise invalid_descriptor("resolved request latent identity is invalid")
        if not isinstance(self.kv_visible_len, int):
            if not callable(getattr(self.kv_visible_len, "ready", None)) or not (
                0
                <= self.kv_published_len
                <= self.kv_committed_len
                <= self.kv_initialized_len
                <= self.kv_reserved_len
            ):
                raise invalid_descriptor("resolved request deferred KV extent is invalid")
            return
        if not (
            0
            <= self.kv_published_len
            <= self.kv_committed_len
            <= self.kv_visible_len
            <= self.kv_initialized_len
            <= self.kv_reserved_len
        ):
            raise invalid_descriptor("resolved request KV extents are not contained")

    @property
    def kv_length(self) -> int:
        return self.kv_visible_len


@dataclass(frozen=True, slots=True)
class KvControlUpdate:
    session_id: int
    visible_len: int
    committed_len: int
    rewind: bool


@dataclass(frozen=True, slots=True)
class _RequestAssignment:
    operation: Operation
    candidate: RequestRow
    base: RequestRow | None
    selected: VersionRef
    runtime: RequestRuntime


@dataclass(frozen=True, slots=True)
class _RequestPublication:
    table_token: object
    step_id: int
    assignments: tuple[_RequestAssignment, ...]


@dataclass(slots=True)
class RequestRow:
    """Bounded host protocol state for one scheduler-assigned request slot."""

    request_key: RequestKey
    request_pool_idx: int
    admission_digest: str
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    version: int = 0
    resolved_op_id: int = 0
    resolved_digest: str = ""
    committed_point: int = 0
    committed_op_id: int = 0
    committed_digest: str = ""
    public_event_limit: int = 0
    applied_control_seq: int = 0
    control_digests: dict[tuple[int, str], str] = field(default_factory=dict)
    resolved_versions: dict[tuple[int, int], VersionRef] = field(default_factory=dict)
    resolved_runtime: dict[tuple[int, int], RequestRuntime] = field(default_factory=dict)
    resolved_operations: dict[int, VersionRef] = field(default_factory=dict)
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
            point=FixedPoint(self.committed_point, str(self.committed_digest)),
        )

    def resolved_version(self) -> VersionRef:
        return VersionRef(
            request_key=self.request_key,
            producer_op_id=self.resolved_op_id,
            point=FixedPoint(self.version, str(self.resolved_digest)),
        )

    def install_runtime(self, runtime: RequestRuntime) -> None:
        if runtime.latent_product is not None and runtime.latent_product.request_key != self.request_key:
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
        return self.resolved_runtime.get(self.point_key(version))

    def selected_for_operation(self, op_id: int) -> VersionRef | None:
        return self.resolved_operations.get(int(op_id))

    def semantic_parent_for_operation(self, op_id: int) -> VersionRef | None:
        parent = self.declared_parents.get(int(op_id))
        if parent is None or parent.is_fixed():
            return parent
        return self.selected_for_operation(parent.producer_op_id)

    def execution_runtime_for_operation(
        self,
        op_id: int,
        point_index: int,
    ) -> RequestRuntime | None:
        return self.resolved_runtime.get((int(op_id), int(point_index)))


class RequestTable:
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
        self._rows: list[RequestRow | None] = [None] * (size + 1)
        self._slots_by_session: dict[int, int] = {}

    def get(self, session_id: int) -> RequestRow:
        row = self.peek(session_id)
        if row is None:
            raise invalid_descriptor(f"unknown session {session_id}")
        return row

    def peek(self, session_id: int) -> RequestRow | None:
        slot = self._slots_by_session.get(int(session_id))
        return None if slot is None else self._rows[slot]

    def row_at(self, request_pool_idx: int) -> RequestRow | None:
        slot = self._validate_slot(request_pool_idx)
        return self._rows[slot]

    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._slots_by_session))

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, int) and session_id in self._slots_by_session

    def stage_partition(
        self,
        operations: Sequence[Operation],
        admissions: Sequence[Admission],
        request_pool_indices: Sequence[int],
    ) -> tuple[tuple[RequestRow, ...], tuple[RequestRow | None, ...]]:
        """Build isolated row candidates without changing the live table."""

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

        candidates: list[RequestRow] = []
        bases: list[RequestRow | None] = []
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
                candidate = self._admission_row(admission)
            else:
                if occupant is not base or int(base.request_pool_idx) != slot:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} names request-pool index {slot}; "
                        f"session index is {base.request_pool_idx}"
                    )
                if admission is not None and (
                    admission.digest != base.admission_digest
                    or int(admission.request_pool_idx) != int(base.request_pool_idx)
                ):
                    raise invalid_descriptor(
                        f"session {session_id} admission conflicts with committed state"
                    )
                candidate = copy.copy(base)
            self._validate_operation(candidate, operation, slot)
            candidates.append(candidate)
            bases.append(base)
        return tuple(candidates), tuple(bases)

    def prepare_publication(
        self,
        *,
        step_id: int,
        operations: Sequence[Operation],
        candidates: Sequence[RequestRow],
        bases: Sequence[RequestRow | None],
        selected_versions: Mapping[int, VersionRef],
        runtimes: Mapping[int, RequestRuntime],
    ) -> _RequestPublication:
        """Validate and freeze one direct request-row publication."""

        if len(operations) != len(candidates) or len(operations) != len(bases):
            raise RuntimeError("request-row publication columns are not aligned")
        assignments: list[_RequestAssignment] = []
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
            point = selected.point
            if not isinstance(point, FixedPoint):
                raise RuntimeError("published request version must be fixed")
            additions = 1 if operation.advances_state and isinstance(point.point_index, int) else 0
            if len(row.resolved_versions) + additions > self.history_capacity:
                raise invalid_descriptor("request history capacity is exhausted")
            parent = operation.parent
            parent_matches = (
                row.committed_version() == parent
                if parent.is_fixed()
                else row.selected_for_operation(parent.producer_op_id) is not None
            )
            if row.request_key != operation.request_key or not parent_matches:
                raise RuntimeError(f"request {session_id} candidate has a stale parent")
            if not parent.is_fixed() and row.selected_for_operation(parent.producer_op_id) is None:
                raise RuntimeError("resolved request parent is missing")
            assignments.append(
                _RequestAssignment(
                    operation=operation,
                    candidate=row,
                    base=base,
                    selected=selected,
                    runtime=runtime,
                )
            )
        return _RequestPublication(
            table_token=self,
            step_id=int(step_id),
            assignments=tuple(assignments),
        )

    def publish(self, publication: _RequestPublication) -> None:
        """Replace the validated live rows with their complete candidates."""

        if publication.table_token is not self:
            raise RuntimeError("request-row publication belongs to another table")
        for assignment in publication.assignments:
            row = assignment.candidate
            base = assignment.base
            slot = int(row.request_pool_idx)
            if self._rows[slot] is not base:
                raise RuntimeError(f"request {row.session_id} changed before row assignment")
            self._rows[slot] = row
            if base is None:
                self._slots_by_session[int(row.session_id)] = slot
        for assignment in publication.assignments:
            operation = assignment.operation
            row = assignment.candidate
            selected = assignment.selected
            runtime = assignment.runtime
            point = cast(FixedPoint, selected.point)
            row.version = point.point_index
            row.resolved_op_id = selected.producer_op_id
            row.resolved_digest = point.semantic_digest
            if operation.advances_state and isinstance(point.point_index, int):
                key = row.point_key(selected)
                row.resolved_versions[key] = selected
                row.resolved_runtime[key] = runtime
            elif not operation.advances_state:
                row.resolved_runtime[row.point_key(selected)] = runtime
                row.install_runtime(runtime)
            row.resolved_operations[int(operation.op_id)] = selected
            row.declared_parents[int(operation.op_id)] = operation.parent
            row.last_op_id = int(operation.op_id)
            row.last_step_id = int(publication.step_id)

    def apply_controls(self, controls: Sequence[Control]) -> tuple[KvControlUpdate, ...]:
        updates: list[KvControlUpdate] = []
        for control in controls:
            update: KvControlUpdate | None = None
            if isinstance(control, Commit):
                update = self._apply_commit(control)
            elif isinstance(control, Close):
                update = self._apply_close(control)
            if update is not None:
                updates.append(update)
        return tuple(updates)

    def finalize_predicated(
        self,
        session_id: int,
        op_id: int,
        parent: VersionRef,
    ) -> tuple[VersionRef, RequestRuntime]:
        row = self.get(session_id)
        selected = parent if parent.is_fixed() else row.selected_for_operation(parent.producer_op_id)
        runtime = None if selected is None else row.runtime_for(selected)
        if selected is None or runtime is None:
            raise invalid_descriptor(f"predicated operation {op_id} lost its resolved parent")
        point = selected.point
        if not isinstance(point, FixedPoint):
            raise invalid_descriptor(f"predicated operation {op_id} resolved to a device point")
        selected = VersionRef(
            request_key=selected.request_key,
            producer_op_id=selected.producer_op_id,
            point=FixedPoint(point.point_index, str(point.semantic_digest)),
        )
        key = row.point_key(selected)
        row.resolved_versions[key] = selected
        row.resolved_runtime[key] = runtime
        row.resolved_operations[int(op_id)] = selected
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
        row.resolved_digest = point.semantic_digest
        row.install_runtime(runtime)
        return selected, runtime

    def drop(self, session_id: int) -> None:
        selected = int(session_id)
        slot = self._slots_by_session.pop(selected, None)
        if slot is not None:
            self._rows[slot] = None

    def snapshot_committed(self, session_ids: set[int]) -> tuple[RequestRow, ...]:
        snapshots: list[RequestRow] = []
        for session_id in sorted(int(value) for value in session_ids):
            snapshot = copy.deepcopy(self.get(session_id))
            snapshot.version = snapshot.committed_point
            snapshot.resolved_op_id = snapshot.committed_op_id
            snapshot.resolved_digest = str(snapshot.committed_digest)
            committed = snapshot.committed_version()
            runtime = snapshot.runtime_for(committed)
            if runtime is None:
                raise invalid_descriptor("committed request runtime state is missing")
            snapshot.install_runtime(runtime)
            key = snapshot.point_key(committed)
            snapshot.resolved_versions = {key: committed}
            snapshot.resolved_runtime = {key: runtime}
            snapshot.resolved_operations = {snapshot.committed_op_id: committed}
            snapshot.declared_parents = {snapshot.committed_op_id: committed}
            snapshots.append(snapshot)
        return tuple(snapshots)

    def restore_rows(
        self,
        rows: Sequence[RequestRow],
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
            if not row.admission_digest:
                raise invalid_descriptor("request snapshot admission identity is missing")
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

    def _admission_row(self, admission: Admission) -> RequestRow:
        slot = self._validate_slot(admission.request_pool_idx)
        prefix_len = 0 if admission.und is None else int(admission.und.kv.prefix_len)
        row = RequestRow(
            request_key=admission.request_key,
            request_pool_idx=slot,
            admission_digest=admission.digest,
            sampling=None if admission.und is None else admission.und.sampling,
            image=None if admission.gen_admission is None else admission.gen_admission.image,
            negative_token_ids=() if admission.und is None else admission.und.negative_token_ids,
            finish_token_ids=() if admission.und is None else admission.und.finish_token_ids,
            resolved_digest=admission.digest,
            committed_digest=admission.digest,
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
            kv_reserved_len=prefix_len,
            kv_initialized_len=prefix_len,
            kv_visible_len=prefix_len,
            kv_committed_len=prefix_len,
            kv_published_len=0,
        )
        return row

    @staticmethod
    def _validate_operation(row: RequestRow, operation: Operation, slot: int) -> None:
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
        elif row.selected_for_operation(parent.producer_op_id) is None:
            raise invalid_descriptor(
                f"operation {operation.op_id} names an unresolved device parent"
            )

    def _control_identity(
        self,
        row: RequestRow,
        control: Commit | Close,
    ) -> tuple[bool, tuple[int, str], str]:
        kind = "commit" if isinstance(control, Commit) else "close"
        identity = (int(control.control_seq), kind)
        digest = control_content_digest(control)
        existing = row.control_digests.get(identity)
        if existing is not None:
            if existing != digest:
                raise invalid_descriptor(
                    f"control identity {identity} conflicts with its committed content"
                )
            return True, identity, digest
        if len(row.control_digests) >= self.history_capacity:
            raise invalid_descriptor("request control history capacity is exhausted")
        if int(control.control_seq) != row.applied_control_seq + 1:
            raise invalid_descriptor(
                f"control sequence {control.control_seq} does not follow {row.applied_control_seq}"
            )
        return False, identity, digest

    def _apply_commit(self, control: Commit) -> KvControlUpdate | None:
        row = self.get(control.request_key.session_id)
        if control.request_key != row.request_key:
            raise invalid_descriptor("commit control has a stale request key")
        duplicate, identity, digest = self._control_identity(row, control)
        if duplicate:
            return None
        if row.terminal_cutoff is not None:
            raise invalid_descriptor("commit control targets a closed request")
        if control.expected_parent != row.committed_version():
            raise invalid_descriptor("commit control expected parent is not current")
        selected = control.selected
        if not selected.is_fixed():
            raise invalid_descriptor("commit control selected point is not fixed")
        if row.resolved_versions.get(row.point_key(selected)) != selected:
            raise invalid_descriptor("commit control selected point was not resolved")
        if row.semantic_parent_for_operation(selected.producer_op_id) != control.expected_parent:
            raise invalid_descriptor("commit control selected point has a different parent")
        runtime = row.runtime_for(selected)
        if runtime is None:
            raise invalid_descriptor("commit control selected point lost its runtime")
        point = cast(FixedPoint, selected.point)
        if int(control.public_event_limit) < row.public_event_limit:
            raise invalid_descriptor("commit control regresses the public event limit")
        row.committed_point = int(point.point_index)
        row.committed_op_id = int(selected.producer_op_id)
        row.committed_digest = point.semantic_digest
        row.public_event_limit = int(control.public_event_limit)
        row.applied_control_seq = int(control.control_seq)
        row.control_digests[identity] = digest
        return KvControlUpdate(
            session_id=row.session_id,
            visible_len=runtime.kv_visible_len,
            committed_len=runtime.kv_visible_len,
            rewind=False,
        )

    def _apply_close(self, control: Close) -> KvControlUpdate | None:
        row = self.get(control.request_key.session_id)
        if control.request_key != row.request_key:
            raise invalid_descriptor("close control has a stale request key")
        duplicate, identity, digest = self._control_identity(row, control)
        if duplicate:
            return None
        cutoff = control.cutoff
        if not cutoff.is_fixed():
            raise invalid_descriptor("close control cutoff is not fixed")
        point = cast(FixedPoint, cutoff.point)
        reachable = cutoff == row.committed_version() or row.resolved_versions.get(
            row.point_key(cutoff)
        ) == cutoff
        if not reachable:
            raise invalid_descriptor("close control cutoff is not on the resolved lineage")
        runtime = row.runtime_for(cutoff)
        if runtime is None:
            raise invalid_descriptor("close control cutoff lost its runtime")
        row.committed_point = int(point.point_index)
        row.committed_op_id = int(cutoff.producer_op_id)
        row.committed_digest = point.semantic_digest
        row.version = int(point.point_index)
        row.resolved_op_id = int(cutoff.producer_op_id)
        row.resolved_digest = point.semantic_digest
        row.install_runtime(runtime)
        row.terminal_cutoff = cutoff
        row.applied_control_seq = int(control.control_seq)
        row.control_digests[identity] = digest
        key = row.point_key(cutoff)
        row.resolved_versions = {key: cutoff}
        row.resolved_runtime = {key: runtime}
        row.resolved_operations = {int(cutoff.producer_op_id): cutoff}
        row.declared_parents = {int(cutoff.producer_op_id): cutoff}
        return KvControlUpdate(
            session_id=row.session_id,
            visible_len=runtime.kv_visible_len,
            committed_len=runtime.kv_visible_len,
            rewind=True,
        )


__all__ = [
    "MAX_REQUEST_HISTORY_POINTS",
    "KvControlUpdate",
    "RequestRow",
    "RequestRuntime",
    "RequestTable",
]
