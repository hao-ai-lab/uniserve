"""Standard transaction executor: lowering, reservation, replay seam, commit.

Dormant deliverable wiring Stages 3, 4, and 8 of
``specs/unified_forward_execution.md`` into one end-to-end transaction:

```text
ExecutionEngine.execute
    -> prepare: resolve roles -> lower rows -> select capacity -> reserve
    -> launch:  adapter seam -> derive deltas from declared commit exprs
                -> residency commit (or abort on any launch failure)
```

Every fallible step happens in ``prepare`` (typed noncommitted errors: the
engine retains no record and the step may retry); ``launch`` failures abort
the reservation and poison the epoch through the engine, per the spec's
pre-launch/post-launch split.

The model side stays behind :class:`ResidentAdapter` — one call receiving
the packed device tables. The capacity-only graph runtime (Stage 7) replays
a captured adapter against bucket-owned storage; unit tests drive a
deterministic stub. Session deltas and committed cache extents are derived
from the family schema's declared commit expressions — the same closed
source that produced the reservation — never from operation-kind branches.

Cache-role lifecycles follow the registration: request-lifetime roles are
created at admission, branch-lifetime roles on first use (the schema's
``EMPTY_WHEN`` open expression), all owned here rather than by models.
Nothing in production routes through this module until the vertical slice
activates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts.cache_schema import (
    CommitExprKind,
    FamilyCacheRegistration,
    RoleInitializationKind,
)
from ..contracts.execution import (
    ExecuteBatch,
    ExecuteRow,
    RowResult,
    RowStatus,
    SequenceStep,
    SessionDelta,
    TerminalStatus,
    operation_tag,
)
from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..execution.engine import EngineBackpressure, PreLaunchRejection
from ..execution.lowering import (
    LoweredBatch,
    LoweringError,
    RoleSequences,
    lower_rows,
    select_capacity,
)
from ..runtime.immutable_session import RequestSession
from ..runtime.transactional_residency import (
    Residency,
    ResidencyExhausted,
    ResidencyReservation,
    StaleSequenceError,
)

__all__ = [
    "AdapterRowOutcome",
    "PreparedTransaction",
    "ResidentAdapter",
    "StandardTransactionExecutor",
]


@dataclass(frozen=True, slots=True)
class AdapterRowOutcome:
    """Compact per-row output of one packed adapter traversal."""

    sampled_tokens: tuple[int, ...]
    accepted_candidates: int = 0


class ResidentAdapter(Protocol):
    """One packed traversal over the transaction's device tables."""

    def forward(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
    ) -> tuple[AdapterRowOutcome, ...]: ...


@dataclass(slots=True)
class PreparedTransaction:
    """Executor-owned pre-launch state (opaque to the engine)."""

    batch: ExecuteBatch
    sessions: tuple[RequestSession, ...]
    lowered: LoweredBatch
    capacity: GraphCapacity
    reservation: ResidencyReservation
    segments: SegmentTableArrays
    residency_arrays: ResidencyBatchArrays


class StandardTransactionExecutor:
    """The model-backed half of one transaction over real residency."""

    def __init__(
        self,
        *,
        residency: Residency,
        registration: FamilyCacheRegistration,
        capacities: tuple[GraphCapacity, ...],
        adapter: ResidentAdapter,
        page_tokens: int,
    ) -> None:
        self._residency = residency
        self._registration = registration
        self._capacities = capacities
        self._adapter = adapter
        self._page_tokens = page_tokens
        # role state per request incarnation: role_id -> sequence_id
        self._roles: dict[tuple[int, int], dict[int, int]] = {}

    # ------------------------------------------------------------------ #
    # Pre-launch: every fallible step, typed noncommitted failures.
    # ------------------------------------------------------------------ #

    def prepare(
        self,
        batch: ExecuteBatch,
        sessions: tuple[RequestSession, ...],
    ) -> PreparedTransaction:
        try:
            lowered_rows = tuple(
                (row, self._resolve_roles(row, session))
                for row, session in zip(batch.rows, sessions)
            )
            lowered = lower_rows(lowered_rows, self._registration)
            capacity = select_capacity(
                lowered.demand(page_tokens=self._page_tokens),
                self._capacities,
            )
            segments = lowered.fill_segment_table(capacity)
            segments.validate(capacity)
            reservation = self._residency.reserve(lowered.plan)
        except ResidencyExhausted as error:
            raise EngineBackpressure(str(error)) from error
        except (LoweringError, StaleSequenceError) as error:
            raise PreLaunchRejection(str(error)) from error
        try:
            residency_arrays = reservation.batch_arrays(
                capacity.residency, lowered.write_token_begins
            )
            residency_arrays.validate(
                capacity.residency, page_tokens=self._page_tokens
            )
        except Exception:
            reservation.abort()
            raise
        return PreparedTransaction(
            batch=batch,
            sessions=sessions,
            lowered=lowered,
            capacity=capacity,
            reservation=reservation,
            segments=segments,
            residency_arrays=residency_arrays,
        )

    # ------------------------------------------------------------------ #
    # Post-acceptance: one replay; failure aborts and poisons upstream.
    # ------------------------------------------------------------------ #

    def launch(
        self,
        prepared: PreparedTransaction,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]:
        try:
            outcomes = self._adapter.forward(
                prepared.segments,
                prepared.residency_arrays,
                prepared.capacity,
            )
            if len(outcomes) != len(prepared.batch.rows):
                raise RuntimeError(
                    "adapter returned the wrong number of row outcomes"
                )
            committed = self._committed_extents(prepared, outcomes)
            prepared.reservation.commit(committed)
            return self._derive_results(prepared, outcomes, committed)
        except Exception:
            prepared.reservation.abort()
            raise

    # ------------------------------------------------------------------ #

    def _resolve_roles(
        self,
        row: ExecuteRow,
        session: RequestSession,
    ) -> RoleSequences:
        """Resolve (creating per the registration's lifecycle) role sequences."""

        key = (row.session.request_id, row.session.incarnation)
        role_map = self._roles.setdefault(key, {})
        sequences: dict[int, object] = {}
        for role in self._registration.schema.roles:
            sequence_id = role_map.get(role.role_id)
            if sequence_id is None:
                create = (
                    role.initialization.kind
                    is not RoleInitializationKind.EMPTY_WHEN
                    or self._role_opens_for(row, role.role_id)
                )
                if not create:
                    continue
                ref = self._residency.create_sequence(
                    row.session,
                    domain_id=role.domain_id,
                    role_id=role.role_id,
                    lifetime=role.lifetime,
                )
                role_map[role.role_id] = ref.sequence_id
                sequences[role.role_id] = ref
            else:
                sequences[role.role_id] = self._residency.sequence_ref(sequence_id)
        return RoleSequences(sequences)

    def _role_opens_for(self, row: ExecuteRow, role_id: int) -> bool:
        """Branch-lifetime roles open when their operation references them."""

        tag = operation_tag(row.operation)
        return any(
            region.operation_tag is tag
            and role_id in region.role.referenced_role_ids()
            for region in self._registration.schema.regions
        )

    def _committed_extents(
        self,
        prepared: PreparedTransaction,
        outcomes: tuple[AdapterRowOutcome, ...],
    ) -> tuple[int, ...]:
        """Evaluate each binding's declared commit expression."""

        lowered = prepared.lowered
        extents: list[int] = []
        bindings = [
            binding for row in lowered.plan.rows for binding in row.bindings
        ]
        for binding, commit, row_id in zip(
            bindings, lowered.binding_commits, lowered.binding_row_ids
        ):
            if commit.kind is CommitExprKind.ZERO:
                extents.append(0)
            elif commit.kind is CommitExprKind.ALL_RESERVED:
                extents.append(binding.reserve_rows)
            elif commit.kind is CommitExprKind.ACCEPTED_CANDIDATE_PREFIX:
                accepted = outcomes[row_id].accepted_candidates
                if not 0 <= accepted <= binding.reserve_rows:
                    raise RuntimeError(
                        "accepted candidates exceed the reserved tail"
                    )
                extents.append(accepted)
            else:  # RESULT_AFFINE — bounded literal expressions only for now.
                extents.append(
                    min(commit.result_affine.constant, binding.reserve_rows)
                    if commit.result_affine is not None
                    else 0
                )
        return tuple(extents)

    def _derive_results(
        self,
        prepared: PreparedTransaction,
        outcomes: tuple[AdapterRowOutcome, ...],
        committed: tuple[int, ...],
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]:
        lowered = prepared.lowered
        committed_by_row: dict[int, int] = {}
        for extent, row_id in zip(committed, lowered.binding_row_ids):
            committed_by_row[row_id] = committed_by_row.get(row_id, 0) + extent
        results: list[RowResult] = []
        deltas: list[SessionDelta] = []
        for row, session, outcome in zip(
            prepared.batch.rows, prepared.sessions, outcomes
        ):
            results.append(
                RowResult(
                    row_id=row.row_id,
                    request_id=row.session.request_id,
                    incarnation=row.session.incarnation,
                    status=RowStatus.OK,
                    sampled_tokens=outcome.sampled_tokens,
                    accepted_candidates=outcome.accepted_candidates,
                    logprobs=(),
                    terminal=TerminalStatus.ACTIVE,
                )
            )
            samples = isinstance(row.operation, SequenceStep)
            deltas.append(
                SessionDelta(
                    source=session.ref,
                    next_version=session.ref.session_version + 1,
                    history_length_after=session.history_length
                    + committed_by_row.get(row.row_id, 0),
                    flow_coordinate_after=session.flow_coordinate,
                    rng_advance=len(outcome.sampled_tokens) if samples else 1,
                    history_append=outcome.sampled_tokens,
                    cache_leases_added=(),
                    cache_leases_released=(),
                    product_leases_added=(),
                    product_leases_released=(),
                    terminal=TerminalStatus.ACTIVE,
                )
            )
        return tuple(results), tuple(deltas)
