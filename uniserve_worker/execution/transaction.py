"""Transaction preparation, resident execution, rank fan-out, and conformance.

Preparation performs every fallible lowering and residency operation before
acceptance. Launch performs one resident traversal, derives effects from the
closed family schema, and commits or aborts the reservation as one transaction.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from uniserve_worker.contracts.cache_schema import (
    CommitExprKind,
    FamilyCacheRegistration,
    RoleInitializationKind,
)
from uniserve_worker.contracts.execution import (
    ExecuteBatch,
    ExecuteRow,
    OperationTag,
    RowResult,
    RowStatus,
    SequenceStep,
    SessionDelta,
    TerminalStatus,
    operation_tag,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchArrays
from uniserve_worker.execution.engine import (
    EngineBackpressure,
    EngineExecutionError,
    PreLaunchRejection,
)
from uniserve_worker.execution.lowering import (
    GraphCapacity,
    LoweredBatch,
    LoweringError,
    RoleSequences,
    SegmentTableArrays,
    lower_rows,
    select_capacity,
)
from uniserve_worker.runtime.immutable_session import RequestSession
from uniserve_worker.runtime.transactional_residency import (
    Residency,
    ResidencyExhausted,
    ResidencyReservation,
    StaleSequenceError,
)

__all__ = [
    "AdapterPayload",
    "AdapterRowOutcome",
    "ConformanceCase",
    "ConformanceManifest",
    "DistributedConfigurationError",
    "ManifestError",
    "PreparedTransaction",
    "RankDisagreement",
    "RankFanOutExecutor",
    "RankMember",
    "ResidentAdapter",
    "StandardTransactionExecutor",
    "build_manifest",
    "case_set_hash",
    "generate_cases",
    "validate_manifest",
]


@dataclass(frozen=True, slots=True)
class AdapterRowOutcome:
    """Compact per-row output of one packed adapter traversal."""

    sampled_tokens: tuple[int, ...]
    accepted_candidates: int = 0


@dataclass(frozen=True, slots=True)
class AdapterPayload:
    """Packed numerical payload columns (canonical token order)."""

    token_ids: tuple[int, ...]
    positions: tuple[int, ...]


class ResidentAdapter(Protocol):
    """One packed traversal over the transaction's device tables."""

    def forward(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        payload: AdapterPayload,
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
            for row in batch.rows:
                for lease in row.product_leases:
                    self._residency.validate_product(lease)
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
            residency_arrays.validate(capacity.residency, page_tokens=self._page_tokens)
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
                AdapterPayload(
                    token_ids=prepared.lowered.token_ids,
                    positions=prepared.lowered.positions,
                ),
            )
            if len(outcomes) != len(prepared.batch.rows):
                raise RuntimeError("adapter returned the wrong number of row outcomes")
            committed = self._committed_extents(prepared, outcomes)
            published = prepared.reservation.commit(committed)
            return self._derive_results(prepared, outcomes, committed, published)
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
                    role.initialization.kind is not RoleInitializationKind.EMPTY_WHEN
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
            region.operation_tag is tag and role_id in region.role.referenced_role_ids()
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
        bindings = [binding for row in lowered.plan.rows for binding in row.bindings]
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
                    raise RuntimeError("accepted candidates exceed the reserved tail")
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
        published: tuple,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]:
        lowered = prepared.lowered
        committed_by_row: dict[int, int] = {}
        for extent, row_id in zip(committed, lowered.binding_row_ids):
            committed_by_row[row_id] = committed_by_row.get(row_id, 0) + extent
        # Published product leases in plan order map back to their rows.
        published_by_row: dict[int, list] = {}
        cursor = 0
        for demand_row in lowered.plan.rows:
            for _ in demand_row.products:
                published_by_row.setdefault(demand_row.row_id, []).append(published[cursor])
                cursor += 1
        results: list[RowResult] = []
        deltas: list[SessionDelta] = []
        for row, session, outcome in zip(prepared.batch.rows, prepared.sessions, outcomes):
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
                    product_leases_added=tuple(published_by_row.get(row.row_id, ())),
                    product_leases_released=(),
                    terminal=TerminalStatus.ACTIVE,
                )
            )
        return tuple(results), tuple(deltas)


# ---------------------
# Distributed rank fan-out executor
# ---------------------


class DistributedConfigurationError(RuntimeError):
    """Ranks disagree about static configuration before readiness."""


class RankDisagreement(EngineExecutionError):
    """Ranks produced conflicting logical results after launch."""


@dataclass(frozen=True, slots=True)
class RankMember:
    """One rank's local execution half plus its static identity."""

    rank: int
    configuration_fingerprint: str
    executor: StandardTransactionExecutor


@dataclass(slots=True)
class _GroupPrepared:
    prepared: tuple[PreparedTransaction, ...]


class RankFanOutExecutor:
    """All-rank prepare/launch agreement behind one TransactionExecutor."""

    def __init__(self, members: tuple[RankMember, ...], *, result_rank: int = 0) -> None:
        if not members:
            raise DistributedConfigurationError("a rank group needs members")
        ranks = [member.rank for member in members]
        if sorted(ranks) != list(range(len(members))):
            raise DistributedConfigurationError(
                f"rank identities must be dense from zero; got {sorted(ranks)}"
            )
        fingerprints = {member.configuration_fingerprint for member in members}
        if len(fingerprints) != 1:
            raise DistributedConfigurationError(
                f"all ranks must share one configuration fingerprint; got {sorted(fingerprints)}"
            )
        if result_rank not in ranks:
            raise DistributedConfigurationError(
                f"designated result rank {result_rank} is not a member"
            )
        self._members = tuple(sorted(members, key=lambda member: member.rank))
        self._result_rank = result_rank

    # ------------------------------------------------------------------ #

    def prepare(
        self,
        batch: ExecuteBatch,
        sessions: tuple[RequestSession, ...],
    ) -> _GroupPrepared:
        """All-rank prepare agreement; any failure leaves nothing reserved."""

        prepared: list[PreparedTransaction] = []
        try:
            for member in self._members:
                prepared.append(member.executor.prepare(batch, sessions))
        except Exception as error:
            for transaction in prepared:
                transaction.reservation.abort()
            if isinstance(error, EngineExecutionError):
                raise
            raise PreLaunchRejection(str(error)) from error
        return _GroupPrepared(prepared=tuple(prepared))

    def launch(
        self,
        prepared: _GroupPrepared,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]:
        """Group-atomic launch: rank failure or disagreement poisons upstream."""

        outcomes: list[tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]] = []
        try:
            for member, transaction in zip(self._members, prepared.prepared):
                outcomes.append(member.executor.launch(transaction))
        except Exception:
            # The failing rank aborted its own reservation; abort the rest
            # (committed ranks cannot be uncommitted — interpretation of the
            # group state is unsafe, which is exactly why the engine poisons).
            for member, transaction in zip(self._members, prepared.prepared):
                transaction.reservation.abort()
            raise
        designated = outcomes[self._result_rank]
        for member, outcome in zip(self._members, outcomes):
            if outcome != designated:
                raise RankDisagreement(f"rank {member.rank} finalized conflicting logical results")
        return designated


# ---------------------
# Mechanical adapter conformance gate
# ---------------------

# Frozen serial-benchmark reference points (docs/benchmark-protocol.md,
# snapshot 20260715T0855Z). Later comparisons name these exact values.
_BENCHMARK_REFERENCES: dict[str, dict[str, float]] = {
    "qwen3_sharegpt_r16": {
        "output_tokens_per_s": 1857.14,
        "mean_ttft_ms": 104.19,
        "mean_tpot_ms": 23.99,
    },
    "sensenova_mjhq_t2i_c1": {"mean_image_latency_ms": 3759.051},
    "sensenova_mjhq_t2i_c32": {"images_per_s": 0.240},
    "bagel_mjhq_t2i_c1": {"mean_image_latency_ms": 6913.954},
}


class ManifestError(ValueError):
    """A manifest does not match its regenerated case set."""


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """One generated operation-composition and row-order case."""

    case_id: str
    family: str
    operations: tuple[OperationTag, ...]  # row order

    def encode(self) -> str:
        tags = ",".join(str(int(tag)) for tag in self.operations)
        return f"{self.family}:{tags}"


@dataclass(frozen=True, slots=True)
class ConformanceManifest:
    family: str
    advertised_operations: tuple[int, ...]
    case_ids: tuple[str, ...]
    case_set_hash: str
    benchmark_references: dict[str, dict[str, float]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def generate_cases(
    family: str,
    advertised: frozenset[OperationTag],
) -> tuple[ConformanceCase, ...]:
    """Every nonempty advertised subset in every row order, mechanically."""

    operations = sorted(advertised, key=int)
    cases: list[ConformanceCase] = []
    for size in range(1, len(operations) + 1):
        for subset in itertools.combinations(operations, size):
            for order in itertools.permutations(subset):
                tags = "-".join(tag.name.lower() for tag in order)
                cases.append(
                    ConformanceCase(
                        case_id=f"{family}/{tags}",
                        family=family,
                        operations=tuple(order),
                    )
                )
    return tuple(cases)


def case_set_hash(cases: tuple[ConformanceCase, ...]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.encode().encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_manifest(
    family: str,
    advertised: frozenset[OperationTag],
) -> ConformanceManifest:
    cases = generate_cases(family, advertised)
    return ConformanceManifest(
        family=family,
        advertised_operations=tuple(sorted(int(tag) for tag in advertised)),
        case_ids=tuple(case.case_id for case in cases),
        case_set_hash=case_set_hash(cases),
        benchmark_references=_BENCHMARK_REFERENCES,
    )


def validate_manifest(manifest: ConformanceManifest) -> None:
    """Regenerate the case set and reject any drift (readiness rule)."""

    regenerated = generate_cases(
        manifest.family,
        frozenset(OperationTag(tag) for tag in manifest.advertised_operations),
    )
    if tuple(case.case_id for case in regenerated) != manifest.case_ids:
        raise ManifestError(f"manifest {manifest.family} lists a stale case set")
    if case_set_hash(regenerated) != manifest.case_set_hash:
        raise ManifestError(f"manifest {manifest.family} hash does not match the regenerated set")
