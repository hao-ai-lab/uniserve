"""The execution engine: one data plane for every model-backed operation.

Consolidates the execution layer named by ``specs/unified_forward_execution.md``
into the single engine module its completion criteria require:

* the transactional :class:`ExecutionEngine` (exactly-once step lifecycle,
  replay window, canonical fingerprints, session deltas, admin commands) and
  its :class:`TransactionExecutor` seam;
* schema-driven lowering of typed operation rows into segment tables,
  capacity demand, and residency reservations;
* the standard and distributed (rank fan-out) transaction executors over the
  family adapters, plus the mechanical adapter conformance gate;
* generic row/segment plan construction, device batch construction, group
  planning, and result projection for the live worker step
  (:class:`ModelRunner`), covering sequence extension, one-position advance,
  candidate verification, flow steps, encode, and materialize operations;
* numerical sampling stays in ``nn.sampler``; graph execution stays in
  ``execution.graph``; family compute stays in ``models``.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch

from uniserve_worker.backends.attention import (
    get_attention_backend,
    normalize_attention_backend_name,
)
from uniserve_worker.contracts.batch_policy import BatchPolicy
from uniserve_worker.contracts.batches import ExecuteBatch as WireExecuteBatch
from uniserve_worker.contracts.batches import UniForwardBatch
from uniserve_worker.contracts.cache_schema import (
    CacheEffect,
    CacheRegionSpec,
    CommitExpr,
    CommitExprKind,
    ExtentOperand,
    FamilyCacheRegistration,
    RoleInitializationKind,
    RoleSelectorKind,
)
from uniserve_worker.contracts.caps import Caps
from uniserve_worker.contracts.execution import (
    DropSession,
    EncodeStep,
    EngineCommand,
    EngineRef,
    ExecuteBatch,
    ExecuteResult,
    ExecuteRow,
    ExecutionContractError,
    MaterializeStep,
    OperationTag,
    Quiesce,
    Resume,
    RowResult,
    RowStatus,
    SequenceStep,
    SessionDelta,
    TerminalStatus,
    canonical_payload_fingerprint,
    operation_tag,
    validate_execute_batch,
)
from uniserve_worker.contracts.execution import (
    FlowStep as FlowOperation,
)
from uniserve_worker.contracts.forward_batch import (
    BranchSpec,
    CacheSpanPlan,
    CfgPlan,
    CommitInputs,
    CommitRowPlan,
    DenoiseBranchKey,
    DenoiseInputs,
    DenoisePostprocessEntry,
    DenoiseRowPlan,
    EagerFallbackReason,
    EagerFallbackWarning,
    EncodeInputs,
    EncodeRowPlan,
    ForwardBatch,
    ForwardGraphPolicy,
    ForwardModality,
    ForwardOutputKind,
    ForwardOutputSlot,
    ForwardPlan,
    ForwardPostprocessPolicy,
    ForwardResult,
    ForwardResultProjection,
    ForwardRowPlan,
    ForwardRuntimeHandles,
    ForwardSegmentClass,
    ForwardSegmentPlan,
    ForwardShapeSummary,
    KvSource,
    KvWritePolicy,
    SegmentSpec,
    StrictForwardGraphError,
    TextPostprocessEntry,
    TextTokenSpanPlan,
    VisiblePolicy,
    coerce_forward_result,
)
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.contracts.model_protocols import FlowContext, ModelHooks, UniModel
from uniserve_worker.contracts.op_kinds import TARGET_VERIFY_UND, VAE_ENCODE, VIT_ENCODE
from uniserve_worker.contracts.outputs import (
    CommitOutput,
    DeferredForwardOutput,
    EncodeOutput,
    FlowOutput,
    ForwardOutput,
    ForwardOutputBase,
    TextTokenOutput,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchArrays, ResidencyBatchCapacity
from uniserve_worker.contracts.resource_plan import LatentTokens, ResourcePlan
from uniserve_worker.contracts.segment_table import GraphCapacity, SegmentTableArrays
from uniserve_worker.foundation.env import env_flag
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.foundation.profiling import profile_range
from uniserve_worker.foundation.runtime_config import get_execution_config
from uniserve_worker.nn.diffusion import (
    CfgParams,
    FlowMatchSchedule,
    ScheduleDirection,
    combine_cfg,
    euler_step,
    init_latent,
    x_pred_to_velocity,
)
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.cfg import CfgPlan as DiffusionCfgPlan
from uniserve_worker.nn.sampler import (
    DeferredBatchedSamplingResult,
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
    sample_one_from_logits,
    score_prompt_token_logprobs,
)
from uniserve_worker.runtime.forward_batch_builder import ForwardBatchBuilder
from uniserve_worker.runtime.host_staging import (
    canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from uniserve_worker.runtime.image_params import required_image_height, required_image_width
from uniserve_worker.runtime.image_utils import pil_image_to_png_b64, to_uint8_image
from uniserve_worker.runtime.immutable_session import DropOutcome, RequestSession, SessionRegistry
from uniserve_worker.runtime.paged_text_cache import copy_paged_text_cache_spans
from uniserve_worker.runtime.request_session import RequestSessionTable
from uniserve_worker.runtime.request_state import RequestState, RequestStateTable
from uniserve_worker.runtime.residency_manager import ResidencyLeaseManager
from uniserve_worker.runtime.resources import ResourceRuntime
from uniserve_worker.runtime.tensor_views import coalesce_one_token_rows
from uniserve_worker.runtime.transactional_residency import (
    ProductDemand,
    ReservationPlan,
    Residency,
    ResidencyExhausted,
    ResidencyReservation,
    RowDemand,
    SequenceBinding,
    StaleSequenceError,
)
from uniserve_worker.spec import speculative_sample_target_only

if TYPE_CHECKING:
    from uniserve_worker.backends.attention.text_dispatch import TextBackendGate
    from uniserve_worker.contracts.batches import TextBatch, UniForwardBatch
    from uniserve_worker.contracts.forward_batch import ForwardBatch, ForwardGraphPolicy
    from uniserve_worker.contracts.forward_stats import ForwardStats
    from uniserve_worker.contracts.model_protocols import FlowCapable
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.request_state import RequestState, RequestStateTable
    from uniserve_worker.runtime.residency import ResidencyManager

"""Transactional ``ExecutionEngine``: one data plane with exact retry.

This section owns engine lifecycle state, the fixed replay window with
`(engine_epoch, step_id)` identity and canonical payload fingerprints,
cumulative acknowledgement eviction, exactly-once duplicate handling, exact
session resolution and provisional admission, atomic session-delta commit,
typed retryable errors, the closed administrative seam, and epoch poisoning.

The model-backed half — residency reservation, graph refresh, one replay,
finalization against device results — sits behind one internal
:class:`TransactionExecutor` seam, implemented over real residency and the
capacity-only graph runtime by :class:`StandardTransactionExecutor` and the
rank fan-out executor below.

Semantics enforced here:

* Execution identity is ``(engine_epoch, step_id)``; the canonical payload
  fingerprint excludes cumulative acknowledgement, so a retry may advance
  acknowledgement without changing transaction identity.
* A retained duplicate with the same fingerprint returns the original
  receipt without re-executing; a different fingerprint is a fatal protocol
  violation that poisons the epoch.
* A step at or below ``acknowledged_through`` is stale and cannot re-execute.
  The first unseen ``step_id`` must be the next engine sequence number.
* A full replay window yields retryable backpressure before session
  admission; an accepted identity never returns to an unobserved state
  within the epoch.
* One logical engine owns one unresolved transaction at a time.
* Commands execute only at a transaction boundary.
"""


__all__ = [
    "AdminOutcome",
    "EngineBackpressure",
    "EngineExecutionError",
    "EnginePoisoned",
    "EngineState",
    "ExecutionEngine",
    "PayloadConflict",
    "PreLaunchRejection",
    "StaleStep",
    "TransactionExecutor",
]


class EngineState(IntEnum):
    READY = 1
    QUIESCED = 2
    POISONED = 3


class EngineExecutionError(RuntimeError):
    """Base class for typed engine failures."""


class EngineBackpressure(EngineExecutionError):
    """Retryable: the replay window or the single transaction slot is busy."""


class StaleStep(EngineExecutionError):
    """The step is at or below cumulative acknowledgement, or out of order."""


class PayloadConflict(EngineExecutionError):
    """A retained step arrived with a different canonical fingerprint."""


class EnginePoisoned(EngineExecutionError):
    """The epoch is poisoned; the engine requires controlled reconstruction."""


class PreLaunchRejection(EngineExecutionError):
    """Typed noncommitted pre-launch failure: no record, retry permitted."""


class TransactionExecutor(Protocol):
    """Internal seam for the model-backed half of one transaction.

    ``prepare`` performs every fallible step — lowering, capacity selection,
    residency reservation, lease pinning, metadata refresh — and may raise
    :class:`PreLaunchRejection` or :class:`EngineBackpressure`; the engine
    treats those as noncommitted (no replay record, the step may retry).

    ``launch`` replays the prepared transaction and returns row results and
    authoritative deltas in row order. Acceptance has happened by then: any
    exception from ``launch`` poisons the epoch, and the executor must
    abort its reservation before raising.
    """

    def prepare(
        self,
        batch: ExecuteBatch,
        sessions: tuple[RequestSession, ...],
    ) -> object: ...

    def launch(
        self,
        prepared: object,
    ) -> tuple[tuple[RowResult, ...], tuple[SessionDelta, ...]]: ...


class _RecordState(IntEnum):
    SUBMITTED = 1
    COMMITTED = 2


@dataclass
class _ReplayRecord:
    step_id: int
    fingerprint: str
    state: _RecordState
    result: ExecuteResult | None


class _Receipt:
    """Engine-owned receipt: no callbacks, sessions, or graph state."""

    def __init__(self, engine: "ExecutionEngine", step_id: int) -> None:
        self._engine = engine
        self._step_id = step_id

    def ready(self) -> bool:
        record = self._engine._records.get(self._step_id)
        return record is not None and record.state is _RecordState.COMMITTED

    def result(self) -> ExecuteResult:
        record = self._engine._records.get(self._step_id)
        if record is None or record.result is None:
            raise EngineExecutionError(f"step {self._step_id} has no durable result in the window")
        return record.result


class AdminOutcome(IntEnum):
    APPLIED = 1
    STALE = 2
    UNSUPPORTED = 3


class ExecutionEngine:
    """The sole model-backed data-plane interface.

    ``execute`` is the only execution method and ``apply`` the only
    administrative seam; there are no public methods for planning, staging,
    per-operation forwarding, graph selection, or postprocessing.
    """

    def __init__(
        self,
        *,
        engine: EngineRef,
        executor: TransactionExecutor,
        advertised_operations: frozenset[OperationTag],
        replay_window: int = 8,
    ) -> None:
        if replay_window <= 0:
            raise ExecutionContractError("replay window needs at least one record")
        self.engine = engine
        self.sessions = SessionRegistry(engine)
        self._executor = executor
        self._advertised = advertised_operations
        self._window_capacity = replay_window
        self._records: dict[int, _ReplayRecord] = {}
        # -1 is the "none acknowledged" sentinel; acknowledgement is a step id.
        self._acknowledged_through = -1
        self._next_step = 0
        self._state = EngineState.READY
        self._unresolved_step: int | None = None

    # ------------------------------------------------------------------ #
    # Data plane.
    # ------------------------------------------------------------------ #

    def execute(self, batch: ExecuteBatch) -> _Receipt:
        self._require_state(EngineState.READY)
        if batch.engine_epoch != self.engine.engine_epoch:
            raise ExecutionContractError(
                f"batch names epoch {batch.engine_epoch}; engine is {self.engine.engine_epoch}"
            )
        if batch.acknowledged_through >= batch.step_id:
            raise ExecutionContractError("a batch cannot acknowledge its own or a future step")
        self._apply_acknowledgement(batch.acknowledged_through)
        retained = self._records.get(batch.step_id)
        if retained is not None:
            fingerprint = canonical_payload_fingerprint(batch)
            if fingerprint != retained.fingerprint:
                self._poison(f"step {batch.step_id} retried with a conflicting payload")
            return _Receipt(self, batch.step_id)
        if batch.step_id <= self._acknowledged_through:
            raise StaleStep(
                f"step {batch.step_id} is at or below acknowledgement "
                f"{self._acknowledged_through} and cannot re-execute"
            )
        if batch.step_id != self._next_step:
            raise StaleStep(f"first unseen step must be {self._next_step}; got {batch.step_id}")
        if self._unresolved_step is not None:
            raise EngineBackpressure(f"transaction {self._unresolved_step} is still unresolved")
        if len(self._records) >= self._window_capacity:
            raise EngineBackpressure("replay window is full; acknowledge completed steps first")
        # Pre-launch phase: every failure through prepare() is noncommitted
        # and leaves no record, so the scheduler may retry the same step
        # after correction (or after backpressure clears).
        validate_execute_batch(batch, advertised_operations=self._advertised)
        provisional, snapshots = self._resolve_sessions(batch.rows)
        prepared = self._executor.prepare(batch, snapshots)
        # Acceptance: from here the identity is retained and never returns to
        # an unobserved state within the epoch.
        fingerprint = canonical_payload_fingerprint(batch)
        record = _ReplayRecord(
            step_id=batch.step_id,
            fingerprint=fingerprint,
            state=_RecordState.SUBMITTED,
            result=None,
        )
        self._records[batch.step_id] = record
        self._next_step = batch.step_id + 1
        self._unresolved_step = batch.step_id
        try:
            row_results, deltas = self._executor.launch(prepared)
            result = self._finalize(batch, row_results, deltas, provisional)
        except EnginePoisoned:
            raise
        except Exception as error:  # noqa: BLE001 — any post-acceptance failure
            self._poison(f"transaction {batch.step_id} failed after acceptance: {error}")
        record.result = result
        record.state = _RecordState.COMMITTED
        self._unresolved_step = None
        return _Receipt(self, batch.step_id)

    # ------------------------------------------------------------------ #
    # Administration (transaction boundaries only).
    # ------------------------------------------------------------------ #

    def apply(self, command: EngineCommand) -> AdminOutcome:
        if self._state is EngineState.POISONED:
            raise EnginePoisoned("poisoned epochs accept no commands")
        if self._unresolved_step is not None:
            raise EngineBackpressure("commands execute at transaction boundaries")
        if isinstance(command, DropSession):
            outcome = self.sessions.drop(command)
            return AdminOutcome.APPLIED if outcome is DropOutcome.DROPPED else AdminOutcome.STALE
        if isinstance(command, Quiesce):
            self._state = EngineState.QUIESCED
            return AdminOutcome.APPLIED
        if isinstance(command, Resume):
            if self._state is EngineState.QUIESCED:
                self._state = EngineState.READY
            return AdminOutcome.APPLIED
        # This engine instance has no residency-backed command executor.
        return AdminOutcome.UNSUPPORTED

    @property
    def state(self) -> EngineState:
        return self._state

    # ------------------------------------------------------------------ #
    # Internal transitions.
    # ------------------------------------------------------------------ #

    def _require_state(self, required: EngineState) -> None:
        if self._state is EngineState.POISONED:
            raise EnginePoisoned("the epoch is poisoned; reconstruct the engine")
        if self._state is not required:
            raise EngineBackpressure(f"engine is {self._state.name}, not READY")

    def _apply_acknowledgement(self, acknowledged_through: int) -> None:
        # Effective acknowledgement never regresses: a retry carrying an older
        # envelope is a no-op, so repeating a retained step stays harmless.
        if acknowledged_through <= self._acknowledged_through:
            return
        for step_id in sorted(self._records):
            if step_id > acknowledged_through:
                break
            record = self._records[step_id]
            if record.state is not _RecordState.COMMITTED:
                raise ExecutionContractError(
                    f"acknowledgement cannot release unresolved step {step_id}"
                )
            del self._records[step_id]
        self._acknowledged_through = acknowledged_through

    def _resolve_sessions(
        self,
        rows: tuple[ExecuteRow, ...],
    ) -> tuple[dict[int, RequestSession], tuple[RequestSession, ...]]:
        provisional: dict[int, RequestSession] = {}
        snapshots: list[RequestSession] = []
        for row in rows:
            if row.admission is not None:
                snapshot = self.sessions.admit(row.admission)
                provisional[row.row_id] = snapshot
            else:
                snapshot = self.sessions.resolve(row.session)
            snapshots.append(snapshot)
        return provisional, tuple(snapshots)

    def _finalize(
        self,
        batch: ExecuteBatch,
        row_results: tuple[RowResult, ...],
        deltas: tuple[SessionDelta, ...],
        provisional: dict[int, RequestSession],
    ) -> ExecuteResult:
        if len(row_results) != len(batch.rows) or len(deltas) != len(batch.rows):
            raise ExecutionContractError("executor must return one result and one delta per row")
        for index, (row, result) in enumerate(zip(batch.rows, row_results)):
            if result.row_id != index:
                raise ExecutionContractError(f"row result {index} is out of scheduler order")
            if (
                result.request_id != row.session.request_id
                or result.incarnation != row.session.incarnation
            ):
                raise ExecutionContractError(
                    f"row result {index} names a foreign request incarnation"
                )
        for row, delta in zip(batch.rows, deltas):
            self.sessions.apply(delta, provisional=provisional.get(row.row_id))
        return ExecuteResult(
            engine_epoch=batch.engine_epoch,
            step_id=batch.step_id,
            row_results=row_results,
            session_deltas=deltas,
        )

    def _poison(self, reason: str) -> None:
        self._state = EngineState.POISONED
        self._unresolved_step = None
        raise EnginePoisoned(reason)


# ---------------------
# Schema-driven lowering: typed rows to segment tables and reservations
# ---------------------


class LoweringError(ValueError):
    """A row cannot be lowered under the family's closed cache schema."""


@dataclass(frozen=True, slots=True)
class RoleSequences:
    """One row's cache-role resolution: role id to current sequence reference.

    The engine owns this mapping (roles live in `RequestSession` leases); the
    lowerer only consumes it. Branch-lifetime roles appear once per role id.
    """

    sequences: dict[int, object]  # role_id -> CacheSequenceRef

    def resolve(self, role_id: int):
        sequence = self.sequences.get(role_id)
        if sequence is None:
            raise LoweringError(f"row resolves no sequence for role {role_id}")
        return sequence


@dataclass(frozen=True, slots=True)
class LoweredSegment:
    """One canonical segment draft (packed spans assigned at fill time)."""

    row_id: int
    operation_tag: OperationTag
    route_id: int
    query_count: int
    context_length: int
    attention_pattern: int
    attention_region_id: int
    kv_group: int
    binding_index: int
    cache_effect: CacheEffect
    cache_write_count: int
    branch_id: int
    branch_count: int
    candidate_count: int
    result_slot: int


@dataclass(frozen=True, slots=True)
class LoweredBatch:
    """Segments, reservation plan, and aggregate demand for one transaction.

    ``binding_commits`` and ``binding_row_ids`` align with the plan's binding
    order: the declared commit expression and owning row of each reserved
    binding, so delta derivation consumes the same closed schema source that
    produced the reservation.
    """

    segments: tuple[LoweredSegment, ...]
    plan: ReservationPlan
    rows: int
    tokens: int
    branches: int
    candidate_tokens: int
    write_token_begins: tuple[int, ...]
    binding_commits: tuple[CommitExpr, ...]
    binding_row_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]

    def demand(self, *, page_tokens: int) -> GraphCapacity:
        """The aggregate capacity vector this transaction requires."""

        bindings = sum(len(row.bindings) for row in self.plan.rows)
        page_references = 0
        for row in self.plan.rows:
            for binding in row.bindings:
                total = binding.sequence.committed_rows + binding.reserve_rows
                page_references += -(-total // page_tokens)
        return GraphCapacity(
            rows=self.rows,
            segments=len(self.segments),
            tokens=self.tokens,
            branches=self.branches,
            candidate_tokens=self.candidate_tokens,
            position_axes=1,
            visibility_payload_entries=0,
            residency=ResidencyBatchCapacity(
                bindings=bindings,
                page_references=page_references,
                tokens=self.tokens,
            ),
        )

    def fill_segment_table(self, capacity: GraphCapacity) -> SegmentTableArrays:
        """Pack the canonical table for one bucket (validated by the caller)."""

        if len(self.segments) > capacity.segments:
            raise LoweringError("transaction exceeds the bucket segment capacity")
        columns: dict[str, list[int]] = {
            name: [0] * capacity.segments for name in SegmentTableArrays.__dataclass_fields__
        }
        token_cursor = 0
        for index, segment in enumerate(self.segments):
            columns["segment_active"][index] = 1
            columns["row_id"][index] = segment.row_id
            columns["operation_tag"][index] = int(segment.operation_tag)
            columns["route_id"][index] = segment.route_id
            columns["token_begin"][index] = token_cursor
            columns["token_count"][index] = segment.query_count
            columns["query_begin"][index] = token_cursor
            columns["query_count"][index] = segment.query_count
            columns["context_length"][index] = segment.context_length
            columns["position_begin"][index] = token_cursor
            columns["position_count"][index] = segment.query_count
            columns["branch_id"][index] = segment.branch_id
            columns["branch_count"][index] = segment.branch_count
            columns["attention_pattern"][index] = segment.attention_pattern
            columns["attention_region_id"][index] = segment.attention_region_id
            columns["kv_group"][index] = segment.kv_group
            columns["kv_read_index"][index] = segment.binding_index
            columns["kv_write_index"][index] = (
                segment.binding_index if segment.cache_effect is not CacheEffect.READ_ONLY else 0
            )
            columns["cache_effect"][index] = int(segment.cache_effect)
            columns["cache_write_count"][index] = segment.cache_write_count
            columns["candidate_begin"][index] = 0
            columns["candidate_count"][index] = segment.candidate_count
            columns["result_slot"][index] = segment.result_slot
            token_cursor += segment.query_count
        local: dict[int, int] = {}
        for index, segment in enumerate(self.segments):
            columns["local_segment_id"][index] = local.get(segment.row_id, 0)
            local[segment.row_id] = local.get(segment.row_id, 0) + 1
        return SegmentTableArrays(**columns)


def lower_rows(
    rows: tuple[tuple[ExecuteRow, RoleSequences], ...],
    registration: FamilyCacheRegistration,
) -> LoweredBatch:
    """Lower one transaction's rows under the family's closed cache schema."""

    domain_by_role = {role.role_id: role.domain_id for role in registration.schema.roles}
    segments: list[LoweredSegment] = []
    demands: list[RowDemand] = []
    write_token_begins: list[int] = []
    binding_commits: list[CommitExpr] = []
    binding_row_ids: list[int] = []
    token_ids: list[int] = []
    positions: list[int] = []
    token_cursor = 0
    region_counter = 0
    max_branches = 0
    candidate_tokens = 0
    for row, roles in rows:
        operands = _operand_values(row)
        tag = operation_tag(row.operation)
        regions = [region for region in registration.schema.regions if region.operation_tag is tag]
        if not regions:
            raise LoweringError(f"row {row.row_id}: the family schema lowers no {tag.name} regions")
        bindings: list[SequenceBinding] = []
        stacked_rows: dict[int, int] = {}
        emitted = 0
        for region in regions:
            query_rows = region.query_rows.evaluate(operands)
            if query_rows == 0:
                continue
            instances = _instances(region, operands)
            max_branches = max(max_branches, len(instances) if len(instances) > 1 else 0)
            for branch_id, role_id in instances:
                emitted += 1
                region_counter += 1
                reserve_rows = region.reserve_rows.evaluate(operands)
                if role_id is None:
                    binding_index = 0
                    kv_group = 0
                    context_length = 0
                else:
                    sequence = roles.resolve(role_id)
                    stacked = stacked_rows.get(role_id, 0)
                    context_length = sequence.committed_rows + stacked
                    if region.cache_effect in (
                        CacheEffect.PERSISTENT_APPEND,
                        CacheEffect.TENTATIVE_APPEND,
                    ):
                        stacked_rows[role_id] = stacked + reserve_rows
                    bindings.append(
                        SequenceBinding(
                            sequence=sequence,
                            effect=region.cache_effect,
                            reserve_rows=(
                                reserve_rows
                                if region.cache_effect is not CacheEffect.READ_ONLY
                                else 0
                            ),
                        )
                    )
                    binding_index = _global_binding_index(demands, bindings)
                    kv_group = domain_by_role[role_id]
                    write_token_begins.append(token_cursor)
                    binding_commits.append(region.commit_rows)
                    binding_row_ids.append(row.row_id)
                is_tentative = region.cache_effect is CacheEffect.TENTATIVE_APPEND
                if is_tentative:
                    candidate_tokens = max(candidate_tokens, query_rows)
                run_cursor = 0
                for run in region.route_runs:
                    extent = run.extent.evaluate(operands)
                    if extent == 0:
                        continue
                    segments.append(
                        LoweredSegment(
                            row_id=row.row_id,
                            operation_tag=tag,
                            route_id=run.route_id,
                            query_count=extent,
                            context_length=context_length,
                            attention_pattern=int(region.attention_pattern),
                            attention_region_id=region_counter - 1,
                            kv_group=kv_group,
                            binding_index=binding_index,
                            cache_effect=region.cache_effect,
                            cache_write_count=(
                                extent if region.cache_effect is not CacheEffect.READ_ONLY else 0
                            ),
                            branch_id=branch_id if len(instances) > 1 else 0,
                            branch_count=(len(instances) if len(instances) > 1 else 0),
                            candidate_count=extent if is_tentative else 0,
                            result_slot=row.row_id,
                        )
                    )
                    run_cursor += extent
                if run_cursor != query_rows:
                    raise LoweringError(
                        f"row {row.row_id}: route runs cover {run_cursor} of "
                        f"{query_rows} query rows"
                    )
                ids, region_positions = _region_payload(
                    row.operation, region, query_rows, context_length
                )
                token_ids.extend(ids)
                positions.extend(region_positions)
                token_cursor += query_rows
        if emitted == 0:
            raise LoweringError(f"row {row.row_id}: no region evaluates to any query rows")
        demands.append(
            RowDemand(
                row_id=row.row_id,
                bindings=tuple(bindings),
                products=_published_products(row, operands),
                input_products=tuple(lease.lease_id for lease in row.product_leases),
            )
        )
    return LoweredBatch(
        segments=tuple(segments),
        plan=ReservationPlan(rows=tuple(demands)),
        rows=len(rows),
        tokens=token_cursor,
        branches=max_branches,
        candidate_tokens=candidate_tokens,
        write_token_begins=tuple(write_token_begins),
        binding_commits=tuple(binding_commits),
        binding_row_ids=tuple(binding_row_ids),
        token_ids=tuple(token_ids),
        positions=tuple(positions),
    )


def select_capacity(
    demand: GraphCapacity,
    capacities: tuple[GraphCapacity, ...],
) -> GraphCapacity:
    """Smallest configured capacity that dominates the demand (spec rule)."""

    dominating = [capacity for capacity in capacities if capacity.dominates(demand)]
    if not dominating:
        raise LoweringError("no configured graph capacity dominates the demand")
    return min(
        dominating,
        key=lambda capacity: (
            capacity.tokens,
            capacity.segments,
            capacity.rows,
            capacity.residency.page_references,
        ),
    )


# --------------------------------------------------------------------------- #


def _operand_values(row: ExecuteRow) -> dict[ExtentOperand, int]:
    operation = row.operation
    values = {
        ExtentOperand.INPUT_TOKEN_COUNT: 0,
        ExtentOperand.CANDIDATE_COUNT: 0,
        ExtentOperand.IMAGE_TOKEN_COUNT: 0,
        ExtentOperand.PRODUCT_ROW_COUNT: 0,
        ExtentOperand.CFG_BRANCH_COUNT: 0,
    }
    if isinstance(operation, SequenceStep):
        values[ExtentOperand.INPUT_TOKEN_COUNT] = len(operation.input_tokens)
        if operation.verification is not None:
            values[ExtentOperand.CANDIDATE_COUNT] = len(operation.verification.candidate_tokens)
        if row.product_leases:
            # Generated-image feedback appends an explicitly encoded region:
            # a sequence row carrying exactly one conditioning product lends
            # that product's extent to the image-token operand.
            if len(row.product_leases) != 1:
                raise LoweringError(
                    f"row {row.row_id}: sequence rows carry at most one conditioning product"
                )
            values[ExtentOperand.IMAGE_TOKEN_COUNT] = row.product_leases[0].extent_rows
    elif isinstance(operation, FlowOperation):
        values[ExtentOperand.CFG_BRANCH_COUNT] = len(operation.branch_coefficients)
        extent = _product_extent(row, operation.input_product)
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = extent
        values[ExtentOperand.PRODUCT_ROW_COUNT] = extent
    elif isinstance(operation, EncodeStep):
        tokens = operation.grid[0] * operation.grid[1] * operation.grid[2]
        values[ExtentOperand.IMAGE_TOKEN_COUNT] = tokens
        values[ExtentOperand.PRODUCT_ROW_COUNT] = tokens
    elif isinstance(operation, MaterializeStep):
        values[ExtentOperand.PRODUCT_ROW_COUNT] = _product_extent(row, operation.input_product)
    return values


def _region_payload(
    operation: object,
    region: CacheRegionSpec,
    query_rows: int,
    context_length: int,
) -> tuple[list[int], list[int]]:
    """Packed token ids and positions for one region instance.

    Sequence regions carry the operation's real token spans and linear
    positions; verification regions carry the candidate span at candidate
    positions; non-token regions pack zero ids with region-local positions.
    """

    if isinstance(operation, SequenceStep):
        if (
            region.cache_effect is CacheEffect.TENTATIVE_APPEND
            and operation.verification is not None
        ):
            return (
                list(operation.verification.candidate_tokens),
                list(operation.verification.candidate_positions),
            )
        if operation.input_tokens:
            begin = operation.position_begin
            return (
                list(operation.input_tokens),
                list(range(begin, begin + len(operation.input_tokens))),
            )
    return [0] * query_rows, list(range(context_length, context_length + query_rows))


def _published_products(
    row: ExecuteRow,
    operands: dict[ExtentOperand, int],
) -> tuple[ProductDemand, ...]:
    """Encode and materialize rows publish one typed product at commit."""

    operation = row.operation
    if isinstance(operation, EncodeStep):
        return (
            ProductDemand(
                schema_id=operation.output_schema,
                rows=operands[ExtentOperand.IMAGE_TOKEN_COUNT],
                producer=row.session,
            ),
        )
    if isinstance(operation, MaterializeStep):
        return (
            ProductDemand(
                schema_id=operation.output_schema,
                rows=operands[ExtentOperand.PRODUCT_ROW_COUNT],
                producer=row.session,
            ),
        )
    return ()


def _product_extent(row: ExecuteRow, lease_id: int) -> int:
    for lease in row.product_leases:
        if lease.lease_id == lease_id:
            return lease.extent_rows
    raise LoweringError(f"row {row.row_id} names input product {lease_id} without its lease")


def _instances(
    region: CacheRegionSpec,
    operands: dict[ExtentOperand, int],
) -> list[tuple[int, int | None]]:
    """``(branch_id, role_id-or-None)`` per region instance."""

    selector = region.role
    if selector.kind is RoleSelectorKind.NO_CACHE:
        return [(0, None)]
    if selector.kind is RoleSelectorKind.FIXED:
        return [(0, selector.role_id)]
    branch_count = operands[ExtentOperand.CFG_BRANCH_COUNT]
    if branch_count <= 0:
        raise LoweringError("branch-role regions require a CFG branch count")
    if branch_count > len(selector.branch_role_ids):
        raise LoweringError(f"{branch_count} CFG branches exceed the declared branch roles")
    return [(branch, selector.branch_role_ids[branch]) for branch in range(branch_count)]


def _global_binding_index(
    demands: list[RowDemand],
    bindings: list[SequenceBinding],
) -> int:
    """1-based plan-order binding index (0 is the sink binding)."""

    return sum(len(row.bindings) for row in demands) + len(bindings)


# ---------------------
# Standard transaction executor over family adapters
# ---------------------


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


# ---------------------
# Decode token/position relays (device-resident sequence feedback)
# ---------------------


class TextDecodeRelay:
    """Publishes and consumes device-resident decode token/position relays."""

    def publish_sample(
        self,
        state: "RequestState",
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> torch.Tensor:
        device_token = token_tensor.detach().reshape(1)
        if device_token.device.type == "cuda":
            device_token.record_stream(torch.cuda.current_stream(device_token.device))
        state.decode_relay.token_id = None if token_id is None else int(token_id)
        state.decode_relay.token_tensor = device_token
        return device_token

    def publish_position(
        self,
        state: "RequestState",
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> torch.Tensor:
        device_position = position_tensor.detach().reshape(1)
        if device_position.device.type == "cuda":
            device_position.record_stream(torch.cuda.current_stream(device_position.device))
        state.decode_relay.position_id = int(position_id)
        state.decode_relay.position_tensor = device_position
        return device_position

    def publish_positions(
        self,
        states: Sequence["RequestState"],
        *,
        position_ids: Sequence[int],
        device: torch.device | str,
    ) -> torch.Tensor:
        """Publish one contiguous batch of device-resident position relays.

        CUDA scalar construction from Python values performs a blocking host-to-device
        transfer on the current stream. Decode calls this after graph replay, so doing
        that once per row serializes the host behind every forward. Stage the complete
        position vector in pinned memory and enqueue one non-blocking copy instead.
        """

        if len(states) != len(position_ids):
            raise invalid_descriptor("decode position relay states and ids must align")
        resolved_device = canonical_device(device)
        count = len(position_ids)
        if resolved_device.type == "cuda":
            cpu = cpu_int_staging_buffer(
                count,
                dtype=torch.long,
                pin=True,
                name="decode_position_relay",
            )
            fill_cpu_ints(cpu, [int(position_id) for position_id in position_ids])
            positions = copy_cpu_to_device(
                cpu,
                device=resolved_device,
                non_blocking=is_pinned(cpu),
                slot=None,
                name="decode_position_relay",
            )
        else:
            positions = torch.tensor(
                [int(position_id) for position_id in position_ids],
                dtype=torch.long,
                device=resolved_device,
            )
        for row, (state, position_id) in enumerate(zip(states, position_ids, strict=True)):
            self.publish_position(
                state,
                position_id=int(position_id),
                position_tensor=positions[row : row + 1],
            )
        return positions

    def publish_deferred_sample_id_if_current(
        self,
        state: "RequestState",
        *,
        token_id: int,
        relay_token_tensor: torch.Tensor,
    ) -> bool:
        current = state.decode_relay.token_tensor
        if not _same_tensor(current, relay_token_tensor):
            return False
        state.decode_relay.token_id = int(token_id)
        return True

    def consume_token(
        self,
        state: "RequestState",
        *,
        expected_token_id: int | None,
        device: torch.device,
        token_source: str = "wire",
        require: bool = False,
        stats: "ForwardStats | None" = None,
    ) -> torch.Tensor | None:
        source = str(token_source or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        from_last_sampled = source == "last_sampled"
        relay = state.decode_relay
        relay_token_id = getattr(relay, "token_id", None)
        relay_token_tensor = getattr(relay, "token_tensor", None)
        if relay_token_id is None and not from_last_sampled:
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if (
            not from_last_sampled
            and expected_token_id is not None
            and relay_token_id is not None
            and int(relay_token_id) != int(expected_token_id)
        ):
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if not isinstance(relay_token_tensor, torch.Tensor):
            _bump_stat(stats, "text_decode_token_relay_misses")
            return self._missing_token(require=require)
        if relay_token_tensor.dtype != torch.long or not _same_device(
            relay_token_tensor.device,
            device,
        ):
            _bump_stat(stats, "text_decode_token_relay_misses")
            if from_last_sampled or require:
                raise invalid_descriptor(
                    "decode op requested token_source='last_sampled' but the relay tensor is on the wrong device"
                )
            return None
        return relay_token_tensor.reshape(1)

    def consume_position(
        self,
        state: "RequestState",
        *,
        expected_position_id: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        relay = state.decode_relay
        relay_position_id = getattr(relay, "position_id", None)
        relay_position_tensor = getattr(relay, "position_tensor", None)
        if (
            relay_position_id is None
            or int(relay_position_id) != int(expected_position_id)
            or not isinstance(relay_position_tensor, torch.Tensor)
            or relay_position_tensor.dtype != torch.long
            or not _same_device(relay_position_tensor.device, device)
        ):
            return None
        return relay_position_tensor.reshape(1)

    def replace_inputs(
        self,
        text: "TextBatch",
        request_states: "RequestStateTable",
        device: torch.device,
    ) -> dict[int, torch.Tensor] | None:
        replacements: dict[int, torch.Tensor] = {}
        flat_idx = 0
        for req_id, tokens, op in zip(text.req_ids, text.token_ids, text.ops):
            source = str(op.get("token_source") or "wire")
            if source not in {"wire", "last_sampled"}:
                raise invalid_descriptor(f"unsupported text token_source {source!r}")
            if source == "last_sampled":
                if len(tokens) != 1:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but does not have exactly one token"
                    )
                state = request_states.get(int(req_id))
                relay_token = self.consume_token(
                    state,
                    expected_token_id=None,
                    device=device,
                    token_source=source,
                    require=True,
                )
                if relay_token is None:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but no relay token is available"
                    )
                replacements[flat_idx] = relay_token.reshape(1)
            flat_idx += len(tokens)
        return replacements or None

    def resolve_decode_batch(
        self,
        text: "TextBatch",
        request_states: "RequestStateTable",
        device: torch.device,
        *,
        stats: "ForwardStats | None" = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        relay_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        position_complete = True
        for req_id, tokens, pos_range, op in zip(
            text.req_ids,
            text.token_ids,
            text.pos_ranges,
            text.ops,
        ):
            if len(tokens) != 1:
                _bump_stat(stats, "text_decode_token_relay_misses")
                return None, None
            state = request_states.get(int(req_id))
            source = str(op.get("token_source") or "wire")
            token = self.consume_token(
                state,
                expected_token_id=int(tokens[0]),
                device=device,
                token_source=source,
                require=source == "last_sampled",
                stats=stats,
            )
            if token is None:
                return None, None
            relay_rows.append(token.reshape(1))
            position = self.consume_position(
                state,
                expected_position_id=int(pos_range[0]),
                device=device,
            )
            if position is None:
                position_complete = False
                continue
            position_rows.append(position.reshape(1))
        _bump_stat(stats, "text_decode_token_relay_hits", len(relay_rows))
        if position_complete and len(position_rows) == len(relay_rows):
            _bump_stat(stats, "text_decode_position_relay_hits", len(position_rows))
            relay_positions = coalesce_one_token_rows(position_rows)
        else:
            _bump_stat(stats, "text_decode_position_relay_misses")
            relay_positions = None
        return coalesce_one_token_rows(relay_rows), relay_positions

    def attach_last_sampled_to_op(self, op: dict[str, Any], state: "RequestState") -> None:
        relay = state.decode_relay
        tensor = relay.token_tensor
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
            )
        op["token_tensor"] = tensor
        if relay.token_id is not None:
            op["token_ids"] = [int(relay.token_id)]

    @staticmethod
    def _missing_token(*, require: bool) -> None:
        if require:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
            )
        return None


def _same_tensor(lhs: Any, rhs: torch.Tensor) -> bool:
    if not isinstance(lhs, torch.Tensor):
        return False
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    return int(lhs.data_ptr()) == int(rhs.data_ptr())


def _same_device(lhs: torch.device | str, rhs: torch.device | str) -> bool:
    return canonical_device(lhs) == canonical_device(rhs)


def _bump_stat(stats: "ForwardStats | None", attr: str, delta: int = 1) -> None:
    if stats is None:
        return
    setattr(stats, attr, int(getattr(stats, attr)) + int(delta))


# ---------------------
# Decode burst execution
# ---------------------

StepOnce = Callable[..., list[Any]]


class DecodeBurstExecutor:
    """Runs one or more decode bursts through relay-backed one-token steps."""

    def __init__(self, step_once: StepOnce, *, relay_placeholder_token_id: int = -1) -> None:
        self._step_once = step_once
        self._relay_placeholder_token_id = int(relay_placeholder_token_id)

    def run(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        if len(first_ops) == 1:
            return [
                self._run_one(
                    dict(first_ops[0]),
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                )
            ]
        return self._run_many(
            first_ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )

    def _run_one(
        self,
        first_op: dict[str, Any],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        count = _positive_int(first_op.get("decode_token_count") or 1, "decode_token_count")
        stop_ids = set(_int_list(first_op.get("decode_stop_token_ids") or []))
        terminal_stop = first_op.get("decode_stop_terminal") is True
        tokens: list[int] = []
        last: dict[str, Any] = {}
        op = dict(first_op)
        op["decode_token_count"] = 1
        op["decode_stop_token_ids"] = []

        def resolve(out: Any) -> dict[str, Any]:
            result = _seq_result_dict(out)
            tok = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
            tokens.append(tok)
            return result

        pending: Any = None
        launched = 0
        while launched < count:
            out = self._launch_one(op, request_states, model)
            launched += 1
            if pending is not None:
                last = resolve(pending)
                pending = None
                if not terminal_stop and tokens[-1] in stop_ids:
                    out = None
                    break
            pending = out
            if launched >= count:
                break
            next_pos = _next_decode_position(op)
            op = self._next_relay_op(first_op, next_pos)
        if pending is not None:
            last = resolve(pending)

        if terminal_stop:
            tokens = _truncate_at_stop(tokens, stop_ids)
        result: dict[str, Any] = (
            {"req_id": _positive_int(first_op.get("req_id"), "req_id", minimum=0)}
            if terminal_stop
            else dict(last)
        )
        result["sampled_token_id"] = tokens[-1]
        result["sampled_token_ids"] = tokens
        return result

    def _run_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: Any,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for op in first_ops:
            op_dict = dict(op)
            count = _positive_int(op_dict.get("decode_token_count") or 1, "decode_token_count")
            first = dict(op_dict)
            first["decode_token_count"] = 1
            first["decode_stop_token_ids"] = []
            states.append(
                {
                    "op": op_dict,
                    "last_op": first,
                    "requested": count,
                    "launched": 0,
                    "stop_ids": set(_int_list(op_dict.get("decode_stop_token_ids") or [])),
                    "terminal_stop": op_dict.get("decode_stop_terminal") is True,
                    "tokens": [],
                    "last": None,
                    "pending": None,
                    "done": False,
                }
            )

        while any(not state["done"] for state in states):
            iter_ops: list[dict[str, Any]] = []
            iter_indexes: list[int] = []
            for index, state in enumerate(states):
                if state["done"] or int(state["launched"]) >= int(state["requested"]):
                    continue
                if int(state["launched"]) == 0:
                    op = dict(state["last_op"])
                else:
                    op = self._next_relay_op(state["op"], _next_decode_position(state["last_op"]))
                state["last_op"] = op
                iter_indexes.append(index)
                iter_ops.append(op)
            if not iter_ops:
                break
            iter_outputs = self._launch_many(iter_ops, request_states, model)
            for index, output in zip(iter_indexes, iter_outputs, strict=True):
                state = states[index]
                previous = state["pending"]
                state["pending"] = output
                state["launched"] = int(state["launched"]) + 1
                if previous is None:
                    continue
                result = _seq_result_dict(previous)
                token = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
                state["tokens"].append(token)
                state["last"] = result
                if not state["terminal_stop"] and token in state["stop_ids"]:
                    state["pending"] = None
                    state["done"] = True

        out: list[dict[str, Any]] = []
        for state in states:
            pending = state["pending"]
            if pending is not None:
                result = _seq_result_dict(pending)
                token = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
                state["tokens"].append(token)
                state["last"] = result
                state["pending"] = None
            tokens = [int(token) for token in state["tokens"]]
            if state["terminal_stop"]:
                tokens = _truncate_at_stop(tokens, state["stop_ids"])
            last = dict(state["last"] or {})
            if not tokens:
                raise invalid_descriptor("decode burst did not produce a sampled token")
            if state["terminal_stop"]:
                last = {"req_id": _positive_int(state["op"].get("req_id"), "req_id", minimum=0)}
            last["sampled_token_id"] = tokens[-1]
            if int(state["requested"]) > 1:
                last["sampled_token_ids"] = tokens
            out.append(last)
        return out

    def _launch_one(self, op: dict[str, Any], request_states: Any, model: Any) -> Any:
        return self._launch_many([op], request_states, model)[0]

    def _launch_many(self, ops: list[dict[str, Any]], request_states: Any, model: Any) -> list[Any]:
        text = UniForwardBatch.from_ops(ops).as_text()
        if text.mode != ForwardMode.DECODE:
            raise invalid_descriptor("decode burst can only launch decode ops")
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=True,
            defer_sampling=False,
            tensor_store=None,
        )

    def _next_relay_op(self, first_op: Mapping[str, Any], next_pos: int) -> dict[str, Any]:
        op = dict(first_op)
        op["new_block_ids"] = []
        op["token_ids"] = [self._relay_placeholder_token_id]
        op["token_source"] = "last_sampled"
        op["pos_range"] = [next_pos, next_pos + 1]
        op["decode_token_count"] = 1
        op["decode_stop_token_ids"] = []
        return op


def _seq_result_dict(output: Any) -> dict[str, Any]:
    if hasattr(output, "finalize") and callable(output.finalize):
        finalized = output.finalize()
        if isinstance(finalized, Mapping):
            return dict(finalized)
    if isinstance(output, ForwardOutputBase):
        return dict(output.to_seq_result())
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported text burst output type {type(output).__name__}")


def _positive_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise invalid_descriptor(f"{where} must be an integer >= {minimum}")
    return int(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise invalid_descriptor("decode_stop_token_ids must be a list")
    out: list[int] = []
    for idx, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise invalid_descriptor(f"decode_stop_token_ids[{idx}] must be a non-negative integer")
        out.append(int(item))
    return out


def _truncate_at_stop(tokens: list[int], stop_ids: set[int]) -> list[int]:
    if not stop_ids:
        return tokens
    out: list[int] = []
    for token in tokens:
        out.append(int(token))
        if int(token) in stop_ids:
            break
    return out


def _next_decode_position(op: Mapping[str, Any]) -> int:
    pos = op.get("pos_range") or [0, 0]
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
    return _positive_int(pos[1], "decode burst op.pos_range[1]", minimum=0)


# ---------------------
# Deferred text results
# ---------------------

_DECODE_RELAY = TextDecodeRelay()


class DeferredTextSeqResult:
    """One text seq-result whose CPU token id is finalized at response time."""

    req_id: int
    _row: int
    _state: RequestState
    _sampling_result: DeferredBatchedSamplingResult
    _relay_token_tensor: torch.Tensor
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        row: int,
        state: "RequestState",
        sampling_result: DeferredBatchedSamplingResult,
        relay_token_tensor: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_row", int(row))
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_sampling_result", sampling_result)
        object.__setattr__(self, "_relay_token_tensor", relay_token_tensor)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTextSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            sample = self._sampling_result.finalize().samples[self._row]
            tok, lp, top = sample
            _DECODE_RELAY.publish_deferred_sample_id_if_current(
                self._state,
                token_id=int(tok),
                relay_token_tensor=self._relay_token_tensor,
            )
            result: dict[str, Any] = {
                "req_id": self.req_id,
                "sampled_token_id": int(tok),
            }
            if lp is not None:
                result["sampled_logprob"] = lp
            if top:
                result["top_logprobs"] = top
            object.__setattr__(self, "_finalized", result)
            finalized = result
        return dict(finalized)

    def materialize_sampled_token_id(self) -> int:
        if self._finalized is not None:
            return int(self._finalized["sampled_token_id"])
        token_ids = self._sampling_result.token_ids()
        tok = int(token_ids[self._row])
        _DECODE_RELAY.publish_deferred_sample_id_if_current(
            self._state,
            token_id=tok,
            relay_token_tensor=self._relay_token_tensor,
        )
        return tok

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        ready = getattr(self._sampling_result, "ready", None)
        return bool(ready()) if callable(ready) else True

    def cuda_ready_group_key(self) -> int:
        return id(self._sampling_result)

    def cuda_ready_elapsed_us(self) -> int | None:
        elapsed = getattr(self._sampling_result, "cuda_ready_elapsed_us", None)
        return elapsed() if callable(elapsed) else None


class DeferredDecodeBurstSeqResult:
    """Decode-burst result whose final sampled token is still event-backed."""

    req_id: int
    _prefix_token_ids: tuple[int, ...]
    _pending: Any
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        prefix_token_ids: Sequence[int],
        pending: Any,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(
            self, "_prefix_token_ids", tuple(int(token) for token in prefix_token_ids)
        )
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            token_id = self._materialize_pending_token_id()
            token_ids = [*self._prefix_token_ids, token_id]
            finalized = {
                "req_id": self.req_id,
                "sampled_token_id": int(token_ids[-1]),
                "sampled_token_ids": [int(token) for token in token_ids],
            }
            object.__setattr__(self, "_finalized", finalized)
        return dict(finalized)

    def _materialize_pending_token_id(self) -> int:
        return _materialize_pending_token_id(self._pending)

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        ready = getattr(self._pending, "ready", None)
        return bool(ready()) if callable(ready) else True

    def cuda_ready_group_key(self) -> int:
        key = getattr(self._pending, "cuda_ready_group_key", None)
        return int(key()) if callable(key) else id(self._pending)

    def cuda_ready_elapsed_us(self) -> int | None:
        elapsed = getattr(self._pending, "cuda_ready_elapsed_us", None)
        return elapsed() if callable(elapsed) else None


class DeferredTerminalDecodeBurstSeqResult:
    """Terminal-stop decode-burst result with all sampled tokens deferred."""

    req_id: int
    _pending_tokens: tuple[Any, ...]
    _stop_token_ids: frozenset[int]
    _finalized: dict[str, Any] | None

    def __init__(
        self,
        *,
        req_id: int,
        pending_tokens: Sequence[Any],
        stop_token_ids: Iterable[int],
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_pending_tokens", tuple(pending_tokens))
        object.__setattr__(
            self, "_stop_token_ids", frozenset(int(token) for token in stop_token_ids)
        )
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTerminalDecodeBurstSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        finalized = self._finalized
        if finalized is None:
            token_ids: list[int] = []
            for pending in self._pending_tokens:
                token_id = _materialize_pending_token_id(pending)
                token_ids.append(token_id)
                if token_id in self._stop_token_ids:
                    break
            if not token_ids:
                raise invalid_descriptor("terminal decode burst did not produce a sampled token")
            finalized = {
                "req_id": self.req_id,
                "sampled_token_id": int(token_ids[-1]),
                "sampled_token_ids": [int(token) for token in token_ids],
            }
            object.__setattr__(self, "_finalized", finalized)
        return dict(finalized)

    def ready(self) -> bool:
        if self._finalized is not None:
            return True
        for pending in self._pending_tokens:
            ready = getattr(pending, "ready", None)
            if callable(ready) and not bool(ready()):
                return False
        return True

    def cuda_ready_group_key(self) -> int:
        keys: list[int] = []
        for pending in self._pending_tokens:
            key = getattr(pending, "cuda_ready_group_key", None)
            keys.append(int(key()) if callable(key) else id(pending))
        return hash(tuple(keys))

    def cuda_ready_elapsed_us(self) -> int | None:
        if not self.ready():
            return None
        total = 0
        seen: set[int] = set()
        for pending in self._pending_tokens:
            key = getattr(pending, "cuda_ready_group_key", None)
            group_key = int(key()) if callable(key) else id(pending)
            if group_key in seen:
                continue
            seen.add(group_key)
            elapsed = getattr(pending, "cuda_ready_elapsed_us", None)
            value = elapsed() if callable(elapsed) else None
            if isinstance(value, int) and value > 0:
                total += value
        return total if total > 0 else None


def _materialize_pending_token_id(pending: Any) -> int:
    materialize = getattr(pending, "materialize_sampled_token_id", None)
    if callable(materialize):
        return int(materialize())
    finalize = getattr(pending, "finalize", None)
    finalized = finalize() if callable(finalize) else pending
    if isinstance(finalized, Mapping):
        return int(finalized["sampled_token_id"])
    return int(dict(finalized)["sampled_token_id"])


# ---------------------
# Candidate verification (speculative token acceptance)
# ---------------------

_KV_LANE = "text"
_SAMPLED_TOKEN_DEVICE_KEY = "sampled_token_device"
_SAMPLED_POSITION_DEVICE_KEY = "sampled_position_device"


def verify_speculative_tokens(
    driver: "TextDriver",
    model: Any,
    text: Any,
    request_states: "RequestStateTable",
) -> list[TextTokenOutput]:
    """Verify the draft tokens on ``text`` against the target model.

    Groups the per-row draft sequences by length, runs one rectangular
    ``target_verify`` forward per length through the system-built attention plan,
    and applies the accept rule per row — system policy over the thin model.
    """

    if text.mode != ForwardMode.DECODE:
        raise invalid_descriptor("spec_token_ids are only supported on decode text ops")
    if any(len(tokens) != 1 for tokens in text.token_ids):
        raise invalid_descriptor("speculative decode requires one committed input token per op")
    if driver.builder is None or driver.kv_pool is None:
        raise invalid_descriptor(
            "speculative verify requires the system ForwardBatchBuilder and KV pool"
        )
    ctx = get_forward_context()
    device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))

    grouped: dict[int, list[tuple[int, dict[str, Any], tuple[int, ...]]]] = {}
    for idx, (op, spec, tokens, pos_range) in enumerate(
        zip(text.ops, text.spec_token_ids, text.token_ids, text.pos_ranges)
    ):
        length = 1 + len(spec)
        start = int(pos_range[0])
        committed = [int(token) for token in tokens]
        extended = dict(op)
        extended["kind"] = TARGET_VERIFY_UND
        extended["token_ids"] = committed + [int(token) for token in spec]
        extended["pos_range"] = [start, start + length]
        extended.pop("spec_token_ids", None)
        grouped.setdefault(length, []).append((idx, extended, tuple(int(token) for token in spec)))

    results: list[dict[str, Any] | None] = [None] * len(text.ops)
    builder, kv_pool = driver._system_forward_runtime()
    for length, rows in grouped.items():
        extended_ops = [op for _, op, _ in rows]
        verify_text = UniForwardBatch.from_ops(extended_ops).as_text()
        fb = builder.build_text(
            verify_text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
        )
        if fb.input_ids is None or fb.positions is None:
            raise invalid_descriptor("speculative verification batch is missing text inputs")
        input_ids = fb.input_ids.reshape(len(rows), length)
        positions = fb.positions.reshape(len(rows), length)
        with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
            logits = model.forward(input_ids, positions, fb)
        for row, (original_idx, op, spec) in enumerate(rows):
            results[original_idx] = _verify_spec_row(
                logits[row],
                op,
                spec,
                request_states.get(int(op["req_id"])),
                stats=ctx.stats,
            )
    if any(result is None for result in results):
        raise invalid_descriptor("speculative verification missed a result row")

    cleaned: list[TextTokenOutput] = []
    for op, result in zip(text.ops, (r for r in results if r is not None)):
        cleaned.append(_finalize_row(op, dict(result), request_states))
    return cleaned


def _finalize_row(
    op: dict[str, Any],
    result: dict[str, Any],
    request_states: "RequestStateTable",
) -> TextTokenOutput:
    token_tensor = result.pop(_SAMPLED_TOKEN_DEVICE_KEY, None)
    position_tensor = result.pop(_SAMPLED_POSITION_DEVICE_KEY, None)
    req_id = result.get("req_id")
    token_id = result.get("sampled_token_id")
    if (
        isinstance(token_tensor, torch.Tensor)
        and isinstance(req_id, int)
        and isinstance(token_id, int)
    ):
        state = request_states.get(int(req_id))
        device_token = token_tensor.detach().reshape(1)
        if device_token.device.type == "cuda":
            device_token.record_stream(torch.cuda.current_stream(device_token.device))
        state.decode_relay.token_id = int(token_id)
        state.decode_relay.token_tensor = device_token
        if isinstance(position_tensor, torch.Tensor):
            pos_range = op.get("pos_range") or (0, 0)
            accepted = int(result.get("num_accepted_tokens") or 0)
            device_position = position_tensor.detach().reshape(1)
            if device_position.device.type == "cuda":
                device_position.record_stream(torch.cuda.current_stream(device_position.device))
            state.decode_relay.position_id = int(pos_range[0]) + 1 + accepted
            state.decode_relay.position_tensor = device_position
    return _spec_verify_token_output(result)


def _verify_spec_row(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    *,
    stats: ForwardStats | None = None,
) -> dict[str, Any]:
    sampling = dict(state.sampling or {})
    recent = list(op.get("recent_tokens") or [])
    allowed = op.get("allowed_tokens")
    suppress = op.get("suppress_tokens")
    n_logprobs = int(sampling.get("n_logprobs", 0) or 0)
    if _can_use_greedy_spec_verify_fast_path(sampling, recent, allowed, suppress):
        return _verify_spec_row_greedy(logits, op, spec, state, stats=stats)
    if _can_use_sglang_target_only_spec_verify(sampling):
        return _verify_spec_row_target_only(
            logits,
            op,
            spec,
            state,
            sampling,
            recent=recent,
            allowed=allowed,
            suppress=suppress,
            stats=stats,
        )
    return _verify_spec_row_sequential(
        logits,
        op,
        spec,
        state,
        sampling,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
        n_logprobs=n_logprobs,
        stats=stats,
    )


def _verify_spec_row_greedy(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    *,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    chosen = torch.argmax(logits, dim=-1)
    accepted = _accepted_greedy_prefix(chosen, spec)
    sampled_token_tensor = chosen[accepted : accepted + 1]
    sampled_token = int(sampled_token_tensor.detach().to("cpu").item())
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "greedy_device")
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": sampled_token,
        "num_accepted_tokens": int(accepted),
        _SAMPLED_TOKEN_DEVICE_KEY: sampled_token_tensor,
        _SAMPLED_POSITION_DEVICE_KEY: chosen.new_full((1,), next_pos, dtype=torch.long),
    }


def _verify_spec_row_target_only(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    sampling: dict[str, Any],
    *,
    recent: list[int],
    allowed: Any,
    suppress: Any,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    spec_sample = speculative_sample_target_only(
        logits[: len(spec) + 1].reshape(len(spec) + 1, -1),
        spec,
        sampling,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
    )
    accepted = int(spec_sample.num_accepted_tokens)
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "sglang_target_only")
    return {
        "req_id": int(op["req_id"]),
        "sampled_token_id": int(spec_sample.sampled_token_id),
        "num_accepted_tokens": int(accepted),
        _SAMPLED_TOKEN_DEVICE_KEY: spec_sample.sampled_token_device,
        _SAMPLED_POSITION_DEVICE_KEY: spec_sample.sampled_token_device.new_full(
            (1,), next_pos, dtype=torch.long
        ),
    }


def _verify_spec_row_sequential(
    logits: torch.Tensor,
    op: dict[str, Any],
    spec: tuple[int, ...],
    state: Any,
    sampling: dict[str, Any],
    *,
    recent: list[int],
    allowed: Any,
    suppress: Any,
    n_logprobs: int,
    stats: ForwardStats | None,
) -> dict[str, Any]:
    accepted = 0
    sampled_token = None
    sampled_token_tensor = None
    sampled_logprob = None
    top_logprobs = None
    for pos in range(len(spec) + 1):
        requested_logprobs = n_logprobs if n_logprobs > 0 else 0
        sampling_for_pos = dict(sampling)
        sampling_for_pos["n_logprobs"] = requested_logprobs
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits[pos].reshape(1, -1),
            [sampling_for_pos],
            [recent],
            [allowed],
            [suppress],
            generators=[state.device_rng(logits.device, stream="text_sampling")],
        )
        token, logprob, top = sampling_result.samples[0]
        if pos < len(spec) and int(token) == int(spec[pos]):
            accepted += 1
            recent.append(int(token))
            continue
        sampled_token = int(token)
        sampled_token_tensor = sampling_result.device_tokens[:1]
        sampled_logprob = logprob
        top_logprobs = top
        break
    if sampled_token is None:
        raise invalid_descriptor("speculative verification did not produce a sampled token")
    next_pos = _advance_spec_kv(state, op, accepted)
    _record_spec_verify_stats(stats, len(spec), accepted, "sequential_target_sample")
    result: dict[str, Any] = {
        "req_id": int(op["req_id"]),
        "sampled_token_id": sampled_token,
        "num_accepted_tokens": int(accepted),
    }
    if sampled_logprob is not None:
        result["sampled_logprob"] = sampled_logprob
    if top_logprobs:
        result["top_logprobs"] = top_logprobs
    if sampled_token_tensor is not None:
        result[_SAMPLED_TOKEN_DEVICE_KEY] = sampled_token_tensor
        result[_SAMPLED_POSITION_DEVICE_KEY] = sampled_token_tensor.new_full(
            (1,), next_pos, dtype=torch.long
        )
    return result


def _accepted_greedy_prefix(chosen: torch.Tensor, spec: tuple[int, ...]) -> int:
    if not spec:
        return 0
    draft = torch.tensor(spec, dtype=chosen.dtype, device=chosen.device)
    mismatch = torch.nonzero(chosen[: len(spec)] != draft, as_tuple=False)
    return int(mismatch[0].item()) if int(mismatch.numel()) > 0 else len(spec)


def _advance_spec_kv(state: Any, op: dict[str, Any], accepted: int) -> int:
    base_len = int((op.get("pos_range") or (0, 0))[0])
    next_pos = base_len + 1 + int(accepted)
    state.set_kv_length(next_pos, lane=_KV_LANE)
    return next_pos


def _spec_verify_token_output(result: dict[str, Any]) -> TextTokenOutput:
    req_id = result.get("req_id")
    token_id = result.get("sampled_token_id")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise invalid_descriptor("speculative verify output req_id must be an integer")
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise invalid_descriptor("speculative verify output sampled_token_id must be an integer")
    num_accepted = result.get("num_accepted_tokens")
    return TextTokenOutput(
        req_id=int(req_id),
        sampled_token_id=int(token_id),
        sampled_logprob=result.get("sampled_logprob"),
        top_logprobs=result.get("top_logprobs") or None,
        num_accepted_tokens=int(num_accepted) if num_accepted is not None else None,
    )


def _can_use_greedy_spec_verify_fast_path(
    sampling: dict[str, Any],
    recent: list[int],
    allowed: Any,
    suppress: Any,
) -> bool:
    if allowed or suppress or sampling.get("logit_bias"):
        return False
    if _generated_logprobs_requested(sampling):
        return False
    if float(sampling.get("temperature", 0.0) or 0.0) > 0.0:
        return False
    if float(sampling.get("min_p", 0.0) or 0.0) > 0.0:
        return False
    if int(sampling.get("top_k", 0) or 0) > 0:
        return False
    if float(sampling.get("top_p", 1.0) or 1.0) < 1.0:
        return False
    repetition = float(sampling.get("repetition_penalty", 1.0) or 1.0)
    frequency = float(sampling.get("frequency_penalty", 0.0) or 0.0)
    presence = float(sampling.get("presence_penalty", 0.0) or 0.0)
    return not recent or (repetition == 1.0 and frequency == 0.0 and presence == 0.0)


def _can_use_sglang_target_only_spec_verify(sampling: dict[str, Any]) -> bool:
    if _generated_logprobs_requested(sampling):
        return False
    return float(sampling.get("temperature", 0.0) or 0.0) > 0.0


def _generated_logprobs_requested(sampling: dict[str, Any]) -> bool:
    return (
        bool(sampling.get("return_logprobs", False))
        or int(sampling.get("n_logprobs", 0) or 0) > 0
        or bool(sampling.get("logprob_token_ids"))
    )


def _record_spec_verify_stats(
    stats: ForwardStats | None,
    draft_tokens: int,
    accepted_tokens: int,
    path: str,
) -> None:
    if stats is None:
        return
    draft = max(0, int(draft_tokens))
    accepted = max(0, min(int(accepted_tokens), draft))
    stats.spec_verify_rows += 1
    stats.spec_verify_draft_tokens += draft
    stats.spec_verify_accepted_tokens += accepted
    stats.spec_verify_rejected_tokens += max(0, draft - accepted)
    stats.spec_verify_committed_tokens += accepted + 1
    stats.record_spec_path(path)


# ---------------------
# Text sequence execution
# ---------------------


# Wire token id carried by pipelined-burst ``last_sampled`` ops whose real token
# still lives only in the device relay tensor. Deliberately invalid: any path
# that embeds the wire token instead of consuming the relay fails loudly.
_RELAY_PLACEHOLDER_TOKEN_ID = -1
_GRAPH_RUNNER_UNSET = object()


class _DecodeBurstGraphMiss(Exception):
    pass


@dataclass(frozen=True)
class TextForwardLogits:
    """Neural text forward result before sampling or request-state mutation."""

    logits: torch.Tensor
    req_ids: tuple[int, ...]
    cuda_ready_start_event: torch.cuda.Event | None = None


def text_input_id_replacements_from_relays(
    text: "TextBatch",
    request_states: RequestStateTable,
    device: torch.device,
) -> dict[int, torch.Tensor] | None:
    """Return flat input-token replacements for ``last_sampled`` decode rows.

    The async decode fast path keeps the last sampled token resident on device;
    a mixed extend+decode forward replaces only the decode rows' placeholder
    tokens by flat index (the contiguous-relay override is the pure-decode case).
    """

    return _DECODE_RELAY.replace_inputs(text, request_states, device)


def sample_logits_result(
    *,
    req_id: int,
    state: RequestState,
    logits: torch.Tensor,
    op: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one token from model-produced logits using the canonical pipeline."""

    if not isinstance(logits, torch.Tensor):
        raise invalid_descriptor("text logits output must contain a tensor")
    if logits.ndim == 0:
        raise invalid_descriptor("text logits tensor must have a vocabulary dimension")
    vocab_logits = logits.float()
    if vocab_logits.ndim > 1:
        vocab_logits = vocab_logits.reshape(-1, vocab_logits.shape[-1])[-1]
    sp = dict(state.sampling or {})
    tok, lp, top = sample_one_from_logits(
        vocab_logits,
        sp,
        recent=op.get("recent_tokens") or [],
        allowed=op.get("allowed_tokens"),
        suppress=op.get("suppress_tokens"),
        n_logprobs=int(sp.get("n_logprobs", 0) or 0),
        generator=state.device_rng(vocab_logits.device, stream="text_sampling"),
    )
    result: dict[str, Any] = {"req_id": int(req_id), "sampled_token_id": tok}
    if lp is not None:
        result["sampled_logprob"] = lp
    if top:
        result["top_logprobs"] = top
    return result


class TextDriver:
    """Run the system-managed text forward and own the post-model sampler.

    Constructed by the runner with the system collaborators it orchestrates:
    the ``ForwardBatchBuilder`` (GPU snapshot + residency + plan), the
    ``TextBackendGate`` (batched-vs-per-op), and the system-owned ``kv_pool``.
    The optional ``graph_runner`` captures/replays the decode/prefill graphs
    around the graph-unaware model.
    """

    def __init__(
        self,
        *,
        builder: "ForwardBatchBuilder | None" = None,
        gate: "TextBackendGate | None" = None,
        kv_pool: "PagedKVPool | None" = None,
        graph_runner: Any | None = None,
    ) -> None:
        self.builder = builder
        self.gate = gate
        self.kv_pool = kv_pool
        self.graph_runner = graph_runner

    def _system_forward_runtime(self) -> tuple["ForwardBatchBuilder", "PagedKVPool"]:
        if self.builder is None or self.kv_pool is None:
            raise invalid_descriptor("system-managed text forward requires a builder and KV pool")
        return self.builder, self.kv_pool

    @torch.inference_mode()
    def step(
        self,
        fb: "UniForwardBatch",
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            with profile_range("uniserve.text.prompt_logprobs"):
                return self._step_prompt_prefill(
                    text,
                    ops,
                    request_states,
                    model,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
        if any(text.spec_token_ids):
            pass

            with profile_range("uniserve.text.speculative_verify"):
                return verify_speculative_tokens(self, model, text, request_states)
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            with profile_range("uniserve.text.decode_burst"):
                return DecodeBurstExecutor(
                    self._step_once,
                    relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
                ).run(
                    ops,
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                )
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            # Compute in an order that keeps graph token-bucket padding legal
            # for the final row, but return results in the wire op order (the
            # response finalizer matches per-seq results to ops positionally).
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                results = self._step_once(
                    reordered,
                    list(reordered.ops),
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
                return [results[row_by_op[id(op)]] for op in ops]
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

    def forward_logits(
        self,
        fb: "UniForwardBatch",
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution and leave postprocessing to the forward stack."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            logits_batch, req_ids = self._forward_with_optional_padding_reorder(
                text,
                ops,
                request_states,
                model,
                store_position_relays=False,
            )
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def forward_logits_graph(
        self,
        fb: "UniForwardBatch",
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution only when a CUDA graph handles the batch."""

        if graph_runner is None and self.builder is not None:
            return None
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            if graph_runner is None:
                graph_result = self._forward(
                    model,
                    text,
                    request_states,
                    store_position_relays=False,
                    require_graph=True,
                )
            else:
                graph_result = self._forward_graph_with_optional_padding_reorder(
                    text,
                    ops,
                    request_states,
                    model,
                    graph_runner=graph_runner,
                    store_position_relays=False,
                )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def forward_graph_result(
        self,
        fb: "UniForwardBatch",
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> ForwardResult | None:
        """Run a text batch only when graph-backed execution can cover it."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            outputs = self._decode_burst_graph_many(
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
            )
            if outputs is None:
                return None
            return ForwardResult(runtime_outputs=tuple(outputs))
        text_result = self.forward_logits_graph(
            fb,
            request_states,
            model,
            graph_runner=graph_runner,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
        )
        if text_result is None:
            return None
        expected_req_ids = tuple(int(op["req_id"]) for op in ops)
        req_ids = tuple(int(req_id) for req_id in text_result.req_ids)
        if req_ids != expected_req_ids:
            raise invalid_descriptor("text graph result req_ids must align with forward ops")
        return ForwardResult(
            text_logits=text_result.logits,
            text_cuda_ready_start_event=text_result.cuda_ready_start_event,
        )

    def _step_once(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            forward_result = self._forward(model, text, request_states)
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        logits_batch, req_ids = forward_result
        # KV-length advance is system-owned now (derived from seq_lens), not the
        # model's job.
        self._advance_kv_lengths(text, request_states)
        if defer_sampling and tensor_store is not None:
            with profile_range("uniserve.text.publish_logits"):
                return self._publish_logits(ops, req_ids, logits_batch, tensor_store)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    def _decode_burst(
        self,
        first_op: dict[str, Any],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            [first_op],
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )[0]

    def _decode_burst_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            list(first_ops),
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )

    def _decode_burst_graph_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]] | None:
        def step_once_graph(
            text: "TextBatch",
            ops: list[Mapping[str, Any]],
            request_states: RequestStateTable,
            model: Any,
            *,
            defer_cpu_results: bool = False,
            defer_sampling: bool = False,
            tensor_store: Any | None = None,
        ) -> list[Any]:
            del tensor_store
            outputs = self._step_once_graph(
                text,
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
                defer_sampling=defer_sampling,
            )
            if outputs is None:
                raise _DecodeBurstGraphMiss
            return outputs

        try:
            return DecodeBurstExecutor(
                step_once_graph,
                relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
            ).run(
                list(first_ops),
                request_states,
                model,
                defer_cpu_results=defer_cpu_results,
            )
        except _DecodeBurstGraphMiss:
            return None

    def _step_once_graph(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> list[Any] | None:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            graph_result = self._forward(
                model,
                text,
                request_states,
                graph_runner=graph_runner,
                require_graph=True,
            )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        self._advance_kv_lengths(text, request_states)
        if defer_sampling:
            return None
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    # ---- forward ---------------------------------------------------------

    def _forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if self.builder is None or self.kv_pool is None:
            # Self-managing text models (the HF day-zero fallback and the
            # composed multimodal programs whose KV is intrinsically coupled to
            # their modality FSM) declare no ``kv_cache_spec``; the system owns no
            # pool for them. They expose their own per-op text logits and the
            # driver still owns the post-model sampler.
            if require_graph:
                return self._model_owned_kv_forward_graph(model, text, request_states)
            return self._model_owned_kv_forward(model, text, request_states)
        ctx = get_forward_context()
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        batched = self.gate is not None and self.gate.batched_capable(
            text, attention_preference=ctx.attention_preference
        )
        if batched:
            return self._forward_batched(
                model,
                text,
                request_states,
                ctx,
                device,
                store_position_relays=store_position_relays,
                graph_runner=graph_runner,
                require_graph=require_graph,
            )
        if require_graph:
            return None
        return self._forward_per_op(
            model,
            text,
            request_states,
            ctx,
            device,
            store_position_relays=store_position_relays,
        )

    def _forward_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]]:
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                forward_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                )
                if forward_result is None:
                    raise invalid_descriptor("reordered eager text forward did not produce logits")
                logits, req_ids = forward_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        forward_result = self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
        )
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        return forward_result

    def _forward_graph_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if text.mode == ForwardMode.MIXED:
            reordered = graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                graph_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                    graph_runner=graph_runner,
                    require_graph=True,
                )
                if graph_result is None:
                    return None
                logits, req_ids = graph_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        return self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
            graph_runner=graph_runner,
            require_graph=True,
        )

    def _model_owned_kv_forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run a self-managing model's per-op text logits and stack them.

        Accepts a batched ``run_text_logits_batch`` or a per-op
        ``run_text_logits``; both return a raw logits tensor per op, coerced to
        one ``[vocab]`` row. Req ids come from the op order, not the tensors.

        The driver owns the decode-relay lookup: ``last_sampled`` ops get the
        device relay tensor attached as ``op['token_tensor']`` (and the resolved
        id when the CPU copy has landed) so the model side can consume the
        sampled token without a GPU synchronize and without reaching into
        system request state.
        """

        ops = self._model_owned_ops(text, request_states)
        outputs = list(model.run_text_logits_batch(ops))
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    def _model_owned_kv_forward_graph(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]] | None:
        graph_logits = getattr(model, "try_run_graph_logits_batch", None)
        if not callable(graph_logits):
            return None
        ops = self._model_owned_ops(text, request_states)
        outputs = graph_logits(ops)
        if outputs is None:
            return None
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    @staticmethod
    def _model_owned_ops(
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> list[dict[str, Any]]:
        ops = [dict(op) for op in text.ops]
        for op in ops:
            if str(op.get("token_source") or "wire") != "last_sampled":
                continue
            _DECODE_RELAY.attach_last_sampled_to_op(op, request_states.get(int(op["req_id"])))
        return ops

    @staticmethod
    def _coerce_logits_row(logits: Any) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise invalid_descriptor("self-managing text model must return a logits tensor")
        return logits.reshape(-1, logits.shape[-1])[-1]

    def _forward_batched(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        stats = ctx.stats
        builder, kv_pool = self._system_forward_runtime()
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        start = component_timer_start(stats)
        # Pure decode uses the contiguous-relay override (fast); a mixed
        # extend+decode batch replaces only its last_sampled decode rows by index.
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode == ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(
                text, request_states, device
            )
        elif text.mode == ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(
                text, request_states, device
            )
        record_component_elapsed(stats, "text_decode_relay", start)
        start = component_timer_start(stats)
        padded = self._graph_padded_num_tokens(text, ctx, graph_runner=active_graph_runner)
        fb = builder.build_text(
            text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
            input_ids_override=relay_input_ids,
            positions_override=relay_positions,
            input_ids_replacements=relay_replacements,
            padded_num_tokens=padded,
        )
        record_component_elapsed(stats, "text_build_batch", start)
        input_ids, positions = self._reshape_inputs(fb, text)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.model_forward"):
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
                logits = self._run_model_forward(
                    model,
                    input_ids,
                    positions,
                    fb,
                    ctx,
                    graph_runner=active_graph_runner,
                    require_graph=require_graph,
                )
        if logits is None:
            return None
        record_component_elapsed(stats, "text_model_forward", start)
        if store_position_relays:
            start = component_timer_start(stats)
            self._store_decode_position_relays(text, fb, request_states)
            record_component_elapsed(stats, "text_decode_position_store", start)
        return logits, [int(req_id) for req_id in text.req_ids]

    def _forward_per_op(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
    ) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        next_positions: list[tuple[int, int]] = []
        builder, kv_pool = self._system_forward_runtime()
        for req_id, tokens, pos_range, op in zip(
            text.req_ids, text.token_ids, text.pos_ranges, text.ops
        ):
            state = request_states.get(int(req_id))
            relay = self._per_op_relay_input(op, tokens, state, device)
            fb = builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=int(req_id),
                mode=text.mode,
                device=device,
                kv_pool=kv_pool,
                request_states=request_states,
                input_ids_override=relay,
            )
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan, kv_pool=kv_pool)):
                logits = model.forward(fb.input_ids, fb.positions, fb)
            rows.append(logits.reshape(-1, logits.shape[-1])[-1])
            next_positions.append((int(req_id), int(pos_range[1])))
        if store_position_relays and text.mode == ForwardMode.DECODE:
            for (req_id, position), op_positions in zip(next_positions, text.pos_ranges):
                tensor = torch.tensor([position], dtype=torch.long, device=device)
                self._store_position_relay(
                    request_states.get(int(req_id)),
                    position_id=position,
                    position_tensor=tensor,
                )
        return torch.stack(rows, dim=0), [int(req_id) for req_id in text.req_ids]

    def _run_model_forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        fb: "ForwardBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> torch.Tensor | None:
        # System-owned CUDA graphs capture/replay around the graph-unaware model;
        # a miss (or graphs disabled) falls through to the eager forward.
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is not None:
            logits = active_graph_runner.maybe_run(model, input_ids, positions, fb, ctx)
            if logits is not None:
                return logits
        if require_graph:
            return None
        return model.forward(input_ids, positions, fb)

    def _reshape_inputs(
        self, fb: "ForwardBatch", text: "TextBatch"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the model input geometry (flat varlen vs rectangular) for the mode.

        Decode is rectangular ``[batch, 1]`` (the batched-decode kernels); ragged
        or padded extend is flat ``[total]`` (gathered logits via
        ``last_token_indices``); equal-length extend is rectangular ``[batch, L]``;
        mixed stays flat varlen.
        """

        mode = text.mode
        if fb.input_ids is None or fb.positions is None:
            raise invalid_descriptor("text forward batch is missing input ids or positions")
        if mode == ForwardMode.DECODE:
            return fb.input_ids.reshape(fb.batch_size, 1), fb.positions.reshape(fb.batch_size, 1)
        if mode == ForwardMode.EXTEND:
            return fb.input_ids, fb.positions
        return fb.input_ids, fb.positions

    def _graph_padded_num_tokens(
        self,
        text: "TextBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
    ) -> int | None:
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is None:
            return None
        return active_graph_runner.padded_num_tokens(
            text, attention_preference=ctx.attention_preference
        )

    def _advance_kv_lengths(self, text: "TextBatch", request_states: RequestStateTable) -> None:
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane=_KV_LANE)

    # ---- deferred sampling -----------------------------------------------

    def _publish_logits(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
        logits_batch: torch.Tensor,
        tensor_store: Any,
    ) -> list[dict[str, Any]]:
        if logits_batch.ndim != 2 or int(logits_batch.shape[0]) != len(ops):
            raise invalid_descriptor("deferred-sampler logits must be shaped [ops, vocab]")
        if logits_batch.is_cuda:
            torch.cuda.synchronize(logits_batch.device)
        results: list[dict[str, Any]] = []
        for row, op in enumerate(ops):
            handle = tensor_store.publish(logits_batch[row].contiguous(), "logits")
            result: dict[str, Any] = {"req_id": int(op["req_id"]), "logits_handle": int(handle)}
            locator = tensor_store.locator_of(handle)
            if locator is not None:
                result["locator"] = base64.b64encode(locator).decode("ascii")
            results.append(result)
        return results

    # ---- sampling --------------------------------------------------------

    def _step_prompt_prefill(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> list[Any]:
        if any(str(op.get("kind")) != "prefill_und" for op in ops):
            raise invalid_descriptor("prompt scoring is valid only for prefill_und operations")

        final_logits: list[torch.Tensor] = []
        prompt_scores: list[list[list[tuple[int, float, int]]] | None] = []
        for row, (op, req_id, tokens, pos_range) in enumerate(
            zip(ops, text.req_ids, text.token_ids, text.pos_ranges, strict=True)
        ):
            del row
            state = request_states.get(int(req_id))
            logits = self._forward_prompt_op(
                model,
                op,
                tokens,
                pos_range,
                int(req_id),
                request_states,
            )
            rows = logits.reshape(-1, logits.shape[-1])
            if bool(op.get("return_all_logits")) and int(rows.shape[0]) != len(tokens):
                raise invalid_descriptor(
                    "prompt-scoring model output must contain one logits row per input token"
                )
            chunk_scores = self._score_prompt_chunk(state, rows, tokens)
            prompt_scores.append(chunk_scores or None)
            final_logits.append(rows[-1])
            state.set_kv_length(int(pos_range[1]), lane=_KV_LANE)

        logits_batch = torch.stack(final_logits, dim=0)
        if defer_sampling:
            if tensor_store is None:
                raise invalid_descriptor("deferred prompt sampling requires a tensor store")
            results = self._publish_logits(ops, text.req_ids, logits_batch, tensor_store)
            for result, prompt_score in zip(results, prompt_scores, strict=True):
                if prompt_score is not None:
                    result["prompt_logprobs"] = prompt_score
            return results

        outputs: list[TextTokenOutput] = []
        for row, (op, req_id, prompt_score) in enumerate(
            zip(ops, text.req_ids, prompt_scores, strict=True)
        ):
            state = request_states.get(int(req_id))
            sample = sample_one_from_logits(
                logits_batch[row],
                dict(state.sampling or {}),
                recent=op.get("recent_tokens") or [],
                allowed=op.get("allowed_tokens"),
                suppress=op.get("suppress_tokens"),
                n_logprobs=int(state.sampling.get("n_logprobs", 0) or 0),
                generator=state.device_rng(
                    logits_batch.device,
                    stream="text_sampling",
                ),
            )
            self._store_sampled_token_relay(
                state,
                token_id=int(sample.token_id),
                token_tensor=torch.tensor(
                    [int(sample.token_id)], dtype=torch.long, device=logits_batch.device
                ),
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=(
                        [
                            (int(item[0]), float(item[1]), int(item[2]))
                            for item in sample.top_logprobs
                        ]
                        if sample.top_logprobs
                        else None
                    ),
                    prompt_logprobs=prompt_score,
                )
            )
        return outputs

    def _forward_prompt_op(
        self,
        model: Any,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        pos_range: tuple[int, int],
        req_id: int,
        request_states: RequestStateTable,
    ) -> torch.Tensor:
        if not tokens:
            raise invalid_descriptor("prompt-scoring prefill operation has no token ids")
        if self.builder is None or self.kv_pool is None:
            predecessor = getattr(model, "prompt_predecessor_logits", None)
            if bool(op.get("return_all_logits")) and callable(predecessor):
                previous_logits = predecessor(req_id)
                if isinstance(previous_logits, torch.Tensor) and previous_logits.ndim > 0:
                    request_states.get(req_id).prompt_last_logits = previous_logits.reshape(
                        -1, previous_logits.shape[-1]
                    )[-1].detach()
            logits = model.run_text_logits(dict(op))
        else:
            ctx = get_forward_context()
            device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
            batch = self.builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=req_id,
                mode=ForwardMode.EXTEND,
                device=device,
                kv_pool=self.kv_pool,
                request_states=request_states,
            )
            batch.return_all_logits = bool(op.get("return_all_logits"))
            with use_forward_context(
                replace(ctx, attention_plan=batch.attn_plan, kv_pool=self.kv_pool)
            ):
                logits = model.forward(batch.input_ids, batch.positions, batch)
        if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
            raise invalid_descriptor("prompt-scoring model output must be a logits tensor")
        return logits

    @staticmethod
    def _score_prompt_chunk(
        state: RequestState,
        logits: torch.Tensor,
        tokens: tuple[int, ...],
    ) -> list[list[tuple[int, float, int]]]:
        sampling = dict(state.sampling or {})
        if not (
            bool(sampling.get("return_prompt_logprobs"))
            or int(sampling.get("n_prompt_logprobs", 0) or 0) > 0
        ):
            return []
        predictors: list[torch.Tensor] = []
        targets: list[int] = []
        if state.prompt_last_logits is not None:
            predictors.append(state.prompt_last_logits.reshape(1, -1))
            targets.append(int(tokens[0]))
        if len(tokens) > 1:
            predictors.append(logits[:-1])
            targets.extend(int(token_id) for token_id in tokens[1:])
        state.prompt_last_logits = logits[-1].detach()
        if not predictors:
            return []
        return score_prompt_token_logprobs(
            torch.cat(predictors, dim=0),
            targets,
            n_logprobs=int(sampling.get("n_prompt_logprobs", 0) or 0),
            logprob_token_ids=sampling.get("logprob_token_ids") or (),
        )

    def _sample_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
        logits_batch: torch.Tensor,
        request_states: RequestStateTable,
        stats: ForwardStats | None,
        start: int,
        *,
        defer_cpu_results: bool = False,
        cuda_ready_start_event: torch.cuda.Event | None = None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op, req_id in zip(ops, req_ids):
            state = request_states.get(req_id)
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        with profile_range("uniserve.text.apply_sampling"):
            sampling_result = apply_sampling_batched_with_device_tokens(
                logits_batch,
                params,
                recent,
                allowed,
                suppress,
                generators=generators,
                defer_cpu=defer_cpu_results,
                enable_cuda_timing=cuda_ready_start_event is not None,
            )
        if is_deferred_sampling_result(sampling_result):
            sampling_result.set_ready_start_event(cuda_ready_start_event)
            out: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(req_id)
                relay_token_tensor = sampling_result.device_tokens[row : row + 1]
                self._store_sampled_token_relay(
                    state, token_id=None, token_tensor=relay_token_tensor
                )
                out.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=relay_token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return out

        immediate_result = finalize_sampling_result(sampling_result)
        samples = immediate_result.samples
        out = []
        for row, (req_id, (tok, lp, top)) in enumerate(zip(req_ids, samples)):
            self._store_sampled_token_relay(
                request_states.get(req_id),
                token_id=int(tok),
                token_tensor=immediate_result.device_tokens[row : row + 1],
            )
            out.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=tok,
                    sampled_logprob=lp,
                    top_logprobs=(
                        [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                        if top
                        else None
                    ),
                )
            )
        record_component_elapsed(stats, "text_sample", start)
        return out

    # ---- decode relays ---------------------------------------------------

    def _per_op_relay_input(
        self,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        state: RequestState,
        device: torch.device,
    ) -> torch.Tensor | None:
        source = str(op.get("token_source") or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        if source != "last_sampled":
            return None
        if len(tokens) != 1:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but does not have exactly one token"
            )
        return _DECODE_RELAY.consume_token(
            state,
            expected_token_id=None,
            device=device,
            token_source=source,
            require=True,
        )

    def _decode_relay_tensors(
        self,
        text: "TextBatch",
        request_states: RequestStateTable,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if text.mode != ForwardMode.DECODE:
            return None, None
        return _DECODE_RELAY.resolve_decode_batch(
            text,
            request_states,
            device,
            stats=get_forward_context().stats,
        )

    def _store_decode_position_relays(
        self,
        text: "TextBatch",
        fb: "ForwardBatch",
        request_states: RequestStateTable,
    ) -> None:
        if text.mode != ForwardMode.DECODE:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        if fb.positions is None:
            raise invalid_descriptor("decode position relay requires position ids")
        next_positions = fb.positions.reshape(-1) + 1
        for row, (req_id, pos_range) in enumerate(zip(text.req_ids, text.pos_ranges)):
            self._store_position_relay(
                request_states.get(int(req_id)),
                position_id=int(pos_range[1]),
                position_tensor=next_positions[row : row + 1],
            )

    @staticmethod
    def _store_sampled_token_relay(
        state: RequestState,
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_sample(state, token_id=token_id, token_tensor=token_tensor)

    @staticmethod
    def _store_position_relay(
        state: RequestState,
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_position(
            state,
            position_id=position_id,
            position_tensor=position_tensor,
        )


def _can_decode_burst(
    text: "TextBatch", ops: list[Mapping[str, Any]], *, defer_sampling: bool
) -> bool:
    if defer_sampling or text.mode != ForwardMode.DECODE:
        return False
    if any(text.spec_token_ids):
        return False
    try:
        return any(
            _positive_int(op.get("decode_token_count") or 1, "decode_token_count") > 1 for op in ops
        )
    except Exception:
        raise


def _record_cuda_ready_start_event(
    kv_pool: "PagedKVPool | None",
    *,
    stats: ForwardStats | None,
    defer_cpu_results: bool,
) -> torch.cuda.Event | None:
    if stats is None or not defer_cpu_results:
        return None
    tensor = getattr(kv_pool, "k", None)
    device = getattr(tensor, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event


# ---------------------
# Flow-step session (single denoise update)
# ---------------------


class FlowSession:
    """Owns branch prediction and latent update semantics for one denoise step."""

    def __init__(
        self,
        model: Any,
        step: Any,
        *,
        combine_velocity: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
        accept_update: Callable[[Any, Any, torch.Tensor], None],
    ) -> None:
        self.model = model
        self.step = step
        self._combine_velocity = combine_velocity
        self._accept_update = accept_update

    def prepare_step(self, op: Mapping[str, Any] | None = None) -> Any:
        del op
        return self.step

    def predict_velocity(self, branch: str) -> torch.Tensor:
        velocity = self.model.predict_velocity(
            self.step,
            self.step.t,
            self.step.latent,
            branch,
        )
        if not isinstance(velocity, torch.Tensor) or velocity.shape != self.step.latent.shape:
            raise invalid_descriptor(
                f"{branch} velocity must be a tensor matching the denoise latent"
            )
        return velocity

    def apply_update(self, velocities: Mapping[str, torch.Tensor]) -> FlowOutput:
        velocity = self._combine_velocity(self.step, velocities)
        updated = euler_step(self.step.latent, velocity, self.step.t, self.step.t_next)
        self._accept_update(self.model, self.step, updated)
        done = self.step.step_index + 1 >= self.step.total_steps
        return FlowOutput(
            req_id=self.step.req_id,
            denoise_done=done,
            num_steps_done=self.step.step_index + 1,
        )

    def release(self) -> None:
        release = getattr(self.step, "release", None)
        if callable(release):
            release()


# ---------------------
# Flow-step execution (denoise)
# ---------------------

# Fallback latent geometry for the model-neutral generic path, used only when an
# op/image descriptor supplies neither an explicit ``latent_shape`` nor the
# per-field overrides. Production models compute their own latent geometry and
# never reach this fallback.
_DEFAULT_LATENT_DOWNSAMPLE = 16
_DEFAULT_LATENT_CHANNELS = 4


@dataclass(frozen=True)
class PreparedFlowStep:
    req_id: int
    state: RequestState
    op: Mapping[str, Any]
    latent: torch.Tensor
    t: torch.Tensor
    t_next: torch.Tensor
    step_index: int
    total_steps: int
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_interval: tuple[float, float]
    cfg_renorm_type: str
    cfg_renorm_min: float
    # Optional scheduler-provided branch bound. Pure T2I deliberately sends
    # branch_count=1 even when model defaults have guidance scales > 1.
    cfg_branch_count: int | None = None
    # Names the model's text/image CFG convention. Construction coerces bools
    # and strings to the enum so the execution path always consumes one type.
    image_scale_applies_to_text: CfgRecipe = CfgRecipe.ADDITIVE_DELTAS
    extra: Any = None

    def __post_init__(self) -> None:
        recipe = CfgRecipe.coerce(self.image_scale_applies_to_text)
        if recipe is not self.image_scale_applies_to_text:
            object.__setattr__(self, "image_scale_applies_to_text", recipe)
        if self.cfg_branch_count is not None:
            branch_count = int(self.cfg_branch_count)
            if branch_count < 1:
                raise ValueError("cfg_branch_count must be >= 1")
            object.__setattr__(self, "cfg_branch_count", branch_count)


class FlowExecutor:
    """Execute one model-neutral flow-matching denoise step."""

    def __init__(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        # ``device``/``dtype`` take effect only on the model-neutral generic
        # ``FlowContext``/``_latent`` path (see ``_finish_prepared_step``).
        # Production diffusion models return a ``PreparedFlowStep`` and run
        # on their own device/dtype, and the runner constructs ``FlowExecutor()``
        # with no arguments -- so these defaults are inert for them.
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def step(
        self, req_id: int, state: RequestState, model: "FlowCapable", op: Mapping[str, Any]
    ) -> FlowOutput:
        return self.step_many([(req_id, state, op)], model)[0]

    @torch.inference_mode()
    def step_many(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "FlowCapable",
        *,
        graph_mode: str = "auto",
    ) -> list[FlowOutput]:
        step_counts = [_denoise_step_count(op) for _req_id, _state, op in items]
        if any(count > 1 for count in step_counts):
            return self._step_many_burst(items, model, step_counts, graph_mode=graph_mode)
        prepared = [
            (int(req_id), state, op, self._prepare(req_id, state, model, op))
            for req_id, state, op in items
        ]
        if all(isinstance(item[3], PreparedFlowStep) for item in prepared):
            return self._flow_steps(
                model,
                [item[3] for item in prepared if isinstance(item[3], PreparedFlowStep)],
                graph_mode=graph_mode,
            )
        return [
            self._finish_prepared_step(req_id, state, model, op, ctx)
            for req_id, state, op, ctx in prepared
        ]

    @torch.inference_mode()
    def forward_result(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "FlowCapable",
        *,
        row_indices: Sequence[int] | None = None,
        graph_mode: str = "auto",
    ) -> ForwardResult | None:
        step_counts = [_denoise_step_count(op) for _req_id, _state, op in items]
        if any(count > 1 for count in step_counts):
            return None
        rows = (
            tuple(range(len(items)))
            if row_indices is None
            else tuple(int(row) for row in row_indices)
        )
        if len(rows) != len(items):
            raise invalid_descriptor("denoise row_indices must align with denoise items")
        prepared = [
            (int(row_index), int(req_id), state, op, self._prepare(req_id, state, model, op))
            for row_index, (req_id, state, op) in zip(rows, items, strict=True)
        ]
        flow_items = [
            (row_index, step)
            for row_index, _req_id, _state, _op, step in prepared
            if isinstance(step, PreparedFlowStep)
        ]
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        updates: dict[int, DenoisePostprocessEntry] = {}
        if flow_items:
            flow_entries = self._flow_forward_entries(
                model,
                flow_items,
                graph_mode=graph_mode,
            )
            if flow_entries is None:
                return None
            flow_velocities, flow_updates = flow_entries
            velocities.update(flow_velocities)
            updates.update(flow_updates)
        for row_index, req_id, state, op, ctx in prepared:
            if isinstance(ctx, PreparedFlowStep):
                continue
            if graph_mode == "require":
                return None
            entry, branch_velocities = self._generic_forward_entry(
                row_index,
                req_id,
                state,
                model,
                op,
                ctx,
            )
            updates[int(row_index)] = entry
            velocities.update(branch_velocities)
        return ForwardResult(denoise_velocities=velocities, denoise_updates=updates)

    def _step_many_burst(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "FlowCapable",
        step_counts: Sequence[int],
        *,
        graph_mode: str,
    ) -> list[FlowOutput]:
        outputs: list[FlowOutput | None] = [None] * len(items)
        active: list[dict[str, Any]] = []
        for index, ((req_id, state, op), step_count) in enumerate(
            zip(items, step_counts, strict=True)
        ):
            op_dict = dict(op)
            cursor = int(op_dict.get("timestep_idx", state.schedule_cursor) or 0)
            active.append(
                {
                    "index": index,
                    "req_id": int(req_id),
                    "state": state,
                    "op": op_dict,
                    "cursor": cursor,
                    "remaining": int(step_count),
                }
            )

        while active:
            prepared: list[
                tuple[dict[str, Any], Mapping[str, Any], FlowContext | PreparedFlowStep]
            ] = []
            for item in active:
                op = dict(item["op"])
                op["timestep_idx"] = int(item["cursor"])
                prepared.append(
                    (
                        item,
                        op,
                        self._prepare(int(item["req_id"]), item["state"], model, op),
                    )
                )

            if all(isinstance(ctx, PreparedFlowStep) for _item, _op, ctx in prepared):
                step_outputs = self._flow_steps(
                    model,
                    [ctx for _item, _op, ctx in prepared if isinstance(ctx, PreparedFlowStep)],
                    graph_mode=graph_mode,
                )
            else:
                step_outputs = [
                    self._finish_prepared_step(
                        int(item["req_id"]),
                        item["state"],
                        model,
                        op,
                        ctx,
                    )
                    for item, op, ctx in prepared
                ]

            next_active: list[dict[str, Any]] = []
            for (item, _op, _ctx), output in zip(prepared, step_outputs, strict=True):
                outputs[int(item["index"])] = output
                item["remaining"] = int(item["remaining"]) - 1
                item["cursor"] = int(output.num_steps_done)
                if not output.denoise_done and int(item["remaining"]) > 0:
                    next_active.append(item)
            active = next_active

        if any(output is None for output in outputs):
            raise invalid_descriptor("denoise burst did not produce an output for every op")
        return [output for output in outputs if output is not None]

    def _prepare(
        self,
        req_id: int,
        state: RequestState,
        model: "FlowCapable",
        op: Mapping[str, Any],
    ) -> FlowContext | PreparedFlowStep:
        del req_id
        prepared = _prepare_flow(model, state, op)
        if isinstance(prepared, PreparedFlowStep):
            return prepared
        if not isinstance(prepared, FlowContext):
            raise invalid_descriptor("prepare_flow(state, op) must return FlowContext")
        return prepared

    def _finish_prepared_step(
        self,
        req_id: int,
        state: RequestState,
        model: "FlowCapable",
        op: Mapping[str, Any],
        prepared: FlowContext | PreparedFlowStep,
    ) -> FlowOutput:
        if isinstance(prepared, PreparedFlowStep):
            return self._flow_steps(model, [prepared])[0]
        # Model-neutral generic flow-matching path. It is the contract for models
        # that return a FlowContext (and is exercised by the synthetic
        # velocity-only test model); the production diffusion models instead
        # return a PreparedFlowStep above and build their own
        # schedule with model-specific direction/shift-domain defaults. As a
        # consequence the op-level schedule_direction/schedule_shift/flow_shift
        # keys read below (and the equivalent DiffusionConfig fields) are INERT
        # for those production models -- callers must not assume they take effect.
        image = dict(state.image or {})
        image.update(op.get("image") or {})
        steps = int(op.get("num_steps") or image.get("steps") or image.get("num_steps") or 50)
        if steps <= 0:
            raise invalid_descriptor("denoise steps must be positive")
        schedule = FlowMatchSchedule(
            num_steps=steps,
            shift=float(image.get("schedule_shift", image.get("flow_shift", 1.0))),
            direction=ScheduleDirection(str(image.get("schedule_direction", "ascending"))),
        )
        cursor = int(op.get("timestep_idx", state.schedule_cursor) or 0)
        t, t_next = schedule.pair(cursor, device=self.device, dtype=self.dtype)
        latent = self._latent(state, image, op)
        cfg = CfgParams.from_mapping(op.get("cfg") or state.cfg_geometry)
        velocities = []
        for branch_index in range(cfg.branch_count):
            branch = _branch_name(branch_index, cfg.branch_count)
            velocity = model.predict_velocity(prepared, t, latent, branch)
            if not isinstance(velocity, torch.Tensor):
                raise invalid_descriptor(
                    "predict_velocity(ctx, t, latent, branch) must return a tensor"
                )
            velocity = _maybe_convert_parameterization(model, velocity, latent, t)
            if velocity.shape != latent.shape:
                raise invalid_descriptor(
                    f"velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}"
                )
            velocities.append(velocity)
        velocity = combine_cfg(velocities, cfg)
        state.latent = euler_step(latent, velocity, t, t_next)
        _accept_flow_update(model, prepared, state.latent)
        done = cursor + 1 >= steps
        return FlowOutput(req_id=req_id, denoise_done=done, num_steps_done=cursor + 1)

    def _flow_steps(
        self,
        model: "FlowCapable",
        steps: Sequence[PreparedFlowStep],
        *,
        graph_mode: str = "auto",
    ) -> list[FlowOutput]:
        branches_by_step = [flow_branches(step) for step in steps]
        batch_predict = _flow_batch_predictor(model)
        if batch_predict is not None:
            predicted = _call_flow_batch_predictor(
                batch_predict,
                steps,
                branches_by_step,
                graph_mode=graph_mode,
            )
        else:
            predicted = None
        if predicted is None and graph_mode == "require":
            raise invalid_descriptor("flow graph mode required a graphable batch")
        if predicted is None:
            branch_outputs = [
                {
                    branch: self._predict_flow_branch(model.predict_velocity, step, branch)
                    for branch in branches
                }
                for step, branches in zip(steps, branches_by_step)
            ]
        else:
            branch_outputs = _validate_batched_flow_outputs(steps, branches_by_step, predicted)
        outputs = []
        for step, velocities in zip(steps, branch_outputs):
            session = FlowSession(
                model,
                step,
                combine_velocity=combine_flow_velocity,
                accept_update=_accept_flow_update,
            )
            outputs.append(session.apply_update(velocities))
        return outputs

    def _predict_flow_branch(
        self, predict: Any, step: PreparedFlowStep, branch: str
    ) -> torch.Tensor:
        velocity = predict(step, step.t, step.latent, branch)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
            raise invalid_descriptor(
                f"{branch} velocity must be a tensor matching the denoise latent"
            )
        return velocity

    def _flow_forward_entries(
        self,
        model: "FlowCapable",
        items: Sequence[tuple[int, PreparedFlowStep]],
        *,
        graph_mode: str = "auto",
    ) -> tuple[dict[DenoiseBranchKey, torch.Tensor], dict[int, DenoisePostprocessEntry]] | None:
        steps = [step for _row_index, step in items]
        branches_by_step = [flow_branches(step) for step in steps]
        batch_predict = _flow_batch_predictor(model)
        predicted = (
            _call_flow_batch_predictor(
                batch_predict,
                steps,
                branches_by_step,
                graph_mode=graph_mode,
            )
            if batch_predict is not None
            else None
        )
        if predicted is None and graph_mode == "require":
            return None
        if predicted is None:
            branch_outputs = [
                {
                    branch: self._predict_flow_branch(model.predict_velocity, step, branch)
                    for branch in branches
                }
                for step, branches in zip(steps, branches_by_step, strict=True)
            ]
        else:
            branch_outputs = _validate_batched_flow_outputs(steps, branches_by_step, predicted)
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        updates: dict[int, DenoisePostprocessEntry] = {}
        for (row_index, step), branches, outputs in zip(
            items,
            branches_by_step,
            branch_outputs,
            strict=True,
        ):
            for branch_id, branch in enumerate(branches):
                velocities[DenoiseBranchKey(int(row_index), int(branch_id))] = outputs[branch]

            def combine_step_velocity(
                values: Mapping[Any, torch.Tensor],
                current_step: PreparedFlowStep = step,
            ) -> torch.Tensor:
                return combine_flow_velocity(current_step, values)

            def accept_step_update(
                latent: torch.Tensor,
                current_model: Any = model,
                current_step: PreparedFlowStep = step,
            ) -> None:
                _accept_flow_update(current_model, current_step, latent)

            updates[int(row_index)] = DenoisePostprocessEntry(
                row_index=int(row_index),
                req_id=int(step.req_id),
                step_index=int(step.step_index),
                total_steps=int(step.total_steps),
                branch_names=tuple(branches),
                latent=step.latent,
                t=step.t,
                t_next=step.t_next,
                combine_velocity=combine_step_velocity,
                accept_update=accept_step_update,
            )
        return velocities, updates

    def _generic_forward_entry(
        self,
        row_index: int,
        req_id: int,
        state: RequestState,
        model: "FlowCapable",
        op: Mapping[str, Any],
        prepared: FlowContext,
    ) -> tuple[DenoisePostprocessEntry, dict[DenoiseBranchKey, torch.Tensor]]:
        image = dict(state.image or {})
        image.update(op.get("image") or {})
        steps = int(op.get("num_steps") or image.get("steps") or image.get("num_steps") or 50)
        if steps <= 0:
            raise invalid_descriptor("denoise steps must be positive")
        schedule = FlowMatchSchedule(
            num_steps=steps,
            shift=float(image.get("schedule_shift", image.get("flow_shift", 1.0))),
            direction=ScheduleDirection(str(image.get("schedule_direction", "ascending"))),
        )
        cursor = int(op.get("timestep_idx", state.schedule_cursor) or 0)
        t, t_next = schedule.pair(cursor, device=self.device, dtype=self.dtype)
        latent = self._latent(state, image, op)
        cfg = CfgParams.from_mapping(op.get("cfg") or state.cfg_geometry)
        branch_names = tuple(
            _branch_name(index, cfg.branch_count) for index in range(cfg.branch_count)
        )
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        for branch_id, branch in enumerate(branch_names):
            velocity = model.predict_velocity(prepared, t, latent, branch)
            if not isinstance(velocity, torch.Tensor):
                raise invalid_descriptor(
                    "predict_velocity(ctx, t, latent, branch) must return a tensor"
                )
            velocity = _maybe_convert_parameterization(model, velocity, latent, t)
            if velocity.shape != latent.shape:
                raise invalid_descriptor(
                    f"velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}"
                )
            velocities[DenoiseBranchKey(int(row_index), int(branch_id))] = velocity

        def combine_generic_velocity(values: Mapping[Any, torch.Tensor]) -> torch.Tensor:
            return combine_cfg([values[branch] for branch in branch_names], cfg)

        def accept_generic_update(latent_value: torch.Tensor) -> None:
            _accept_flow_update(model, prepared, latent_value)

        entry = DenoisePostprocessEntry(
            row_index=int(row_index),
            req_id=int(req_id),
            step_index=int(cursor),
            total_steps=int(steps),
            branch_names=branch_names,
            latent=latent,
            t=t,
            t_next=t_next,
            combine_velocity=combine_generic_velocity,
            accept_update=accept_generic_update,
        )
        return entry, velocities

    def _latent(
        self, state: RequestState, image: Mapping[str, Any], op: Mapping[str, Any]
    ) -> torch.Tensor:
        if isinstance(state.latent, torch.Tensor):
            return state.latent.to(device=self.device, dtype=self.dtype)
        shape = op.get("latent_shape") or image.get("latent_shape")
        if shape is None:
            h = required_image_height(image)
            w = required_image_width(image)
            downsample = int(
                image.get("latent_downsample", _DEFAULT_LATENT_DOWNSAMPLE)
                or _DEFAULT_LATENT_DOWNSAMPLE
            )
            channels = int(
                image.get("latent_channels", _DEFAULT_LATENT_CHANNELS) or _DEFAULT_LATENT_CHANNELS
            )
            shape = (channels, max(1, h // downsample), max(1, w // downsample))
        shape_tuple = tuple(int(v) for v in shape)
        if state.rng is None:
            # The generator must live on the same device as the sampled latent;
            # ``torch.randn(generator=rng, device=...)`` requires rng.device to
            # match. Building a CPU generator while sampling on CUDA raises.
            state.rng = torch.Generator(device=self.device)
            state.rng.manual_seed(int(image.get("seed", 0) or 0))
        state.latent = init_latent(
            shape_tuple,
            rng=state.rng,
            device=self.device,
            dtype=self.dtype,
            scale=float(image.get("latent_scale", 1.0) or 1.0),
        )
        return state.latent


def _branch_name(branch_index: int, branch_count: int) -> str:
    del branch_count
    return f"branch_{branch_index}"


def _prepare_flow(
    model: Any,
    state: RequestState,
    op: Mapping[str, Any],
) -> FlowContext | PreparedFlowStep:
    return model.prepare_flow(state, op)


def _denoise_step_count(op: Mapping[str, Any]) -> int:
    raw = op.get("denoise_step_count") or 1
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("denoise_step_count must be a positive integer") from exc
    if value <= 0:
        raise invalid_descriptor("denoise_step_count must be positive")
    return value


def _flow_batch_predictor(model: Any) -> Any | None:
    return model.predict_flow_velocity_batch


def _call_flow_batch_predictor(
    predictor: Any,
    steps: Sequence[PreparedFlowStep],
    branches_by_step: Sequence[Sequence[str]],
    *,
    graph_mode: str,
) -> list[dict[str, torch.Tensor]] | None:
    if graph_mode == "auto":
        return predictor(steps, branches_by_step)
    if not _accepts_graph_mode(predictor):
        return None if graph_mode == "require" else predictor(steps, branches_by_step)
    return predictor(steps, branches_by_step, graph_mode=graph_mode)


def _accepts_graph_mode(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "graph_mode" in signature.parameters


def _accept_flow_update(
    model: Any,
    ctx: FlowContext | PreparedFlowStep,
    latent: torch.Tensor,
) -> None:
    accept = _flow_update_acceptor(model)
    if accept is not None:
        accept(ctx, latent)
        return
    state = getattr(ctx, "state", None)
    if state is not None:
        state.latent = latent


def _flow_update_acceptor(model: Any) -> Any | None:
    return model.accept_flow_update


def _flow_cfg_plan(step: PreparedFlowStep) -> DiffusionCfgPlan:
    """Single source of truth for a step's branch set and combination weights.

    CFG runs only while the timestep lies within ``cfg_interval`` ``[lo, hi]``
    (inclusive). The default ``(0.0, 1.0)`` enables CFG for the whole trajectory;
    a restricted window such as ``(0.0, 0.5)`` disables CFG above the upper bound
    instead of being short-circuited by ``lo == 0``. The returned plan drives
    both which branches the model evaluates and how they are weighted, so the two
    cannot disagree.
    """
    if step.cfg_branch_count == 1:
        return DiffusionCfgPlan(branches=(Branch.COND,))
    t_value = float(step.t.detach().float().item())
    lo, hi = step.cfg_interval
    use_cfg = lo <= t_value <= hi
    return build_flow_cfg_plan(
        cfg_text_scale=step.cfg_text_scale,
        cfg_img_scale=step.cfg_img_scale,
        recipe=step.image_scale_applies_to_text,
        renorm=step.cfg_renorm_type,
        renorm_min=step.cfg_renorm_min,
        use_cfg=use_cfg,
    )


def flow_branches(step: PreparedFlowStep) -> tuple[str, ...]:
    return flow_cfg_plan(step).branches


def flow_cfg_plan(step: PreparedFlowStep) -> DiffusionCfgPlan:
    return _flow_cfg_plan(step)


def flow_cfg_branch_count(op: Mapping[str, Any]) -> int | None:
    cfg = op.get("cfg")
    if not isinstance(cfg, Mapping) or cfg.get("branch_count") is None:
        return None
    try:
        branch_count = int(cfg["branch_count"])
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("cfg.branch_count must be a positive integer") from exc
    if branch_count < 1:
        raise invalid_descriptor("cfg.branch_count must be a positive integer")
    return branch_count


def combine_flow_velocity(
    step: PreparedFlowStep,
    outputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return flow_cfg_plan(step).combine(outputs)


def _validate_batched_flow_outputs(
    steps: Sequence[PreparedFlowStep],
    branches_by_step: Sequence[Sequence[str]],
    predicted: Any,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(predicted, Sequence) or isinstance(predicted, (str, bytes, bytearray)):
        raise invalid_descriptor("predict_flow_velocity_batch must return one mapping per step")
    if len(predicted) != len(steps):
        raise invalid_descriptor("predict_flow_velocity_batch returned the wrong number of steps")
    out: list[dict[str, torch.Tensor]] = []
    for step, branches, values in zip(steps, branches_by_step, predicted):
        if not isinstance(values, Mapping):
            raise invalid_descriptor("predict_flow_velocity_batch entries must be mappings")
        checked: dict[str, torch.Tensor] = {}
        for branch in branches:
            velocity = values.get(branch)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
                raise invalid_descriptor(
                    f"{branch} velocity must be a tensor matching the denoise latent"
                )
            checked[branch] = velocity
        out.append(checked)
    return out


def _maybe_convert_parameterization(
    model: Any,
    prediction: torch.Tensor,
    latent: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    parameterization = _velocity_parameterization(model)
    if parameterization == "velocity":
        return prediction
    if parameterization == "x_pred":
        return x_pred_to_velocity(prediction, latent, t)
    raise invalid_descriptor(f"unsupported velocity_parameterization {parameterization!r}")


def _velocity_parameterization(model: Any) -> str:
    return str(model.velocity_parameterization() or "velocity")


# ---------------------
# Encode-step execution
# ---------------------


class EncodeDriver:
    """Run model-provided image/text encoder primitives for ENCODE ops."""

    @torch.inference_mode()
    def step(self, fb: UniForwardBatch, model: Any) -> list[EncodeOutput]:
        # The encode view validates req_id/kind/mm_hash once; ``ops`` is the raw
        # op mapping retained so the model encode hooks still receive their
        # unparsed pixel/grid payloads.
        ops = fb.as_encode().ops
        return self._run_many(model, ops)

    @torch.inference_mode()
    def forward_result(
        self,
        fb: UniForwardBatch,
        model: Any,
        *,
        row_indices: tuple[int, ...] | list[int] | None = None,
    ) -> ForwardResult:
        ops = fb.as_encode().ops
        rows = (
            tuple(range(len(ops)))
            if row_indices is None
            else tuple(int(row) for row in row_indices)
        )
        if len(rows) != len(ops):
            raise invalid_descriptor("encode row_indices must align with encode ops")
        outputs = dict(zip(rows, self._run_many(model, ops), strict=True))
        return ForwardResult(encode_outputs=outputs)

    def _run_many(self, model: Any, ops: tuple[Mapping[str, Any], ...]) -> list[EncodeOutput]:
        encode_many = getattr(model, "encode_many", None)
        if callable(encode_many):
            outputs = list(encode_many(ops))
        else:
            outputs = [self._run_one(model, op) for op in ops]
        if len(outputs) != len(ops):
            raise invalid_descriptor(
                f"model returned {len(outputs)} encode outputs for {len(ops)} ops"
            )
        return [_coerce_encode_output(output) for output in outputs]

    def _run_one(self, model: Any, op: Mapping[str, Any]) -> Any:
        kind = str(op.get("kind"))
        if mode_for_op(kind) != ForwardMode.ENCODE:
            raise invalid_descriptor(f"unsupported encode op {op.get('kind')!r}")
        if kind == VIT_ENCODE:
            return model.encode_image(op.get("pixels"), op.get("grid"), op=op)
        if kind == VAE_ENCODE:
            return model.encode_latents(op.get("pixels"), op.get("grid"), op=op)
        raise invalid_descriptor(f"unsupported encode op {op.get('kind')!r}")


def _coerce_encode_output(output: Mapping[str, Any] | EncodeOutput) -> EncodeOutput:
    # The model->driver boundary stays flexible: encode adapters may return either
    # a typed ``EncodeOutput`` or a wire mapping. The driver normalizes both to
    # ``EncodeOutput`` so the runner consumes one uniform ``ForwardOutput`` list.
    if isinstance(output, EncodeOutput):
        return output
    if not isinstance(output, Mapping):
        raise invalid_descriptor("encode adapter outputs must be mappings")
    req_id = output.get("req_id")
    handle = output.get("encoder_handle")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise invalid_descriptor("encode output req_id must be an integer")
    if not isinstance(handle, int) or isinstance(handle, bool):
        raise invalid_descriptor("encode output encoder_handle must be an integer")
    return EncodeOutput(
        req_id=int(req_id),
        encoder_handle=int(handle),
        num_tokens=_optional_int(output.get("num_tokens")),
        image_hw=_image_hw(output.get("image_hw")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor("encode output num_tokens must be an integer")
    return int(value)


def _image_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise invalid_descriptor("encode output image_hw must be [height, width]")
    return (int(value[0]), int(value[1]))


# ---------------------
# Materialize-step execution (image decode)
# ---------------------


class ImageDecodeDriver:
    @torch.inference_mode()
    def step(
        self, req_id: int, state: RequestState, model: Any, op: Mapping[str, Any]
    ) -> CommitOutput:
        result = model.decode_image(state.latent, req_id=int(req_id), state=state, op=op)
        out = dict(result) if isinstance(result, Mapping) else _image_to_result(int(req_id), result)
        logits = out.pop("logits", None)
        if logits is not None:
            sampled = sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
            sampled.pop("req_id", None)
            out.update(sampled)
        return _commit_output_from_dict(int(req_id), out)

    @torch.inference_mode()
    def forward_result(
        self,
        items: list[tuple[int, RequestState, Mapping[str, Any]]]
        | tuple[tuple[int, RequestState, Mapping[str, Any]], ...],
        model: Any,
        *,
        row_indices: tuple[int, ...] | list[int] | None = None,
    ) -> ForwardResult:
        rows = (
            tuple(range(len(items)))
            if row_indices is None
            else tuple(int(row) for row in row_indices)
        )
        if len(rows) != len(items):
            raise invalid_descriptor("commit row_indices must align with commit items")
        outputs: dict[int, Any] = {}
        for row, (req_id, state, op) in zip(rows, items, strict=True):
            outputs[int(row)] = model.decode_image(
                state.latent, req_id=int(req_id), state=state, op=op
            )
        return ForwardResult(commit_outputs=outputs)


def _commit_output_from_dict(req_id: int, out: Mapping[str, Any]) -> CommitOutput:
    image_hw = out.get("image_hw")
    return CommitOutput(
        req_id=req_id,
        image_png_b64=out.get("image_png_b64"),
        image_hw=(int(image_hw[0]), int(image_hw[1])) if image_hw is not None else None,
        sampled_token_id=out.get("sampled_token_id"),
        sampled_logprob=out.get("sampled_logprob"),
        top_logprobs=out.get("top_logprobs"),
        num_tokens=out.get("num_tokens"),
        locator=out.get("locator"),
    )


def _image_to_result(req_id: int, image: Any) -> dict[str, Any]:
    save = getattr(image, "save", None)
    if callable(save):
        width, height = getattr(image, "size", (None, None))
        out = {"req_id": req_id, "image_png_b64": pil_image_to_png_b64(image)}
        if width is not None and height is not None:
            out["image_hw"] = [int(height), int(width)]
        return out
    if isinstance(image, torch.Tensor):
        if image.ndim not in (3, 4):
            raise invalid_descriptor("decode_image tensor output must be CHW or NCHW")
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency failure is environment-specific.
            raise invalid_descriptor("PIL is required to encode tensor image outputs") from exc
        # decode_image returns already-normalized [0, 1] image space (not the
        # diffusion [-1, 1] latent convention), so decode against that range.
        pil = Image.fromarray(to_uint8_image(image, value_range=(0.0, 1.0)))
        return _image_to_result(req_id, pil)
    raise invalid_descriptor("decode_image must return a mapping, PIL image, or image tensor")


# ---------------------
# Canonical plan construction (rows, segments, output slots)
# ---------------------

_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})
_SEGMENT_PRODUCING_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.DENOISE})


class Route(str, Enum):
    """Planning route for a resource-admitted worker op group."""

    PER_MODE = "per_mode"
    FORWARD = "forward"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ForwardAdmissionDecision:
    route: Route
    reason: str
    modes: tuple[ForwardMode, ...]

    @property
    def use_forward(self) -> bool:
        return self.route is Route.FORWARD


@dataclass(frozen=True)
class ForwardAdmissionRouter:
    @classmethod
    def from_runtime_config(cls) -> "ForwardAdmissionRouter":
        return cls()

    def decide(self, ops: Sequence[Mapping[str, object]]) -> ForwardAdmissionDecision:
        modes = tuple(mode_for_op(str(op.get("kind"))) for op in ops)
        if not ops:
            return ForwardAdmissionDecision(Route.PER_MODE, "empty batch", modes)
        if any(mode not in _SEGMENT_PRODUCING_MODES for mode in modes):
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "group contains an operation without a model-execution segment",
                modes,
            )
        if any(_has_values(op.get("spec_token_ids")) for op in ops):
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "candidate expansion uses its dedicated execution path",
                modes,
            )
        return ForwardAdmissionDecision(
            Route.FORWARD,
            "group is representable by one segment table",
            modes,
        )

    def partition_supported(
        self,
        ops: Sequence[Mapping[str, object]],
    ) -> tuple[list[tuple[int, Mapping[str, object]]], list[tuple[int, Mapping[str, object]]]]:
        supported: list[tuple[int, Mapping[str, object]]] = []
        delegated: list[tuple[int, Mapping[str, object]]] = []
        for index, op in enumerate(ops):
            mode = mode_for_op(str(op.get("kind")))
            target = supported if mode in _SEGMENT_PRODUCING_MODES else delegated
            target.append((index, op))
        return supported, delegated


class ForwardPlanBuilder:
    """Build immutable control plans from admitted worker op groups."""

    def build(
        self,
        group: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]],
        *,
        request_states: Any = None,
        step_id: int | None = None,
        graph_policy: "ForwardGraphPolicy | None" = None,
        runtime_handles: ForwardRuntimeHandles | None = None,
    ) -> ForwardPlan:
        if not group:
            raise invalid_descriptor("forward plan group must not be empty")
        rows: list[ForwardRowPlan] = []
        segments: list[ForwardSegmentPlan] = []
        output_slots: list[ForwardOutputSlot] = []
        for row_index, item in enumerate(group):
            original_index, op = _group_item(row_index, item)
            row = self._row_plan(row_index, original_index, op, request_states)
            rows.append(row)
            segments.extend(self._segments_for_row(row, len(segments)))
            output_slots.append(self._output_slot(row))
        forward_mode = _summary_mode(tuple(row.mode for row in rows))
        shape = ForwardShapeSummary.from_parts(
            forward_mode=forward_mode,
            rows=rows,
            segments=segments,
        )
        handles = runtime_handles or ForwardRuntimeHandles(request_states=request_states)
        plan = ForwardPlan(
            step_id=step_id,
            rows=tuple(rows),
            segments=tuple(segments),
            output_slots=tuple(output_slots),
            shape=shape,
            graph_policy=graph_policy,
            runtime_handles=handles,
        )
        plan.validate()
        return plan

    def _row_plan(
        self,
        row_index: int,
        original_index: int,
        op: Mapping[str, Any],
        request_states: Any,
    ) -> ForwardRowPlan:
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor("forward op kind must be a string")
        req_id = _int_field(op, "req_id")
        mode = mode_for_op(kind)
        token_span = self._text_span(op, mode)
        cache_span = self._cache_span(op, request_states, req_id, token_span)
        return ForwardRowPlan(
            row_index=row_index,
            original_index=original_index,
            req_id=req_id,
            op=MappingProxyType(dict(op)),
            mode=mode,
            token_span=token_span,
            cache_span=cache_span,
            denoise=self._denoise_plan(op, mode),
            commit=self._commit_plan(op, mode),
            encode=self._encode_plan(op, mode),
        )

    @staticmethod
    def _text_span(op: Mapping[str, Any], mode: ForwardMode) -> TextTokenSpanPlan | None:
        if mode not in _TEXT_MODES:
            return None
        tokens = tuple(int(token) for token in (op.get("token_ids") or ()))
        start, end = _pos_range(op, len(tokens))
        return TextTokenSpanPlan(
            token_ids=tokens,
            position_start=start,
            position_end=end,
            token_source=str(op.get("token_source") or "wire"),
            last_token_only=mode
            in {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT},
        )

    @staticmethod
    def _cache_span(
        op: Mapping[str, Any],
        request_states: Any,
        req_id: int,
        token_span: TextTokenSpanPlan | None,
    ) -> CacheSpanPlan | None:
        if token_span is None:
            return None
        state = None
        if request_states is not None:
            get = getattr(request_states, "get", None)
            if callable(get):
                try:
                    state = get(req_id)
                except Exception:
                    state = None
        state_blocks = tuple(int(block) for block in getattr(state, "block_ids", ()) or ())
        new_blocks = tuple(int(block) for block in (op.get("new_block_ids") or ()))
        block_ids = state_blocks
        if new_blocks and not block_ids[-len(new_blocks) :] == new_blocks:
            block_ids = (*block_ids, *new_blocks)
        return CacheSpanPlan(
            block_ids=block_ids,
            base_len=int(token_span.position_start),
            append_len=token_span.q_len,
            pool_identity=str(op.get("kv_pool") or "text"),
            persistent=True,
        )

    @staticmethod
    def _denoise_plan(op: Mapping[str, Any], mode: ForwardMode) -> DenoiseRowPlan | None:
        if mode is not ForwardMode.DENOISE:
            return None
        cfg = dict(op.get("cfg") or {})
        branch_count = int(cfg.get("branch_count") or op.get("branch_count") or 1)
        if branch_count < 1:
            raise invalid_descriptor("denoise branch count must be positive")
        return DenoiseRowPlan(
            step_index=int(op.get("timestep_idx") or 0),
            total_steps=max(1, int(op.get("num_steps") or op.get("total_steps") or 1)),
            branch_count=branch_count,
            branch_ids=tuple(_plan_branch_name(i, branch_count) for i in range(branch_count)),
            image_token_count=max(1, _image_token_count(op)),
            latent_handle=_plan_optional_int(op.get("latent_handle")),
            grid_hw=_grid_hw(op.get("grid_hw")),
            cfg=MappingProxyType(cfg),
        )

    @staticmethod
    def _commit_plan(op: Mapping[str, Any], mode: ForwardMode) -> CommitRowPlan | None:
        if mode is not ForwardMode.COMMIT:
            return None
        return CommitRowPlan(
            latent_handle=_plan_optional_int(op.get("latent_handle")),
            fold_back=bool(op.get("fold_back", False)),
            image_token_count=max(1, _image_token_count(op)),
        )

    @staticmethod
    def _encode_plan(op: Mapping[str, Any], mode: ForwardMode) -> EncodeRowPlan | None:
        if mode is not ForwardMode.ENCODE:
            return None
        return EncodeRowPlan(
            kind=str(op.get("kind")),
            out_handle=_plan_optional_int(op.get("out_handle") or op.get("encoder_handle")),
            mm_hash=_plan_optional_int(op.get("mm_hash")),
            num_tokens=max(1, int(op.get("num_tokens") or op.get("image_token_count") or 1)),
        )

    @staticmethod
    def _segments_for_row(
        row: ForwardRowPlan,
        next_segment_index: int,
    ) -> list[ForwardSegmentPlan]:
        segments: list[ForwardSegmentPlan] = []
        if row.token_span is not None and row.token_span.q_len > 0:
            segment_class = (
                ForwardSegmentClass.DECODE
                if row.mode is ForwardMode.DECODE
                else ForwardSegmentClass.EXTEND
            )
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.TEXT,
                    segment_class=segment_class,
                    q_len=row.token_span.q_len,
                    prefix_len=row.token_span.position_start,
                    visible_policy=VisiblePolicy.CAUSAL,
                    branch_id=0,
                    position_source=row.token_span.token_source,
                    kv_write_policy=KvWritePolicy.PERSISTENT,
                )
            )
            next_segment_index += 1
        if row.denoise is not None:
            for branch_index in range(row.denoise.branch_count):
                segments.append(
                    ForwardSegmentPlan(
                        segment_index=next_segment_index,
                        row_index=row.row_index,
                        mode=row.mode,
                        modality=ForwardModality.GENERATION,
                        segment_class=ForwardSegmentClass.DENOISE,
                        q_len=row.denoise.image_token_count,
                        prefix_len=0,
                        visible_policy=VisiblePolicy.BIDIRECTIONAL,
                        branch_id=branch_index,
                        position_source="generation_grid",
                        kv_write_policy=KvWritePolicy.TRANSIENT,
                    )
                )
                next_segment_index += 1
        if row.commit is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.COMMIT_INPUT,
                    q_len=row.commit.image_token_count,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
            next_segment_index += 1
        if row.encode is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.ENCODE_INPUT,
                    q_len=row.encode.num_tokens,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
        return segments

    @staticmethod
    def _output_slot(row: ForwardRowPlan) -> ForwardOutputSlot:
        if row.mode in _TEXT_MODES:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.TEXT_TOKEN,
                result_projection=ForwardResultProjection.LAST_TEXT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.SAMPLE,
            )
        if row.mode is ForwardMode.DENOISE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.DENOISE_STEP,
                result_projection=ForwardResultProjection.DENOISE_BRANCHES,
                postprocess_policy=ForwardPostprocessPolicy.LATENT_UPDATE,
            )
        if row.mode is ForwardMode.COMMIT:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.COMMIT,
                result_projection=ForwardResultProjection.COMMIT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.COMMIT_DECODE,
            )
        if row.mode is ForwardMode.ENCODE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.ENCODE,
                result_projection=ForwardResultProjection.ENCODE_ROW,
                postprocess_policy=ForwardPostprocessPolicy.ENCODE_PUBLISH,
            )
        return ForwardOutputSlot(
            row_index=row.row_index,
            req_id=row.req_id,
            kind=ForwardOutputKind.COMBINED,
            result_projection=ForwardResultProjection.RUNTIME_OUTPUT,
            postprocess_policy=ForwardPostprocessPolicy.NONE,
        )


def _summary_mode(modes: tuple[ForwardMode, ...]) -> ForwardMode:
    first = modes[0]
    if any(mode is not first for mode in modes):
        return ForwardMode.MIXED
    return first


def _group_item(
    fallback_index: int,
    item: Mapping[str, Any] | tuple[int, Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    if isinstance(item, tuple) and len(item) == 2:
        index, op = item
        if not isinstance(op, Mapping):
            raise invalid_descriptor("forward group item op must be a mapping")
        return int(index), op
    if not isinstance(item, Mapping):
        raise invalid_descriptor("forward group item must be an op mapping")
    return fallback_index, item


def _int_field(op: Mapping[str, Any], field_name: str) -> int:
    value = op.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"forward op {field_name} must be an integer")
    return int(value)


def _plan_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _pos_range(op: Mapping[str, Any], token_count: int) -> tuple[int, int]:
    raw = op.get("pos_range")
    if raw is None:
        return 0, int(token_count)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("text op pos_range must be [start, end]")
    start, end = int(raw[0]), int(raw[1])
    if end < start:
        raise invalid_descriptor("text op pos_range end must be >= start")
    return start, end


def _image_token_count(op: Mapping[str, Any]) -> int:
    for key in ("image_token_count", "latent_tokens", "num_tokens"):
        if op.get(key) is not None:
            return int(op[key])
    shape = op.get("latent_shape")
    if isinstance(shape, (list, tuple)) and shape:
        total = 1
        for value in shape:
            total *= max(1, int(value))
        return total
    return 1


def _grid_hw(raw: Any) -> tuple[int, int]:
    if raw is None:
        return (0, 0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("grid_hw must be [height, width]")
    return (int(raw[0]), int(raw[1]))


def _plan_branch_name(index: int, count: int) -> str:
    if count == 1:
        return "cond"
    if index == 0:
        return "cond"
    return f"branch_{index}"


def _has_values(raw: object) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(raw)


# ---------------------
# Graph-policy accounting
# ---------------------

logger = logging.getLogger(__name__)


class EagerFallbackRecorder:
    """Rate-limit warning logs while counting every eager fallback."""

    def __init__(self) -> None:
        self._warned: set[tuple[Any, ...]] = set()
        self.counts: dict[str, int] = {}

    def record(self, warning: EagerFallbackWarning, *, stats: Any | None = None) -> None:
        key = (
            warning.reason.value,
            warning.topology_id,
            repr(warning.capacity_key),
            warning.mode.value,
            tuple(mode.value for mode in warning.op_modes),
        )
        self.counts[warning.reason.value] = self.counts.get(warning.reason.value, 0) + 1
        _bump_stats(stats, warning)
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(
            "forward eager fallback: reason=%s mode=%s rows=%d tokens=%d padded_rows=%d padded_tokens=%d topology=%s backend=%s",
            warning.reason.value,
            warning.mode.value,
            warning.rows,
            warning.tokens,
            warning.padded_rows,
            warning.padded_tokens,
            warning.topology_id,
            warning.backend,
        )


def _bump_stats(stats: Any | None, warning: EagerFallbackWarning) -> None:
    if stats is None:
        return
    for attr in ("cuda_graph_fallbacks", "forward_eager_fallbacks"):
        try:
            setattr(stats, attr, int(getattr(stats, attr, 0)) + 1)
        except Exception:
            pass
    try:
        setattr(
            stats,
            "forward_eager_tokens",
            int(getattr(stats, "forward_eager_tokens", 0)) + warning.tokens,
        )
        setattr(
            stats, "forward_eager_rows", int(getattr(stats, "forward_eager_rows", 0)) + warning.rows
        )
    except Exception:
        pass


# ---------------------
# Device batch construction
# ---------------------


class UnifiedForwardBatchBuilder:
    """Build the single device snapshot consumed by the executor."""

    def __init__(
        self,
        *,
        runtime_builder: Any | None = None,
        kv_pool: Any | None = None,
        request_states: Any | None = None,
        default_device: torch.device | str = "cpu",
    ) -> None:
        self.runtime_builder = runtime_builder
        self.kv_pool = kv_pool
        self.request_states = request_states
        self.default_device = torch.device(default_device)

    def build(
        self,
        plan: ForwardPlan,
        *,
        device: torch.device | str | None = None,
    ) -> ForwardBatch:
        plan.validate()
        target_device = torch.device(device) if device is not None else self.default_device
        runtime_batch = self._try_runtime_text_batch(plan, target_device)
        if runtime_batch is not None:
            return runtime_batch
        return self._build_generic(plan, target_device)

    def _try_runtime_text_batch(
        self,
        plan: ForwardPlan,
        device: torch.device,
    ) -> ForwardBatch | None:
        if self.runtime_builder is None or self.kv_pool is None or self.request_states is None:
            return None
        if not plan.rows or any(row.mode not in _TEXT_MODES for row in plan.rows):
            return None
        text = UniForwardBatch.from_ops(plan.ops).as_text(
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED
        )
        return self.runtime_builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
            request_states=self.request_states,
        )

    def _build_generic(self, plan: ForwardPlan, device: torch.device) -> ForwardBatch:
        input_ids: list[int] = []
        positions: list[int] = []
        is_gen: list[bool] = []
        last_token_indices: list[int] = []
        token_offset = 0
        for row in plan.rows:
            if row.token_span is None:
                continue
            for offset, token in enumerate(row.token_span.token_ids):
                input_ids.append(int(token))
                positions.append(int(row.token_span.position_start + offset))
                is_gen.append(False)
            if row.token_span.q_len:
                token_offset += row.token_span.q_len
                last_token_indices.append(token_offset - 1)
        padded_tokens = max(plan.shape.padded_token_count, len(input_ids))
        if padded_tokens > len(input_ids):
            pad = padded_tokens - len(input_ids)
            input_ids.extend([0] * pad)
            positions.extend([0] * pad)
            is_gen.extend([True] * pad)
        input_tensor = (
            torch.tensor(input_ids, dtype=torch.long, device=device) if input_ids else None
        )
        position_tensor = (
            torch.tensor(positions, dtype=torch.long, device=device) if positions else None
        )
        is_gen_tensor = torch.tensor(is_gen, dtype=torch.bool, device=device) if is_gen else None
        last_token_tensor = (
            torch.tensor(last_token_indices, dtype=torch.long, device=device)
            if last_token_indices
            else None
        )
        return ForwardBatch(
            forward_mode=plan.forward_mode,
            req_ids=plan.req_ids,
            op_modes=plan.op_modes,
            ops=plan.ops,
            device=device,
            input_ids=input_tensor,
            positions=position_tensor,
            last_token_indices=last_token_tensor,
            num_token_non_padded=sum(
                row.token_span.q_len for row in plan.rows if row.token_span is not None
            ),
            padded_num_tokens=padded_tokens,
            is_gen=is_gen_tensor,
            segments=_segment_specs(plan),
            denoise=_denoise_inputs(plan, device),
            encode=_encode_inputs(plan),
            commit=_commit_inputs(plan),
            sampling=None,
        )


def _segment_specs(plan: ForwardPlan) -> tuple[SegmentSpec, ...]:
    specs: list[SegmentSpec] = []
    start = 0
    for segment in plan.segments:
        specs.append(
            SegmentSpec(
                start=start,
                length=int(segment.q_len),
                visible_policy=segment.visible_policy,
                branch_id=int(segment.branch_id),
                is_gen=segment.modality is ForwardModality.GENERATION,
            )
        )
        start += int(segment.q_len)
    return tuple(specs)


def _denoise_inputs(plan: ForwardPlan, device: torch.device) -> DenoiseInputs | None:
    row = next((row for row in plan.rows if row.denoise is not None), None)
    if row is None or row.denoise is None:
        return None
    branches = tuple(
        BranchSpec(
            name=name,
            kv_source=KvSource.SCRATCH,
            kv_handle=0,
            kv_len=row.denoise.image_token_count,
            position=branch_index,
        )
        for branch_index, name in enumerate(row.denoise.branch_ids)
    )
    cfg = CfgPlan(branches=branches)
    return DenoiseInputs(
        latent_handle=int(row.denoise.latent_handle or 0),
        step_index=row.denoise.step_index,
        total_steps=row.denoise.total_steps,
        t=_scalar_tensor(row.op.get("t"), device),
        t_next=_scalar_tensor(row.op.get("t_next"), device),
        grid_hw=row.denoise.grid_hw,
        branches=branches,
        cfg=cfg,
        rng_handle=int(row.op.get("rng_handle") or 0),
    )


def _encode_inputs(plan: ForwardPlan) -> EncodeInputs | None:
    row = next((row for row in plan.rows if row.encode is not None), None)
    if row is None or row.encode is None:
        return None
    return EncodeInputs(
        kind=row.encode.kind,
        out_handle=int(row.encode.out_handle or 0),
        mm_hash=row.encode.mm_hash,
        cond_pos=int(row.op.get("cond_pos") or 0),
    )


def _commit_inputs(plan: ForwardPlan) -> CommitInputs | None:
    row = next((row for row in plan.rows if row.commit is not None), None)
    if row is None or row.commit is None:
        return None
    return CommitInputs(
        latent_handle=int(row.commit.latent_handle or 0),
        fold_back=bool(row.commit.fold_back),
    )


def _scalar_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        return torch.tensor([float(value)], dtype=torch.float32, device=device)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("denoise scalar tensors must be numeric") from exc


# ---------------------
# Model descriptor
# ---------------------


@dataclass(frozen=True)
class ForwardModelModules:
    embed_text: Callable[..., Any] | None = None
    embed_generation: Callable[..., Any] | None = None
    decoder: Callable[..., Any] | None = None
    logits: Callable[..., Any] | None = None
    velocity: Callable[..., Any] | None = None
    encode: Callable[..., Any] | None = None
    commit: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ForwardModelDescriptor:
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    vocab_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    attention_scale: float = 1.0
    kv_page_size: int | None = None
    supports_text: bool = False
    supports_denoise: bool = False
    supports_encode: bool = False
    supports_commit: bool = False
    graph_capture: Mapping[str, Any] = field(default_factory=dict)
    modules: ForwardModelModules = field(default_factory=ForwardModelModules)
    variant: str = "default"

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise invalid_descriptor(f"forward model descriptor {name} must be positive")
        if self.supports_text:
            if int(self.vocab_size) <= 0:
                raise invalid_descriptor("text-capable descriptor must declare vocab_size")
            if self.modules.decoder is None and self.modules.logits is None:
                raise invalid_descriptor("text-capable descriptor is missing text neural surfaces")
        if self.supports_denoise and self.modules.velocity is None:
            raise invalid_descriptor("denoise-capable descriptor is missing velocity surface")
        if self.supports_encode and self.modules.encode is None:
            raise invalid_descriptor("encode-capable descriptor is missing encode surface")
        if self.supports_commit and self.modules.commit is None:
            raise invalid_descriptor("commit-capable descriptor is missing commit surface")


def descriptor_from_model(model: Any) -> ForwardModelDescriptor:
    device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
    dtype = _dtype(getattr(model, "dtype", None))
    modules = ForwardModelModules(
        embed_text=_first_callable(model, ("embed_tokens", "packed_text_embeddings")),
        embed_generation=_first_callable(
            model, ("embed_generation", "prepare_generation_embeddings")
        ),
        decoder=_first_callable(model, ("forward", "decoder_forward", "packed_decoder_forward")),
        logits=_first_callable(model, ("compute_logits", "logits", "lm_head"))
        or _first_overridden_callable(model, ("run_text_logits_batch", "run_text_logits")),
        velocity=_first_callable(model, ("predict_velocity", "velocity", "project_velocity")),
        encode=_first_callable(model, ("encode_image", "encode_latents")),
        commit=_first_callable(model, ("decode_image", "commit")),
    )
    supports_text = (
        callable(getattr(model, "forward", None))
        or _overrides(model, "run_text_logits_batch")
        or _overrides(model, "run_text_logits")
    )
    raw_vocab_size = int(_attr(model, ("vocab_size",), 0))
    descriptor = ForwardModelDescriptor(
        device=device,
        dtype=dtype,
        hidden_size=max(1, int(_attr(model, ("hidden_size", "d_model"), 1))),
        vocab_size=max(1 if supports_text else 0, raw_vocab_size),
        num_layers=max(1, int(_attr(model, ("num_layers", "n_layers"), 1))),
        num_q_heads=max(1, int(_attr(model, ("num_q_heads", "num_attention_heads"), 1))),
        num_kv_heads=max(1, int(_attr(model, ("num_kv_heads", "num_key_value_heads"), 1))),
        head_dim=max(1, int(_attr(model, ("head_dim",), 1))),
        attention_scale=float(_attr(model, ("attention_scale",), 1.0)),
        kv_page_size=_descriptor_optional_int(_attr(model, ("block_size", "kv_page_size"), None)),
        supports_text=supports_text,
        supports_denoise=_overrides(model, "predict_velocity"),
        supports_encode=_overrides(model, "encode_image") or _overrides(model, "encode_latents"),
        supports_commit=_overrides(model, "decode_image"),
        graph_capture={
            "step": callable(getattr(model, "query_geometry", None)),
        },
        modules=modules,
        variant=type(model).__name__,
    )
    descriptor.validate()
    return descriptor


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    parsed = getattr(torch, str(value), None)
    return parsed if isinstance(parsed, torch.dtype) else torch.float32


def _attr(model: Any, names: tuple[str, ...], default: Any) -> Any:
    config = getattr(model, "config", None)
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
        if config is not None and hasattr(config, name):
            return getattr(config, name)
    return default


def _first_callable(model: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(model, name, None)
        if callable(value):
            return value
    return None


def _first_overridden_callable(model: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(model, name, None)
        if callable(value) and _overrides(model, name):
            return value
    return None


def _descriptor_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _overrides(model: Any, name: str) -> bool:
    from uniserve_worker.contracts.model_protocols import ModelHooks

    hook = getattr(type(model), name, None)
    default = getattr(ModelHooks, name, None)
    return hook is not None and hook is not default


# ---------------------
# Family adapter dispatch
# ---------------------

_TEXT_DRIVER_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


@runtime_checkable
class ForwardModelAdapter(Protocol):
    """Small neural execution seam consumed by ``ForwardExecutor``."""

    descriptor: ForwardModelDescriptor

    def forward(self, batch: ForwardBatch) -> ForwardResult: ...


@dataclass(frozen=True)
class ForwardAdapterContext:
    """Per-group execution context bound around one adapter forward."""

    dispatch_batch: UniForwardBatch
    group: tuple[tuple[int, Mapping[str, Any]], ...]
    defer_text_cpu_results: bool = False


class WorkerForwardAdapter:
    """Adapter over the worker's model and system execution services."""

    def __init__(
        self,
        *,
        model: Any,
        request_states: Any,
        text_driver: Any,
        denoise_driver: Any,
        encode_driver: Any,
        image_decode_driver: Any,
        descriptor: ForwardModelDescriptor | None = None,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
        mixed_proof_callback: Any | None = None,
    ) -> None:
        self.model = model
        self.request_states = request_states
        self.text_driver = text_driver
        self.denoise_driver = denoise_driver
        self.encode_driver = encode_driver
        self.image_decode_driver = image_decode_driver
        self.descriptor = descriptor or descriptor_from_model(model)
        self.defer_sampling = bool(defer_sampling)
        self.tensor_store = tensor_store
        self.mixed_proof_callback = mixed_proof_callback
        self._context: ForwardAdapterContext | None = None
        self._has_text_forward = hasattr(model, "forward")
        self._has_text_logits_batch = _overrides_model_hook(model, "run_text_logits_batch")
        self._is_text_capable = self._has_text_forward or self._has_text_logits_batch
        self._has_predict_velocity = _overrides_model_hook(model, "predict_velocity")
        self._has_decode_image = _overrides_model_hook(model, "decode_image")
        self._has_encode = _overrides_model_hook(model, "encode_image") or _overrides_model_hook(
            model, "encode_latents"
        )
        self._whole_batch_forward = bool(getattr(model, "whole_batch_forward", False))

    @contextmanager
    def bind(
        self,
        *,
        dispatch_batch: UniForwardBatch,
        group: Sequence[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> Iterator[None]:
        previous = self._context
        self._context = ForwardAdapterContext(
            dispatch_batch=dispatch_batch,
            group=tuple(group),
            defer_text_cpu_results=bool(defer_text_cpu_results),
        )
        try:
            yield
        finally:
            self._context = previous

    def forward(self, batch: ForwardBatch) -> ForwardResult:
        context = self._context
        if context is None:
            fb = UniForwardBatch.from_ops(batch.ops)
            group = tuple(enumerate(batch.ops))
            defer_text_cpu_results = False
        else:
            fb = context.dispatch_batch
            group = context.group
            defer_text_cpu_results = context.defer_text_cpu_results
        result = self.dispatch(
            fb,
            list(group),
            defer_text_cpu_results=defer_text_cpu_results,
        )
        if isinstance(result, ForwardResult):
            return result
        outputs = result
        if len(outputs) != len(batch.ops):
            raise invalid_descriptor(
                f"adapter returned {len(outputs)} outputs for {len(batch.ops)} forward ops"
            )
        return ForwardResult(runtime_outputs=tuple(outputs))

    def dispatch(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        *,
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any]:
        if self._whole_batch_forward and (
            fb.mode is ForwardMode.MIXED or fb.mode in _TEXT_DRIVER_MODES
        ):
            result = self._run_model_forward(fb)
        elif _can_run_private_forward_adapter(self.model, fb):
            if fb.mode is ForwardMode.MIXED and callable(self.mixed_proof_callback):
                self.mixed_proof_callback(fb, group)
            with profile_range("uniserve.forward_adapter.forward"):
                result = _run_private_forward_adapter(
                    model=self.model,
                    batch=fb,
                    group=group,
                    request_states=self.request_states,
                    defer_text_cpu_results=defer_text_cpu_results,
                )
        elif fb.mode is ForwardMode.MIXED:
            result = self._run_mixed_mode(fb, group, defer_text_cpu_results)
        elif fb.mode in _TEXT_DRIVER_MODES:
            result = self._run_text_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.DENOISE:
            result = self._run_denoise_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.COMMIT:
            result = self._run_commit_mode(fb, group, defer_text_cpu_results)
        elif fb.mode is ForwardMode.ENCODE:
            result = self._run_encode_mode(fb, group, defer_text_cpu_results)
        else:
            result = None
        if result is None:
            result = self._run_whole_batch_forward(fb, group, defer_text_cpu_results)
        if result is None:
            raise capability_mismatch(
                f"model advertises ops for mode {fb.mode.value!r} but implements no "
                f"matching forward adapter path"
            )
        return result

    def can_run_forward(self, fb: UniForwardBatch) -> bool:
        """Whether this adapter can execute ``fb`` as one unified forward."""
        if fb.mode is not ForwardMode.MIXED:
            return True
        if self._whole_batch_forward or _can_run_private_forward_adapter(self.model, fb):
            return True
        return all(mode in _TEXT_DRIVER_MODES for mode in fb.op_modes)

    def _run_mixed_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        if self._whole_batch_forward:
            return self._run_model_forward(fb)
        if not all(mode in _TEXT_DRIVER_MODES for mode in fb.op_modes):
            return None
        if callable(self.mixed_proof_callback):
            self.mixed_proof_callback(fb, group)
        return self._run_text_driver(fb, defer_text_cpu_results)

    def _run_text_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del group
        if self._whole_batch_forward:
            return self._run_model_forward(fb)
        if self._is_text_capable:
            return self._run_text_driver(fb, defer_text_cpu_results)
        return None

    def _run_text_driver(
        self,
        fb: UniForwardBatch,
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any]:
        forward_logits = getattr(self.text_driver, "forward_logits", None)
        if callable(forward_logits):
            with profile_range("uniserve.forward_adapter.text_logits"):
                text_result = forward_logits(
                    fb,
                    self.request_states,
                    self.model,
                    defer_cpu_results=defer_text_cpu_results,
                    defer_sampling=self.defer_sampling,
                )
            if text_result is not None:
                req_ids = tuple(int(req_id) for req_id in getattr(text_result, "req_ids", ()))
                expected_req_ids = tuple(int(op["req_id"]) for op in fb.ops)
                if req_ids != expected_req_ids:
                    raise invalid_descriptor(
                        "text logits result req_ids must align with forward ops"
                    )
                return ForwardResult(
                    text_logits=text_result.logits,
                    text_cuda_ready_start_event=getattr(
                        text_result,
                        "cuda_ready_start_event",
                        None,
                    ),
                )
        with profile_range("uniserve.forward_adapter.text_driver"):
            return self.text_driver.step(
                fb,
                self.request_states,
                self.model,
                defer_cpu_results=defer_text_cpu_results,
                defer_sampling=self.defer_sampling,
                tensor_store=self.tensor_store,
            )

    def _run_denoise_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_predict_velocity:
            return None
        forward_result = getattr(self.denoise_driver, "forward_result", None)
        items = [
            (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op) for _, op in group
        ]
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.denoise_forward"):
                kwargs: dict[str, Any] = {"row_indices": tuple(range(len(group)))}
                if _accepts_keyword(forward_result, "graph_mode"):
                    kwargs["graph_mode"] = "eager"
                result = forward_result(
                    items,
                    self.model,
                    **kwargs,
                )
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.denoise_driver"):
            kwargs = (
                {"graph_mode": "eager"}
                if _accepts_keyword(self.denoise_driver.step_many, "graph_mode")
                else {}
            )
            return self.denoise_driver.step_many(items, self.model, **kwargs)

    def _run_commit_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_decode_image:
            return None
        items = [
            (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op) for _, op in group
        ]
        forward_result = getattr(self.image_decode_driver, "forward_result", None)
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.commit_forward"):
                result = forward_result(items, self.model, row_indices=tuple(range(len(group))))
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.image_decode"):
            return [
                self.image_decode_driver.step(
                    int(op["req_id"]),
                    self.request_states.get(int(op["req_id"])),
                    self.model,
                    op,
                )
                for _, op in group
            ]

    def _run_encode_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> ForwardResult | list[Any] | None:
        del group, defer_text_cpu_results
        if not self._has_encode:
            return None
        forward_result = getattr(self.encode_driver, "forward_result", None)
        if callable(forward_result):
            with profile_range("uniserve.forward_adapter.encode_forward"):
                result = forward_result(fb, self.model, row_indices=tuple(range(len(fb.ops))))
            if result is not None:
                return result
        with profile_range("uniserve.forward_adapter.encode_driver"):
            return self.encode_driver.step(fb, self.model)

    def _run_whole_batch_forward(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del group, defer_text_cpu_results
        if not self._whole_batch_forward:
            return None
        return self._run_model_forward(fb)

    def _run_model_forward(self, fb: UniForwardBatch) -> list[Any]:
        with profile_range("uniserve.forward_adapter.model_forward"):
            result = self.model.forward(fb)
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
            raise invalid_descriptor("model forward must return one result per op")
        return list(result)


def _overrides_model_hook(model: Any, name: str) -> bool:
    hook = getattr(type(model), name, None)
    default = getattr(ModelHooks, name, None)
    return hook is not None and hook is not default


def _can_run_private_forward_adapter(model: Any, fb: UniForwardBatch) -> bool:
    hook = getattr(model, "_run_forward_adapter", None)
    if not callable(hook):
        return False
    modes = tuple(getattr(fb, "op_modes", ()))
    if not modes:
        return False
    mode_set = set(modes)
    if not mode_set <= _SEGMENT_PRODUCING_MODES:
        return False
    if not mode_set & _SEGMENT_PRODUCING_MODES:
        return False
    return not any(_has_values(op.get("spec_token_ids")) for op in fb.ops)


def _run_private_forward_adapter(
    *,
    model: Any,
    batch: UniForwardBatch,
    group: Sequence[tuple[int, Mapping[str, Any]]],
    request_states: Any,
    defer_text_cpu_results: bool,
) -> ForwardResult | list[Any]:
    hook = getattr(model, "_run_forward_adapter", None)
    if not callable(hook):
        raise capability_mismatch("model does not implement a private forward adapter")
    kwargs: dict[str, Any] = {"request_states": request_states, "group": list(group)}
    if _accepts_deferred_text_cpu_results(hook):
        kwargs["defer_text_cpu_results"] = bool(defer_text_cpu_results)
    result = hook(batch, **kwargs)
    if isinstance(result, ForwardResult):
        return result
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        raise invalid_descriptor("private forward adapter must return one result per operation")
    if len(result) != len(group):
        raise invalid_descriptor("private forward adapter returned the wrong number of results")
    return list(result)


def _accepts_deferred_text_cpu_results(hook: Any) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "defer_text_cpu_results" in signature.parameters


def _accepts_keyword(hook: Any, name: str) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


# ---------------------
# Eager forward runner
# ---------------------


class ForwardRunner(ABC):
    @abstractmethod
    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult: ...


class EagerForwardRunner(ForwardRunner):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult:
        del plan
        if forward_fn is not None:
            return coerce_forward_result(forward_fn(batch))
        if self.model is None or not callable(getattr(self.model, "forward", None)):
            raise TypeError("eager forward requires a model.forward(batch) callable")
        return coerce_forward_result(self.model.forward(batch))


# ---------------------
# One-selection forward executor
# ---------------------


class ForwardExecutor:
    def __init__(
        self,
        *,
        model: Any | None = None,
        graph_runner: Any | None = None,
        eager_runner: EagerForwardRunner | None = None,
        graph_policy: ForwardGraphPolicy | None = None,
        fallback_recorder: EagerFallbackRecorder | None = None,
    ) -> None:
        self.model = model
        self.graph_runner = graph_runner
        self.eager_runner = eager_runner or EagerForwardRunner(model)
        self.graph_policy = graph_policy or ForwardGraphPolicy()
        self.fallback_recorder = fallback_recorder or EagerFallbackRecorder()

    def execute(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], Any] | None = None,
    ) -> ForwardResult:
        policy = plan.graph_policy or self.graph_policy
        stats = get_forward_context().stats
        if policy.graph_selection_delegated:
            return self.eager_runner.run(batch, plan, forward_fn=forward_fn)
        if policy.prefer_graph and self.graph_runner is not None:
            try:
                graph_result = self.graph_runner.run(
                    batch,
                    plan,
                    allow_capture=policy.allow_capture,
                )
            except Exception as exc:
                warning = self._warning(EagerFallbackReason.REPLAY_FAILURE, plan)
                if policy.strict:
                    raise StrictForwardGraphError(warning) from exc
                self.fallback_recorder.record(warning, stats=stats)
            else:
                if graph_result is not None:
                    return graph_result
                warning = self._warning(EagerFallbackReason.GRAPH_MISS, plan)
                if policy.strict:
                    raise StrictForwardGraphError(warning)
                self.fallback_recorder.record(warning, stats=stats)
        else:
            warning = self._warning(EagerFallbackReason.GRAPH_DISABLED, plan)
            if policy.strict:
                raise StrictForwardGraphError(warning)
            self.fallback_recorder.record(warning, stats=stats)
        return self.eager_runner.run(batch, plan, forward_fn=forward_fn)

    @staticmethod
    def _warning(reason: EagerFallbackReason, plan: ForwardPlan) -> EagerFallbackWarning:
        return EagerFallbackWarning(
            reason=reason,
            mode=plan.forward_mode,
            op_modes=plan.op_modes,
            tokens=plan.shape.token_count,
            rows=plan.shape.row_count,
            padded_tokens=plan.shape.padded_token_count,
            padded_rows=plan.shape.padded_row_count,
        )


# ---------------------
# Result projection and postprocess
# ---------------------


class ForwardPostprocessor:
    def apply(self, plan: ForwardPlan, result: ForwardResult) -> list[Any]:
        result.validate_for_plan(plan)
        if result.runtime_outputs is not None:
            runtime_outputs = self._normalize_outputs(plan, result.runtime_outputs)
            side_effects = plan.runtime_handles.get("postprocess_side_effects")
            if callable(side_effects):
                side_effects(runtime_outputs)
            return runtime_outputs
        if self._can_apply_text_batch(plan, result):
            batch_outputs = self._normalize_outputs(plan, self._apply_text_batch(plan, result))
            side_effects = plan.runtime_handles.get("postprocess_side_effects")
            if callable(side_effects):
                side_effects(batch_outputs)
            return batch_outputs
        text_outputs = self._apply_text_entries(plan, result) if result.text_postprocess else {}
        outputs: list[Any] = []
        text_row = 0
        for slot in plan.output_slots:
            row = plan.rows[slot.row_index]
            if slot.kind is ForwardOutputKind.TEXT_TOKEN:
                if slot.row_index in text_outputs:
                    outputs.append(text_outputs[slot.row_index])
                    continue
                if result.text_logits is None:
                    raise invalid_descriptor("text output slot requires logits")
                logits_rows = result.text_logits.reshape(-1, result.text_logits.shape[-1])
                outputs.append(self._sample_text(plan, row.req_id, row.op, logits_rows[text_row]))
                text_row += 1
            elif slot.kind is ForwardOutputKind.DENOISE_STEP:
                outputs.append(self._apply_denoise(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.COMMIT:
                outputs.append(self._apply_commit(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.ENCODE:
                outputs.append(self._apply_encode(plan, slot.row_index, result))
            else:
                outputs.append({"req_id": row.req_id})
        outputs = self._normalize_outputs(plan, outputs)
        side_effects = plan.runtime_handles.get("postprocess_side_effects")
        if callable(side_effects):
            side_effects(outputs)
        return outputs

    def _normalize_outputs(
        self,
        plan: ForwardPlan,
        outputs: tuple[Any, ...] | list[Any],
    ) -> list[Any]:
        normalizer = plan.runtime_handles.get("output_normalizer")
        if callable(normalizer):
            return [normalizer(output) for output in outputs]
        return list(outputs)

    @staticmethod
    def _can_apply_text_batch(plan: ForwardPlan, result: ForwardResult) -> bool:
        return (
            isinstance(result.text_logits, torch.Tensor)
            and bool(plan.output_slots)
            and all(slot.kind is ForwardOutputKind.TEXT_TOKEN for slot in plan.output_slots)
            and plan.runtime_handles.get("dispatch_batch") is not None
        )

    def _apply_text_batch(self, plan: ForwardPlan, result: ForwardResult) -> list[Any]:
        text = plan.runtime_handles.get("dispatch_batch").as_text(
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED
        )
        req_ids = [int(row.req_id) for row in plan.rows]
        if result.text_logits is None:
            raise invalid_descriptor("text batch postprocess requires logits")
        logits_batch = self._text_logits_rows(result.text_logits, len(req_ids))
        if (
            plan.runtime_handles.get("defer_sampling")
            and plan.runtime_handles.tensor_store is not None
        ):
            published_outputs = self._publish_logits(
                list(text.ops),
                req_ids,
                logits_batch,
                plan.runtime_handles.tensor_store,
            )
            self._publish_decode_position_relays(text, logits_batch.device, plan)
            self._advance_text_kv_lengths(text, plan)
            return published_outputs
        sampled_outputs = self._sample_text_logits_batch(
            plan,
            list(text.ops),
            req_ids,
            logits_batch,
            defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results")),
            cuda_ready_start_event=result.text_cuda_ready_start_event,
        )
        self._publish_decode_position_relays(text, logits_batch.device, plan)
        self._advance_text_kv_lengths(text, plan)
        return sampled_outputs

    @staticmethod
    def _text_logits_rows(logits: torch.Tensor, row_count: int) -> torch.Tensor:
        if logits.ndim != 2:
            logits = logits.reshape(-1, logits.shape[-1])
        if int(logits.shape[0]) < int(row_count):
            raise invalid_descriptor("text logits row count is smaller than text output slots")
        return logits[: int(row_count)]

    def _sample_text_logits_batch(
        self,
        plan: ForwardPlan,
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        *,
        defer_cpu_results: bool,
        cuda_ready_start_event: torch.cuda.Event | None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return [
                TextTokenOutput(
                    req_id=req_id,
                    sampled_token_id=int(torch.argmax(logits_batch[row].float()).item()),
                )
                for row, req_id in enumerate(req_ids)
            ]
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op, req_id in zip(ops, req_ids, strict=True):
            state = request_states.get(int(req_id))
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_batch,
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=defer_cpu_results,
            enable_cuda_timing=cuda_ready_start_event is not None,
        )
        if is_deferred_sampling_result(sampling_result):
            sampling_result.set_ready_start_event(cuda_ready_start_event)
            outputs: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(int(req_id))
                relay_token_tensor = sampling_result.device_tokens[row : row + 1]
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None,
                    token_tensor=relay_token_tensor,
                )
                outputs.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=relay_token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return outputs

        immediate_result = finalize_sampling_result(sampling_result)
        outputs = []
        for row, (req_id, (tok, lp, top)) in enumerate(
            zip(req_ids, immediate_result.samples, strict=True)
        ):
            _DECODE_RELAY.publish_sample(
                request_states.get(int(req_id)),
                token_id=int(tok),
                token_tensor=immediate_result.device_tokens[row : row + 1],
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(tok),
                    sampled_logprob=lp,
                    top_logprobs=(
                        [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                        if top
                        else None
                    ),
                )
            )
        record_component_elapsed(stats, "text_sample", start)
        return outputs

    @staticmethod
    def _publish_logits(
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        tensor_store: Any,
    ) -> list[dict[str, Any]]:
        if logits_batch.ndim != 2 or int(logits_batch.shape[0]) != len(ops):
            raise invalid_descriptor("deferred-sampler logits must be shaped [ops, vocab]")
        if logits_batch.is_cuda:
            torch.cuda.synchronize(logits_batch.device)
        results: list[dict[str, Any]] = []
        for row, op in enumerate(ops):
            handle = tensor_store.publish(logits_batch[row].contiguous(), "logits")
            result: dict[str, Any] = {"req_id": int(op["req_id"]), "logits_handle": int(handle)}
            locator = tensor_store.locator_of(handle)
            if locator is not None:
                result["locator"] = base64.b64encode(locator).decode("ascii")
            results.append(result)
        return results

    @staticmethod
    def _publish_decode_position_relays(text: Any, device: torch.device, plan: ForwardPlan) -> None:
        if text.mode is not ForwardMode.DECODE:
            return
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        states = [request_states.get(int(req_id)) for req_id in text.req_ids]
        positions = [int(pos_range[1]) for pos_range in text.pos_ranges]
        _DECODE_RELAY.publish_positions(states, position_ids=positions, device=device)
        record_component_elapsed(stats, "text_decode_position_store", start)

    @staticmethod
    def _advance_text_kv_lengths(text: Any, plan: ForwardPlan) -> None:
        if text.mode not in _TEXT_MODES:
            return
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges, strict=True):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane="text")

    @staticmethod
    def _sample_text(
        plan: ForwardPlan, req_id: int, op: Any, logits: torch.Tensor
    ) -> dict[str, Any]:
        request_states = plan.runtime_handles.request_states
        if request_states is not None:
            state = request_states.get(int(req_id))
            return sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        token = int(torch.argmax(logits.float()).item())
        return {"req_id": int(req_id), "sampled_token_id": token}

    def _apply_text_entries(self, plan: ForwardPlan, result: ForwardResult) -> dict[int, Any]:
        entries = tuple(result.text_postprocess or ())
        if not entries:
            return {}
        if result.text_logits is None:
            raise invalid_descriptor("text postprocess entries require text logits")
        logits_rows = self._text_logits_rows(result.text_logits, len(entries))
        entries_by_index = sorted(entries, key=lambda entry: int(entry.logits_index))
        if [int(entry.logits_index) for entry in entries_by_index] != list(
            range(len(entries_by_index))
        ):
            raise invalid_descriptor("text postprocess logits indices must be contiguous")
        request_states = plan.runtime_handles.request_states
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator | None] = []
        for entry in entries_by_index:
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            params.append(dict(getattr(state, "sampling", {}) or {}))
            recent.append(row.op.get("recent_tokens") or [])
            allowed.append(row.op.get("allowed_tokens"))
            suppress.append(row.op.get("suppress_tokens"))
            generators.append(
                None
                if state is None
                else state.device_rng(logits_rows.device, stream="text_sampling")
            )
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_rows[: len(entries_by_index)],
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=bool(plan.runtime_handles.get("defer_text_cpu_results")),
        )
        deferred: Any | None
        if is_deferred_sampling_result(sampling_result):
            deferred = sampling_result
            samples = []
            device_tokens = sampling_result.device_tokens
        else:
            deferred = None
            immediate_result = finalize_sampling_result(sampling_result)
            samples = immediate_result.samples
            device_tokens = immediate_result.device_tokens
        promotions = [
            entry.kv_promotion for entry in entries_by_index if entry.kv_promotion is not None
        ]
        if promotions:
            num_layers = max(int(entry.num_layers) for entry in entries_by_index)
            copy_paged_text_cache_spans(
                promotions,
                num_layers=num_layers,
                missing_message="forward text K/V span is missing from staged cache",
            )
        outputs: dict[int, Any] = {}
        for sample_index, entry in enumerate(entries_by_index):
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            logits = logits_rows[int(entry.logits_index) : int(entry.logits_index) + 1].unsqueeze(0)
            self._publish_text_entry_state(entry, logits)
            token_tensor = device_tokens[sample_index : sample_index + 1]
            position_tensor = torch.tensor(
                [int(entry.position_id)],
                dtype=torch.long,
                device=logits_rows.device,
            )
            if state is not None:
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None if deferred is not None else int(samples[sample_index].token_id),
                    token_tensor=token_tensor,
                )
                _DECODE_RELAY.publish_position(
                    state,
                    position_id=int(entry.position_id),
                    position_tensor=position_tensor,
                )
            output: Any
            if deferred is not None:
                if state is None:
                    raise invalid_descriptor("deferred text postprocess requires request state")
                output = DeferredTextSeqResult(
                    req_id=int(entry.req_id),
                    row=sample_index,
                    state=state,
                    sampling_result=deferred,
                    relay_token_tensor=token_tensor,
                )
            else:
                sample = samples[sample_index]
                top_logprobs = (
                    [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                    if sample.top_logprobs is not None
                    else None
                )
                output = TextTokenOutput(
                    req_id=int(entry.req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=top_logprobs,
                )
            outputs[int(row.row_index)] = output
        return outputs

    @staticmethod
    def _publish_text_entry_state(entry: TextPostprocessEntry, logits: torch.Tensor) -> None:
        state = entry.program_state
        cond = getattr(state, "cond", None)
        if cond is not None:
            cond.t_index = int(entry.position_id) - 1
            cond.last_logits = logits
            cond.last_token_id = int(entry.last_input_token)
        persistent_cache = entry.persistent_cache
        if persistent_cache is not None:
            persistent_cache.length = int(entry.kv_new_length)
        if (
            entry.staged_cache is not None
            and persistent_cache is not None
            and entry.staged_cache is not persistent_cache
            and entry.mark_staging_advanced is not None
        ):
            entry.mark_staging_advanced(
                entry.staged_cache, persistent_cache, int(entry.kv_new_length)
            )

    @staticmethod
    def _apply_denoise(plan: ForwardPlan, row_index: int, result: ForwardResult) -> FlowOutput:
        row = plan.rows[row_index]
        denoise = row.denoise
        if denoise is None:
            raise invalid_descriptor("denoise output slot references a non-denoise row")
        velocities = result.denoise_velocities or {}
        update = (result.denoise_updates or {}).get(int(row_index))
        if update is not None:
            branch_velocities: dict[Any, torch.Tensor] = {}
            for branch_id, branch_name in enumerate(update.branch_names):
                key = DenoiseBranchKey(row_index, branch_id)
                velocity = velocities.get(key)
                if velocity is None:
                    raise invalid_descriptor("denoise output is missing branch velocity")
                branch_velocities[branch_name] = velocity
            combined = update.combine_velocity(branch_velocities)
            if not isinstance(combined, torch.Tensor):
                raise invalid_descriptor("denoise combined velocity must be a tensor")
            if tuple(combined.shape) != tuple(update.latent.shape):
                raise invalid_descriptor("denoise combined velocity shape must match latent shape")
            updated = euler_step(update.latent, combined, update.t, update.t_next)
            update.accept_update(updated)
            done = update.step_index + 1 >= update.total_steps
            return FlowOutput(
                req_id=update.req_id,
                denoise_done=done,
                num_steps_done=update.step_index + 1,
            )
        for branch_id in range(denoise.branch_count):
            key = DenoiseBranchKey(row_index, branch_id)
            if key not in velocities:
                raise invalid_descriptor("denoise output is missing branch velocity")
        done = denoise.step_index + 1 >= denoise.total_steps
        return FlowOutput(
            req_id=row.req_id,
            denoise_done=done,
            num_steps_done=denoise.step_index + 1,
        )

    @staticmethod
    def _apply_commit(plan: ForwardPlan, row_index: int, result: ForwardResult) -> CommitOutput:
        row = plan.rows[row_index]
        if row.commit is None:
            raise invalid_descriptor("commit output slot references a non-commit row")
        if result.commit_outputs is None:
            return CommitOutput(req_id=row.req_id)
        return _commit_output_from_value(
            int(row.req_id),
            plan.runtime_handles.request_states.get(int(row.req_id))
            if plan.runtime_handles.request_states is not None
            else None,
            row.op,
            result.commit_outputs[int(row_index)],
        )

    @staticmethod
    def _apply_encode(plan: ForwardPlan, row_index: int, result: ForwardResult) -> EncodeOutput:
        row = plan.rows[row_index]
        if row.encode is None:
            raise invalid_descriptor("encode output slot references a non-encode row")
        if result.encode_outputs is None:
            return EncodeOutput(req_id=row.req_id, encoder_handle=0)
        return _coerce_encode_output(result.encode_outputs[int(row_index)])


def _commit_output_from_value(
    req_id: int,
    state: Any,
    op: Mapping[str, Any],
    value: Any,
) -> CommitOutput:
    out = dict(value) if isinstance(value, Mapping) else _image_to_result(req_id, value)
    logits = out.pop("logits", None)
    if logits is not None:
        if state is None:
            raise invalid_descriptor("commit logits require request state for sampling")
        sampled = sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        sampled.pop("req_id", None)
        out.update(sampled)
    return _commit_output_from_dict(req_id, out)


# ---------------------
# Step execution over parsed batches
# ---------------------

_GRAPH_SELECTION_DELEGATED_MODES = frozenset({ForwardMode.COMMIT, ForwardMode.ENCODE})

_STREAM_OVERLAP_MODES = frozenset(
    {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT}
)


def _overlap_eligible(group: list[tuple[int, Mapping[str, Any]]]) -> bool:
    """Plan/forward stream overlap covers text-only groups.

    Flow, encode, and commit groups run packed graph programs with their own
    buffer ownership; their prepare phases stay on the forward stream.
    """

    return all(mode_for_op(op["kind"]) in _STREAM_OVERLAP_MODES for _, op in group)


@dataclass(frozen=True)
class ForwardStepOptions:
    defer_text_cpu_results: bool = False


class ForwardGroupPlanner:
    """Plans one parsed worker step into executable op groups."""

    def __init__(
        self,
        batch_policy: BatchPolicy,
        *,
        log_text_mixed_split: Callable[[list[Mapping[str, Any]], Any], None],
        can_run_forward: Callable[[UniForwardBatch], bool] | None = None,
    ) -> None:
        self.batch_policy = batch_policy
        self._log_text_mixed_split = log_text_mixed_split
        self._can_run_forward = can_run_forward

    def groups(self, ops: list[Mapping[str, Any]]) -> list[list[tuple[int, Mapping[str, Any]]]]:
        if self.batch_policy.supports_mixed_modes:
            router = ForwardAdmissionRouter.from_runtime_config()
            forward_rows, delegated_rows = router.partition_supported(ops)
            forward_ops = [op for _index, op in forward_rows]
            decision = router.decide(forward_ops)
            if decision.use_forward:
                batch = UniForwardBatch.from_ops(forward_ops)
                if self._can_run_forward is None or self._can_run_forward(batch):
                    delegated_groups = self._mode_ordered_indexed_groups(delegated_rows)
                    return self._order_groups([*delegated_groups, forward_rows])
                raise capability_mismatch(
                    "admitted segment group has no whole-batch executor",
                    details={
                        "modes": [mode.value for mode in decision.modes],
                        "reason": decision.reason,
                    },
                )
            self._log_text_mixed_split(forward_ops, decision)
            return self._mode_ordered_groups(ops)
        return self._contiguous_groups(ops)

    def _contiguous_groups(
        self,
        ops: list[Mapping[str, Any]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        for idx, op in enumerate(ops):
            mode = self._validated_mode(idx, op)
            if groups:
                modes = [mode_for_op(item[1]["kind"]) for item in groups[-1]]
                if self.batch_policy.allows_group([*modes, mode]):
                    groups[-1].append((idx, op))
                    continue
            groups.append([(idx, op)])
        return groups

    def _mode_ordered_groups(
        self,
        ops: list[Mapping[str, Any]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        return self._mode_ordered_indexed_groups(list(enumerate(ops)))

    def _mode_ordered_indexed_groups(
        self,
        indexed_ops: list[tuple[int, Mapping[str, Any]]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        buckets: dict[ForwardMode, list[tuple[int, Mapping[str, Any]]]] = {}
        first_seen: list[ForwardMode] = []
        for idx, op in indexed_ops:
            mode = self._validated_mode(idx, op)
            if mode not in buckets:
                buckets[mode] = []
                first_seen.append(mode)
            buckets[mode].append((idx, op))

        ordered_modes: list[ForwardMode] = []
        for mode in self.batch_policy.mode_order:
            if mode in buckets:
                ordered_modes.append(mode)
        for mode in first_seen:
            if mode not in ordered_modes:
                ordered_modes.append(mode)

        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        max_batch_ops = self.batch_policy.max_batch_ops
        for mode in ordered_modes:
            items = buckets[mode]
            for start in range(0, len(items), max_batch_ops):
                groups.append(items[start : start + max_batch_ops])
        return groups

    def _order_groups(
        self,
        groups: list[list[tuple[int, Mapping[str, Any]]]],
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        rank = {mode: index for index, mode in enumerate(self.batch_policy.mode_order)}
        fallback = len(rank)

        def group_key(group: list[tuple[int, Mapping[str, Any]]]) -> tuple[int, int]:
            modes = [self._validated_mode(index, op) for index, op in group]
            return min((rank.get(mode, fallback) for mode in modes), default=fallback), min(
                (index for index, _op in group),
                default=0,
            )

        return sorted(groups, key=group_key)

    @staticmethod
    def _validated_mode(idx: int, op: Mapping[str, Any]) -> ForwardMode:
        if not isinstance(op, Mapping):
            raise invalid_descriptor(f"execute batch.ops[{idx}] must be a map")
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor(f"execute batch.ops[{idx}].kind must be a string")
        return mode_for_op(kind)


class ForwardStepExecutor:
    """Owns per-step grouping, dispatch context, and result alignment."""

    def __init__(self, runner: Any, *, group_planner: ForwardGroupPlanner) -> None:
        self.runner = runner
        self.group_planner = group_planner

    def execute(self, parsed: WireExecuteBatch, options: ForwardStepOptions) -> dict[str, Any]:
        forward_stats = ForwardStats() if env_flag("UNISERVE_FORWARD_METRICS") else None
        self.runner._register_new_reqs(parsed.new_reqs)
        ops = parsed.ops
        results: list[dict[str, Any] | None] = [None] * len(ops)
        with profile_range("uniserve.runner.group_ops"):
            groups = self.group_planner.groups(list(ops))
        for group in groups:
            self._run_group(
                group,
                results,
                forward_stats=forward_stats,
                defer_text_cpu_results=options.defer_text_cpu_results,
            )
        if any(result is None for result in results):
            raise invalid_descriptor("runner missed at least one op result")
        out: dict[str, Any] = {"step_id": parsed.step_id, "per_seq": results}
        if forward_stats is not None:
            out["forward_stats"] = forward_stats.to_wire()
        return out

    def _run_group(
        self,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[dict[str, Any] | None],
        *,
        forward_stats: ForwardStats | None,
        defer_text_cpu_results: bool,
    ) -> None:
        indices = [idx for idx, _ in group]
        fb = UniForwardBatch.from_ops([op for _, op in group])
        with profile_range(f"uniserve.runner.group.{fb.mode.value}"):
            self.runner._accountant.account_group(group)
            if forward_stats is not None:
                self.runner._record_group_shape(forward_stats, fb)
            ctx = ForwardContext(
                attention_backend=self.runner.attention_backend,
                attention_preference=self.runner.attention_preference,
                stats=forward_stats,
            )
            group_start = time.perf_counter_ns() if forward_stats is not None else 0
            stream_ctx = self.runner._forward_stream_context(fb)
            with torch.inference_mode(), use_forward_context(ctx), stream_ctx:
                outputs = self._execute_unified_group(
                    fb,
                    group,
                    results,
                    forward_stats=forward_stats,
                    defer_text_cpu_results=defer_text_cpu_results,
                )
            if forward_stats is not None:
                forward_stats.record_mode_wall_time(
                    fb.mode.value,
                    time.perf_counter_ns() - group_start,
                )
            if len(outputs) != len(group):
                raise invalid_descriptor(
                    f"model returned {len(outputs)} outputs for {len(group)} ops"
                )
            for idx, result in zip(indices, outputs):
                results[idx] = result

    def _execute_unified_group(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[dict[str, Any] | None],
        *,
        forward_stats: ForwardStats | None,
        defer_text_cpu_results: bool,
    ) -> list[Any]:
        indices = [idx for idx, _ in group]

        def normalize_output(output: Any) -> Any:
            return _to_seq_result(output)

        def postprocess_side_effects(group_results: list[Any]) -> None:
            staged_results = list(results)
            for idx, result in zip(indices, group_results):
                staged_results[idx] = result
            self.runner._advance_state(fb, group_results)
            self.runner._stamp_conditioning_locators(fb, group, staged_results)
            for row, idx in enumerate(indices):
                group_results[row] = staged_results[idx]

        runtime_handles = ForwardRuntimeHandles(
            request_states=self.runner.request_states,
            residency=self.runner.residency,
            tensor_store=self.runner.tensor_store,
            values={
                "dispatch_batch": fb,
                "defer_text_cpu_results": defer_text_cpu_results,
                "defer_sampling": self.runner.defer_sampling,
                "output_normalizer": normalize_output,
                "postprocess_side_effects": postprocess_side_effects,
            },
        )
        device = torch.device(str(getattr(self.runner.model, "device", "cpu") or "cpu"))

        def prepare() -> tuple[Any, Any]:
            plan = self.runner.forward_plan_builder.build(
                group,
                request_states=self.runner.request_states,
                graph_policy=_graph_policy_for_group(self.runner.forward_graph_policy, fb),
                runtime_handles=runtime_handles,
            )
            return plan, self.runner.unified_forward_batch_builder.build(plan, device=device)

        overlap = getattr(self.runner, "plan_stream_overlap", None)
        if overlap is None or device.type != "cuda" or not _overlap_eligible(group):
            overlap = None
        prepared = overlap.prepare(prepare) if overlap is not None else None
        plan, batch = prepared.value if prepared is not None else prepare()

        with self.runner.forward_adapter.bind(
            dispatch_batch=fb,
            group=group,
            defer_text_cpu_results=defer_text_cpu_results,
        ):
            if overlap is not None and prepared is not None:
                result = overlap.launch(
                    prepared,
                    lambda: self.runner.forward_executor.execute(batch, plan),
                    retain=batch,
                )
            else:
                result = self.runner.forward_executor.execute(batch, plan)
        return self.runner.forward_postprocessor.apply(plan, result)


def _to_seq_result(output: ForwardOutput | Mapping[str, Any]) -> Any:
    if isinstance(output, ForwardOutputBase):
        return output.to_seq_result()
    if isinstance(output, DeferredForwardOutput):
        return output.to_seq_result()
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported forward output type {type(output).__name__}")


def _graph_policy_for_group(policy: Any, fb: UniForwardBatch) -> Any:
    if getattr(fb, "mode", None) not in _GRAPH_SELECTION_DELEGATED_MODES:
        return policy
    if policy is None:
        return None
    return replace(policy, graph_selection_delegated=True)


# ---------------------
# Plan/forward stream overlap
# ---------------------

_T = TypeVar("_T")


@dataclass(frozen=True)
class PreparedOnPlanStream:
    """One prepare-phase result plus the event that fences its consumption."""

    value: Any
    plan_done: torch.cuda.Event


@dataclass(frozen=True)
class _InflightForward:
    plan_done: torch.cuda.Event
    forward_done: torch.cuda.Event
    retained: Any


class PlanStreamOverlap:
    """Runs batch preparation on a dedicated stream, fenced against reuse.

    ``max_inflight`` must equal the staging-ring depth: it is the reuse period
    of the pinned/device staging buffers, and the coordinator's fences are what
    make that reuse safe across streams.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_inflight: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("PlanStreamOverlap requires a CUDA device")
        self.device = device
        self.plan_stream = torch.cuda.Stream(device=device)
        self.max_inflight = max(1, int(max_inflight))
        self._inflight: deque[_InflightForward] = deque()

    def prepare(self, build: Callable[[], _T]) -> PreparedOnPlanStream:
        """Run ``build`` under ``plan_stream`` and record its completion event.

        Applies the ring-reuse fences before running: host-sync on the oldest
        retired prepare (pinned-buffer WAR) and a device-side wait on its
        forward (device-buffer WAR), then releases that batch's references.
        """

        while len(self._inflight) >= self.max_inflight:
            oldest = self._inflight.popleft()
            oldest.plan_done.synchronize()
            self.plan_stream.wait_event(oldest.forward_done)
        with torch.cuda.stream(self.plan_stream):
            value = build()
            plan_done = torch.cuda.Event()
            plan_done.record(self.plan_stream)
        return PreparedOnPlanStream(value=value, plan_done=plan_done)

    def launch(
        self,
        prepared: PreparedOnPlanStream,
        run: Callable[[], _T],
        *,
        retain: Any,
    ) -> _T:
        """Launch the forward on the current stream after the plan event.

        ``retain`` (the prepared batch) is held until the ring-reuse fence for
        its slot passes, keeping plan-stream allocations alive while the
        forward may still read them.
        """

        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(prepared.plan_done)
        result = run()
        forward_done = torch.cuda.Event()
        forward_done.record(stream)
        self._inflight.append(
            _InflightForward(
                plan_done=prepared.plan_done,
                forward_done=forward_done,
                retained=retain,
            )
        )
        return result

    def drain(self) -> None:
        """Retire every in-flight forward (host-blocking); used by tests."""

        while self._inflight:
            entry = self._inflight.popleft()
            entry.plan_done.synchronize()
            entry.forward_done.synchronize()


# ---------------------
# Worker model runner assembly
# ---------------------

# Modes routed to the typed text driver.

_MIXED_PROOF_LOG = logging.getLogger("uniserve.mixed_proof")
# Per-step mixed-forward trace (opt-in). Fallback warnings are always emitted.
_MIXED_PROOF_ENABLED = env_flag("UNISERVE_MIXED_PROOF_LOG")


def _model_max_context_len(model: Any) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "max_position_embeddings", None)
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


@dataclass
class RunnerComponents:
    """Optional per-modality execution overrides for ``ModelRunner``.

    Each is constructed with a default when left ``None``; the text driver is
    additionally wired to the system text-execution stack built in ``__init__``.
    """

    denoise_driver: FlowExecutor | None = None
    encode_driver: EncodeDriver | None = None
    image_decode_driver: ImageDecodeDriver | None = None
    text_driver: TextDriver | None = None


@dataclass
class RunnerConfig:
    """Non-driver ``ModelRunner`` configuration knobs.

    Groups the sampler-stage split (``defer_sampling``/``tensor_store``),
    multimodal processor, batch policy, and attention-backend selection.
    """

    batch_policy: BatchPolicy | None = None
    attention_backend: Any | None = None
    multimodal_processor: Any | None = None
    defer_sampling: bool = False
    tensor_store: Any | None = None
    simulation: bool = False


@dataclass
class _ResolvedRunnerDependencies:
    batch_policy: BatchPolicy | None
    attention_backend: Any | None
    denoise_driver: FlowExecutor | None
    encode_driver: EncodeDriver | None
    image_decode_driver: ImageDecodeDriver | None
    text_driver: TextDriver | None
    multimodal_processor: Any | None
    defer_sampling: bool
    tensor_store: Any | None
    simulation: bool


@dataclass(frozen=True)
class _TextExecutionStack:
    builder: ForwardBatchBuilder | None
    gate: Any | None
    graph_runner: Any | None


class ModelRunner:
    """Dispatches wire ops through modality drivers and resource accounting."""

    def __init__(
        self,
        model: UniModel,
        request_states: RequestSessionTable | None = None,
        *,
        components: RunnerComponents | None = None,
        config: RunnerConfig | None = None,
        resource_runtime: ResourceRuntime | None = None,
        residency: "ResidencyManager | None" = None,
        batch_policy: BatchPolicy | None = None,
        attention_backend: Any | None = None,
        denoise_driver: FlowExecutor | None = None,
        encode_driver: EncodeDriver | None = None,
        image_decode_driver: ImageDecodeDriver | None = None,
        text_driver: TextDriver | None = None,
        multimodal_processor: Any | None = None,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ):
        if not isinstance(model, ModelHooks):
            raise capability_mismatch("runner model must inherit ModelHooks")
        dependencies = self._resolve_dependencies(
            components=components,
            config=config,
            batch_policy=batch_policy,
            attention_backend=attention_backend,
            denoise_driver=denoise_driver,
            encode_driver=encode_driver,
            image_decode_driver=image_decode_driver,
            text_driver=text_driver,
            multimodal_processor=multimodal_processor,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

        self.model = model
        self.residency = residency
        self.request_states = request_states or RequestSessionTable()
        # Deferred sampling: text decode/extend ops publish logits to
        # ``tensor_store`` and return handles — a separate Sampler worker samples.
        # Off = sample inline (default).
        self.defer_sampling = (
            bool(dependencies.defer_sampling) and dependencies.tensor_store is not None
        )
        self.tensor_store = dependencies.tensor_store
        self.simulation = bool(dependencies.simulation)
        self.batch_policy = dependencies.batch_policy or self._model_batch_policy()
        self.attention_backend, self.attention_preference = self._resolve_attention_backend(
            dependencies.attention_backend
        )
        self.denoise_driver = dependencies.denoise_driver or FlowExecutor()
        self.encode_driver = dependencies.encode_driver or EncodeDriver()
        self.image_decode_driver = dependencies.image_decode_driver or ImageDecodeDriver()
        self._init_text_execution(
            model,
            residency,
            dependencies.text_driver,
        )
        self._init_unified_forward_execution(model, residency)
        self.multimodal_processor = dependencies.multimodal_processor
        self._init_resource_accounting(resource_runtime, residency)
        # CUDA Green Context SM partitioning. ``None`` unless runtime config
        # enables it and the model runs on a CUDA device.
        self.stream_manager = self._maybe_build_stream_manager()
        self._group_planner = ForwardGroupPlanner(
            self.batch_policy,
            log_text_mixed_split=self._log_text_mixed_split,
            can_run_forward=self.forward_adapter.can_run_forward,
        )
        self._step_executor = ForwardStepExecutor(self, group_planner=self._group_planner)

    def _init_unified_forward_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> None:
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        kv_pool = residency.kv if residency is not None else None
        self.forward_plan_builder = ForwardPlanBuilder()
        self.unified_forward_batch_builder = UnifiedForwardBatchBuilder(
            runtime_builder=self.forward_batch_builder,
            kv_pool=kv_pool,
            request_states=self.request_states,
            default_device=device,
        )
        self.forward_graph_policy = ForwardGraphPolicy(
            prefer_graph=bool(get_execution_config().cuda_graph),
            strict=not self.simulation,
        )
        self.forward_graph_runner = self._build_forward_graph_runner(model)
        self.forward_fallback_recorder = EagerFallbackRecorder()
        self.forward_adapter = WorkerForwardAdapter(
            model=model,
            request_states=self.request_states,
            text_driver=self.text_driver,
            denoise_driver=self.denoise_driver,
            encode_driver=self.encode_driver,
            image_decode_driver=self.image_decode_driver,
            defer_sampling=self.defer_sampling,
            tensor_store=self.tensor_store,
            mixed_proof_callback=self._log_mixed_proof if _MIXED_PROOF_ENABLED else None,
        )
        self.forward_executor = ForwardExecutor(
            model=self.forward_adapter,
            graph_runner=self.forward_graph_runner,
            graph_policy=self.forward_graph_policy,
            fallback_recorder=self.forward_fallback_recorder,
        )
        self.forward_postprocessor = ForwardPostprocessor()
        self.plan_stream_overlap = self._maybe_build_plan_stream_overlap(device)

    def _maybe_build_plan_stream_overlap(self, device: torch.device):
        """Build the plan/forward stream-overlap coordinator when enabled.

        Gated behind ``UNISERVE_STREAM_OVERLAP=1`` per
        ``specs/intra-worker-stream-overlap.md``; requires a CUDA device. The
        in-flight bound is the staging-ring reuse period so the coordinator's
        WAR fences cover pinned and device staging-buffer recycling.
        """
        if not env_flag("UNISERVE_STREAM_OVERLAP"):
            return None
        if device.type != "cuda":
            return None
        pass

        ring_depth = getattr(self.forward_batch_builder, "staging_ring_depth", 3)
        overlap = PlanStreamOverlap(device, max_inflight=int(ring_depth))
        logger.info(
            "intra-worker stream overlap enabled (plan stream, max_inflight=%d)",
            overlap.max_inflight,
        )
        return overlap

    def _build_forward_graph_runner(self, model: UniModel) -> Any | None:
        from uniserve_worker.execution.graph import Dispatch
        from uniserve_worker.execution.graph.path import Batch, Flow, Segment

        paths = []
        if self.text_graph_runner is not None or callable(
            getattr(model, "try_run_graph_logits_batch", None)
        ):
            paths.append(
                Batch(
                    driver=self.text_driver,
                    model=model,
                    states=self.request_states,
                    executor=self.text_graph_runner,
                )
            )
        # Dispatch follows a specialization chain: constrained sequence batches
        # use their narrower graph program before the general segment program.
        # Heterogeneous compositions fall through to Segment unchanged.
        paths.append(
            Segment(
                executor=model.segment_executor,
                states=self.request_states,
                publisher=self.image_decode_driver,
            )
        )
        paths.append(
            Flow(
                driver=self.denoise_driver,
                model=model,
                states=self.request_states,
            )
        )
        return Dispatch(paths=tuple(paths))

    def _init_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
        text_driver: TextDriver | None,
    ) -> None:
        # System text execution: a thin model declares its KV geometry and the
        # runtime owns builder/gate/graph/sampler around the graph-unaware model.
        text_stack = self._build_text_execution(model, residency)
        self.forward_batch_builder = text_stack.builder
        self.text_gate = text_stack.gate
        self.text_graph_runner = text_stack.graph_runner
        self.text_driver = text_driver or TextDriver(
            builder=self.forward_batch_builder,
            gate=self.text_gate,
            kv_pool=residency.kv if residency is not None else None,
        )

    def _init_resource_accounting(
        self,
        resource_runtime: ResourceRuntime | None,
        residency: "ResidencyManager | None",
    ) -> None:
        resource_plan = self._model_resource_plan()
        classes = resource_plan.classes()
        self.resource_runtime = resource_runtime or ResourceRuntime(
            classes,
            totals=self._model_resource_totals(classes),
        )
        # The accountant holds ``resource_plan`` as the single source of truth;
        # ``ModelRunner.resource_plan`` forwards to it so a runtime reassignment
        # is seen by both.
        self._accountant = ResidencyLeaseManager(
            self.resource_runtime,
            self.request_states,
            resource_plan,
            residency=residency,
        )

    @staticmethod
    def _resolve_dependencies(
        *,
        components: RunnerComponents | None,
        config: RunnerConfig | None,
        batch_policy: BatchPolicy | None,
        attention_backend: Any | None,
        denoise_driver: FlowExecutor | None,
        encode_driver: EncodeDriver | None,
        image_decode_driver: ImageDecodeDriver | None,
        text_driver: TextDriver | None,
        multimodal_processor: Any | None,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> _ResolvedRunnerDependencies:
        components = components or RunnerComponents()
        config = config or RunnerConfig()
        return _ResolvedRunnerDependencies(
            batch_policy=batch_policy if batch_policy is not None else config.batch_policy,
            attention_backend=(
                attention_backend if attention_backend is not None else config.attention_backend
            ),
            denoise_driver=(
                denoise_driver if denoise_driver is not None else components.denoise_driver
            ),
            encode_driver=(
                encode_driver if encode_driver is not None else components.encode_driver
            ),
            image_decode_driver=(
                image_decode_driver
                if image_decode_driver is not None
                else components.image_decode_driver
            ),
            text_driver=(text_driver if text_driver is not None else components.text_driver),
            multimodal_processor=(
                multimodal_processor
                if multimodal_processor is not None
                else config.multimodal_processor
            ),
            defer_sampling=defer_sampling or config.defer_sampling,
            tensor_store=tensor_store if tensor_store is not None else config.tensor_store,
            simulation=bool(config.simulation),
        )

    @staticmethod
    def _resolve_attention_backend(attention_backend: Any | None) -> tuple[Any | None, str | None]:
        if isinstance(attention_backend, str) or attention_backend is None:
            attention_preference = normalize_attention_backend_name(attention_backend or "auto")
            if attention_preference != "auto":
                get_attention_backend(attention_preference)
            return None, attention_preference
        return attention_backend, getattr(attention_backend, "name", None)

    @property
    def resource_plan(self) -> ResourcePlan:
        return self._accountant.resource_plan

    @resource_plan.setter
    def resource_plan(self, plan: ResourcePlan) -> None:
        self._accountant.resource_plan = plan

    def _build_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> _TextExecutionStack:
        """Build the system text builder + backend gate + CUDA-graph runner.

        A thin model declares ``kv_cache_spec`` and the runtime owns its KV pool
        (``residency.kv``); the gate decides batched-paged vs per-op-dense from
        the model geometry + the system pool's storage flag, and the graph runner
        captures/replays decode/prefill graphs around the graph-unaware model.
        Models without ``kv_cache_spec`` (self-managed KV) get none of these.
        """

        if residency is None or residency.kv is None:
            return _TextExecutionStack(None, None, None)
        if self._model_kv_cache_spec(model) is None:
            return _TextExecutionStack(None, None, None)
        import torch

        from uniserve_worker.backends.attention.text_dispatch import TextBackendGate

        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        max_context_len = _model_max_context_len(model)
        gate = TextBackendGate(
            head_dim=int(getattr(model, "head_dim", residency.kv.head_dim)),
            block_size=int(residency.kv.block_size),
            device_type=device.type,
            paged_storage_ok=bool(getattr(residency.kv, "supports_paged_attention_storage", True)),
        )
        graph_runner = None
        if device.type == "cuda":
            from uniserve_worker.execution.graph import Executor

            graph_runner = Executor(
                kv_pool=residency.kv,
                num_blocks=int(residency.kv.num_blocks),
                block_size=int(residency.kv.block_size),
                device=device,
                attention_preference=self.attention_preference,
                max_context_len=max_context_len,
            )
            try:
                graph_runner.warmup(model)
            except Exception:  # noqa: BLE001 - a warmup failure must never block serving.
                logger.warning(
                    "text CUDA-graph warmup failed; falling back to eager", exc_info=True
                )
        return _TextExecutionStack(
            ForwardBatchBuilder(max_context_len=max_context_len),
            gate,
            graph_runner,
        )

    @staticmethod
    def _model_kv_cache_spec(model: UniModel) -> Any | None:
        return model.kv_cache_spec()

    def _maybe_build_stream_manager(self):
        if not get_execution_config().green_contexts:
            return None
        import torch

        device = torch.device(str(getattr(self.model, "device", "cpu") or "cpu"))
        if device.type != "cuda":
            return None
        try:
            from uniserve_worker.runtime.stream_manager import StreamManager

            gpu_id = device.index if device.index is not None else torch.cuda.current_device()
            manager = StreamManager(int(gpu_id))
            logger.info(
                "green contexts enabled: %d SMs, %d stream groups (partitioned=%s)",
                manager.total_sms,
                len(manager.stream_groups),
                manager.using_green_contexts,
            )
            return manager
        except Exception:  # noqa: BLE001 - never let a stream-setup failure block serving.
            logger.warning("StreamManager init failed; green contexts disabled", exc_info=True)
            return None

    def _forward_stream_context(self, fb: UniForwardBatch):
        """Context manager that runs a single-mode group on its SM partition.

        Prefill/verify groups run on the prefill (large-SM) partition; decode
        groups on the decode partition sized by the running decode batch; mixed
        and non-text groups run full-SM (no partitioning helps a mixed forward).
        A no-op ``nullcontext`` when green contexts are disabled.
        """
        from contextlib import nullcontext

        if self.stream_manager is None:
            return nullcontext()
        mode = fb.mode
        if mode == ForwardMode.EXTEND or mode == ForwardMode.VERIFY_DRAFT:
            stream = self.stream_manager.select_streams(0)[0]
        elif mode == ForwardMode.DECODE:
            stream = self.stream_manager.select_streams(len(fb.ops))[1]
        else:
            stream = self.stream_manager.default_stream()
        import torch

        return torch.cuda.stream(stream)

    def drop_request(self, req_id: int) -> None:
        self.model.drop_request(req_id)
        self._accountant.release_request(int(req_id))
        self.request_states.drop(req_id)

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        parsed = WireExecuteBatch.from_wire(batch)
        return self._step_executor.execute(
            parsed,
            ForwardStepOptions(defer_text_cpu_results=defer_text_cpu_results),
        )

    def _register_new_reqs(self, new_reqs: tuple[Mapping[str, Any], ...]) -> None:
        """Create/refresh request state and account resident blocks for new reqs.

        A block-accounting failure rolls back any request *this* call freshly
        created (existing requests are left untouched) before re-raising.
        """
        for nr in new_reqs:
            req_id = nr["req_id"]
            existed = req_id in self.request_states
            state = self.request_states.create_or_update(req_id, dict(nr))
            try:
                self._accountant.account_blocks(req_id, state.block_ids, append_to_state=False)
            except Exception:
                if not existed:
                    self._accountant.release_request(int(req_id))
                    self.request_states.drop(req_id)
                raise
            self.model.on_new_request(req_id, state)

    def _stamp_conditioning_locators(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[Any],
    ) -> None:
        """Mode A: stamp the und->gen conditioning locator on a text result that
        begins an image, so the gen pool can fetch the conditioning KV.

        Delegates the decision to the model's ``maybe_publish_conditioning`` hook
        (a no-op unless a data-plane handoff is bound). Only fires for text-decode
        results carrying an inline sampled token (the image-start trigger)."""
        if fb.mode not in _TEXT_DRIVER_MODES:
            return
        for idx, op in group:
            result = results[idx]
            if not isinstance(result, dict):
                continue
            sampled = result.get("sampled_token_id")
            if sampled is None:
                continue
            locator = self.model.maybe_publish_conditioning(int(op["req_id"]), int(sampled))
            if locator:
                result["locator"] = locator

    def _log_mixed_proof(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
    ) -> None:
        # Per-step "mixed forward executed" trace (opt-in via the proof flag).
        n_ext = sum(1 for m in fb.op_modes if m == ForwardMode.EXTEND)
        n_dec = sum(1 for m in fb.op_modes if m == ForwardMode.DECODE)
        n_den = sum(1 for m in fb.op_modes if m == ForwardMode.DENOISE)
        n_tok = sum(len(op.get("token_ids") or []) for _, op in group)
        _MIXED_PROOF_LOG.info(
            "MIXED FORWARD executed: %d ops (%d extend + %d decode + %d denoise), %d tokens",
            len(fb.ops),
            n_ext,
            n_dec,
            n_den,
            n_tok,
        )

    def _log_text_mixed_split(self, ops: list[Mapping[str, Any]], decision: Any) -> None:
        # Warn when a prefill+decode text mix is split to per-mode groups.
        modes = {mode_for_op(str(op.get("kind"))) for op in ops}
        if {ForwardMode.EXTEND, ForwardMode.DECODE} <= modes:
            _MIXED_PROOF_LOG.warning(
                "MIXED TEXT SPLIT: scheduler co-batched %d ops with both prefill+decode "
                "and runner is splitting per-mode (use_forward=%s)",
                len(ops),
                decision.use_forward,
            )

    def _model_batch_policy(self) -> BatchPolicy:
        policy = self.model.batch_policy()
        if isinstance(policy, BatchPolicy):
            return policy
        raise invalid_descriptor("model.batch_policy() must return BatchPolicy")

    def _model_resource_plan(self) -> ResourcePlan:
        return self.model.resource_plan

    def _model_resource_totals(self, classes: tuple[str, ...]) -> dict[str, int]:
        caps = self._model_caps()
        totals = {cls: 0 for cls in classes}
        if "kv_block" in totals:
            totals["kv_block"] = int(caps.num_blocks if caps is not None else 0)
        if "scratch" in totals:
            totals["scratch"] = int(caps.scratch_capacity_tokens if caps is not None else 0)
        if "image_latent" in totals:
            totals["image_latent"] = int(caps.max_latent_size if caps is not None else 0)
        if "encoder_output" in totals:
            totals["encoder_output"] = int(
                caps.encoder_cache_budget
                if caps is not None and caps.encoder_cache_budget is not None
                else 0
            )
        if "adapter" in totals:
            totals["adapter"] = 0
        return totals

    def _model_caps(self) -> Caps | None:
        return self.model.caps() if isinstance(self.model, ModelHooks) else None

    def _advance_state(self, fb: UniForwardBatch, results: list[Any]) -> None:
        if fb.mode == ForwardMode.MIXED:
            # Advance each op by its own mode; results align with ``fb.op_modes``.
            for mode, result in zip(fb.op_modes, results):
                self._advance_op_state(mode, result)
            return
        for result in results:
            self._advance_op_state(fb.mode, result)

    def _advance_op_state(self, mode: ForwardMode, result: Any) -> None:
        if mode == ForwardMode.DENOISE:
            self.request_states.advance_denoise(
                int(result["req_id"]),
                int(result["num_steps_done"]) if result.get("num_steps_done") is not None else None,
            )
        elif mode == ForwardMode.COMMIT:
            req_id = int(result["req_id"])
            self._accountant.release_generation(req_id, committed=True)

    def _record_group_shape(self, stats: ForwardStats, fb: UniForwardBatch) -> None:
        if fb.mode is ForwardMode.MIXED:
            for mode, op in zip(fb.op_modes, fb.ops):
                stats.record_mode_shape(
                    mode.value,
                    ops=1,
                    tokens=self._op_token_count(op),
                )
            return
        stats.record_mode_shape(
            fb.mode.value,
            ops=len(fb.ops),
            tokens=sum(self._op_token_count(op) for op in fb.ops),
        )

    def _op_token_count(self, op: Mapping[str, Any]) -> int:
        mode = mode_for_op(str(op.get("kind")))
        if mode in _TEXT_DRIVER_MODES:
            if mode == ForwardMode.DECODE:
                try:
                    return max(1, int(op.get("decode_token_count") or 1))
                except (TypeError, ValueError):
                    raise invalid_descriptor(
                        "decode_token_count must be a positive integer"
                    ) from None
            tokens = op.get("token_ids") or []
            return len(tokens) if isinstance(tokens, (list, tuple)) else 0
        if mode == ForwardMode.DENOISE:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            cfg = op.get("cfg")
            # Token throughput count matches denoise_driver branch iterations.
            branch_count = int(cfg.get("branch_count") or 1) if isinstance(cfg, Mapping) else 1
            step_count = max(1, int(op.get("denoise_step_count") or 1))
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return (
                self._accountant.latent_units(op, state.image, latent_rule)
                * max(1, branch_count)
                * step_count
            )
        if mode == ForwardMode.COMMIT:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return self._accountant.latent_units(op, state.image, latent_rule)
        return 0
