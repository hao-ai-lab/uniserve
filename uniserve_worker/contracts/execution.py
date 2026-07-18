"""Canonical host execution contracts for the target `ExecutionEngine`.

Dormant Stage 1 deliverable from ``specs/unified_forward_execution.md``
("Canonical Host And Device Contracts"). This module owns every host-visible
typed value of the target data plane:

* engine and session identity (`EngineRef`, `SessionRef`) and the sealed
  operation algebra tag (`OperationTag`) shared with the cache companion;
* the immutable transaction shape (`ExecuteBatch`, `ExecuteRow`, the four
  operation variants, `CandidateVerification`, `NewSession` admission);
* typed leases (`CacheLease`, `ProductLease`);
* durable outcomes (`ExecuteResult`, `RowResult`, `SessionDelta`) and the
  `ExecuteReceipt` protocol;
* the closed administrative `EngineCommand` union;
* exhaustive pre-launch validation and the canonical payload fingerprint.

Host contracts contain no CUDA tensors, backend instances, allocators,
physical page identifiers, untyped dictionaries, or extension maps — the
module is torch-free by construction and guarded by the layering tests.

Execution identity is ``(engine_epoch, step_id)``; the canonical payload
fingerprint covers ordered row identities, exact session references,
admissions, operation values, and leases, and **excludes** the cumulative
``acknowledged_through`` envelope so a retry may advance acknowledgement
without changing transaction identity. The byte encoding is specified exactly
(see `_Encoder`) and mirrored by the Rust twin
(`crates/foundation/core/src/execution_identity.rs`); shared test vectors in
``crates/protocol/vocab/execution_fingerprint.toml`` pin byte-for-byte
agreement.

Nothing in production consumes this module yet: the current wire batch in
`contracts.batches` and the mutable session table remain authoritative until
the whole vertical transaction slice activates (parent migration strategy).
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Protocol, Union

__all__ = [
    "CacheLease",
    "CandidateVerification",
    "DropSession",
    "EncodeStep",
    "EngineCommand",
    "EngineRef",
    "ExecuteBatch",
    "ExecuteReceipt",
    "ExecuteResult",
    "ExecuteRow",
    "ExecutionContractError",
    "FlowStep",
    "MaterializeStep",
    "NewSession",
    "Operation",
    "OperationTag",
    "OverlayLoad",
    "OverlayUnload",
    "PrefixCacheReset",
    "ProductLease",
    "ProductLifetime",
    "Quiesce",
    "ReleaseLease",
    "RepresentationKind",
    "Resume",
    "RowResult",
    "RowStatus",
    "SamplingSpec",
    "SequenceStep",
    "SessionDelta",
    "SessionRef",
    "TerminalStatus",
    "TransferKind",
    "canonical_payload_fingerprint",
    "operation_tag",
    "validate_execute_batch",
]


class ExecutionContractError(ValueError):
    """An execute payload violates the closed host contract (pre-launch)."""


# --------------------------------------------------------------------------- #
# Identity.
# --------------------------------------------------------------------------- #


class OperationTag(IntEnum):
    """The parent's sealed model-backed operation algebra.

    Canonical integer tags shared with the Rust side and with the cache
    companion's `FamilyCacheSchema` lowering. Adding a fifth variant is a
    cross-language protocol change, never a string registration.
    """

    SEQUENCE_STEP = 1
    FLOW_STEP = 2
    ENCODE_STEP = 3
    MATERIALIZE_STEP = 4


@dataclass(frozen=True, slots=True)
class EngineRef:
    """Deployment identity plus engine epoch. A new epoch is a new engine."""

    deployment_id: int
    engine_epoch: int


@dataclass(frozen=True, slots=True)
class SessionRef:
    """Exact logical session identity.

    ``incarnation`` makes request-identifier reuse safe: a new incarnation
    never aliases stale rows, results, drops, leases, or products from an
    older incarnation. ``session_version`` is engine-scoped and exact — a
    mismatch is a pre-launch stale-row error, never a reconciliation.
    """

    engine: EngineRef
    request_id: int
    incarnation: int
    session_version: int


# --------------------------------------------------------------------------- #
# Leases.
# --------------------------------------------------------------------------- #


class ProductLifetime(IntEnum):
    TRANSITION = 1
    REQUEST = 2
    SESSION = 3
    CACHE = 4
    ACKNOWLEDGED_OUTPUT = 5


class TransferKind(IntEnum):
    LOCAL_RESIDENCY = 1
    CUDA_IPC = 2
    SHARED_MEMORY = 3
    MOONCAKE = 4


@dataclass(frozen=True, slots=True)
class CacheLease:
    """Scheduler-visible exact reusable prefix.

    Carries the verified identity digest, schema, charge, version, epoch, and
    an opaque residency handle. Residency validates these fields against the
    committed prefix record; the lease never carries physical block or page
    identifiers.
    """

    lease_id: int
    engine_epoch: int
    identity_digest: bytes
    identity_schema: int
    charge: int
    version: int
    residency_handle: int


@dataclass(frozen=True, slots=True)
class ProductLease:
    """Versioned claim on one committed typed product."""

    lease_id: int
    schema_id: int
    producer: SessionRef
    product_version: int
    extent_rows: int
    lifetime: ProductLifetime
    transfer: TransferKind


# --------------------------------------------------------------------------- #
# Sealed operation algebra.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    """Ordered candidate span evaluated in the same target forward.

    Acceptance and rejection processing occur after replay; rejected KV stays
    provisional and is discarded by the reservation delta.
    """

    candidate_tokens: tuple[int, ...]
    candidate_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SequenceStep:
    """One sequence extension, one-position advance, or target verification.

    Extension versus advance is inferred from lengths and token counts; there
    are no phase types. A field asking the engine to loop several
    autoregressive positions is invalid by construction — one row is one model
    iteration.
    """

    input_tokens: tuple[int, ...]
    history_length: int
    position_begin: int
    requested_outputs: int
    verification: CandidateVerification | None = None


@dataclass(frozen=True, slots=True)
class FlowStep:
    """One exact flow or denoising schedule coordinate (no internal loop)."""

    schedule_id: int
    step_index: int
    total_steps: int
    input_product: int
    branch_coefficients: tuple[float, ...]
    conditioning_products: tuple[int, ...]
    output_schema: int


class RepresentationKind(IntEnum):
    """Closed encoder representation kinds (no model hook names)."""

    IMAGE_PATCH = 1
    IMAGE_LATENT = 2
    AUDIO_FEATURE = 3
    VIDEO_FRAME = 4


@dataclass(frozen=True, slots=True)
class EncodeStep:
    """One resident encoder transform between typed product schemas."""

    kind: RepresentationKind
    input_product: int
    output_schema: int
    grid: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MaterializeStep:
    """One resident neural materialization of a durable representation.

    Non-neural serialization and container encoding stay outside
    `ExecutionEngine`.
    """

    input_product: int
    output_schema: int


Operation = Union[SequenceStep, FlowStep, EncodeStep, MaterializeStep]

_OPERATION_TAGS: dict[type, OperationTag] = {
    SequenceStep: OperationTag.SEQUENCE_STEP,
    FlowStep: OperationTag.FLOW_STEP,
    EncodeStep: OperationTag.ENCODE_STEP,
    MaterializeStep: OperationTag.MATERIALIZE_STEP,
}


def operation_tag(operation: Operation) -> OperationTag:
    tag = _OPERATION_TAGS.get(type(operation))
    if tag is None:
        raise ExecutionContractError(
            f"{type(operation).__name__} is not a sealed operation variant"
        )
    return tag


# --------------------------------------------------------------------------- #
# Batch and rows.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """Static per-session sampling configuration (frozen at admission)."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float


@dataclass(frozen=True, slots=True)
class NewSession:
    """Typed admission used only at session version 0."""

    request_id: int
    incarnation: int
    sampling: SamplingSpec
    base_seed: int
    max_history_tokens: int


@dataclass(frozen=True, slots=True)
class ExecuteRow:
    """One request's single model iteration inside one transaction.

    ``row_id`` must equal the row's zero-based position; rows are never
    regrouped on the host. The row carries logical lengths and typed values
    but no physical storage identifiers.
    """

    row_id: int
    session: SessionRef
    operation: Operation
    admission: NewSession | None
    cache_leases: tuple[CacheLease, ...]
    product_leases: tuple[ProductLease, ...]
    scheduler_op_id: int


@dataclass(frozen=True, slots=True)
class ExecuteBatch:
    """One frozen transaction: identity, acknowledgement envelope, rows.

    ``acknowledged_through`` is cumulative envelope metadata excluded from the
    payload fingerprint. There is no independent ``new_reqs`` list that can
    drift from operation rows.
    """

    engine_epoch: int
    step_id: int
    acknowledged_through: int
    rows: tuple[ExecuteRow, ...]


# --------------------------------------------------------------------------- #
# Results, deltas, receipt.
# --------------------------------------------------------------------------- #


class RowStatus(IntEnum):
    OK = 0
    INVALID_OUTPUT = 1


class TerminalStatus(IntEnum):
    ACTIVE = 0
    FINISHED_STOP = 1
    FINISHED_LENGTH = 2
    DROPPED = 3


@dataclass(frozen=True, slots=True)
class RowResult:
    """Compact durable outcome for one row, in scheduler order."""

    row_id: int
    request_id: int
    incarnation: int
    status: RowStatus
    sampled_tokens: tuple[int, ...]
    accepted_candidates: int
    logprobs: tuple[float, ...]
    terminal: TerminalStatus


@dataclass(frozen=True, slots=True)
class SessionDelta:
    """The authoritative logical transition for one committed row.

    The server never infers state changes from operation kind or reconciles a
    partially mutated object; the delta is the only session mutation channel.
    """

    source: SessionRef
    next_version: int
    history_length_after: int
    flow_coordinate_after: int
    rng_advance: int
    history_append: tuple[int, ...]
    cache_leases_added: tuple[CacheLease, ...]
    cache_leases_released: tuple[int, ...]
    product_leases_added: tuple[ProductLease, ...]
    product_leases_released: tuple[int, ...]
    terminal: TerminalStatus


@dataclass(frozen=True, slots=True)
class ExecuteResult:
    """Durable transaction outcome: ordered row results plus deltas."""

    engine_epoch: int
    step_id: int
    row_results: tuple[RowResult, ...]
    session_deltas: tuple[SessionDelta, ...]


class ExecuteReceipt(Protocol):
    """Engine-owned handle to one accepted transaction.

    Carries no callback, session reference, graph tensor, bucket object,
    reservation object, or user-extensible payload.
    """

    def ready(self) -> bool: ...

    def result(self) -> ExecuteResult: ...


# --------------------------------------------------------------------------- #
# Administrative commands (closed union; transaction-boundary only).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DropSession:
    """Versioned drop; a stale drop is a typed no-op and can never remove a
    newer incarnation that reused the request identifier."""

    engine: EngineRef
    request_id: int
    incarnation: int
    min_committed_version: int


@dataclass(frozen=True, slots=True)
class ReleaseLease:
    engine: EngineRef
    lease_id: int
    version: int


@dataclass(frozen=True, slots=True)
class PrefixCacheReset:
    engine: EngineRef


@dataclass(frozen=True, slots=True)
class OverlayLoad:
    engine: EngineRef
    overlay_slot: int
    overlay_id: int


@dataclass(frozen=True, slots=True)
class OverlayUnload:
    engine: EngineRef
    overlay_slot: int


@dataclass(frozen=True, slots=True)
class Quiesce:
    engine: EngineRef


@dataclass(frozen=True, slots=True)
class Resume:
    engine: EngineRef


EngineCommand = Union[
    DropSession,
    ReleaseLease,
    PrefixCacheReset,
    OverlayLoad,
    OverlayUnload,
    Quiesce,
    Resume,
]


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def validate_execute_batch(
    batch: ExecuteBatch,
    *,
    advertised_operations: frozenset[OperationTag],
) -> None:
    """Prove every pre-launch batch invariant the contract level can check.

    Raises :class:`ExecutionContractError` on the first violation: identity
    bounds, canonical row order, unique request incarnations, admissions only
    at version 0, sealed and advertised operation tags, and per-variant
    scalar/extent bounds. Session-version resolution, lease pinning, and
    capacity selection are engine obligations at their owning seams.
    """

    if batch.engine_epoch < 0 or batch.step_id < 0:
        raise ExecutionContractError("batch identity fields must be non-negative")
    if batch.acknowledged_through < -1:
        raise ExecutionContractError(
            "acknowledged_through is a step id, or -1 for none acknowledged"
        )
    if batch.acknowledged_through >= batch.step_id:
        raise ExecutionContractError(
            "a batch cannot acknowledge its own or a future step"
        )
    if not batch.rows:
        raise ExecutionContractError("an execute batch carries at least one row")
    seen_incarnations: set[tuple[int, int]] = set()
    for index, row in enumerate(batch.rows):
        if row.row_id != index:
            raise ExecutionContractError(
                f"row_id {row.row_id} must equal its position {index}; "
                "rows are never regrouped on the host"
            )
        if row.session.engine.engine_epoch != batch.engine_epoch:
            raise ExecutionContractError(
                f"row {index} session names epoch "
                f"{row.session.engine.engine_epoch}, batch is {batch.engine_epoch}"
            )
        key = (row.session.request_id, row.session.incarnation)
        if key in seen_incarnations:
            raise ExecutionContractError(
                f"request incarnation {key} appears in more than one row"
            )
        seen_incarnations.add(key)
        _validate_admission(index, row)
        tag = operation_tag(row.operation)
        if tag not in advertised_operations:
            raise ExecutionContractError(
                f"row {index} carries {tag.name}, outside the advertised set"
            )
        _validate_operation(index, row.operation)
        for lease in row.cache_leases:
            if lease.engine_epoch != batch.engine_epoch:
                raise ExecutionContractError(
                    f"row {index} cache lease {lease.lease_id} names a foreign epoch"
                )
            if len(lease.identity_digest) != 32:
                raise ExecutionContractError(
                    f"row {index} cache lease {lease.lease_id} digest must be 32 bytes"
                )


def _validate_admission(index: int, row: ExecuteRow) -> None:
    if row.admission is not None:
        if row.session.session_version != 0:
            raise ExecutionContractError(
                f"row {index} admission is only legal at session version 0"
            )
        if (
            row.admission.request_id != row.session.request_id
            or row.admission.incarnation != row.session.incarnation
        ):
            raise ExecutionContractError(
                f"row {index} admission identity disagrees with its session"
            )
    elif row.session.session_version == 0:
        raise ExecutionContractError(
            f"row {index} references an unadmitted version-0 session without "
            "a typed admission"
        )


def _validate_operation(index: int, operation: Operation) -> None:
    if isinstance(operation, SequenceStep):
        if not operation.input_tokens and operation.verification is None:
            raise ExecutionContractError(
                f"row {index} sequence step evaluates no positions"
            )
        if operation.history_length < 0 or operation.position_begin < 0:
            raise ExecutionContractError(
                f"row {index} sequence step has negative logical coordinates"
            )
        if operation.requested_outputs <= 0:
            raise ExecutionContractError(
                f"row {index} sequence step requests no output"
            )
        verification = operation.verification
        if verification is not None:
            if not verification.candidate_tokens:
                raise ExecutionContractError(
                    f"row {index} verification carries no candidates"
                )
            if len(verification.candidate_tokens) != len(
                verification.candidate_positions
            ):
                raise ExecutionContractError(
                    f"row {index} verification tokens and positions disagree"
                )
    elif isinstance(operation, FlowStep):
        if not 0 <= operation.step_index < operation.total_steps:
            raise ExecutionContractError(
                f"row {index} flow step coordinate is outside its schedule"
            )
        if not operation.branch_coefficients:
            raise ExecutionContractError(
                f"row {index} flow step declares no branches"
            )
        for coefficient in operation.branch_coefficients:
            if not math.isfinite(coefficient):
                raise ExecutionContractError(
                    f"row {index} flow step carries a non-finite coefficient"
                )
    elif isinstance(operation, EncodeStep):
        if any(dim <= 0 for dim in operation.grid):
            raise ExecutionContractError(
                f"row {index} encode step grid must be positive"
            )
    elif isinstance(operation, MaterializeStep):
        pass
    else:  # pragma: no cover - operation_tag() already rejects foreign types
        raise ExecutionContractError(
            f"row {index} carries an unsealed operation type"
        )


# --------------------------------------------------------------------------- #
# Canonical payload fingerprint.
# --------------------------------------------------------------------------- #


class _Encoder:
    """Exact byte encoding shared with the Rust twin.

    Layout (all integers little-endian):

    * ``0x01`` + i64        — integer field
    * ``0x02`` + f64        — float field (IEEE 754 bits)
    * ``0x03`` + u32 + raw  — bytes field
    * ``0x04`` + u32        — sequence header (element count; elements follow)
    * ``0x05`` + u32        — struct header (field count; fields follow in
                               declaration order)
    * ``0x06`` + u32        — enum value
    * ``0x07``              — absent optional

    The encoding is positional: field names never enter the digest, so
    renames are free but reorders and retypes are protocol changes — exactly
    the property a canonical fingerprint needs.
    """

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def integer(self, value: int) -> None:
        self._parts.append(b"\x01" + struct.pack("<q", value))

    def real(self, value: float) -> None:
        self._parts.append(b"\x02" + struct.pack("<d", value))

    def raw(self, value: bytes) -> None:
        self._parts.append(b"\x03" + struct.pack("<I", len(value)) + value)

    def sequence(self, count: int) -> None:
        self._parts.append(b"\x04" + struct.pack("<I", count))

    def structure(self, field_count: int) -> None:
        self._parts.append(b"\x05" + struct.pack("<I", field_count))

    def enum(self, value: int) -> None:
        self._parts.append(b"\x06" + struct.pack("<I", value))

    def absent(self) -> None:
        self._parts.append(b"\x07")

    def digest(self) -> str:
        return hashlib.sha256(b"".join(self._parts)).hexdigest()


def _encode_value(encoder: _Encoder, value: object) -> None:
    if isinstance(value, bool):
        raise ExecutionContractError("booleans do not appear in payload identity")
    if isinstance(value, IntEnum):
        encoder.enum(int(value))
    elif isinstance(value, int):
        encoder.integer(value)
    elif isinstance(value, float):
        encoder.real(value)
    elif isinstance(value, bytes):
        encoder.raw(value)
    elif value is None:
        encoder.absent()
    elif isinstance(value, tuple):
        encoder.sequence(len(value))
        for item in value:
            _encode_value(encoder, item)
    elif hasattr(type(value), "__dataclass_fields__"):
        value_fields = fields(value)  # type: ignore[arg-type]
        encoder.structure(len(value_fields) + 1)
        if type(value) in _OPERATION_TAGS:
            encoder.enum(int(_OPERATION_TAGS[type(value)]))
        else:
            encoder.enum(0)
        for field in value_fields:
            _encode_value(encoder, getattr(value, field.name))
    else:
        raise ExecutionContractError(
            f"{type(value).__name__} values cannot join the payload fingerprint"
        )


def canonical_payload_fingerprint(batch: ExecuteBatch) -> str:
    """Canonical execution-payload identity (sha256 hex).

    Covers ordered row identities, exact session references, admissions,
    operation values, and product/cache leases. Excludes the cumulative
    acknowledgement and any transport call identifiers, so a retry that only
    advances acknowledgement keeps the same transaction identity.
    """

    encoder = _Encoder()
    encoder.structure(3)
    encoder.integer(batch.engine_epoch)
    encoder.integer(batch.step_id)
    encoder.sequence(len(batch.rows))
    for row in batch.rows:
        _encode_value(encoder, row)
    return encoder.digest()
