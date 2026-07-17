"""Immutable request sessions, exact versioning, and counter-addressed RNG.

Dormant Stage 2 deliverable from ``specs/unified_forward_execution.md``
("Immutable Sessions, Versioning, And RNG"). This module is the target that
replaces the mutable `runtime.request_session.RequestSessionTable` at its
deletion gate; nothing in production consumes it until the vertical
transaction slice activates.

A :class:`RequestSession` is a frozen snapshot of logical values only — no
tensors, physical identifiers, pools, graph slots, dictionaries, callbacks,
or ``torch.Generator`` instances. The :class:`SessionRegistry` stores complete
snapshots and replaces one snapshot with another only through validated
:class:`~uniserve_worker.contracts.execution.SessionDelta` application:

* admission constructs a **provisional version-0 snapshot without publishing
  it**; the first successful transaction commits version 1;
* rows resolve against exact ``(request_id, incarnation, version)`` identity —
  a mismatch is a typed stale error, never a reconciliation;
* drops are incarnation-safe and version-guarded, so a delayed drop cannot
  remove a newly admitted incarnation that reused the request identifier;
* aborting a transaction is simply *not applying* its deltas — committed
  snapshots are never mutated in place, so retry reproduces identical state.

Randomness is counter-addressed and stateless at execution time: a draw is
keyed by stable logical coordinates (:class:`RngCoordinates`), the session
stores only the next logical counter (:class:`RngState`), and commit applies
the exact ``rng_advance`` from the validated delta. Batch order, padding
capacity, route participation, and unrelated requests cannot change a row's
random values; rank-local randomness must name an explicit shard coordinate.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace
from enum import IntEnum

from ..contracts.execution import (
    CacheLease,
    DropSession,
    EngineRef,
    NewSession,
    ProductLease,
    SamplingSpec,
    SessionDelta,
    SessionRef,
    TerminalStatus,
)

__all__ = [
    "DrawPurpose",
    "DropOutcome",
    "RequestSession",
    "RngCoordinates",
    "RngState",
    "SessionRegistry",
    "SessionTransitionError",
    "StaleSessionError",
    "draw_key",
]


class SessionTransitionError(ValueError):
    """A delta violates the closed transition contract."""


class StaleSessionError(SessionTransitionError):
    """A row names a session identity that is not the committed snapshot."""


# --------------------------------------------------------------------------- #
# Counter-addressed RNG.
# --------------------------------------------------------------------------- #


class DrawPurpose(IntEnum):
    TOKEN_SAMPLE = 1
    FLOW_NOISE = 2
    INITIAL_LATENT = 3


@dataclass(frozen=True, slots=True)
class RngCoordinates:
    """Stable logical coordinates of one random draw.

    Every field is semantic; incidental facts (batch order, padding, rank
    process order) are deliberately unrepresentable. Rank-local randomness
    names its semantic shard through ``shard``.
    """

    base_seed: int
    request_id: int
    incarnation: int
    stage_id: int
    logical_position: int
    branch_id: int
    candidate_id: int
    purpose: DrawPurpose
    draw_index: int
    shard: int = 0


def draw_key(coordinates: RngCoordinates) -> int:
    """Map one coordinate tuple to a stable 64-bit generator key."""

    packed = struct.pack(
        "<10q",
        coordinates.base_seed,
        coordinates.request_id,
        coordinates.incarnation,
        coordinates.stage_id,
        coordinates.logical_position,
        coordinates.branch_id,
        coordinates.candidate_id,
        int(coordinates.purpose),
        coordinates.draw_index,
        coordinates.shard,
    )
    return int.from_bytes(hashlib.sha256(packed).digest()[:8], "little")


@dataclass(frozen=True, slots=True)
class RngState:
    """The session's logical randomness cursor — never a mutable generator."""

    base_seed: int
    next_counter: int

    def advanced(self, rng_advance: int) -> "RngState":
        if rng_advance < 0:
            raise SessionTransitionError("rng_advance must be non-negative")
        return replace(self, next_counter=self.next_counter + rng_advance)


# --------------------------------------------------------------------------- #
# Immutable session snapshot.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RequestSession:
    """One frozen logical request snapshot (see module docs for exclusions)."""

    ref: SessionRef
    sampling: SamplingSpec
    history_length: int
    flow_coordinate: int
    rng: RngState
    recent_tokens: tuple[int, ...]
    max_history_tokens: int
    cache_leases: tuple[CacheLease, ...]
    product_leases: tuple[ProductLease, ...]
    terminal: TerminalStatus


def _admitted_session(engine: EngineRef, admission: NewSession) -> RequestSession:
    return RequestSession(
        ref=SessionRef(
            engine=engine,
            request_id=admission.request_id,
            incarnation=admission.incarnation,
            session_version=0,
        ),
        sampling=admission.sampling,
        history_length=0,
        flow_coordinate=0,
        rng=RngState(base_seed=admission.base_seed, next_counter=0),
        recent_tokens=(),
        max_history_tokens=admission.max_history_tokens,
        cache_leases=(),
        product_leases=(),
        terminal=TerminalStatus.ACTIVE,
    )


def _apply_delta(session: RequestSession, delta: SessionDelta) -> RequestSession:
    if delta.source != session.ref:
        raise StaleSessionError(
            f"delta names {delta.source}, committed snapshot is {session.ref}"
        )
    if delta.next_version != session.ref.session_version + 1:
        raise SessionTransitionError(
            f"delta advances to version {delta.next_version}; expected "
            f"{session.ref.session_version + 1}"
        )
    if session.terminal is not TerminalStatus.ACTIVE:
        raise SessionTransitionError("terminal sessions accept no further deltas")
    if delta.history_length_after < session.history_length:
        raise SessionTransitionError("logical history cannot shrink")
    released_cache = set(delta.cache_leases_released)
    held_cache = {lease.lease_id for lease in session.cache_leases}
    if not released_cache <= held_cache:
        raise SessionTransitionError(
            f"delta releases unheld cache leases {sorted(released_cache - held_cache)}"
        )
    released_products = set(delta.product_leases_released)
    held_products = {lease.lease_id for lease in session.product_leases}
    if not released_products <= held_products:
        raise SessionTransitionError(
            f"delta releases unheld product leases "
            f"{sorted(released_products - held_products)}"
        )
    tokens = session.recent_tokens + delta.history_append
    if len(tokens) > session.max_history_tokens:
        tokens = tokens[-session.max_history_tokens:]
    return RequestSession(
        ref=replace(session.ref, session_version=delta.next_version),
        sampling=session.sampling,
        history_length=delta.history_length_after,
        flow_coordinate=delta.flow_coordinate_after,
        rng=session.rng.advanced(delta.rng_advance),
        recent_tokens=tokens,
        max_history_tokens=session.max_history_tokens,
        cache_leases=tuple(
            lease
            for lease in session.cache_leases
            if lease.lease_id not in released_cache
        )
        + delta.cache_leases_added,
        product_leases=tuple(
            lease
            for lease in session.product_leases
            if lease.lease_id not in released_products
        )
        + delta.product_leases_added,
        terminal=delta.terminal,
    )


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #


class DropOutcome(IntEnum):
    DROPPED = 1
    STALE = 2
    ABSENT = 3


class SessionRegistry:
    """Engine-scoped store of committed immutable session snapshots.

    The registry holds at most one committed snapshot per request identifier.
    Provisional version-0 admissions are values returned to the engine, not
    registry state — they become visible only when their first delta commits,
    so a pre-launch abort leaves no trace.
    """

    def __init__(self, engine: EngineRef) -> None:
        self.engine = engine
        self._sessions: dict[int, RequestSession] = {}

    def admit(self, admission: NewSession) -> RequestSession:
        """Construct a provisional version-0 snapshot without publishing it."""

        committed = self._sessions.get(admission.request_id)
        if committed is not None and committed.ref.incarnation == admission.incarnation:
            raise SessionTransitionError(
                f"request {admission.request_id} incarnation "
                f"{admission.incarnation} is already committed"
            )
        return _admitted_session(self.engine, admission)

    def resolve(self, ref: SessionRef) -> RequestSession:
        """Exact-identity lookup; any mismatch is a typed stale error."""

        session = self._sessions.get(ref.request_id)
        if session is None:
            raise StaleSessionError(f"no committed session for {ref}")
        if session.ref != ref:
            raise StaleSessionError(
                f"row names {ref}, committed snapshot is {session.ref}"
            )
        return session

    def apply(
        self,
        delta: SessionDelta,
        *,
        provisional: RequestSession | None = None,
    ) -> RequestSession:
        """Validate one delta and atomically swap the snapshot.

        A version-0 source must supply its provisional admission snapshot;
        this is the ``0 -> 1`` commit that first publishes the session.
        """

        if delta.source.session_version == 0:
            if provisional is None:
                raise SessionTransitionError(
                    "version-0 deltas commit a provisional admission snapshot"
                )
            if provisional.ref != delta.source:
                raise StaleSessionError(
                    f"provisional snapshot {provisional.ref} does not match "
                    f"delta source {delta.source}"
                )
            committed = self._sessions.get(delta.source.request_id)
            if (
                committed is not None
                and committed.ref.incarnation == delta.source.incarnation
            ):
                raise SessionTransitionError(
                    f"request {delta.source.request_id} incarnation "
                    f"{delta.source.incarnation} is already committed"
                )
            source = provisional
        else:
            source = self.resolve(delta.source)
        replacement = _apply_delta(source, delta)
        self._sessions[replacement.ref.request_id] = replacement
        return replacement

    def drop(self, command: DropSession) -> DropOutcome:
        """Incarnation-safe, version-guarded session removal."""

        if command.engine != self.engine:
            return DropOutcome.STALE
        session = self._sessions.get(command.request_id)
        if session is None:
            return DropOutcome.ABSENT
        if session.ref.incarnation != command.incarnation:
            return DropOutcome.STALE
        if session.ref.session_version < command.min_committed_version:
            return DropOutcome.STALE
        del self._sessions[command.request_id]
        return DropOutcome.DROPPED

    def snapshot_count(self) -> int:
        return len(self._sessions)
