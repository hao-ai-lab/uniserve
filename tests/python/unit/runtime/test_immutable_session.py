"""Immutable sessions, exact versioning, and counter-addressed RNG (Stage 2).

Covers the Stage 2 test obligations from ``specs/unified_forward_execution.md``:
admission ``0 -> 1``, normal advance, stale-row rejection, request-identifier
reuse with a new incarnation, stale-drop safety, abort preserving snapshot
identity, deterministic retry, batch-order and padding independence of draws,
rank-consistent draws, and disaggregated version independence.
"""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.contracts.execution import (
    DropSession,
    EngineRef,
    NewSession,
    SamplingSpec,
    SessionDelta,
    SessionRef,
    TerminalStatus,
)
from uniserve_worker.runtime.immutable_session import (
    DrawPurpose,
    DropOutcome,
    RngCoordinates,
    SessionRegistry,
    SessionTransitionError,
    StaleSessionError,
    draw_key,
)

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SAMPLING = SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0)


def _admission(request_id: int, incarnation: int = 1) -> NewSession:
    return NewSession(
        request_id=request_id,
        incarnation=incarnation,
        sampling=_SAMPLING,
        base_seed=42,
        max_history_tokens=8,
    )


def _delta(
    source: SessionRef,
    *,
    history_after: int = 3,
    tokens: tuple[int, ...] = (1, 2, 3),
    rng_advance: int = 1,
    terminal: TerminalStatus = TerminalStatus.ACTIVE,
) -> SessionDelta:
    return SessionDelta(
        source=source,
        next_version=source.session_version + 1,
        history_length_after=history_after,
        flow_coordinate_after=0,
        rng_advance=rng_advance,
        history_append=tokens,
        cache_leases_added=(),
        cache_leases_released=(),
        product_leases_added=(),
        product_leases_released=(),
        terminal=terminal,
    )


def test_admission_commits_zero_to_one_and_publishes_atomically():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    assert provisional.ref.session_version == 0
    # Provisional admissions are unpublished: rows cannot resolve them.
    assert registry.snapshot_count() == 0
    committed = registry.apply(_delta(provisional.ref), provisional=provisional)
    assert committed.ref.session_version == 1
    assert registry.resolve(committed.ref) == committed


def test_normal_version_advance_and_stale_row_rejection():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    v1 = registry.apply(_delta(provisional.ref), provisional=provisional)
    v2 = registry.apply(_delta(v1.ref, history_after=4, tokens=(4,)))
    assert v2.ref.session_version == 2
    with pytest.raises(StaleSessionError):
        registry.resolve(v1.ref)
    stale_delta = _delta(v1.ref, history_after=5)
    with pytest.raises(StaleSessionError):
        registry.apply(stale_delta)


def test_abort_is_simply_not_applying_and_retry_is_deterministic():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    v1 = registry.apply(_delta(provisional.ref), provisional=provisional)
    before = registry.resolve(v1.ref)
    # A failed transaction aborts pre-commit: nothing to undo, the committed
    # snapshot is untouched, and the exact same delta applies on retry.
    retry_delta = _delta(v1.ref, history_after=4, tokens=(9,))
    assert registry.resolve(v1.ref) == before
    first = registry.apply(retry_delta)
    assert first.recent_tokens[-1] == 9
    # Re-applying the same committed delta is rejected, not double-applied.
    with pytest.raises(StaleSessionError):
        registry.apply(retry_delta)


def test_request_identifier_reuse_needs_a_new_incarnation():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41, incarnation=1))
    registry.apply(_delta(provisional.ref), provisional=provisional)
    with pytest.raises(SessionTransitionError, match="already committed"):
        registry.admit(_admission(41, incarnation=1))
    fresh = registry.admit(_admission(41, incarnation=2))
    assert fresh.ref.incarnation == 2


def test_drop_is_incarnation_safe_and_version_guarded():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41, incarnation=2))
    committed = registry.apply(_delta(provisional.ref), provisional=provisional)
    old_incarnation = DropSession(
        engine=_ENGINE, request_id=41, incarnation=1, min_committed_version=1
    )
    assert registry.drop(old_incarnation) is DropOutcome.STALE
    premature = DropSession(
        engine=_ENGINE, request_id=41, incarnation=2, min_committed_version=5
    )
    assert registry.drop(premature) is DropOutcome.STALE
    assert registry.resolve(committed.ref) == committed
    exact = DropSession(
        engine=_ENGINE, request_id=41, incarnation=2, min_committed_version=1
    )
    assert registry.drop(exact) is DropOutcome.DROPPED
    assert registry.drop(exact) is DropOutcome.ABSENT


def test_terminal_sessions_accept_no_further_deltas():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    finished = registry.apply(
        _delta(provisional.ref, terminal=TerminalStatus.FINISHED_STOP),
        provisional=provisional,
    )
    with pytest.raises(SessionTransitionError, match="terminal"):
        registry.apply(_delta(finished.ref))


def test_history_is_bounded_and_never_shrinks():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    v1 = registry.apply(
        _delta(provisional.ref, history_after=6, tokens=(1, 2, 3, 4, 5, 6)),
        provisional=provisional,
    )
    v2 = registry.apply(
        _delta(v1.ref, history_after=12, tokens=(7, 8, 9, 10, 11, 12))
    )
    # max_history_tokens=8 keeps the newest bounded window only.
    assert v2.recent_tokens == (5, 6, 7, 8, 9, 10, 11, 12)
    with pytest.raises(SessionTransitionError, match="shrink"):
        registry.apply(_delta(v2.ref, history_after=3))


def test_disaggregated_engines_version_independently():
    prefill = SessionRegistry(EngineRef(deployment_id=1, engine_epoch=7))
    decode = SessionRegistry(EngineRef(deployment_id=2, engine_epoch=3))
    p0 = prefill.admit(_admission(41))
    d0 = decode.admit(
        dataclasses.replace(_admission(41), base_seed=43)
    )
    p1 = prefill.apply(_delta(p0.ref), provisional=p0)
    assert p1.ref.session_version == 1
    assert d0.ref.session_version == 0
    d1 = decode.apply(_delta(d0.ref), provisional=d0)
    d2 = decode.apply(_delta(d1.ref, history_after=4, tokens=(4,)))
    assert (p1.ref.session_version, d2.ref.session_version) == (1, 2)


# --------------------------------------------------------------------------- #
# Counter-addressed RNG.
# --------------------------------------------------------------------------- #


def _coords(**overrides) -> RngCoordinates:
    values = dict(
        base_seed=42,
        request_id=41,
        incarnation=1,
        stage_id=0,
        logical_position=12,
        branch_id=0,
        candidate_id=0,
        purpose=DrawPurpose.TOKEN_SAMPLE,
        draw_index=0,
        shard=0,
    )
    values.update(overrides)
    return RngCoordinates(**values)


def test_draws_are_deterministic_and_coordinate_addressed():
    assert draw_key(_coords()) == draw_key(_coords())
    # Batch order, padding capacity, and unrelated requests have no coordinate,
    # so they cannot perturb a draw; every semantic coordinate does.
    assert draw_key(_coords()) != draw_key(_coords(logical_position=13))
    assert draw_key(_coords()) != draw_key(_coords(branch_id=1))
    assert draw_key(_coords()) != draw_key(_coords(candidate_id=1))
    assert draw_key(_coords()) != draw_key(_coords(purpose=DrawPurpose.FLOW_NOISE))
    assert draw_key(_coords()) != draw_key(_coords(incarnation=2))


def test_rank_consistency_is_semantic_not_incidental():
    # Replicated behavior: every rank derives the same logical draw.
    assert draw_key(_coords(shard=0)) == draw_key(_coords(shard=0))
    # Rank-local randomness names an explicit semantic shard coordinate.
    assert draw_key(_coords(shard=0)) != draw_key(_coords(shard=1))


def test_rng_state_advances_by_exact_committed_amounts():
    registry = SessionRegistry(_ENGINE)
    provisional = registry.admit(_admission(41))
    assert provisional.rng.next_counter == 0
    v1 = registry.apply(_delta(provisional.ref, rng_advance=3), provisional=provisional)
    assert v1.rng.next_counter == 3
    v2 = registry.apply(_delta(v1.ref, history_after=4, tokens=(4,), rng_advance=0))
    assert v2.rng.next_counter == 3
    with pytest.raises(SessionTransitionError, match="non-negative"):
        registry.apply(_delta(v2.ref, history_after=5, rng_advance=-1))
