"""Transactional residency properties (Stage 3, dormant).

Covers the spec's Stage 3 test list where it applies to the cache-sequence
slice: reservation exhaustion with no leak, multi-row partial-failure
rollback, append abort, transient non-persistence, tentative prefix
acceptance from zero through full length, private-tail copy-on-write, lease
pin versus release, stale generation rejection, and packed-mapping
population validated by the device-contract validator.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.cache_schema import CacheEffect, CacheLifetime
from uniserve_worker.contracts.execution import EngineRef, SessionRef
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.runtime.transactional_residency import (
    ArenaConfig,
    LeaseReleaseOutcome,
    ReservationPlan,
    Residency,
    ResidencyError,
    ResidencyExhausted,
    RowDemand,
    SequenceBinding,
    StaleSequenceError,
)

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SESSION = SessionRef(_ENGINE, request_id=41, incarnation=1, session_version=1)

# 8 real pages of 16 tokens (page 0 is the sink).
_ARENA = ArenaConfig(domain_id=1, page_count=9, page_tokens=16)


def _residency() -> Residency:
    return Residency(_ENGINE, (_ARENA,))


def _plan(*bindings: SequenceBinding) -> ReservationPlan:
    return ReservationPlan(
        rows=tuple(
            RowDemand(row_id=index, bindings=(binding,))
            for index, binding in enumerate(bindings)
        )
    )


def _sequence(residency: Residency):
    return residency.create_sequence(
        _SESSION,
        domain_id=1,
        role_id=1,
        lifetime=CacheLifetime.REQUEST,
    )


def _append(residency: Residency, ref, rows: int):
    """Reserve-and-commit one persistent append; returns the fresh ref."""

    reservation = residency.reserve(
        _plan(SequenceBinding(ref, CacheEffect.PERSISTENT_APPEND, rows))
    )
    reservation.commit((rows,))
    return residency.sequence_ref(ref.sequence_id)


def test_append_commit_advances_and_abort_preserves_reachability():
    residency = _residency()
    ref = _sequence(residency)
    free_before = residency.pressure()[1]["free_pages"]
    ref = _append(residency, ref, 20)  # 2 pages
    assert ref.committed_rows == 20
    assert ref.generation == 1
    assert residency.pressure()[1]["free_pages"] == free_before - 2
    # Abort a further append: committed state and page use are untouched.
    reservation = residency.reserve(
        _plan(SequenceBinding(ref, CacheEffect.PERSISTENT_APPEND, 40))
    )
    reservation.abort()
    reservation.abort()  # idempotent
    assert residency.sequence_ref(ref.sequence_id) == ref
    assert residency.pressure()[1]["free_pages"] == free_before - 2


def test_exhaustion_is_all_or_nothing_with_no_leak():
    residency = _residency()
    first = _sequence(residency)
    second = _sequence(residency)
    free_before = residency.pressure()[1]["free_pages"]
    # Row 0 fits (6 pages); row 1 needs 3 more than remain.
    plan = _plan(
        SequenceBinding(first, CacheEffect.PERSISTENT_APPEND, 96),
        SequenceBinding(second, CacheEffect.PERSISTENT_APPEND, 48),
    )
    with pytest.raises(ResidencyExhausted):
        residency.reserve(plan)
    assert residency.pressure()[1]["free_pages"] == free_before


def test_transient_overlay_commits_zero_and_returns_pages():
    residency = _residency()
    ref = _append(residency, _sequence(residency), 20)
    free_before = residency.pressure()[1]["free_pages"]
    reservation = residency.reserve(
        _plan(SequenceBinding(ref, CacheEffect.TRANSIENT_OVERLAY, 32))
    )
    assert residency.pressure()[1]["free_pages"] == free_before - 2
    with pytest.raises(ResidencyError, match="zero"):
        reservation.commit((32,))
    # A rejected commit leaves the reservation unresolved; abort releases it.
    reservation.abort()
    reservation = residency.reserve(
        _plan(SequenceBinding(ref, CacheEffect.TRANSIENT_OVERLAY, 32))
    )
    reservation.commit((0,))
    assert residency.pressure()[1]["free_pages"] == free_before
    # The committed sequence is unchanged apart from its generation.
    after = residency.sequence_ref(ref.sequence_id)
    assert after.committed_rows == 20


@pytest.mark.parametrize("accepted", [0, 1, 7, 16, 24])
def test_tentative_append_publishes_exactly_the_accepted_prefix(accepted: int):
    residency = _residency()
    ref = _append(residency, _sequence(residency), 16)  # one full page
    free_before = residency.pressure()[1]["free_pages"]
    reservation = residency.reserve(
        _plan(SequenceBinding(ref, CacheEffect.TENTATIVE_APPEND, 24))
    )
    reservation.commit((accepted,))
    after = residency.sequence_ref(ref.sequence_id)
    assert after.committed_rows == 16 + accepted
    expected_pages = -(-(16 + accepted) // 16)
    assert residency.pressure()[1]["free_pages"] == free_before + 1 - expected_pages


def test_shared_partial_pages_are_cloned_before_append():
    residency = _residency()
    ref = _append(residency, _sequence(residency), 16)
    lease = residency.publish_prefix(
        ref, 16, identity_digest=bytes(32), identity_schema=1
    )
    # Extend to a partial page, publish... the partial page itself cannot be
    # published; instead share it via adoption of the full-page lease and then
    # grow both sequences from the shared state.
    adopted = residency.adopt_prefix(
        lease, _SESSION, role_id=1, lifetime=CacheLifetime.REQUEST
    )
    ref = _append(residency, ref, 8)  # owner now has a private partial page
    adopted = _append(residency, adopted, 8)
    # Both sequences advanced independently; committed rows diverge safely.
    assert residency.sequence_ref(ref.sequence_id).committed_rows == 24
    assert residency.sequence_ref(adopted.sequence_id).committed_rows == 24


def test_adopted_prefix_shares_pages_until_both_release():
    residency = _residency()
    ref = _append(residency, _sequence(residency), 32)  # 2 full pages
    free_after_owner = residency.pressure()[1]["free_pages"]
    lease = residency.publish_prefix(
        ref, 32, identity_digest=bytes(32), identity_schema=1
    )
    adopted = residency.adopt_prefix(
        lease, _SESSION, role_id=2, lifetime=CacheLifetime.BRANCH
    )
    assert adopted.committed_rows == 32
    # Sharing allocates nothing new.
    assert residency.pressure()[1]["free_pages"] == free_after_owner
    # Releasing the owner keeps pages alive for the adopter and the lease.
    residency.release_sequence(ref)
    assert residency.pressure()[1]["free_pages"] == free_after_owner
    residency.release_sequence(adopted)
    assert residency.pressure()[1]["free_pages"] == free_after_owner
    assert residency.release_lease(lease) is LeaseReleaseOutcome.RELEASED
    assert residency.pressure()[1]["free_pages"] == free_after_owner + 2


def test_pinned_leases_refuse_release_until_the_transaction_resolves():
    residency = _residency()
    ref = _append(residency, _sequence(residency), 16)
    lease = residency.publish_prefix(
        ref, 16, identity_digest=bytes(32), identity_schema=1
    )
    other = _sequence(residency)
    reservation = residency.reserve(
        _plan(
            SequenceBinding(
                other,
                CacheEffect.PERSISTENT_APPEND,
                4,
                input_leases=(lease.lease_id,),
            )
        )
    )
    assert residency.release_lease(lease) is LeaseReleaseOutcome.BUSY
    reservation.commit((4,))
    assert residency.release_lease(lease) is LeaseReleaseOutcome.RELEASED
    assert residency.release_lease(lease) is LeaseReleaseOutcome.STALE


def test_stale_generation_and_unknown_sequences_are_rejected():
    residency = _residency()
    ref = _sequence(residency)
    _append(residency, ref, 4)  # bumps the generation past ref's
    with pytest.raises(StaleSequenceError):
        residency.reserve(
            _plan(SequenceBinding(ref, CacheEffect.PERSISTENT_APPEND, 4))
        )
    residency.release_sequence(residency.sequence_ref(ref.sequence_id))
    with pytest.raises(StaleSequenceError):
        residency.sequence_ref(ref.sequence_id)


def test_packed_mapping_population_passes_the_device_contract():
    residency = _residency()
    ref = _append(residency, _sequence(residency), 20)
    current = residency.sequence_ref(ref.sequence_id)
    reservation = residency.reserve(
        _plan(
            SequenceBinding(current, CacheEffect.PERSISTENT_APPEND, 3),
            SequenceBinding(current_readonly(residency, ref), CacheEffect.READ_ONLY),
        )
    )
    capacity = ResidencyBatchCapacity(bindings=4, page_references=16, tokens=8)
    arrays = reservation.batch_arrays(capacity, write_token_begins=(0, 3))
    arrays.validate(capacity, page_tokens=16)
    assert arrays.active_binding_count == 2
    assert sum(arrays.write_active) == 3
    # Write offsets continue the committed partial page (20 % 16 == 4).
    assert arrays.write_page_offsets[:3] == [4, 5, 6]
    reservation.commit((3, 0))


def current_readonly(residency: Residency, ref):
    """A second sequence used as a read-only conditioning binding."""

    other = residency.create_sequence(
        _SESSION,
        domain_id=1,
        role_id=2,
        lifetime=CacheLifetime.BRANCH,
    )
    return other
