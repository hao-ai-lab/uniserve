"""Transactional cache residency: reserve, commit, abort, and prefix leases.

Dormant Stage 3 deliverable from ``specs/unified_forward_execution.md``
("Residency Reservations, Leases, And Products"), covering the cache-sequence
half of the target ``Residency`` owner: page arenas with domain-scoped sink
pages, all-or-nothing reservation across rows, provisional append tails,
transient overlays, tentative candidate tails, private-tail copy-on-write for
shared partial pages, complete-page prefix leases with pin accounting,
non-failing commit, idempotent abort, and pressure reporting.

Product regions, transfer staging, overlay banks, and distributed shard
leases are later slices of the same owner; this module fixes the ownership
and rollback semantics those build on. Storage here is logical (page
identifiers only): binding physical tensors to arenas is the capacity-only
graph runtime's job, and nothing in production routes through this module
until the vertical slice activates.

Semantics enforced, straight from the spec and its cache companion:

* every write targets reservation-owned storage; committed logical length
  changes only at commit (Law 4);
* reservation is all-or-nothing across every row and binding — a failure
  releases every page the plan acquired (no leak);
* a shared partial page is never written in place: appending through it
  clones the page privately (quantization metadata travels with the clone in
  the real storage binding);
* transient overlays commit zero rows; tentative appends publish exactly the
  accepted prefix; excess provisional pages return to the arena;
* prefix leases pin complete pages only, and release is explicit, versioned,
  and refused while pinned;
* abort restores the exact committed reachability graph and is idempotent.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import IntEnum

from ..contracts.cache_schema import CacheEffect, CacheLifetime, CacheSequenceRef
from ..contracts.execution import CacheLease, EngineRef, SessionRef
from ..contracts.residency_batch import (
    SINK_PAGE_ID,
    ResidencyBatchArrays,
    ResidencyBatchCapacity,
)

__all__ = [
    "ArenaConfig",
    "LeaseReleaseOutcome",
    "Residency",
    "ResidencyError",
    "ResidencyExhausted",
    "ResidencyReservation",
    "ReservationPlan",
    "RowDemand",
    "SequenceBinding",
    "StaleSequenceError",
]


class ResidencyError(RuntimeError):
    """A reservation, lease, or transition violates residency ownership."""


class ResidencyExhausted(ResidencyError):
    """Retryable pre-launch condition: the arena cannot cover the plan."""


class StaleSequenceError(ResidencyError):
    """A binding names a sequence generation that is no longer current."""


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """One domain-scoped page arena. Page ``0`` is the residency-owned sink."""

    domain_id: int
    page_count: int
    page_tokens: int


@dataclass(frozen=True, slots=True)
class SequenceBinding:
    """One row's use of one cache sequence inside a transaction.

    ``reserve_rows`` is the provisional extent beyond the committed length
    (append and tentative effects) or the overlay extent (transient); the
    current transaction writes exactly those rows. Read-only bindings reserve
    nothing.
    """

    sequence: CacheSequenceRef
    effect: CacheEffect
    reserve_rows: int = 0
    input_leases: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RowDemand:
    row_id: int
    bindings: tuple[SequenceBinding, ...]


@dataclass(frozen=True, slots=True)
class ReservationPlan:
    """Typed per-row logical requirements; no callbacks, no physical ids."""

    rows: tuple[RowDemand, ...]


class LeaseReleaseOutcome(IntEnum):
    RELEASED = 1
    BUSY = 2
    STALE = 3


class _Arena:
    def __init__(self, config: ArenaConfig) -> None:
        if config.page_count < 2 or config.page_tokens <= 0:
            raise ResidencyError("an arena needs a sink page and real capacity")
        self.config = config
        self._free: list[int] = list(range(config.page_count - 1, 0, -1))
        self._refcounts: dict[int, int] = {}

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise ResidencyExhausted(
                f"domain {self.config.domain_id} needs {count} pages; "
                f"{len(self._free)} free"
            )
        pages = [self._free.pop() for _ in range(count)]
        for page in pages:
            self._refcounts[page] = 1
        return pages

    def retain(self, page: int) -> None:
        self._refcounts[page] += 1

    def refcount(self, page: int) -> int:
        return self._refcounts.get(page, 0)

    def release(self, page: int) -> None:
        count = self._refcounts.get(page)
        if count is None:
            raise ResidencyError(f"page {page} is not allocated")
        if count == 1:
            del self._refcounts[page]
            self._free.append(page)
        else:
            self._refcounts[page] = count - 1


class _SequenceState:
    __slots__ = (
        "sequence_id",
        "generation",
        "domain_id",
        "role_id",
        "lifetime",
        "session",
        "committed_rows",
        "pages",
        "released",
    )

    def __init__(
        self,
        sequence_id: int,
        session: SessionRef,
        domain_id: int,
        role_id: int,
        lifetime: CacheLifetime,
    ) -> None:
        self.sequence_id = sequence_id
        self.generation = 0
        self.domain_id = domain_id
        self.role_id = role_id
        self.lifetime = lifetime
        self.session = session
        self.committed_rows = 0
        self.pages: list[int] = []
        self.released = False

    def ref(self, engine: EngineRef) -> CacheSequenceRef:
        return CacheSequenceRef(
            engine=engine,
            session=self.session,
            sequence_id=self.sequence_id,
            generation=self.generation,
            domain_id=self.domain_id,
            role_id=self.role_id,
            committed_rows=self.committed_rows,
            lifetime=self.lifetime,
        )


@dataclass(slots=True)
class _PrefixRecord:
    lease_id: int
    sequence_id: int
    rows: int
    pages: tuple[int, ...]
    domain_id: int
    version: int
    pins: int = 0
    released: bool = False


@dataclass(slots=True)
class _BindingReservation:
    binding: SequenceBinding
    state: _SequenceState
    arena: _Arena
    # Shared per-sequence provisional chain for this plan: stacked writing
    # bindings on one sequence extend one chain with consecutive,
    # nonoverlapping extents (the companion's two-regions-one-sequence rule).
    chain: list[int] = field(default_factory=list)
    base_rows: int = 0
    provisional_pages: list[int] = field(default_factory=list)
    cloned_partial: tuple[int, int] | None = None  # (original, clone)
    pinned_leases: tuple[int, ...] = ()

    @property
    def committed_rows(self) -> int:
        return self.state.committed_rows

    def provisional_rows(self) -> int:
        """Maximum physical row extent reserved through this binding.

        Transient overlays begin on a fresh page boundary after the committed
        chain (a shared partial committed page is never written), so their
        extent counts from that aligned begin; explicit holes are legal in the
        packed mapping.
        """

        effect = self.binding.effect
        if effect is CacheEffect.READ_ONLY:
            return self.state.committed_rows
        return self.base_rows + self.binding.reserve_rows

    def read_chain(self) -> list[int]:
        """The page chain this binding reads (shared plan chain)."""

        return list(self.chain)

    def write_locations(self) -> list[tuple[int, int]]:
        """``(page, offset)`` per written row, in logical row order."""

        page_tokens = self.arena.config.page_tokens
        if self.binding.effect is CacheEffect.READ_ONLY:
            return []
        rows = range(self.base_rows, self.base_rows + self.binding.reserve_rows)
        return [
            (self.chain[row // page_tokens], row % page_tokens) for row in rows
        ]


class ResidencyReservation:
    """All-or-nothing ownership of one transaction's provisional storage."""

    def __init__(
        self,
        residency: "Residency",
        bindings: list[_BindingReservation],
    ) -> None:
        self._residency = residency
        self._bindings = bindings
        self._resolved = False

    @property
    def bindings(self) -> tuple[_BindingReservation, ...]:
        return tuple(self._bindings)

    def commit(self, committed_rows: tuple[int, ...]) -> None:
        """Publish exactly the accepted extents; non-failing after validation.

        ``committed_rows[i]`` is binding ``i``'s accepted row count: all
        reserved rows for ordinary appends, the accepted prefix for tentative
        appends, and zero for transient overlays and read-only bindings.
        """

        if self._resolved:
            raise ResidencyError("reservation is already resolved")
        if len(committed_rows) != len(self._bindings):
            raise ResidencyError("commit needs one extent per binding")
        fully_committed: dict[int, bool] = {}
        for reservation, advance in zip(self._bindings, committed_rows):
            effect = reservation.binding.effect
            if effect in (CacheEffect.READ_ONLY, CacheEffect.TRANSIENT_OVERLAY):
                if advance != 0:
                    raise ResidencyError(
                        f"{effect.name} bindings commit zero rows"
                    )
            elif not 0 <= advance <= reservation.binding.reserve_rows:
                raise ResidencyError(
                    f"accepted extent {advance} is outside the reserved tail"
                )
            else:
                sequence_id = reservation.state.sequence_id
                if advance > 0 and not fully_committed.get(sequence_id, True):
                    raise ResidencyError(
                        "a stacked binding cannot publish rows after a "
                        "partially committed earlier extent"
                    )
                fully_committed[sequence_id] = fully_committed.get(
                    sequence_id, True
                ) and advance == reservation.binding.reserve_rows
        # Validation complete; the transition below cannot fail.
        for reservation, advance in zip(self._bindings, committed_rows):
            self._commit_binding(reservation, advance)
        self._residency._release_pins(self._bindings)
        self._resolved = True

    def abort(self) -> None:
        """Release every provisional page; idempotent; committed state intact."""

        if self._resolved:
            return
        for reservation in self._bindings:
            for page in reservation.provisional_pages:
                reservation.arena.release(page)
            if reservation.cloned_partial is not None:
                reservation.arena.release(reservation.cloned_partial[1])
        self._residency._release_pins(self._bindings)
        self._resolved = True

    def batch_arrays(
        self,
        capacity: ResidencyBatchCapacity,
        write_token_begins: tuple[int, ...],
    ) -> ResidencyBatchArrays:
        """Populate the packed device mapping for this reservation.

        ``write_token_begins[i]`` is binding ``i``'s first packed-token index;
        its written rows occupy consecutive token slots. Read-only bindings
        take no write slots (pass their row's token begin; zero rows follow).
        """

        bindings = self._bindings
        if len(write_token_begins) != len(bindings):
            raise ResidencyError("one token begin per binding is required")
        if len(bindings) > capacity.bindings:
            raise ResidencyExhausted(
                f"transaction needs {len(bindings)} bindings; capacity is "
                f"{capacity.bindings}"
            )
        binding_active = [0] * (capacity.bindings + 1)
        binding_domain = [0] * (capacity.bindings + 1)
        binding_committed = [0] * (capacity.bindings + 1)
        binding_provisional = [0] * (capacity.bindings + 1)
        indptr = [0] * (capacity.bindings + 2)
        page_ids: list[int] = []
        write_page_ids = [SINK_PAGE_ID] * capacity.tokens
        write_page_offsets = [0] * capacity.tokens
        write_active = [0] * capacity.tokens
        for index, reservation in enumerate(bindings, start=1):
            chain = reservation.read_chain()
            binding_active[index] = 1
            binding_domain[index] = reservation.state.domain_id
            binding_committed[index] = reservation.committed_rows
            binding_provisional[index] = reservation.provisional_rows()
            page_ids.extend(chain)
            indptr[index + 1] = len(page_ids)
            locations = reservation.write_locations()
            begin = write_token_begins[index - 1]
            if begin + len(locations) > capacity.tokens:
                raise ResidencyExhausted(
                    "write locations exceed the packed token capacity"
                )
            for offset, (page, page_offset) in enumerate(locations):
                slot = begin + offset
                write_page_ids[slot] = page
                write_page_offsets[slot] = page_offset
                write_active[slot] = 1
        if len(page_ids) > capacity.page_references:
            raise ResidencyExhausted(
                f"transaction references {len(page_ids)} pages; capacity is "
                f"{capacity.page_references}"
            )
        for index in range(len(bindings) + 1, capacity.bindings + 2):
            indptr[index] = len(page_ids)
        page_column = page_ids + [SINK_PAGE_ID] * (
            capacity.page_references - len(page_ids)
        )
        return ResidencyBatchArrays(
            active_binding_count=len(bindings),
            active_page_reference_count=len(page_ids),
            binding_active=binding_active,
            binding_domain_id=binding_domain,
            binding_committed_rows=binding_committed,
            binding_provisional_rows=binding_provisional,
            binding_page_indptr=indptr,
            page_ids=page_column,
            write_page_ids=write_page_ids,
            write_page_offsets=write_page_offsets,
            write_active=write_active,
        )

    # ------------------------------------------------------------------ #

    def _commit_binding(self, reservation: _BindingReservation, advance: int) -> None:
        state = reservation.state
        arena = reservation.arena
        effect = reservation.binding.effect
        if effect in (CacheEffect.READ_ONLY, CacheEffect.TRANSIENT_OVERLAY):
            for page in reservation.provisional_pages:
                arena.release(page)
            if reservation.cloned_partial is not None:
                arena.release(reservation.cloned_partial[1])
            if effect is CacheEffect.TRANSIENT_OVERLAY:
                state.generation += 1
            return
        # Stacked bindings commit in plan order: state.pages already reflects
        # every earlier binding's kept extent, and this binding's provisional
        # pages continue that chain directly.
        page_tokens = arena.config.page_tokens
        final_rows = state.committed_rows + advance
        needed_pages = -(-final_rows // page_tokens)
        chain = list(state.pages)
        if reservation.cloned_partial is not None:
            original, clone = reservation.cloned_partial
            if advance > 0:
                chain[-1] = clone
                arena.release(original)
            else:
                arena.release(clone)
        chain.extend(reservation.provisional_pages)
        for page in chain[needed_pages:]:
            arena.release(page)
        state.pages = chain[:needed_pages]
        state.committed_rows = final_rows
        state.generation += 1


class Residency:
    """Sole owner of cache pages, sequences, prefix pins, and reservations."""

    def __init__(self, engine: EngineRef, arenas: tuple[ArenaConfig, ...]) -> None:
        self.engine = engine
        self._arenas = {config.domain_id: _Arena(config) for config in arenas}
        if len(self._arenas) != len(arenas):
            raise ResidencyError("duplicate arena domain")
        self._sequences: dict[int, _SequenceState] = {}
        self._sequence_ids = itertools.count(1)
        self._leases: dict[int, _PrefixRecord] = {}
        self._lease_ids = itertools.count(1)

    # ------------------------------------------------------------------ #
    # Sequences.
    # ------------------------------------------------------------------ #

    def create_sequence(
        self,
        session: SessionRef,
        *,
        domain_id: int,
        role_id: int,
        lifetime: CacheLifetime,
    ) -> CacheSequenceRef:
        if domain_id not in self._arenas:
            raise ResidencyError(f"unknown cache domain {domain_id}")
        state = _SequenceState(
            next(self._sequence_ids), session, domain_id, role_id, lifetime
        )
        self._sequences[state.sequence_id] = state
        return state.ref(self.engine)

    def sequence_ref(self, sequence_id: int) -> CacheSequenceRef:
        return self._state(sequence_id).ref(self.engine)

    def release_sequence(self, ref: CacheSequenceRef) -> None:
        """Idempotent release of a sequence and its page references."""

        state = self._sequences.get(ref.sequence_id)
        if state is None or state.released:
            return
        arena = self._arenas[state.domain_id]
        for page in state.pages:
            arena.release(page)
        state.pages = []
        state.released = True
        del self._sequences[ref.sequence_id]

    # ------------------------------------------------------------------ #
    # Reservation.
    # ------------------------------------------------------------------ #

    def reserve(self, plan: ReservationPlan) -> ResidencyReservation:
        """All-or-nothing reservation across every row and binding.

        Stacked writing bindings on one sequence receive consecutive,
        nonoverlapping provisional extents on one shared chain (the cache
        companion's rule for two regions appending one sequence in a row).
        """

        bindings: list[_BindingReservation] = []
        contexts: dict[int, _BindingReservation] = {}
        try:
            for row in plan.rows:
                for binding in row.bindings:
                    reservation = self._reserve_binding(
                        binding, contexts.get(binding.sequence.sequence_id)
                    )
                    bindings.append(reservation)
                    if binding.effect in (
                        CacheEffect.PERSISTENT_APPEND,
                        CacheEffect.TENTATIVE_APPEND,
                    ):
                        contexts[binding.sequence.sequence_id] = reservation
        except ResidencyError:
            for reservation in bindings:
                for page in reservation.provisional_pages:
                    reservation.arena.release(page)
                if reservation.cloned_partial is not None:
                    reservation.arena.release(reservation.cloned_partial[1])
            self._release_pins(bindings)
            raise
        return ResidencyReservation(self, bindings)

    def _reserve_binding(
        self,
        binding: SequenceBinding,
        stacked_on: "_BindingReservation | None",
    ) -> _BindingReservation:
        state = self._state(binding.sequence.sequence_id)
        if state.generation != binding.sequence.generation:
            raise StaleSequenceError(
                f"sequence {state.sequence_id} is at generation "
                f"{state.generation}; binding names {binding.sequence.generation}"
            )
        if state.domain_id != binding.sequence.domain_id:
            raise ResidencyError("binding names the wrong cache domain")
        arena = self._arenas[state.domain_id]
        page_tokens = arena.config.page_tokens
        pinned = self._pin_leases(binding.input_leases)
        reservation = _BindingReservation(
            binding=binding, state=state, arena=arena, pinned_leases=pinned
        )
        effect = binding.effect
        try:
            if effect is CacheEffect.READ_ONLY:
                if binding.reserve_rows:
                    raise ResidencyError("read-only bindings reserve no rows")
                reservation.chain = list(state.pages)
                reservation.base_rows = state.committed_rows
                return reservation
            if binding.reserve_rows <= 0:
                raise ResidencyError(
                    f"{effect.name} bindings reserve at least one row"
                )
            if effect is CacheEffect.TRANSIENT_OVERLAY:
                if stacked_on is not None:
                    raise ResidencyError(
                        "a transient overlay cannot stack on a writing binding"
                    )
                overlay_pages = arena.allocate(
                    -(-binding.reserve_rows // page_tokens)
                )
                reservation.provisional_pages = overlay_pages
                reservation.chain = list(state.pages) + overlay_pages
                reservation.base_rows = len(state.pages) * page_tokens
                return reservation
            if stacked_on is not None:
                # Continue the shared chain exactly where the earlier writing
                # binding's provisional extent ends.
                base = stacked_on.base_rows + stacked_on.binding.reserve_rows
                chain = stacked_on.chain
            else:
                base = state.committed_rows
                chain = list(state.pages)
                partial = base % page_tokens
                if partial and chain and arena.refcount(chain[-1]) > 1:
                    clone = arena.allocate(1)[0]
                    reservation.cloned_partial = (chain[-1], clone)
                    chain[-1] = clone
            total = base + binding.reserve_rows
            needed = -(-total // page_tokens) - len(chain)
            reservation.provisional_pages = arena.allocate(max(needed, 0))
            chain.extend(reservation.provisional_pages)
            reservation.chain = chain
            reservation.base_rows = base
        except ResidencyError:
            if reservation.cloned_partial is not None:
                arena.release(reservation.cloned_partial[1])
            self._release_pins([reservation])
            raise
        return reservation

    # ------------------------------------------------------------------ #
    # Prefix leases.
    # ------------------------------------------------------------------ #

    def publish_prefix(
        self,
        ref: CacheSequenceRef,
        rows: int,
        *,
        identity_digest: bytes,
        identity_schema: int,
    ) -> CacheLease:
        """Publish a complete-page committed prefix as a scheduler lease."""

        state = self._state(ref.sequence_id)
        if state.generation != ref.generation:
            raise StaleSequenceError("cannot publish from a stale reference")
        page_tokens = self._arenas[state.domain_id].config.page_tokens
        if rows <= 0 or rows > state.committed_rows or rows % page_tokens:
            raise ResidencyError(
                "prefix publication covers a nonempty complete-page committed "
                "prefix only"
            )
        pages = tuple(state.pages[: rows // page_tokens])
        arena = self._arenas[state.domain_id]
        for page in pages:
            arena.retain(page)
        record = _PrefixRecord(
            lease_id=next(self._lease_ids),
            sequence_id=state.sequence_id,
            rows=rows,
            pages=pages,
            domain_id=state.domain_id,
            version=1,
        )
        self._leases[record.lease_id] = record
        return CacheLease(
            lease_id=record.lease_id,
            engine_epoch=self.engine.engine_epoch,
            identity_digest=identity_digest,
            identity_schema=identity_schema,
            charge=rows,
            version=record.version,
            residency_handle=record.lease_id,
        )

    def adopt_prefix(
        self,
        lease: CacheLease,
        session: SessionRef,
        *,
        role_id: int,
        lifetime: CacheLifetime,
    ) -> CacheSequenceRef:
        """Seed a new sequence from an exact published prefix (shared pages)."""

        record = self._lease_record(lease)
        arena = self._arenas[record.domain_id]
        state = _SequenceState(
            next(self._sequence_ids), session, record.domain_id, role_id, lifetime
        )
        for page in record.pages:
            arena.retain(page)
        state.pages = list(record.pages)
        state.committed_rows = record.rows
        self._sequences[state.sequence_id] = state
        return state.ref(self.engine)

    def release_lease(self, lease: CacheLease) -> LeaseReleaseOutcome:
        record = self._leases.get(lease.lease_id)
        if record is None or record.version != lease.version or record.released:
            return LeaseReleaseOutcome.STALE
        if record.pins:
            return LeaseReleaseOutcome.BUSY
        arena = self._arenas[record.domain_id]
        for page in record.pages:
            arena.release(page)
        record.released = True
        del self._leases[lease.lease_id]
        return LeaseReleaseOutcome.RELEASED

    # ------------------------------------------------------------------ #
    # Pressure.
    # ------------------------------------------------------------------ #

    def pressure(self) -> dict[int, dict[str, int]]:
        report: dict[int, dict[str, int]] = {}
        for domain_id, arena in self._arenas.items():
            report[domain_id] = {
                "total_pages": arena.config.page_count - 1,
                "free_pages": arena.free_pages,
            }
        return report

    # ------------------------------------------------------------------ #

    def _state(self, sequence_id: int) -> _SequenceState:
        state = self._sequences.get(sequence_id)
        if state is None:
            raise StaleSequenceError(f"unknown or released sequence {sequence_id}")
        return state

    def _lease_record(self, lease: CacheLease) -> _PrefixRecord:
        record = self._leases.get(lease.lease_id)
        if record is None or record.released:
            raise ResidencyError(f"lease {lease.lease_id} is not published")
        if record.version != lease.version:
            raise StaleSequenceError(f"lease {lease.lease_id} version is stale")
        if lease.engine_epoch != self.engine.engine_epoch:
            raise StaleSequenceError("lease names a foreign engine epoch")
        return record

    def _pin_leases(self, lease_ids: tuple[int, ...]) -> tuple[int, ...]:
        pinned: list[int] = []
        for lease_id in lease_ids:
            record = self._leases.get(lease_id)
            if record is None or record.released:
                raise StaleSequenceError(f"input lease {lease_id} is not published")
            record.pins += 1
            pinned.append(lease_id)
        return tuple(pinned)

    def _release_pins(self, bindings: list[_BindingReservation]) -> None:
        for reservation in bindings:
            for lease_id in reservation.pinned_leases:
                record = self._leases.get(lease_id)
                if record is not None and record.pins > 0:
                    record.pins -= 1
            reservation.pinned_leases = ()
