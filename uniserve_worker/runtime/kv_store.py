"""Worker authority for per-request sequence-KV references.

``KvStore`` owns each request's sequence-KV references: the block table, the
committed per-lane KV lengths, and the prefix-reference boundary (``ref_len``)
— the leading, block-aligned token span whose KV is reused prefix cache and is
therefore read-only for the request. Sharing is reference-based rather than
copy-on-write: the scheduler only shares block-aligned prefixes and never
schedules a write below the boundary, and :meth:`KvStore.validate_write_range`
guards that invariant before any KV mutation. The store also keeps a
block-holder registry for reference reclamation and (env-gated via
``UNISERVE_KV_GUARDS``) cross-request exclusive-writer asserts, and it joins
the step transaction so a failed step restores every touched request's
references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..foundation.env import env_flag
from ..foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    from .request_state import RequestState, RequestStateTable

__all__ = [
    "append_new_block_ids",
    "KvEntry",
    "KvStore",
]

_TERMINAL_LIFECYCLES = frozenset({"committed", "aborted"})


def append_new_block_ids(
    block_ids: list[int],
    new_block_ids: list[int] | tuple[int, ...] | None,
) -> bool:
    """Idempotently ingest a host ``ForwardOp.new_block_ids`` payload.

    This is the single source of truth for the host->worker block-id append
    contract: every consumer of ``new_block_ids`` must route through it so the
    idempotency rule lives in exactly one place. A retried/duplicated op that
    carries the same tail of block ids is a no-op (the tail already matches), so
    appends are not double-applied; any other suffix is appended verbatim.

    Mutates ``block_ids`` in place and returns ``True`` iff blocks were appended.
    """
    new_blocks = [int(block_id) for block_id in (new_block_ids or [])]
    if new_blocks and block_ids[-len(new_blocks) :] != new_blocks:
        block_ids.extend(new_blocks)
        return True
    return False


def _merge_registered_block_ids(block_ids: list[int], registered: list[int]) -> None:
    if not registered:
        return
    if not block_ids:
        block_ids.extend(registered)
        return
    if block_ids == registered:
        return
    if len(registered) > len(block_ids) and registered[: len(block_ids)] == block_ids:
        block_ids[:] = registered
        return
    if len(block_ids) >= len(registered) and block_ids[: len(registered)] == registered:
        return
    append_new_block_ids(block_ids, registered)


@dataclass(slots=True)
class KvEntry:
    """One request's committed sequence-KV references.

    ``block_ids`` is THE block-table list for the request: flow branch caches
    alias the list object, so it is only ever mutated in place and never
    rebound after the entry is created. ``ref_len`` is the prefix-reference
    boundary in tokens (the read-only reused-prefix span); ``lengths`` holds
    the committed per-lane KV lengths.
    """

    block_ids: list[int] = field(default_factory=list)
    ref_len: int = 0
    lengths: dict[str, int] = field(default_factory=dict)


class KvStore:
    """Authority for request block tables, lane lengths, and prefix references.

    The per-request :class:`KvEntry` lives on the request state for direct
    hot-path access; this store owns every policy that mutates or validates it
    (lease ingest, write-range guarding, reclamation) plus the block-holder
    registry, and implements the transactional snapshot/restore surface for
    the step transaction.
    """

    def __init__(self, sessions: "RequestStateTable", *, block_size: int = 0) -> None:
        self.sessions = sessions
        self.block_size = int(block_size)
        # Cross-request exclusive-writer asserts (``UNISERVE_KV_GUARDS``);
        # registry bookkeeping and boundary/capacity checks are always on.
        self.guards = env_flag("UNISERVE_KV_GUARDS")
        self._holders: dict[int, set[int]] = {}

    def bind_block_size(self, block_size: int) -> None:
        self.block_size = int(block_size)

    # ---- lease ingest ------------------------------------------------------

    def admit(
        self,
        req_id: int,
        state: "RequestState",
        new_req: Mapping[str, Any],
        *,
        fresh: bool,
    ) -> None:
        """Ingest one registration's block lease and prefix-reference boundary."""
        entry = state.kv
        entry.ref_len = int(new_req.get("prefix_len") or 0)
        if "block_ids" not in new_req:
            return
        incoming = [int(block_id) for block_id in (new_req.get("block_ids") or [])]
        before = len(entry.block_ids)
        if fresh:
            entry.block_ids[:] = incoming
        else:
            _merge_registered_block_ids(entry.block_ids, incoming)
        self._register(int(req_id), entry.block_ids[before:])

    def ingest_new_blocks(
        self,
        req_id: int,
        state: "RequestState",
        new_block_ids: list[int] | tuple[int, ...] | None,
    ) -> bool:
        """Tail-deduping ingest of a host ``new_block_ids`` payload."""
        entry = state.kv
        before = len(entry.block_ids)
        appended = append_new_block_ids(entry.block_ids, new_block_ids)
        if appended:
            self._register(int(req_id), entry.block_ids[before:])
        return appended

    # ---- write guarding ----------------------------------------------------

    def validate_write_range(
        self,
        req_id: int,
        state: "RequestState",
        base: int,
        end: int,
        *,
        block_ids: Sequence[int] | None = None,
    ) -> None:
        """Validate a declared KV write span before any KV mutation.

        ``base`` must sit at or above the request's prefix-reference boundary
        (the read-only reused span) and ``end`` must stay inside the leased
        block capacity. With ``UNISERVE_KV_GUARDS`` enabled, additionally
        asserts that every block the span touches has this request as its only
        registered holder.
        """
        if end <= base:
            return
        entry = state.kv
        if base < entry.ref_len:
            raise invalid_descriptor(
                f"request {req_id} writes KV at {base} below its prefix "
                f"reference boundary {entry.ref_len}"
            )
        blocks = entry.block_ids if block_ids is None else block_ids
        block_size = self.block_size
        if block_size and blocks and end > len(blocks) * block_size:
            raise invalid_descriptor(
                f"request {req_id} writes KV through {end} beyond its "
                f"{len(blocks)}-block lease"
            )
        if self.guards and block_size:
            self._assert_exclusive_writer(int(req_id), blocks, int(base), int(end))

    def _assert_exclusive_writer(
        self,
        req_id: int,
        blocks: Sequence[int],
        base: int,
        end: int,
    ) -> None:
        block_size = self.block_size
        first = base // block_size
        last = min((end - 1) // block_size, len(blocks) - 1)
        for index in range(first, last + 1):
            holders = self._holders.get(int(blocks[index]))
            if holders is None:
                continue
            if len(holders) > 1 or req_id not in holders:
                raise invalid_descriptor(
                    f"request {req_id} writes KV block {blocks[index]} "
                    "shared with another request"
                )

    # ---- block-holder registry ----------------------------------------------

    def _register(self, req_id: int, blocks: Sequence[int]) -> None:
        for raw in blocks:
            block = int(raw)
            holders = self._holders.get(block)
            if holders is None:
                self._holders[block] = {req_id}
                continue
            if req_id in holders:
                continue
            for stale in [holder for holder in holders if self._is_stale(holder)]:
                holders.discard(stale)
            for holder in holders:
                if self._holder_may_write(holder, block):
                    raise invalid_descriptor(
                        f"block {block} leased to request {req_id} is writable "
                        f"by live request {holder}"
                    )
            holders.add(req_id)

    def _is_stale(self, holder: int) -> bool:
        state = self.sessions.peek(holder)
        return state is None or str(state.lifecycle) in _TERMINAL_LIFECYCLES

    def _holder_may_write(self, holder: int, block: int) -> bool:
        # The writable region is defined by block geometry; a store with no
        # bound block size cannot place the boundary and never raises.
        block_size = self.block_size
        if not block_size:
            return False
        state = self.sessions.peek(holder)
        if state is None:
            return False
        entry = state.kv
        # A block is writable by its holder iff any of its tokens lie at or
        # above the holder's prefix-reference boundary; a non-block-aligned
        # boundary makes its boundary block writable, so sharing it is illegal.
        boundary = entry.ref_len // block_size
        try:
            index = entry.block_ids.index(block)
        except ValueError:
            return False
        return index >= boundary

    # ---- reclamation ---------------------------------------------------------

    def release(self, req_id: int, state: "RequestState") -> None:
        """Reclaim every block reference the dropped request held."""
        self._unregister(int(req_id), state.kv.block_ids)

    def _unregister(self, req_id: int, blocks: Sequence[int]) -> None:
        for raw in blocks:
            block = int(raw)
            holders = self._holders.get(block)
            if holders is None:
                continue
            holders.discard(req_id)
            if not holders:
                del self._holders[block]

    def _unregister_request(self, req_id: int) -> None:
        emptied = []
        for block, holders in self._holders.items():
            holders.discard(req_id)
            if not holders:
                emptied.append(block)
        for block in emptied:
            del self._holders[block]

    # ---- step transaction -----------------------------------------------------

    def snapshot_requests(
        self, request_ids: set[int]
    ) -> dict[int, tuple[KvEntry, list[int], int, dict[str, int]]]:
        snapshot: dict[int, tuple[KvEntry, list[int], int, dict[str, int]]] = {}
        for raw in request_ids:
            request_id = int(raw)
            state = self.sessions.peek(request_id)
            if state is None:
                continue
            entry = state.kv
            snapshot[request_id] = (
                entry,
                list(entry.block_ids),
                int(entry.ref_len),
                dict(entry.lengths),
            )
        return snapshot

    def restore_requests(
        self,
        request_ids: set[int],
        snapshot: dict[int, tuple[KvEntry, list[int], int, dict[str, int]]],
    ) -> None:
        for raw in request_ids:
            request_id = int(raw)
            snap = snapshot.get(request_id)
            if snap is None:
                self._unregister_request(request_id)
                continue
            entry, block_ids, ref_len, lengths = snap
            # Lease ingest only appends (prefixes are preserved), so the tail
            # beyond the snapshot is exactly what the failed step registered.
            self._unregister(request_id, entry.block_ids[len(block_ids) :])
            entry.block_ids[:] = block_ids
            entry.ref_len = ref_len
            entry.lengths.clear()
            entry.lengths.update(lengths)
