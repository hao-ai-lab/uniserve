"""The model-facing tower handoff: one API, two transports.

The understanding (text) tower owns the authoritative conditioning KV; the
generation tower reads a *snapshot* of it once per image, denoises against it,
and writes the finished latent back across the tower boundary at commit. Those
three crossings — stage the conditioning KV, wait it ready, write the commit
latent back — are the entire model-visible surface of the und↔gen split.

``TowerHandoff`` is that surface. A model calls it and never branches on the
deployment mode. The mode is chosen by which implementation is bound at
construction:

* :class:`LocalP2PTowerHandoff` — both towers live in one process on two CUDA
  devices; the crossing is an in-process NVLink peer copy with CUDA-event
  readiness barriers.
* :class:`DataPlaneTowerHandoff` — the towers live in separate worker pools; the
  crossing is the register-once point-to-point data plane (``cuda_ipc`` same-node
  / ``mooncake`` cross-node).

A trivial tower (no transport, or src coordinate == dst coordinate) degrades to a
same-device copy.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..foundation.errors import invalid_descriptor
from ..ops.core import Capabilities, Handoff
from .tower_kv import reshard_kv_snapshot, wait_kv_snapshot_ready

__all__ = [
    "TowerBinding",
    "TowerHandoff",
    "LocalP2PTowerHandoff",
    "ConditioningSnapshot",
    "DataPlaneTowerHandoff",
    "TowerStageReq",
    "TowerPublishReq",
    "TowerCommitReq",
]


@dataclass(frozen=True)
class TowerBinding:
    """The live destination residency + coordinates for one handoff crossing.

    Resolved fresh per crossing from the model's current state so the handoff
    tracks the tower-vs-trivial choice and the destination pool as they are set
    up, rather than capturing them once at construction.
    """

    transport: Any | None
    primary_coord: int
    gen_coord: int
    num_layers: int
    block_size: int
    target_pool: Any
    target_device: Any
    allocate_blocks: Callable[[int], list[int]]

    @property
    def active(self) -> bool:
        return self.transport is not None and int(self.primary_coord) != int(self.gen_coord)


@dataclass(frozen=True)
class TowerStageReq:
    value: Any


@dataclass(frozen=True)
class TowerPublishReq:
    cache: Any
    t_index: int = -1
    last_token_id: int | None = None
    tu_cache: Any = None
    tu_t_index: int = -1
    tu_last_token_id: int | None = None
    iu_cache: Any = None
    iu_t_index: int = -1
    iu_last_token_id: int | None = None


@dataclass(frozen=True)
class TowerCommitReq:
    latent: Any
    device: Any
    dtype: Any


@runtime_checkable
class TowerHandoff(Protocol):
    """Move the per-image conditioning KV across the und<->gen tower boundary.

    Bound to a transport by deployment mode. The model calls these three methods
    and never branches on the mode.
    """

    @property
    def active(self) -> bool:
        """True when a real tower split is in effect (a non-trivial transport)."""
        ...

    def stage_conditioning(self, cache: Any) -> Any:
        """Snapshot a conditioning-KV branch into a writable gen-side replica.

        Copies the conditioning prefix across the tower boundary and waits for the
        copy to finish before the gen tower reads it. Returns the destination cache
        the gen tower denoises against. On a trivial tower this is a same-device copy
        into the scratch pool.
        """
        ...

    def await_ready(self, replica: Any) -> None:
        """Block the gen tower until the staged replica is fully written."""
        ...

    def writeback_commit(self, latent: Any, *, device: Any, dtype: Any) -> Any:
        """Bring the finished latent back to the understanding coordinate.

        Orders the gen->und crossing (a ``Pinned(gen) -> Pinned(primary)`` barrier)
        and returns the latent on the understanding device, ready for re-encode +
        KV writeback. On a trivial tower the latent is already on the model device.
        """
        ...


@dataclass
class LocalP2PTowerHandoff:
    """In-process NVLink peer copy + CUDA-event barriers for tower handoff.

    ``stage_conditioning`` reshards the paged KV snapshot;
    ``await_ready`` waits for the replica to be fully written;
    ``writeback_commit`` brings the finished latent back to the understanding
    device.

    For a trivial tower the binding's ``transport`` is ``None`` and
    ``primary_coord == gen_coord`` — every method degrades to the single-device path.
    """

    name = "local_p2p"
    operator = "tower_handoff"

    bind: Callable[[], TowerBinding]

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({self.name, self.operator, "conditioning", "commit"}))

    @property
    def active(self) -> bool:
        return self.bind().active

    def can_dispatch(self, req: Any, *, mesh: Any | None = None) -> bool:
        del mesh
        return isinstance(req, TowerStageReq)

    def dispatch(self, req: Any, *, mesh: Any | None = None) -> Handoff:
        del mesh
        if not isinstance(req, TowerStageReq):
            raise invalid_descriptor("local tower handoff dispatch expects TowerStageReq")
        return Handoff(
            format="conditioning_cache",
            payload=self._stage_conditioning(req.value),
            provider=self.name,
            operator=self.operator,
        )

    def can_combine(self, handoff: Handoff, *, mesh: Any | None = None) -> bool:
        del mesh
        return handoff.format == "commit_latent" and isinstance(handoff.payload, TowerCommitReq)

    def combine(self, handoff: Handoff, *, mesh: Any | None = None) -> Any:
        del mesh
        if not self.can_combine(handoff):
            raise invalid_descriptor("local tower handoff combine expects commit_latent")
        req = handoff.payload
        return self._writeback_commit(req.latent, device=req.device, dtype=req.dtype)

    def stage_conditioning(self, cache: Any) -> Any:
        return self.dispatch(TowerStageReq(cache)).payload

    def _stage_conditioning(self, cache: Any) -> Any:
        if cache is None:
            return cache
        b = self.bind()
        if b.target_pool is None:
            raise invalid_descriptor("tower handoff: destination KV pool is not initialized")
        if getattr(cache, "pool", None) is b.target_pool:
            return cache
        return reshard_kv_snapshot(
            cache,
            target_pool=b.target_pool,
            allocate_blocks=b.allocate_blocks,
            num_layers=b.num_layers,
            block_size=b.block_size,
            target_device=b.target_device,
            transport=b.transport if b.active else None,
            src_coord=b.primary_coord,
            dst_coord=b.gen_coord,
        )

    def await_ready(self, replica: Any) -> None:
        self._await_ready(replica)

    def _await_ready(self, replica: Any) -> None:
        b = self.bind()
        if not b.active:
            return
        wait_kv_snapshot_ready(replica, transport=b.transport, coord=b.gen_coord)

    def writeback_commit(self, latent: Any, *, device: Any, dtype: Any) -> Any:
        return self.combine(
            Handoff(
                format="commit_latent",
                payload=TowerCommitReq(latent, device, dtype),
                provider=self.name,
                operator=self.operator,
            )
        )

    def _writeback_commit(self, latent: Any, *, device: Any, dtype: Any) -> Any:
        b = self.bind()
        if b.active:
            done = b.transport.record_ready(b.gen_coord)
            b.transport.wait_ready(done, b.primary_coord)
        return latent.to(device=device, dtype=dtype, non_blocking=True)


@dataclass(frozen=True)
class ConditioningSnapshot:
    """The wire form of a conditioning-KV crossing: per-layer (k, v) locators.

    The und pool publishes the cond KV prefix once per image; ``locators`` holds
    2*num_layers opaque data-plane locators (k then v per layer, layer-major) that
    ride the control plane (``SeqResult.locator``) to the gen pool, which fetches
    them into its writable replica. ``t_index``/``last_token_id`` carry the small
    decode-state scalars the gen tower needs to rebuild ``st.cond`` (the position
    index for the image RoPE and the img-start guard) without a second crossing.

    ``to_wire``/``from_wire`` is the ``str`` form for the ``SeqResult.locator``
    wire field: base64 of the pickled snapshot, the same opaque-locator encoding
    the control plane already uses for the decode→sampler edge (every ``Locator``
    is pickle-clean, incl. the ``cuda_ipc`` IPC-handle reduction)."""

    locators: tuple[Any, ...]
    length: int
    num_layers: int
    t_index: int = -1
    last_token_id: int | None = None
    tu_locators: tuple[Any, ...] = ()
    tu_length: int = 0
    tu_t_index: int = -1
    tu_last_token_id: int | None = None
    iu_locators: tuple[Any, ...] = ()
    iu_length: int = 0
    iu_t_index: int = -1
    iu_last_token_id: int | None = None

    def to_wire(self) -> str:
        import base64
        import pickle

        return base64.b64encode(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

    @staticmethod
    def from_wire(s: str) -> "ConditioningSnapshot":
        import base64
        import pickle

        obj = pickle.loads(base64.b64decode(s.encode("ascii")))
        if not isinstance(obj, ConditioningSnapshot):
            raise invalid_descriptor("decoded object is not a ConditioningSnapshot")
        return obj


@dataclass
class DataPlaneTowerHandoff:
    """Cross-process und↔gen handoff over the data plane.

    The two towers live in separate worker pools. The und pool publishes the
    conditioning KV pages and the gen pool fetches them into its registered
    replica (register-once point-to-point, ``cuda_ipc`` same-node /
    ``mooncake`` cross-node — never a collective). ``data_plane`` is the
    per-worker ``Transport`` (``publish``/``fetch``); ``bind`` resolves the
    gen-side destination residency.

    Producer (und) side calls :meth:`publish_conditioning`; consumer (gen) side
    calls :meth:`stage_conditioning` with the resulting snapshot. Cross-pool
    readiness is enforced host-side by ``StageRouter``/``TensorMover``, so the
    worker-side ``await_ready`` is a no-op for a synchronous fetch."""

    name = "data_plane"
    operator = "tower_handoff"

    data_plane: Any
    bind: Callable[[], TowerBinding]

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({self.name, self.operator, "conditioning", "commit"}))

    @property
    def active(self) -> bool:
        return True

    def can_dispatch(self, req: Any, *, mesh: Any | None = None) -> bool:
        del mesh
        return isinstance(req, (TowerPublishReq, TowerStageReq))

    def dispatch(self, req: Any, *, mesh: Any | None = None) -> Handoff:
        del mesh
        if isinstance(req, TowerPublishReq):
            return Handoff(
                format="conditioning_snapshot",
                payload=self._publish_conditioning(
                    req.cache,
                    t_index=req.t_index,
                    last_token_id=req.last_token_id,
                    tu_cache=req.tu_cache,
                    tu_t_index=req.tu_t_index,
                    tu_last_token_id=req.tu_last_token_id,
                    iu_cache=req.iu_cache,
                    iu_t_index=req.iu_t_index,
                    iu_last_token_id=req.iu_last_token_id,
                ),
                provider=self.name,
                operator=self.operator,
            )
        if isinstance(req, TowerStageReq):
            return Handoff(
                format="conditioning_cache",
                payload=self._stage_conditioning(req.value),
                provider=self.name,
                operator=self.operator,
            )
        raise invalid_descriptor("data-plane tower handoff dispatch expects TowerPublishReq or TowerStageReq")

    def can_combine(self, handoff: Handoff, *, mesh: Any | None = None) -> bool:
        del mesh
        return handoff.format == "commit_latent" and isinstance(handoff.payload, TowerCommitReq)

    def combine(self, handoff: Handoff, *, mesh: Any | None = None) -> Any:
        del mesh
        if not self.can_combine(handoff):
            raise invalid_descriptor("data-plane tower handoff combine expects commit_latent")
        req = handoff.payload
        return self._writeback_commit(req.latent, device=req.device, dtype=req.dtype)

    def publish_conditioning(
        self,
        cache: Any,
        *,
        t_index: int = -1,
        last_token_id: int | None = None,
        tu_cache: Any = None,
        tu_t_index: int = -1,
        tu_last_token_id: int | None = None,
        iu_cache: Any = None,
        iu_t_index: int = -1,
        iu_last_token_id: int | None = None,
    ) -> ConditioningSnapshot | None:
        """Producer (und) side: publish ``cache``'s ``[0, length)`` KV prefix.

        Returns the snapshot of per-layer (k, v) locators plus the decode-state
        scalars (``t_index``/``last_token_id``) to thread to the gen pool, or
        ``None`` for an empty/absent cache."""
        return self.dispatch(
            TowerPublishReq(
                cache,
                t_index=int(t_index),
                last_token_id=last_token_id,
                tu_cache=tu_cache,
                tu_t_index=int(tu_t_index),
                tu_last_token_id=tu_last_token_id,
                iu_cache=iu_cache,
                iu_t_index=int(iu_t_index),
                iu_last_token_id=iu_last_token_id,
            )
        ).payload

    def _publish_conditioning(
        self,
        cache: Any,
        *,
        t_index: int = -1,
        last_token_id: int | None = None,
        tu_cache: Any = None,
        tu_t_index: int = -1,
        tu_last_token_id: int | None = None,
        iu_cache: Any = None,
        iu_t_index: int = -1,
        iu_last_token_id: int | None = None,
    ) -> ConditioningSnapshot | None:
        if cache is None:
            return None
        def _publish_branch(branch_cache: Any) -> tuple[tuple[Any, ...], int, int]:
            if branch_cache is None:
                return (), 0, 0
            pool = getattr(branch_cache, "pool", None)
            blocks = list(getattr(branch_cache, "block_ids", []) or [])
            length = int(branch_cache.get_seq_length())
            if pool is None or length <= 0:
                return (), length, 0
            num_layers = int(self.bind().num_layers)
            locators: list[Any] = []
            for layer_idx in range(num_layers):
                k, v = pool.read(layer_idx, blocks, start=0, length=length)
                locators.append(self.data_plane.publish(k))
                locators.append(self.data_plane.publish(v))
            return tuple(locators), length, num_layers

        locators, length, num_layers = _publish_branch(cache)
        if length <= 0 or not locators:
            return ConditioningSnapshot(
                locators=(), length=length, num_layers=0,
                t_index=int(t_index), last_token_id=last_token_id,
            )
        tu_locators, tu_length, tu_num_layers = _publish_branch(tu_cache)
        iu_locators, iu_length, iu_num_layers = _publish_branch(iu_cache)
        if tu_locators and tu_num_layers != num_layers:
            raise invalid_descriptor("data-plane tower handoff: tu branch layer count mismatch")
        if iu_locators and iu_num_layers != num_layers:
            raise invalid_descriptor("data-plane tower handoff: iu branch layer count mismatch")
        return ConditioningSnapshot(
            locators=tuple(locators), length=length, num_layers=num_layers,
            t_index=int(t_index), last_token_id=last_token_id,
            tu_locators=tu_locators, tu_length=tu_length,
            tu_t_index=int(tu_t_index), tu_last_token_id=tu_last_token_id,
            iu_locators=iu_locators, iu_length=iu_length,
            iu_t_index=int(iu_t_index), iu_last_token_id=iu_last_token_id,
        )

    def stage_conditioning(self, snapshot: ConditioningSnapshot | None) -> Any:
        """Consumer (gen) side: fetch a published snapshot into the gen replica.

        Allocates a writable replica in the gen residency and fetches each
        layer's (k, v) from its locator, leaving room to append the transient
        denoise K/V. Returns the destination cache the gen tower denoises against."""
        return self.dispatch(TowerStageReq(snapshot)).payload

    def _stage_conditioning(self, snapshot: ConditioningSnapshot | None) -> Any:
        if snapshot is None:
            return None
        from ..foundation.sizing import ceil_div
        from .paged_text_cache import PagedTextCache

        b = self.bind()
        if b.target_pool is None:
            raise invalid_descriptor("data-plane tower handoff: gen replica pool is not initialized")
        initial_blocks = ceil_div(snapshot.length, b.block_size)
        locators = snapshot.locators
        length = snapshot.length
        num_layers = snapshot.num_layers
        if locators and len(locators) != 2 * num_layers:
            raise invalid_descriptor("data-plane tower handoff: malformed conditioning snapshot")
        block_ids = b.allocate_blocks(initial_blocks)
        replica = PagedTextCache(
            b.target_pool,
            block_ids,
            num_layers=b.num_layers,
            length=length,
            allocate_blocks=b.allocate_blocks,
        )
        for layer_idx in range(num_layers):
            k = self.data_plane.fetch(locators[2 * layer_idx])
            v = self.data_plane.fetch(locators[2 * layer_idx + 1])
            # The fetch materializes on the producer's device (the und GPU); land it
            # on the gen replica's device before the write (a no-op same-GPU).
            if b.target_device is not None and hasattr(k, "to"):
                k = k.to(b.target_device)
                v = v.to(b.target_device)
            b.target_pool.write(layer_idx, replica.block_ids, start=0, k=k, v=v)
        return replica

    def await_ready(self, replica: Any) -> None:
        # Synchronous fetch: readiness is the host-side StageRouter gate (Invariant
        # B), so there is no worker-local event to wait.
        return None

    def publish_commit_latent(self, latent: Any) -> Any:
        """Publish the finished latent for the und pool to fetch and re-encode."""
        return self.data_plane.publish(latent)

    def writeback_commit(self, latent: Any, *, device: Any, dtype: Any) -> Any:
        return self.combine(
            Handoff(
                format="commit_latent",
                payload=TowerCommitReq(latent, device, dtype),
                provider=self.name,
                operator=self.operator,
            )
        )

    def _writeback_commit(self, latent: Any, *, device: Any, dtype: Any) -> Any:
        del device, dtype
        # The gen pool produces the latent and publishes it; the und pool fetches +
        # re-encodes (the two-hop straddle). On this (gen) side the locator is the
        # edge; the und side materializes it on its device.
        return self.publish_commit_latent(latent)
