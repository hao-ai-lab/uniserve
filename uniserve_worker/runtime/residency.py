"""System-owned physical residency for every Rust-leased resource class.

The Rust control plane accounts ``kv_block``, ``image_latent``, ``scratch``,
``encoder_output`` and ``adapter`` as integer/handle leases. The
:class:`ResidencyManager` turns each leased handle into a physical buffer by one
uniform rule: it pairs each resource class with a physical pool addressed by the
handle Rust issues, so the model receives indices, never memory.

:class:`~uniserve_worker.runtime.resources.ResourceRuntime` is the
lease-balance cross-check; the tensors have a single system owner here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .kv_pool import PagedKVPool

if TYPE_CHECKING:
    import torch

    from .resources import ResourceRuntime

__all__ = [
    "KvCacheSpec",
    "GenResidencySpec",
    "KvPool",
    "ScratchKvPool",
    "LatentPool",
    "EncoderCache",
    "ResidencyManager",
    "encoder_handle_from_mm_hash",
]

# The system-facing name for the physical paged KV pool. The implementation is
# ``PagedKVPool`` (paged, FP8-store-capable); the worker runtime constructs and
# holds it, not the model.
KvPool = PagedKVPool


class ScratchKvPool(PagedKVPool):
    """The per-CFG-branch uncond KV pool (``scratch`` resource class).

    Identical storage to ``KvPool``; a distinct system type so the residency
    layer names the uncond-branch scratch residency apart from the request-owned
    text KV.
    """


class LatentPool:
    """System-owned per-request denoise latent (``x_t``) buffers.

    The latent trajectory ``x_t`` is a leased buffer the system owns, addressed
    by a handle (the request id). The generation model reads/writes it through
    the handle; the worker owns the storage and frees it at commit.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, Any] = {}

    def set(self, handle: int, latent: Any) -> None:
        self._buffers[int(handle)] = latent

    def get(self, handle: int) -> Any:
        return self._buffers.get(int(handle))

    def free(self, handle: int) -> None:
        self._buffers.pop(int(handle), None)

    def __contains__(self, handle: int) -> bool:
        return int(handle) in self._buffers


def encoder_handle_from_mm_hash(mm_hash: int | None) -> int:
    """Derive a stable, nonzero 64-bit encoder handle from an image content hash.

    The handle must be deterministic from ``mm_hash`` and nonzero (0 means
    no handle). A SplitMix64 finalizer spreads the hash across the u64 space.
    """
    x = int(mm_hash or 0) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF
    return x or 0x9E3779B97F4A7C15


class EncoderCache:
    """System-owned encoder-output store (``encoder_output`` resource class).

    Handle->embedding residency, keyed by the encoder handle the host issues
    (derived from ``mm_hash`` for cross-request reuse). The worker owns the
    budget and eviction.
    """

    def __init__(self, budget: int = 0) -> None:
        self._store: dict[int, Any] = {}
        self.budget = int(budget)

    def put(self, handle: int, value: Any) -> None:
        self._store[int(handle)] = value

    def get(self, handle: int) -> Any:
        return self._store.get(int(handle))

    def pop(self, handle: int) -> Any:
        return self._store.pop(int(handle), None)

    def __contains__(self, handle: int) -> bool:
        return int(handle) in self._store


@dataclass(frozen=True)
class KvCacheSpec:
    """The KV geometry a model *declares* so the system can own the pool.

    Only the model knows its architecture geometry (kv-head count, head dim,
    layer count, compute/store dtype); the *system* owns the resulting tensors.
    This is the seam: the model states the shape, the worker allocates and holds
    the memory — mirroring SGLang building the ``token_to_kv_pool`` from the
    model config.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: "torch.dtype"
    store_dtype: "torch.dtype | str | None" = None


@dataclass(frozen=True)
class GenResidencySpec:
    """A generation model's full residency geometry, declared for system build.

    A multimodal/generation model needs more than one pool (the text ``kv``, the
    per-CFG-branch uncond ``scratch``, and an optional gen-device ``gen_scratch``
    for the tower split). The model computes the geometry/sizing — only it knows
    the gen_device split and the latent-aware scratch sizing — and the *system*
    (``ResidencyManager.build_gen``) constructs and owns the pools outside
    ``models/``.
    """

    kv: KvCacheSpec
    num_blocks: int
    block_size: int
    device: str
    scratch_num_blocks: int
    gen_scratch_num_blocks: int | None = None
    gen_device: str | None = None
    # Tower coordinate the gen-scratch pool is Pinned to (the gen tower); recorded
    # on the pool so the snapshot reshard knows its destination coordinate.
    gen_tower_coord: int | None = None


class ResidencyManager:
    """Worker-owned physical residency for the leased resource classes.

    Holds the KV pool today; the latent / scratch-KV / encoder-cache pools are
    attached for generation (they share the same handle->buffer rule). The
    optional ``ledger`` is the :class:`ResourceRuntime` count cross-check.
    """

    def __init__(
        self,
        *,
        kv: PagedKVPool | None = None,
        latent: Any | None = None,
        scratch: Any | None = None,
        gen_scratch: Any | None = None,
        encoder: Any | None = None,
        ledger: "ResourceRuntime | None" = None,
    ) -> None:
        self.kv = kv
        # The handle-addressed stores (latent trajectories, encoder embeddings)
        # are pure dict residency — always present so the universal "handle->buffer"
        # rule holds even before a model declares GPU geometry. The sized GPU pools
        # (kv / scratch / gen_scratch) are what build()/build_gen() add.
        self.latent = latent if latent is not None else LatentPool()      # image_latent class
        self.scratch = scratch        # ScratchKvPool: per-CFG-branch uncond KV (scratch class)
        self.gen_scratch = gen_scratch  # gen-device scratch (tower-axis generation pool)
        self.encoder = encoder if encoder is not None else EncoderCache()  # encoder_output class
        self.ledger = ledger

    @classmethod
    def build(
        cls,
        spec: KvCacheSpec | None,
        *,
        num_blocks: int,
        block_size: int,
        device: "torch.device | str",
        ledger: "ResourceRuntime | None" = None,
    ) -> "ResidencyManager":
        """Construct the system KV pool from the model-declared geometry + caps.

        ``spec`` is ``None`` only for models that declare no ``kv_block`` class
        (pure encode/commit workers); the KV pool is then absent and the model
        resolves residency from the other pools.
        """
        kv: PagedKVPool | None = None
        if spec is not None:
            kv = PagedKVPool(
                num_layers=int(spec.num_layers),
                num_blocks=int(num_blocks),
                block_size=int(block_size),
                num_kv_heads=int(spec.num_kv_heads),
                head_dim=int(spec.head_dim),
                device=device,
                dtype=spec.dtype,
                store_dtype=spec.store_dtype,
            )
        return cls(kv=kv, ledger=ledger)

    @classmethod
    def build_gen(cls, spec: GenResidencySpec) -> "ResidencyManager":
        """Construct a generation model's pools from its declared geometry.

        Builds the text ``kv`` pool, the per-CFG-branch uncond ``scratch`` pool,
        and (for a tower/gen-device split) the ``gen_scratch`` pool on the gen
        device. This is where the multimodal models' pool construction lives now
        — in the system, not under ``models/``.
        """

        kv = PagedKVPool(
            num_layers=int(spec.kv.num_layers),
            num_blocks=int(spec.num_blocks),
            block_size=int(spec.block_size),
            num_kv_heads=int(spec.kv.num_kv_heads),
            head_dim=int(spec.kv.head_dim),
            device=spec.device,
            dtype=spec.kv.dtype,
            store_dtype=spec.kv.store_dtype,
        )
        scratch = None
        if int(spec.scratch_num_blocks) > 0:
            scratch = ScratchKvPool(
                num_layers=int(spec.kv.num_layers),
                num_blocks=int(spec.scratch_num_blocks),
                block_size=int(spec.block_size),
                num_kv_heads=int(spec.kv.num_kv_heads),
                head_dim=int(spec.kv.head_dim),
                device=spec.device,
                dtype=spec.kv.dtype,
                store_dtype=spec.kv.store_dtype,
            )
        gen_scratch = None
        if spec.gen_scratch_num_blocks is not None and spec.gen_device is not None:
            gen_scratch = PagedKVPool(
                num_layers=int(spec.kv.num_layers),
                num_blocks=int(spec.gen_scratch_num_blocks),
                block_size=int(spec.block_size),
                num_kv_heads=int(spec.kv.num_kv_heads),
                head_dim=int(spec.kv.head_dim),
                device=spec.gen_device,
                dtype=spec.kv.dtype,
                store_dtype=spec.kv.store_dtype,
                tower_coord=spec.gen_tower_coord,
            )
        return cls(
            kv=kv,
            scratch=scratch,
            gen_scratch=gen_scratch,
            latent=LatentPool(),
            encoder=EncoderCache(),
        )

    def kv_pool(self) -> PagedKVPool:
        if self.kv is None:
            raise RuntimeError("ResidencyManager has no KV pool (model declares no kv_block class)")
        return self.kv
