"""Structural typing protocols for KV-cache, buffer-staging, and model caps.

These ``typing.Protocol`` definitions name the duck-typed surfaces that the
worker probes via ``getattr`` at runtime. They document the member set each
collaborator must expose and let the type checker verify call sites; the
concrete classes (``PagedRequestCache``, ``BatchedPagedRequestCache``,
``PagedTextCache``, ``TextTensorStagingSlot``, the registered models) satisfy
them structurally without an explicit subclass relationship.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

__all__ = [
    'KVCacheView',
    'BatchedKVCacheView',
    'BufferStager',
    'ModelCapabilities',
]

if TYPE_CHECKING:
    import torch

    from .kv_pool import PagedKVPool


@runtime_checkable
class KVCacheView(Protocol):
    """Per-request KV-cache surface for attention and packed forward reads.

    The canonical persistent-length member is ``base_len`` (an int attribute).
    ``PagedRequestCache`` additionally exposes ``length()`` as a method that
    returns the same value; new code reads ``base_len`` directly. Batched views
    (``BatchedPagedRequestCache``) carry per-row ``base_lens`` and a scalar
    ``base_len`` set to ``max(base_lens)``.
    """

    pool: PagedKVPool
    base_len: int

    def get(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None: ...

    def block_table(self, *, device: torch.device | str | None = ...) -> torch.Tensor: ...


@runtime_checkable
class BatchedKVCacheView(KVCacheView, Protocol):
    """Batched KV-cache surface with per-row lengths and ragged append.

    Extends :class:`KVCacheView` with the members the batched-paged attention
    path probes: per-row ``base_lens`` and a varlen append.
    """

    base_lens: Sequence[int]

    def append_varlen(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        query_lens: Sequence[int],
        *,
        block_table: torch.Tensor | None = ...,
        cache_seqlens: torch.Tensor | None = ...,
        cu_seqlens_q: torch.Tensor | None = ...,
    ) -> None: ...


@runtime_checkable
class BufferStager(Protocol):
    """Reusable CPU/device staging-buffer provider.

    ``paged_text_cache`` and the text staging path use a stager to recycle
    pinned host buffers and device buffers across iterations instead of
    allocating per call. ``TextTensorStagingSlot`` is the concrete implementer.
    """

    def int_buffer(self, name: str, numel: int, *, pin: bool) -> torch.Tensor: ...

    def device_buffer(
        self,
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor: ...


@runtime_checkable
class ModelCapabilities(Protocol):
    """Model-level capability surface read by ``RunnerDriver._build_caps``.

    Names the capability attributes required by the model protocol boundary when
    a registered model does not expose a prebuilt ``Caps`` snapshot. The members
    here name what the protocol adapter inspects:

    * ``num_layers`` / ``bytes_per_token`` are required — every registered model
      entry exposes them as a model attribute, so a missing one is a model bug,
      not a default.
    * the remaining members are optional and carry documented defaults in the
      fallback because some real models surface them only through ``caps()``.
    """

    num_layers: int
    bytes_per_token: int
    # Optional fallback fields (documented defaults live in ``_build_caps``):
    supported_ops: tuple[str, ...]
    max_latent_size: int
    latent_downsample: int
    supported_controls: tuple[str, ...]
    adapter_mode: str
