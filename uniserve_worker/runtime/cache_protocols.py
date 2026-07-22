"""Structural typing protocols for buffer-staging and model caps.

These ``typing.Protocol`` definitions name the duck-typed surfaces that the
worker probes via ``getattr`` at runtime. They document the member set each
collaborator must expose and let the type checker verify call sites; the
concrete classes (``TextTensorStagingSlot``, the registered models) satisfy
them structurally without an explicit subclass relationship. The per-batch
paged KV view surface lives in ``contracts.attention_plan.KvView``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

__all__ = [
    "BufferStager",
    "ModelCapabilities",
]

if TYPE_CHECKING:
    import torch


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
    """Model-level capability surface read by ``ModelWorker``.

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
