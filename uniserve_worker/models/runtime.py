"""Imperative model-runner boundary and worker-local runtime geometry."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from uniserve_worker.config import WorkerConfig

from ..execution.batch import DecodeRange, MediaGeometry, OpCode, TensorSpec
from ..execution.bounded_storage import BoundedTensorStorage, TensorSchema
from ..execution.forward_batch import AttentionSelection, ForwardBatch, ForwardOutput
from ..foundation.errors import invalid_descriptor
from ..nn.diffusion.schedule import DiffusionSchedule
from ..nn.mesh import Communicator
from ..nn.parallel_attention import AttentionContextGeometry, AttentionContextWorkspace
from ..transfer.layout import TensorRegion

if TYPE_CHECKING:
    from ..loader.component import CheckpointComponent, ModelBuildContext, ModelConstruction
    from ..runtime.cache_pool import CachePool
    from .generation import GenerationPipeline
    from .inputs import ImageProcessor

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})


@dataclass(frozen=True, slots=True)
class ModuleExecution:
    """Exact numerical signature and caller-owned storage for one capturable call.

    Calls sharing a residency key are serialized and replace its shape together.
    Mutated tensors are restored after compilation warmup before the first replay.
    Coordination groups span every rank participating in this numerical call.
    """

    signature: Hashable
    residency: Hashable
    variant: Hashable
    operation: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]
    mutated: tuple[torch.Tensor, ...] = ()
    groups: tuple[Communicator, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorOutputLayout:
    """Logical result shape and the unique region produced by this rank.

    A missing shape uses the declared capacity. A missing region produces the
    complete tensor. Storage reservation and publication remain runtime-owned.
    """

    shape: tuple[int, ...] | None = None
    region: TensorRegion | None = None


class PositionLayout(StrEnum):
    """Selects temporal-only or temporal-spatial position coordinates."""

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Defines local KV storage and its head interval in the logical model cache."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None

    def __post_init__(self) -> None:
        """Validate physical dimensions, global head coverage, and numeric format."""

        for name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, name)) < 1:
                raise invalid_descriptor(f"cache geometry {name} must be positive")
        if self.kv_head_offset < 0 or self.kv_head_offset + self.num_kv_heads > self.total_kv_heads:
            raise invalid_descriptor("cache head interval exceeds logical model geometry")
        if self.dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor("cache compute dtype is unsupported")
        if self.store_dtype is not None and self.store_dtype not in _KV_DTYPES:
            raise invalid_descriptor("cache storage dtype is unsupported")


@dataclass(frozen=True, slots=True)
class ResourceGeometry:
    """Defines request, cache, latent, feature, and persistent-buffer capacity exposed by a model."""

    kv: bool = True
    encoder_cache_entries: int = 0
    latent_downsample: int | None = None
    request_tensors: Mapping[str, TensorSchema] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate non-negative request, cache, latent, feature, and byte capacities."""

        object.__setattr__(self, "request_tensors", MappingProxyType(dict(self.request_tensors)))
        if self.encoder_cache_entries < 0:
            raise invalid_descriptor("encoder cache capacity must not be negative")
        if self.latent_downsample is not None and self.latent_downsample < 1:
            raise invalid_descriptor("latent downsample must be positive")

    def classes(self) -> tuple[str, ...]:
        """List the scheduler resource classes required by this model geometry."""

        result: list[str] = []
        if self.kv:
            result.append("kv_block")
        if self.encoder_cache_entries:
            result.append("encoder_output")
        if self.latent_downsample is not None:
            result.append("image_latent")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ModuleWarmup:
    """Numerical inputs for one loaded module and optional reusable shape metadata.

    A geometry key must identify the supplied immutable metadata uniquely. The
    public runner retains that metadata for subsequent request execution.
    """

    name: str
    inputs: tuple[object, ...]
    geometry: tuple[Hashable, object] | None = None


class ExecutionModel(nn.Module):
    """Common required geometry for concrete imperative model implementations."""

    architecture: str
    serving_dtype: str = "bfloat16"
    cache_geometry: CacheGeometry
    resource_geometry: ResourceGeometry
    supported_work: frozenset[OpCode]
    vocab_size: int
    hidden_size: int
    text_max_tokens: int
    max_vit_grid_tokens: int = 0
    text_topology: tuple[str, ...] = ("tp",)
    tensorized_mixed: bool = False
    image_processor: ImageProcessor | None = None
    generation: GenerationPipeline | None = None
    media_profile: str | None = None
    scratch_schema: Mapping[str, TensorSchema] = MappingProxyType({})
    context_geometry: AttentionContextGeometry | None = None
    # Loaded submodules with fixed tensor arguments. The public runner owns
    # their input storage, capture streams, executables, and output lifetime.
    capture_inputs: Mapping[str, tuple[TensorSchema, ...]] = MappingProxyType({})
    capture_entries: frozenset[str] = frozenset()

    def module_execution(self, name: str, inputs: tuple[object, ...]) -> ModuleExecution | None:
        """Bind a declared numerical call to stable storage and its exact signature."""

        return None

    # Result declarations contain numerical geometry only. Request identities,
    # allocation, physical locations and reader lifetimes belong to the runtime.
    entry_outputs: Mapping[str, tuple[TensorSpec, ...]] = MappingProxyType({})
    supports_weight_updates: bool = True
    ordered_collective_execution: bool = False
    warmup_inputs: (
        Callable[
            [
                tuple[BoundedTensorStorage, ...],
                BoundedTensorStorage | None,
                AttentionContextWorkspace | None,
                DiffusionSchedule | None,
            ],
            Iterable[ModuleWarmup],
        ]
        | None
    ) = None

    def output_layout(
        self,
        entry: str,
        output_index: int,
        media: MediaGeometry | None,
        decode: DecodeRange | None,
    ) -> TensorOutputLayout | None:
        """Describe numerical result geometry; return None when this rank has no result."""

        return TensorOutputLayout()

    @property
    def product_storage_bytes(self) -> int:
        """Per-request persistent storage for the entry's declared Tensor results."""

        return sum(
            ((output.max_bytes + 255) // 256) * 256
            for outputs in self.entry_outputs.values()
            for output in outputs
        )

    @classmethod
    def build_checkpoint(
        cls, config: dict[str, Any], context: ModelBuildContext
    ) -> ModelConstruction:
        """Declare checkpoint construction and component ownership for this architecture."""

        raise NotImplementedError(f"{cls.__name__} does not declare checkpoint construction")

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare resident tensor ownership for architectures supporting weight updates."""

        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint updates")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """Produce packed hidden rows for the supplied token ids and positions."""

        raise NotImplementedError("model does not implement packed forward execution")

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        """Project hidden rows into the output tensors requested by the batch."""

        raise NotImplementedError("model does not implement output projection")

    def encode(
        self,
        pixels: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        """Encode preprocessed image tensors into model-visible feature rows."""

        raise NotImplementedError("model does not implement vision encoding")

    def encoder_latent(
        self,
        pixels: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        """Encode image tensors into latent payloads for cache or transfer."""

        raise NotImplementedError("model does not implement latent encoding")

    def decode_latent(
        self,
        latents: tuple[torch.Tensor, ...],
        batch: ForwardBatch,
    ) -> ForwardOutput:
        """Decode latent payloads into model-specific media outputs."""

        raise NotImplementedError("model does not implement latent decoding")

    def bind_cache_pool(
        self,
        cache_pool: CachePool,
        selection: AttentionSelection,
    ) -> None:
        """Bind every model-owned attention layer to its process KV allocation."""

        from ..nn.attention import bind_attention_modules

        bind_attention_modules(self, cache_pool, selection)


def active_latent_capacity_tokens(
    per_image_tokens: int, concurrency_token_budget: int | None
) -> int:
    """Reserve at least one image worth of latent tokens within a concurrency budget."""

    per_image = max(0, int(per_image_tokens))
    if per_image == 0:
        return 0
    if concurrency_token_budget is None:
        return per_image
    return max(per_image, int(concurrency_token_budget))


__all__ = [
    "CacheGeometry",
    "ExecutionModel",
    "PositionLayout",
    "ResourceGeometry",
    "WorkerConfig",
    "active_latent_capacity_tokens",
]
