"""Imperative model-runner boundary and worker-local runtime geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..execution.batch import RunKind
from ..execution.forward_batch import AttentionSelection, ForwardBatch, ForwardOutput
from ..foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    from ..runtime.cache_pool import CachePool
    from ..worker.warmup import WarmupContext
    from .generation import GenerationPipeline
    from .inputs import ImageProcessor

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})


class PositionLayout(StrEnum):
    """Selects temporal-only or temporal-spatial position coordinates."""

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Defines KV layer count, head geometry, block size, capacity, and storage dtype."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None

    def __post_init__(self) -> None:
        """Validate positive KV dimensions, capacity, block size, and dtype metadata."""

        for name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, name)) < 1:
                raise invalid_descriptor(f"cache geometry {name} must be positive")
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

    def __post_init__(self) -> None:
        """Validate non-negative request, cache, latent, feature, and byte capacities."""

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
class DedicatedStateGeometry:
    """Fixed request-state geometry for models with a dedicated execution arena."""

    slot_count: int
    persistent_units: int
    max_vae_grid_tokens: int
    rank: int
    size: int

    def __post_init__(self) -> None:
        """Validate slot, persistent-unit, VAE-grid, and rank geometry."""

        if (
            min(
                self.slot_count,
                self.persistent_units,
                self.max_vae_grid_tokens,
                self.size,
            )
            < 1
            or not 0 <= self.rank < self.size
        ):
            raise invalid_descriptor("dedicated model-state geometry is invalid")


@dataclass(frozen=True, slots=True)
class WorkerDeployment:
    """Defines device placement, parallel topology, cache geometry, dtype policy, and batch bounds for a model."""

    device: str
    model_scope: str
    rank: int
    world_size: int
    block_size: int
    kv_token_capacity: int | None
    attention_backend: str | None
    model_dtype: str
    kv_cache_dtype: str | None
    kv_memory_fraction: float
    max_batch_operations: int
    max_batch_tokens: int
    max_request_pool_size: int
    generation_device: str | None
    output_rank: int = 0

    def __post_init__(self) -> None:
        """Validate topology axes, device placement, batch bounds, and dtype policies."""

        if not self.device or not self.model_scope:
            raise invalid_descriptor("worker deployment placement must be named")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise invalid_descriptor("worker deployment process rank is invalid")
        if not 0 <= self.output_rank < self.world_size:
            raise invalid_descriptor("worker deployment output owner is invalid")
        if (
            self.block_size < 1
            or self.max_batch_operations < 1
            or self.max_batch_tokens < 1
            or self.max_request_pool_size < 1
        ):
            raise invalid_descriptor("worker deployment capacities must be positive")
        if not 0 < self.kv_memory_fraction <= 1:
            raise invalid_descriptor("worker deployment KV memory fraction must be in (0, 1]")
        if self.model_dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor("worker model dtype is unsupported")
        if self.kv_cache_dtype is not None and self.kv_cache_dtype not in _KV_DTYPES:
            raise invalid_descriptor("worker KV dtype is unsupported")


class ExecutionModel(nn.Module):
    """Common required geometry for concrete imperative model implementations."""

    architecture: str
    serving_dtype: str = "bfloat16"
    cache_geometry: CacheGeometry
    resource_geometry: ResourceGeometry
    supported_work: frozenset[RunKind]
    vocab_size: int
    hidden_size: int
    text_max_tokens: int
    text_topology: tuple[str, ...] = ("tp",)
    tensorized_mixed: bool = False
    image_processor: ImageProcessor | None = None
    generation: GenerationPipeline | None = None
    media_profile: str | None = None
    supports_weight_updates: bool = True
    ordered_collective_execution: bool = False
    dedicated_state_geometry: DedicatedStateGeometry | None = None

    def warmup(self, context: WarmupContext) -> None:
        """Run model-owned first-use work before request admission."""

    def create_media_runtime(self, unresolved_window: int) -> tuple[object | None, object | None]:
        """Allocate optional model-specific runtime and fixed execution resources."""

        return None, None

    def create_request_state(self) -> object | None:
        """Allocate optional mutable state shared by this model's active requests."""

        return None

    def validate_run(self, runtime, batch) -> None:
        """Validate a model-specific operation batch before execution mutates state."""

        return None

    def run_operation(self, runtime, state) -> bool:
        """Advance one model-specific operation and report whether it completed."""

        return False

    def synchronize_runtime(self) -> None:
        """Wait until model-owned asynchronous work is safe to observe on the host."""

        return None

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
    "DedicatedStateGeometry",
    "ExecutionModel",
    "PositionLayout",
    "ResourceGeometry",
    "WorkerDeployment",
    "active_latent_capacity_tokens",
]
