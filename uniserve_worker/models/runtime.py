"""Imperative model-runner boundary and worker-local runtime geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from uniserve_worker.config import WorkerConfig
from uniserve_worker.protocol.batch import Computation, PipelineStage

from ..execution.forward_batch import AttentionSelection, ForwardBatch, ForwardOutput
from ..execution.model_entry import ModelEntry
from ..foundation.errors import invalid_descriptor
from ..nn.parallel_attention import AttentionContextGeometry
from ..protocol.batch import DecodeRange, DiffusionSamplingParams, TensorSpec
from ..runtime.tensor_buffers import TensorBuffers, TensorSchema
from ..transfer.layout import TensorRegion

if TYPE_CHECKING:
    from ..execution.model_runner import ModelRunner
    from ..loader.component import CheckpointComponent, ModelBuildContext, ModelConstruction
    from ..runtime.kv_cache import KVCache
    from .generation import GenerationPipeline
    from .inputs import ImageProcessor

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})


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
    """Defines local KV storage and its layer/head region in the logical cache."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None
    total_layers: int | None = None
    layer_offset: int = 0

    def __post_init__(self) -> None:
        """Validate physical dimensions, global head coverage, and numeric format."""

        for name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, name)) < 1:
                raise invalid_descriptor(f"cache geometry {name} must be positive")
        total_layers = self.num_layers if self.total_layers is None else self.total_layers
        object.__setattr__(self, "total_layers", total_layers)
        if self.layer_offset < 0 or self.layer_offset + self.num_layers > total_layers:
            raise invalid_descriptor("cache layer interval exceeds logical model geometry")
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


class ExecutionModel(nn.Module):
    """Common required geometry for concrete imperative model implementations."""

    architecture: str
    serving_dtype: str = "bfloat16"
    cache_geometry: CacheGeometry
    resource_geometry: ResourceGeometry
    supported_work: frozenset[Computation]
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
    # Result declarations contain numerical geometry only. Request identities,
    # allocation, physical locations and reader lifetimes belong to the runtime.
    entry_outputs: Mapping[str, tuple[TensorSpec, ...]] = MappingProxyType({})
    bindings: Mapping[str, ModelEntry] = MappingProxyType({})
    pipeline_components: Mapping[PipelineStage, str] = MappingProxyType({})
    num_inference_steps: int = 0
    ordered_collective_execution: bool = False

    def bind_execution(self, runner: ModelRunner) -> None:
        """Bind this rank's numerical callables to their execution owners."""

    def warmup_execution(self, runner: ModelRunner, storage: tuple[TensorBuffers, ...]) -> None:
        """Prepare representative numerical inputs after runtime storage exists."""

    def output_layout(
        self,
        entry: str,
        output_index: int,
        media: DiffusionSamplingParams | None,
        decode: DecodeRange | None,
        num_prompt_tokens: int,
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
        """Declare the resident components populated by checkpoint loading."""

        raise NotImplementedError(f"{type(self).__name__} does not declare checkpoint components")

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
        kv_cache: KVCache,
        selection: AttentionSelection,
    ) -> None:
        """Bind every model-owned attention layer to its process KV allocation."""

        from ..nn.attention import bind_attention_modules

        bind_attention_modules(self, kv_cache, selection)


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
