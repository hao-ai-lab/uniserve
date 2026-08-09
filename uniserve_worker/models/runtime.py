"""Imperative model/executor boundary and worker-local runtime geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from torch import nn

from ..batch import WorkVariant
from ..forward import ForwardRow, RouteId
from ..foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    from .generation import GenerationPipeline
    from .inputs import ImageProcessor

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})


class RowKind(StrEnum):
    TOKEN = "token"
    FLOW = "flow"
    ENCODE = "encode"
    DECODE = "decode"


class DeviceRole(StrEnum):
    PRIMARY = "primary"
    GENERATION = "generation"


class PositionLayout(StrEnum):
    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


@dataclass(frozen=True, slots=True)
class LoweredStage:
    """One neural call produced while lowering a registered operation."""

    route: RouteId
    row: RowKind
    publishes_state: bool = False


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, name)) < 1:
                raise invalid_descriptor(f"cache geometry {name} must be positive")
        if self.dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor("cache compute dtype is unsupported")
        if self.store_dtype is not None and self.store_dtype not in _KV_DTYPES:
            raise invalid_descriptor("cache storage dtype is unsupported")


@dataclass(frozen=True, slots=True)
class ScratchGeometry:
    minimum_blocks: int = 1
    fixed_tokens: int = 0
    mirror_kv: bool = False
    latent_copies: int = 0

    def __post_init__(self) -> None:
        if self.minimum_blocks < 1 or self.fixed_tokens < 0 or self.latent_copies < 0:
            raise invalid_descriptor("scratch geometry bounds are invalid")


@dataclass(frozen=True, slots=True)
class ResourceGeometry:
    kv: bool = True
    encoder_cache_entries: int = 0
    latent_downsample: int | None = None
    scratch: ScratchGeometry | None = None

    def __post_init__(self) -> None:
        if self.encoder_cache_entries < 0:
            raise invalid_descriptor("encoder cache capacity must not be negative")
        if self.latent_downsample is not None and self.latent_downsample < 1:
            raise invalid_descriptor("latent downsample must be positive")

    def classes(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.kv:
            result.append("kv_block")
        if self.encoder_cache_entries:
            result.append("encoder_output")
        if self.latent_downsample is not None:
            result.append("image_latent")
        if self.scratch is not None:
            result.append("scratch")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class WorkerDeployment:
    device: str
    model_scope: str
    tp_rank: int
    tp_size: int
    block_size: int
    kv_token_capacity: int | None
    generation_kv_capacity_tokens: int | None
    attention_backend: str | None
    model_dtype: str
    kv_cache_dtype: str | None
    kv_memory_fraction: float
    max_batch_operations: int
    generation_device: str | None

    def __post_init__(self) -> None:
        if not self.device or not self.model_scope:
            raise invalid_descriptor("worker deployment placement must be named")
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise invalid_descriptor("worker deployment TP rank is invalid")
        if self.block_size < 1 or self.max_batch_operations < 1:
            raise invalid_descriptor("worker deployment capacities must be positive")
        if not 0 < self.kv_memory_fraction <= 1:
            raise invalid_descriptor("worker deployment KV memory fraction must be in (0, 1]")
        if self.model_dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor("worker model dtype is unsupported")
        if self.kv_cache_dtype is not None and self.kv_cache_dtype not in _KV_DTYPES:
            raise invalid_descriptor("worker KV dtype is unsupported")


class ExecutionModel(nn.Module):
    """Concrete models implement physical lowering beside their forward method."""

    architecture: str
    serving_dtype: str = "bfloat16"
    cache_geometry: CacheGeometry
    resource_geometry: ResourceGeometry
    supported_work: frozenset[WorkVariant]
    image_processor: ImageProcessor | None = None
    generation: GenerationPipeline | None = None

    def lower(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool = False,
    ) -> tuple[LoweredStage, ...]:
        raise NotImplementedError

    def route_dtype(self, route: RouteId) -> str:
        raise NotImplementedError

    def route_device_role(self, route: RouteId) -> DeviceRole:
        raise NotImplementedError

    def route_topology(self, route: RouteId) -> tuple[str, ...]:
        raise NotImplementedError

    def route_graph_eligible(self, route: RouteId) -> bool:
        raise NotImplementedError

    def route_max_tokens(self, route: RouteId) -> int:
        raise NotImplementedError

    def route_shape_key(self, route: RouteId, row: ForwardRow) -> tuple[int, ...]:
        raise NotImplementedError

    def route_uses_packed_attention(self, route: RouteId) -> bool:
        return False

    def allows_mixed(self, route: RouteId, rows: frozenset[RowKind]) -> bool:
        return False

    def primary_stage(self, variant: WorkVariant) -> LoweredStage:
        primary = tuple(stage for stage in self.lower(variant) if not stage.publishes_state)
        if len(primary) != 1:
            raise invalid_descriptor(
                f"operation {variant.value!r} requires exactly one primary neural stage"
            )
        return primary[0]

    def state_stages(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool,
    ) -> tuple[LoweredStage, ...]:
        return tuple(
            stage
            for stage in self.lower(variant, retain_image=retain_image)
            if stage.publishes_state
        )


def active_latent_capacity_tokens(per_image_tokens: int, concurrency_token_budget: int | None) -> int:
    per_image = max(0, int(per_image_tokens))
    if per_image == 0:
        return 0
    if concurrency_token_budget is None:
        return per_image
    return max(per_image, int(concurrency_token_budget))


__all__ = [
    "CacheGeometry",
    "DeviceRole",
    "ExecutionModel",
    "LoweredStage",
    "PositionLayout",
    "ResourceGeometry",
    "RowKind",
    "ScratchGeometry",
    "WorkerDeployment",
    "active_latent_capacity_tokens",
]
