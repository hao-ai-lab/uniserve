"""Typed worker capability declaration and wire projection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from .batch import Domain, SamplingOwnership, WorkVariant, protocol_layout_digest
from .foundation.errors import invalid_descriptor

# Two work variants are never admitted onto a configured serving route:
# ``TOKEN_VERIFY`` is speculative acceptance the scheduler drives inside token
# decode, and ``DRAFT`` has no depth-one route. Both are folded out of every
# advertised capability.
_UNCONFIGURED_WORK = frozenset({WorkVariant.TOKEN_VERIFY, WorkVariant.DRAFT})


def configured_work_variants(variants: Iterable[WorkVariant]) -> tuple[WorkVariant, ...]:
    """The admitted work leaves among ``variants``.

    The result excludes the unconfigured work and follows canonical
    ``WorkVariant`` order.
    """

    selected = set(variants)
    return tuple(
        variant
        for variant in WorkVariant
        if variant in selected and variant not in _UNCONFIGURED_WORK
    )


class RequestKind(StrEnum):
    GET_CAPABILITIES = "get_capabilities"
    EXECUTE = "execute"
    POLL_COMPLETIONS = "poll_completions"
    DROP_SESSION = "drop_session"
    SHUTDOWN = "shutdown"
    COPY_KV = "copy_kv"
    RELEASE_PRODUCTS = "release_products"
    GET_PRESSURE = "get_pressure"
    SNAPSHOT_SESSION = "snapshot_session"
    RESTORE_SESSION = "restore_session"


class ResponseKind(StrEnum):
    CAPABILITIES = "capabilities"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"
    PRESSURE = "pressure"
    SNAPSHOT = "snapshot"


class ResourceClass(StrEnum):
    KV_BLOCK = "kv_block"
    ENCODER_OUTPUT = "encoder_output"
    IMAGE_LATENT = "image_latent"


class KvGroupKind(StrEnum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True, slots=True)
class KvGroupSpec:
    group_id: int
    block_offset: int
    num_blocks: int
    kind: KvGroupKind
    window: int
    sink: int

    @classmethod
    def from_wire(cls, value: object, where: str) -> KvGroupSpec:
        data = _map(value, where)
        return cls(
            group_id=_uint(data.get("group_id"), f"{where}.group_id"),
            block_offset=_uint(data.get("block_offset"), f"{where}.block_offset"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            kind=_enum(KvGroupKind, data.get("kind"), f"{where}.kind"),
            window=_uint(data.get("window"), f"{where}.window"),
            sink=_uint(data.get("sink"), f"{where}.sink"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "block_offset": self.block_offset,
            "num_blocks": self.num_blocks,
            "kind": self.kind.value,
            "window": self.window,
            "sink": self.sink,
        }


@dataclass(frozen=True, slots=True)
class RankInfo:
    tp_rank: int = 0
    tp_size: int = 1

    def __post_init__(self) -> None:
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise invalid_descriptor("rank.tp must satisfy 0 <= rank < size")

    @classmethod
    def from_wire(cls, value: object, where: str = "rank") -> RankInfo:
        data = _map(value, where)
        return cls(
            tp_rank=_uint(data.get("tp_rank", 0), f"{where}.tp_rank"),
            tp_size=_uint(data.get("tp_size", 1), f"{where}.tp_size"),
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
        }


@dataclass(frozen=True, slots=True)
class GraphBucketCapability:
    phase: str
    batch_size: int
    token_bucket: int
    attention_form: str
    height: int
    width: int
    cfg_branches: int
    layout: str = ""

    def __post_init__(self) -> None:
        if not self.phase or not self.attention_form:
            raise invalid_descriptor("graph bucket phase and attention form must be non-empty")
        if self.batch_size < 1 or self.token_bucket < 0:
            raise invalid_descriptor("graph bucket batch and token dimensions are invalid")
        if min(self.height, self.width) < 0 or self.cfg_branches < 1:
            raise invalid_descriptor("graph bucket image dimensions are invalid")

    @classmethod
    def from_wire(cls, value: object, where: str) -> GraphBucketCapability:
        data = _map(value, where)
        return cls(
            phase=_str(data.get("phase"), f"{where}.phase"),
            batch_size=_uint(data.get("batch_size"), f"{where}.batch_size"),
            token_bucket=_uint(data.get("token_bucket"), f"{where}.token_bucket"),
            attention_form=_str(data.get("attention_form"), f"{where}.attention_form"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            cfg_branches=_uint(data.get("cfg_branches"), f"{where}.cfg_branches"),
            layout=_str(data.get("layout", ""), f"{where}.layout"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "batch_size": self.batch_size,
            "token_bucket": self.token_bucket,
            "attention_form": self.attention_form,
            "height": self.height,
            "width": self.width,
            "cfg_branches": self.cfg_branches,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class MixedExecutionCapability:
    decode_rows: int
    flow_rows: int
    height: int
    width: int
    cfg_branches: int

    def __post_init__(self) -> None:
        if min(
            self.decode_rows,
            self.flow_rows,
            self.height,
            self.width,
            self.cfg_branches,
        ) < 1:
            raise invalid_descriptor("mixed execution capability dimensions must be positive")

    @classmethod
    def from_wire(cls, value: object, where: str) -> MixedExecutionCapability:
        data = _map(value, where)
        return cls(
            decode_rows=_uint(data.get("decode_rows"), f"{where}.decode_rows"),
            flow_rows=_uint(data.get("flow_rows"), f"{where}.flow_rows"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            cfg_branches=_uint(data.get("cfg_branches"), f"{where}.cfg_branches"),
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "decode_rows": self.decode_rows,
            "flow_rows": self.flow_rows,
            "height": self.height,
            "width": self.width,
            "cfg_branches": self.cfg_branches,
        }


@dataclass(frozen=True, slots=True)
class LaneCapabilities:
    lane_id: str
    domains: tuple[Domain, ...]
    resolved_sm_count: int
    kv_capacity_tokens: int | None
    latent_capacity_units: int | None
    max_batch_operations: int
    max_batch_tokens: int
    max_inflight: int
    graph_buckets: tuple[GraphBucketCapability, ...]
    eager_max_batch_operations: int
    eager_max_batch_tokens: int

    def __post_init__(self) -> None:
        if not self.lane_id or not self.domains or len(set(self.domains)) != len(self.domains):
            raise invalid_descriptor("lane capability identity and domains are invalid")
        for name in (
            "resolved_sm_count",
            "max_batch_operations",
            "max_batch_tokens",
            "max_inflight",
            "eager_max_batch_operations",
            "eager_max_batch_tokens",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"lane capability {name} must be positive")
        for name in ("kv_capacity_tokens", "latent_capacity_units"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise invalid_descriptor(f"lane capability {name} must be positive")
        if len(set(self.graph_buckets)) != len(self.graph_buckets):
            raise invalid_descriptor("lane capability repeats a graph bucket")

    @classmethod
    def from_wire(cls, value: object, where: str) -> LaneCapabilities:
        data = _map(value, where)
        return cls(
            lane_id=_str(data.get("lane_id"), f"{where}.lane_id"),
            domains=tuple(
                _enum(Domain, item, f"{where}.domains[{index}]")
                for index, item in enumerate(_seq(data.get("domains"), f"{where}.domains"))
            ),
            resolved_sm_count=_uint(data.get("resolved_sm_count"), f"{where}.resolved_sm_count"),
            kv_capacity_tokens=_optional_uint(
                data.get("kv_capacity_tokens"), f"{where}.kv_capacity_tokens"
            ),
            latent_capacity_units=_optional_uint(
                data.get("latent_capacity_units"), f"{where}.latent_capacity_units"
            ),
            max_batch_operations=_uint(
                data.get("max_batch_operations"), f"{where}.max_batch_operations"
            ),
            max_batch_tokens=_uint(data.get("max_batch_tokens"), f"{where}.max_batch_tokens"),
            max_inflight=_uint(data.get("max_inflight"), f"{where}.max_inflight"),
            graph_buckets=tuple(
                GraphBucketCapability.from_wire(item, f"{where}.graph_buckets[{index}]")
                for index, item in enumerate(
                    _seq(data.get("graph_buckets", ()), f"{where}.graph_buckets")
                )
            ),
            eager_max_batch_operations=_uint(
                data.get("eager_max_batch_operations"),
                f"{where}.eager_max_batch_operations",
            ),
            eager_max_batch_tokens=_uint(
                data.get("eager_max_batch_tokens"), f"{where}.eager_max_batch_tokens"
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "domains": [domain.value for domain in self.domains],
            "resolved_sm_count": self.resolved_sm_count,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "latent_capacity_units": self.latent_capacity_units,
            "max_batch_operations": self.max_batch_operations,
            "max_batch_tokens": self.max_batch_tokens,
            "max_inflight": self.max_inflight,
            "graph_buckets": [bucket.to_wire() for bucket in self.graph_buckets],
            "eager_max_batch_operations": self.eager_max_batch_operations,
            "eager_max_batch_tokens": self.eager_max_batch_tokens,
        }


@dataclass(frozen=True, slots=True)
class WorkerCapabilities:
    block_size: int
    num_blocks: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    supported_work: tuple[WorkVariant, ...]
    latent_page_units: int
    num_latent_pages: int
    latent_width: int
    latent_dtype: str
    latent_downsample: int
    max_vae_grid_tokens: int
    max_vit_grid_tokens: int
    max_latent_feature_bytes: int
    max_vision_feature_bytes: int
    commit_marker_tokens: int
    gen_rope_advance: int
    max_cfg_branches: int
    bytes_per_token: int
    groups: tuple[KvGroupSpec, ...]
    kv_dtype: str
    model_dtype: str
    attention_backend: str
    rank: RankInfo
    pipeline_depth: int
    encoder_cache_budget: int
    supported_controls: tuple[RequestKind, ...]
    max_batch_operations: int
    max_batch_tokens: int
    max_request_pool_size: int
    max_unresolved_window: int
    incremental_kv_publication: bool
    mixed_buckets: tuple[MixedExecutionCapability, ...]
    sampling_ownership: SamplingOwnership
    resource_classes: tuple[ResourceClass, ...]
    model_identity: str
    weight_digest: str
    protocol_layout_digest: str = ""
    lanes: tuple[LaneCapabilities, ...] = ()

    @property
    def latent_capacity_units(self) -> int:
        return max(0, self.num_latent_pages - 1) * self.latent_page_units

    def __post_init__(self) -> None:
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise invalid_descriptor("model_dtype must use the canonical runtime vocabulary")
        for name in (
            "block_size",
            "num_blocks",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "latent_downsample",
            "bytes_per_token",
            "pipeline_depth",
            "gen_rope_advance",
            "max_cfg_branches",
            "max_batch_operations",
            "max_batch_tokens",
            "max_request_pool_size",
            "max_unresolved_window",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"capabilities.{name} must be positive")
        if self.groups:
            next_offset = 0
            for index, group in enumerate(self.groups):
                if (
                    group.group_id != index
                    or group.block_offset != next_offset
                    or group.num_blocks < 1
                ):
                    raise invalid_descriptor(
                        "capabilities.groups must be a canonical physical page partition"
                    )
                next_offset += group.num_blocks
            if next_offset != self.num_blocks:
                raise invalid_descriptor(
                    "capabilities.groups must cover the physical request page pool"
                )
        for name in (
            "latent_page_units",
            "num_latent_pages",
            "latent_width",
            "max_vae_grid_tokens",
            "max_vit_grid_tokens",
            "max_latent_feature_bytes",
            "max_vision_feature_bytes",
            "encoder_cache_budget",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(f"capabilities.{name} must not be negative")
        if not self.supported_work:
            raise invalid_descriptor("capabilities must support a work variant")
        if len(set(self.supported_work)) != len(self.supported_work):
            raise invalid_descriptor("capabilities repeat a work variant")
        if len(set(self.supported_controls)) != len(self.supported_controls):
            raise invalid_descriptor("capabilities repeat a control")
        if len(set(self.resource_classes)) != len(self.resource_classes):
            raise invalid_descriptor("capabilities repeat a resource class")
        if len({lane.lane_id for lane in self.lanes}) != len(self.lanes):
            raise invalid_descriptor("capabilities repeat a lane id")
        if len(set(self.mixed_buckets)) != len(self.mixed_buckets):
            raise invalid_descriptor("capabilities repeat a mixed-execution bucket")
        if any(
            bucket.decode_rows + bucket.flow_rows > self.max_batch_operations
            for bucket in self.mixed_buckets
        ):
            raise invalid_descriptor("mixed-execution bucket exceeds the operation bound")
        lane_domains = tuple(domain for lane in self.lanes for domain in lane.domains)
        if len(set(lane_domains)) != len(lane_domains):
            raise invalid_descriptor("capabilities repeat a lane domain binding")
        has_latent_geometry = bool(
            self.latent_page_units
            or self.num_latent_pages
            or self.latent_width
            or self.latent_dtype
        )
        if has_latent_geometry or ResourceClass.IMAGE_LATENT in self.resource_classes:
            if (
                self.latent_page_units < 1
                or self.num_latent_pages < 2
                or self.latent_width < 1
                or self.latent_dtype not in {"float16", "bfloat16", "float32"}
            ):
                raise invalid_descriptor(
                    "worker capabilities declare incomplete latent pool geometry"
                )
        addresses_latent = any(
            variant
            in {
                WorkVariant.GEN_TRANSITION,
                WorkVariant.GEN_FLOW,
            }
            for variant in self.supported_work
        )
        if addresses_latent and ResourceClass.IMAGE_LATENT not in self.resource_classes:
            raise invalid_descriptor(
                "worker capabilities advertise latent work without a latent page pool"
            )
        identities = (self.model_identity, self.weight_digest)
        if any(identities) and any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in identities
        ):
            raise invalid_descriptor(
                "capability model identities must be lowercase SHA-256 digests"
            )
        object.__setattr__(self, "protocol_layout_digest", protocol_layout_digest())

    @classmethod
    def from_wire(cls, value: object, where: str = "capabilities") -> WorkerCapabilities:
        data = _map(value, where)
        return cls(
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            num_kv_heads=_uint(data.get("num_kv_heads"), f"{where}.num_kv_heads"),
            head_dim=_uint(data.get("head_dim"), f"{where}.head_dim"),
            supported_work=tuple(
                _enum(WorkVariant, item, f"{where}.supported_work[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_work"), f"{where}.supported_work")
                )
            ),
            latent_page_units=_uint(data.get("latent_page_units"), f"{where}.latent_page_units"),
            num_latent_pages=_uint(data.get("num_latent_pages"), f"{where}.num_latent_pages"),
            latent_width=_uint(data.get("latent_width"), f"{where}.latent_width"),
            latent_dtype=_str(data.get("latent_dtype", ""), f"{where}.latent_dtype"),
            latent_downsample=_uint(data.get("latent_downsample"), f"{where}.latent_downsample"),
            max_vae_grid_tokens=_uint(
                data.get("max_vae_grid_tokens"), f"{where}.max_vae_grid_tokens"
            ),
            max_vit_grid_tokens=_uint(
                data.get("max_vit_grid_tokens"), f"{where}.max_vit_grid_tokens"
            ),
            max_latent_feature_bytes=_uint(
                data.get("max_latent_feature_bytes"), f"{where}.max_latent_feature_bytes"
            ),
            max_vision_feature_bytes=_uint(
                data.get("max_vision_feature_bytes"), f"{where}.max_vision_feature_bytes"
            ),
            commit_marker_tokens=_uint(
                data.get("commit_marker_tokens"), f"{where}.commit_marker_tokens"
            ),
            gen_rope_advance=_uint(data.get("gen_rope_advance"), f"{where}.gen_rope_advance"),
            max_cfg_branches=_uint(data.get("max_cfg_branches"), f"{where}.max_cfg_branches"),
            bytes_per_token=_uint(data.get("bytes_per_token"), f"{where}.bytes_per_token"),
            groups=tuple(
                KvGroupSpec.from_wire(item, f"{where}.groups[{index}]")
                for index, item in enumerate(_seq(data.get("groups", ()), f"{where}.groups"))
            ),
            kv_dtype=_str(data.get("kv_dtype"), f"{where}.kv_dtype"),
            model_dtype=_str(data.get("model_dtype"), f"{where}.model_dtype"),
            attention_backend=_str(data.get("attention_backend"), f"{where}.attention_backend"),
            rank=RankInfo.from_wire(data.get("rank"), f"{where}.rank"),
            pipeline_depth=_uint(data.get("pipeline_depth"), f"{where}.pipeline_depth"),
            encoder_cache_budget=_uint(
                data.get("encoder_cache_budget"), f"{where}.encoder_cache_budget"
            ),
            supported_controls=tuple(
                _enum(RequestKind, item, f"{where}.supported_controls[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_controls", ()), f"{where}.supported_controls")
                )
            ),
            max_batch_operations=_uint(
                data.get("max_batch_operations"), f"{where}.max_batch_operations"
            ),
            max_batch_tokens=_uint(data.get("max_batch_tokens"), f"{where}.max_batch_tokens"),
            max_request_pool_size=_uint(
                data.get("max_request_pool_size"), f"{where}.max_request_pool_size"
            ),
            max_unresolved_window=_uint(
                data.get("max_unresolved_window"), f"{where}.max_unresolved_window"
            ),
            incremental_kv_publication=_bool(
                data.get("incremental_kv_publication"), f"{where}.incremental_kv_publication"
            ),
            mixed_buckets=tuple(
                MixedExecutionCapability.from_wire(item, f"{where}.mixed_buckets[{index}]")
                for index, item in enumerate(
                    _seq(data.get("mixed_buckets", ()), f"{where}.mixed_buckets")
                )
            ),
            sampling_ownership=_enum(
                SamplingOwnership, data.get("sampling_ownership"), f"{where}.sampling_ownership"
            ),
            resource_classes=tuple(
                _enum(ResourceClass, item, f"{where}.resource_classes[{index}]")
                for index, item in enumerate(
                    _seq(data.get("resource_classes", ()), f"{where}.resource_classes")
                )
            ),
            model_identity=_str(data.get("model_identity", ""), f"{where}.model_identity"),
            weight_digest=_str(data.get("weight_digest", ""), f"{where}.weight_digest"),
            lanes=tuple(
                LaneCapabilities.from_wire(item, f"{where}.lanes[{index}]")
                for index, item in enumerate(_seq(data.get("lanes", ()), f"{where}.lanes"))
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "supported_work": [value.value for value in self.supported_work],
            "latent_page_units": self.latent_page_units,
            "num_latent_pages": self.num_latent_pages,
            "latent_width": self.latent_width,
            "latent_dtype": self.latent_dtype,
            "latent_downsample": self.latent_downsample,
            "max_vae_grid_tokens": self.max_vae_grid_tokens,
            "max_vit_grid_tokens": self.max_vit_grid_tokens,
            "max_latent_feature_bytes": self.max_latent_feature_bytes,
            "max_vision_feature_bytes": self.max_vision_feature_bytes,
            "commit_marker_tokens": self.commit_marker_tokens,
            "gen_rope_advance": self.gen_rope_advance,
            "max_cfg_branches": self.max_cfg_branches,
            "bytes_per_token": self.bytes_per_token,
            "groups": [value.to_wire() for value in self.groups],
            "kv_dtype": self.kv_dtype,
            "model_dtype": self.model_dtype,
            "attention_backend": self.attention_backend,
            "rank": self.rank.to_wire(),
            "pipeline_depth": self.pipeline_depth,
            "encoder_cache_budget": self.encoder_cache_budget,
            "supported_controls": [value.value for value in self.supported_controls],
            "max_batch_operations": self.max_batch_operations,
            "max_batch_tokens": self.max_batch_tokens,
            "max_request_pool_size": self.max_request_pool_size,
            "max_unresolved_window": self.max_unresolved_window,
            "incremental_kv_publication": self.incremental_kv_publication,
            "mixed_buckets": [bucket.to_wire() for bucket in self.mixed_buckets],
            "sampling_ownership": self.sampling_ownership.value,
            "resource_classes": [value.value for value in self.resource_classes],
            "model_identity": self.model_identity,
            "weight_digest": self.weight_digest,
            "protocol_layout_digest": self.protocol_layout_digest,
            "lanes": [lane.to_wire() for lane in self.lanes],
        }


_E = TypeVar("_E", bound=StrEnum)


def _enum(kind: type[_E], value: object, where: str) -> _E:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(f"{where} has unknown value {value!r}") from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _uint(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    return None if value is None else _uint(value, where)


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a boolean")
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


__all__ = [name for name in globals() if not name.startswith("_")]
