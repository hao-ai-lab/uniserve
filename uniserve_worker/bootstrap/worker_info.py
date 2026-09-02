"""Worker startup information shared with the scheduler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from ..execution.batch import ForwardMode, SamplingOwnership
from ..foundation.errors import invalid_descriptor
from .role import WorkerRole


class RequestKind(StrEnum):
    GET_INFO = "get_info"
    EXECUTE = "execute"
    POLL_COMPLETIONS = "poll_completions"
    DROP_SESSION = "drop_session"
    SHUTDOWN = "shutdown"
    RELEASE_PRODUCTS = "release_products"
    GET_PRESSURE = "get_pressure"


class ResponseKind(StrEnum):
    INFO = "info"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"
    PRESSURE = "pressure"


class ResourceClass(StrEnum):
    KV_BLOCK = "kv_block"
    ENCODER_OUTPUT = "encoder_output"
    IMAGE_LATENT = "image_latent"


class KvGroupKind(StrEnum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True, slots=True)
class KvGroup:
    num_blocks: int
    kind: KvGroupKind
    window: int
    sink: int

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KvGroup:
        data = _map(value, where)
        kind_data = _map(data.get("kind"), f"{where}.kind")
        return cls(
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            kind=_enum(KvGroupKind, kind_data.get("kind"), f"{where}.kind.kind"),
            window=_uint(kind_data.get("window", 0), f"{where}.kind.window"),
            sink=_uint(kind_data.get("sink", 0), f"{where}.kind.sink"),
        )

    def to_mapping(self) -> dict[str, object]:
        kind: dict[str, object] = {"kind": self.kind.value}
        if self.kind is KvGroupKind.SLIDING_WINDOW:
            kind.update(window=self.window, sink=self.sink)
        return {
            "num_blocks": self.num_blocks,
            "kind": kind,
        }


@dataclass(frozen=True, slots=True)
class RankInfo:
    tp_rank: int = 0
    tp_size: int = 1

    def __post_init__(self) -> None:
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise invalid_descriptor("rank.tp must satisfy 0 <= rank < size")

    @classmethod
    def from_mapping(cls, value: object, where: str = "rank") -> RankInfo:
        data = _map(value, where)
        return cls(
            tp_rank=_uint(data.get("tp_rank", 0), f"{where}.tp_rank"),
            tp_size=_uint(data.get("tp_size", 1), f"{where}.tp_size"),
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
        }


@dataclass(frozen=True, slots=True)
class GraphBucket:
    decode_rows: int
    flow_rows: int
    height: int
    width: int
    cfg_branches: int

    def __post_init__(self) -> None:
        if (
            min(
                self.decode_rows,
                self.flow_rows,
                self.height,
                self.width,
                self.cfg_branches,
            )
            < 1
        ):
            raise invalid_descriptor("mixed execution bucket dimensions must be positive")

    @classmethod
    def from_mapping(cls, value: object, where: str) -> GraphBucket:
        data = _map(value, where)
        return cls(
            decode_rows=_uint(data.get("decode_rows"), f"{where}.decode_rows"),
            flow_rows=_uint(data.get("flow_rows"), f"{where}.flow_rows"),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            cfg_branches=_uint(data.get("cfg_branches"), f"{where}.cfg_branches"),
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "decode_rows": self.decode_rows,
            "flow_rows": self.flow_rows,
            "height": self.height,
            "width": self.width,
            "cfg_branches": self.cfg_branches,
        }


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    worker_role: WorkerRole
    block_size: int
    num_blocks: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    supported_work: tuple[ForwardMode, ...]
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
    groups: tuple[KvGroup, ...]
    kv_dtype: str
    model_dtype: str
    rank: RankInfo
    pipeline_depth: int
    encoder_cache_budget: int
    supported_controls: tuple[RequestKind, ...]
    max_batch_operations: int
    max_batch_tokens: int
    max_request_pool_size: int
    max_unresolved_window: int
    incremental_kv_publication: bool
    mixed_buckets: tuple[GraphBucket, ...]
    sampling_ownership: SamplingOwnership
    resource_classes: tuple[ResourceClass, ...]
    model_name: str
    weight_version: int

    @property
    def latent_capacity_units(self) -> int:
        return max(0, self.num_latent_pages - 1) * self.latent_page_units

    @property
    def uses_kv(self) -> bool:
        return (
            any(
                variant
                in {
                    ForwardMode.TOKEN_EXTEND,
                    ForwardMode.TOKEN_DECODE,
                    ForwardMode.TOKEN_VERIFY,
                    ForwardMode.TRANSFER_KV_PUBLISH,
                    ForwardMode.TRANSFER_KV_INSTALL,
                }
                for variant in self.supported_work
            )
            or ResourceClass.KV_BLOCK in self.resource_classes
        )

    def __post_init__(self) -> None:
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise invalid_descriptor("model_dtype must use the canonical runtime vocabulary")
        for name in (
            "latent_downsample",
            "pipeline_depth",
            "gen_rope_advance",
            "max_cfg_branches",
            "max_batch_operations",
            "max_batch_tokens",
            "max_request_pool_size",
            "max_unresolved_window",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"worker info.{name} must be positive")
        kv_values = (
            self.block_size,
            self.num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.bytes_per_token,
        )
        if self.uses_kv:
            if any(value < 1 for value in kv_values) or not self.groups or not self.kv_dtype:
                raise invalid_descriptor("worker info declares incomplete KV geometry")
            total_blocks = 0
            for group in self.groups:
                if group.num_blocks < 1:
                    raise invalid_descriptor(
                        "worker info.groups must be a canonical physical page partition"
                    )
                total_blocks += group.num_blocks
            if total_blocks != self.num_blocks:
                raise invalid_descriptor(
                    "worker info.groups must cover the physical request page pool"
                )
        elif any(kv_values) or self.groups or self.kv_dtype:
            raise invalid_descriptor("KV-free worker info must carry zero KV geometry")
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
                raise invalid_descriptor(f"worker info.{name} must not be negative")
        if not self.supported_work:
            raise invalid_descriptor("worker info must support a work variant")
        if len(set(self.supported_work)) != len(self.supported_work):
            raise invalid_descriptor("worker info repeats a work variant")
        if len(set(self.supported_controls)) != len(self.supported_controls):
            raise invalid_descriptor("worker info repeats a control")
        if len(set(self.resource_classes)) != len(self.resource_classes):
            raise invalid_descriptor("worker info repeats a resource class")
        if len(set(self.mixed_buckets)) != len(self.mixed_buckets):
            raise invalid_descriptor("worker info repeats a mixed-execution bucket")
        if any(
            bucket.decode_rows + bucket.flow_rows > self.max_batch_operations
            for bucket in self.mixed_buckets
        ):
            raise invalid_descriptor("mixed-execution bucket exceeds the operation bound")
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
                    "worker worker info declares incomplete latent pool geometry"
                )
        addresses_latent = any(
            variant
            in {
                ForwardMode.MEDIA_PREPARE,
                ForwardMode.MEDIA_DENOISE,
            }
            for variant in self.supported_work
        )
        if addresses_latent and ResourceClass.IMAGE_LATENT not in self.resource_classes:
            raise invalid_descriptor("worker info advertise latent work without a latent page pool")
        if not self.model_name or self.weight_version < 0:
            raise invalid_descriptor("worker model name and weight version are invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "info") -> WorkerInfo:
        data = _map(value, where)
        return cls(
            worker_role=_enum(WorkerRole, data.get("worker_role"), f"{where}.worker_role"),
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            num_kv_heads=_uint(data.get("num_kv_heads"), f"{where}.num_kv_heads"),
            head_dim=_uint(data.get("head_dim"), f"{where}.head_dim"),
            supported_work=tuple(
                _enum(ForwardMode, item, f"{where}.supported_work[{index}]")
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
                KvGroup.from_mapping(item, f"{where}.groups[{index}]")
                for index, item in enumerate(_seq(data.get("groups", ()), f"{where}.groups"))
            ),
            kv_dtype=_str(data.get("kv_dtype"), f"{where}.kv_dtype"),
            model_dtype=_str(data.get("model_dtype"), f"{where}.model_dtype"),
            rank=RankInfo.from_mapping(data.get("rank"), f"{where}.rank"),
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
                GraphBucket.from_mapping(item, f"{where}.mixed_buckets[{index}]")
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
            model_name=_str(data.get("model_name", ""), f"{where}.model_name"),
            weight_version=_uint(data.get("weight_version", 0), f"{where}.weight_version"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "worker_role": self.worker_role.value,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "supported_work": [value.value for value in self.supported_work],
            "latent_page_units": self.latent_page_units,
            "num_latent_pages": self.num_latent_pages,
            "latent_width": self.latent_width,
            "latent_dtype": self.latent_dtype or None,
            "latent_downsample": self.latent_downsample,
            "max_vae_grid_tokens": self.max_vae_grid_tokens,
            "max_vit_grid_tokens": self.max_vit_grid_tokens,
            "max_latent_feature_bytes": self.max_latent_feature_bytes,
            "max_vision_feature_bytes": self.max_vision_feature_bytes,
            "commit_marker_tokens": self.commit_marker_tokens,
            "gen_rope_advance": self.gen_rope_advance,
            "max_cfg_branches": self.max_cfg_branches,
            "bytes_per_token": self.bytes_per_token,
            "groups": [value.to_mapping() for value in self.groups],
            "kv_dtype": self.kv_dtype or None,
            "model_dtype": self.model_dtype,
            "rank": self.rank.to_mapping(),
            "pipeline_depth": self.pipeline_depth,
            "encoder_cache_budget": self.encoder_cache_budget,
            "supported_controls": [value.value for value in self.supported_controls],
            "max_batch_operations": self.max_batch_operations,
            "max_batch_tokens": self.max_batch_tokens,
            "max_request_pool_size": self.max_request_pool_size,
            "max_unresolved_window": self.max_unresolved_window,
            "incremental_kv_publication": self.incremental_kv_publication,
            "mixed_buckets": [bucket.to_mapping() for bucket in self.mixed_buckets],
            "sampling_ownership": self.sampling_ownership.value,
            "resource_classes": [value.value for value in self.resource_classes],
            "model_name": self.model_name,
            "weight_version": self.weight_version,
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


from .worker_info_builder import build_worker_info


__all__ = [
    "GraphBucket",
    "KvGroup",
    "KvGroupKind",
    "RankInfo",
    "RequestKind",
    "ResourceClass",
    "ResponseKind",
    "WorkerInfo",
    "build_worker_info",
]
