"""Typed worker capability declaration and wire projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from .foundation.errors import invalid_descriptor
from .spec import OperationType


class RequestKind(StrEnum):
    GET_CAPABILITIES = "get_capabilities"
    EXECUTE = "execute"
    DROP_SESSION = "drop_session"
    SHUTDOWN = "shutdown"
    COPY_KV = "copy_kv"
    LOAD_ADAPTER = "load_adapter"
    UNLOAD_ADAPTER = "unload_adapter"
    RELEASE_PRODUCTS = "release_products"
    RESET_PREFIX_CACHE = "reset_prefix_cache"
    GET_METRICS = "get_metrics"
    GET_PRESSURE = "get_pressure"
    SNAPSHOT_SESSION = "snapshot_session"
    RESTORE_SESSION = "restore_session"


class ResponseKind(StrEnum):
    CAPABILITIES = "capabilities"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"
    METRICS = "metrics"
    PRESSURE = "pressure"
    SNAPSHOT = "snapshot"


class AdapterMode(StrEnum):
    NONE = "none"
    ENGINE_WIDE = "engine_wide"
    PER_REQUEST = "per_request"
    MULTI_ADAPTER = "multi_adapter"


class ResourceClass(StrEnum):
    KV_BLOCK = "kv_block"
    ENCODER_OUTPUT = "encoder_output"
    IMAGE_LATENT = "image_latent"
    SCRATCH = "scratch"
    ADAPTER = "adapter"


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
    pp_rank: int = 0
    pp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1

    def __post_init__(self) -> None:
        for axis in ("tp", "pp", "dp"):
            rank = getattr(self, f"{axis}_rank")
            size = getattr(self, f"{axis}_size")
            if size < 1 or not 0 <= rank < size:
                raise invalid_descriptor(f"rank.{axis} must satisfy 0 <= rank < size")

    @classmethod
    def from_wire(cls, value: object, where: str = "rank") -> RankInfo:
        data = _map(value, where)
        return cls(
            tp_rank=_uint(data.get("tp_rank", 0), f"{where}.tp_rank"),
            tp_size=_uint(data.get("tp_size", 1), f"{where}.tp_size"),
            pp_rank=_uint(data.get("pp_rank", 0), f"{where}.pp_rank"),
            pp_size=_uint(data.get("pp_size", 1), f"{where}.pp_size"),
            dp_rank=_uint(data.get("dp_rank", 0), f"{where}.dp_rank"),
            dp_size=_uint(data.get("dp_size", 1), f"{where}.dp_size"),
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
        }


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    max_batch_operations: int

    def __post_init__(self) -> None:
        if self.max_batch_operations < 1:
            raise invalid_descriptor("max_batch_operations must be positive")


@dataclass(frozen=True, slots=True)
class EngineCaps:
    block_size: int
    num_blocks: int
    num_layers: int
    scratch_capacity_tokens: int
    supported_operation_types: tuple[OperationType, ...]
    max_latent_size: int
    latent_downsample: int
    max_vae_grid_tokens: int
    max_vit_grid_tokens: int
    commit_marker_tokens: int
    gen_rope_advance: int
    max_cfg_branches: int
    bytes_per_token: int
    groups: tuple[KvGroupSpec, ...]
    kv_dtype: str
    model_dtype: str
    attention_backend: str
    quantization: str | None
    rank: RankInfo
    pipeline_depth: int
    encoder_cache_budget: int
    supported_controls: tuple[RequestKind, ...]
    adapter_mode: AdapterMode
    execution_constraints: ExecutionConstraints
    resource_classes: tuple[ResourceClass, ...]
    model_spec_digest: str
    weight_digest: str
    restored_sessions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise invalid_descriptor("model_dtype must use the canonical runtime vocabulary")
        for name in (
            "block_size",
            "num_blocks",
            "num_layers",
            "latent_downsample",
            "bytes_per_token",
            "pipeline_depth",
            "commit_marker_tokens",
            "gen_rope_advance",
            "max_cfg_branches",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"capabilities.{name} must be positive")
        for name in (
            "scratch_capacity_tokens",
            "max_latent_size",
            "max_vae_grid_tokens",
            "max_vit_grid_tokens",
            "encoder_cache_budget",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(f"capabilities.{name} must not be negative")
        if not self.supported_operation_types:
            raise invalid_descriptor("capabilities must support an operation type")
        if len(set(self.supported_operation_types)) != len(self.supported_operation_types):
            raise invalid_descriptor("capabilities repeat an operation type")
        if len(set(self.supported_controls)) != len(self.supported_controls):
            raise invalid_descriptor("capabilities repeat a control")
        if len(set(self.resource_classes)) != len(self.resource_classes):
            raise invalid_descriptor("capabilities repeat a resource class")
        if len(set(self.restored_sessions)) != len(self.restored_sessions) or any(
            value < 0 for value in self.restored_sessions
        ):
            raise invalid_descriptor("capabilities restored sessions are invalid")
        if self.adapter_mode is not AdapterMode.NONE and not {
            RequestKind.LOAD_ADAPTER,
            RequestKind.UNLOAD_ADAPTER,
        } <= set(self.supported_controls):
            raise invalid_descriptor("adapter capability requires load and unload controls")

    @classmethod
    def from_wire(cls, value: object, where: str = "capabilities") -> EngineCaps:
        data = _map(value, where)
        return cls(
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            scratch_capacity_tokens=_uint(
                data.get("scratch_capacity_tokens"), f"{where}.scratch_capacity_tokens"
            ),
            supported_operation_types=tuple(
                _enum(OperationType, item, f"{where}.supported_operation_types[{index}]")
                for index, item in enumerate(
                    _seq(
                        data.get("supported_operation_types"),
                        f"{where}.supported_operation_types",
                    )
                )
            ),
            max_latent_size=_uint(data.get("max_latent_size"), f"{where}.max_latent_size"),
            latent_downsample=_uint(
                data.get("latent_downsample"), f"{where}.latent_downsample"
            ),
            max_vae_grid_tokens=_uint(
                data.get("max_vae_grid_tokens"), f"{where}.max_vae_grid_tokens"
            ),
            max_vit_grid_tokens=_uint(
                data.get("max_vit_grid_tokens"), f"{where}.max_vit_grid_tokens"
            ),
            commit_marker_tokens=_uint(
                data.get("commit_marker_tokens"), f"{where}.commit_marker_tokens"
            ),
            gen_rope_advance=_uint(
                data.get("gen_rope_advance"), f"{where}.gen_rope_advance"
            ),
            max_cfg_branches=_uint(
                data.get("max_cfg_branches"), f"{where}.max_cfg_branches"
            ),
            bytes_per_token=_uint(data.get("bytes_per_token"), f"{where}.bytes_per_token"),
            groups=tuple(
                KvGroupSpec.from_wire(item, f"{where}.groups[{index}]")
                for index, item in enumerate(_seq(data.get("groups", ()), f"{where}.groups"))
            ),
            kv_dtype=_str(data.get("kv_dtype"), f"{where}.kv_dtype"),
            model_dtype=_str(data.get("model_dtype"), f"{where}.model_dtype"),
            attention_backend=_str(
                data.get("attention_backend"), f"{where}.attention_backend"
            ),
            quantization=(
                None
                if data.get("quantization") is None
                else _str(data["quantization"], f"{where}.quantization")
            ),
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
            adapter_mode=_enum(AdapterMode, data.get("adapter_mode"), f"{where}.adapter_mode"),
            execution_constraints=ExecutionConstraints(
                _uint(
                    _map(
                        data.get("execution_constraints"), f"{where}.execution_constraints"
                    ).get("max_batch_operations"),
                    f"{where}.execution_constraints.max_batch_operations",
                )
            ),
            resource_classes=tuple(
                _enum(ResourceClass, item, f"{where}.resource_classes[{index}]")
                for index, item in enumerate(
                    _seq(data.get("resource_classes", ()), f"{where}.resource_classes")
                )
            ),
            model_spec_digest=_str(
                data.get("model_spec_digest", ""), f"{where}.model_spec_digest"
            ),
            weight_digest=_str(data.get("weight_digest", ""), f"{where}.weight_digest"),
            restored_sessions=tuple(
                _uint(item, f"{where}.restored_sessions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("restored_sessions", ()), f"{where}.restored_sessions")
                )
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "scratch_capacity_tokens": self.scratch_capacity_tokens,
            "supported_operation_types": [value.value for value in self.supported_operation_types],
            "max_latent_size": self.max_latent_size,
            "latent_downsample": self.latent_downsample,
            "max_vae_grid_tokens": self.max_vae_grid_tokens,
            "max_vit_grid_tokens": self.max_vit_grid_tokens,
            "commit_marker_tokens": self.commit_marker_tokens,
            "gen_rope_advance": self.gen_rope_advance,
            "max_cfg_branches": self.max_cfg_branches,
            "bytes_per_token": self.bytes_per_token,
            "groups": [value.to_wire() for value in self.groups],
            "kv_dtype": self.kv_dtype,
            "model_dtype": self.model_dtype,
            "attention_backend": self.attention_backend,
            "quantization": self.quantization,
            "rank": self.rank.to_wire(),
            "pipeline_depth": self.pipeline_depth,
            "encoder_cache_budget": self.encoder_cache_budget,
            "supported_controls": [value.value for value in self.supported_controls],
            "adapter_mode": self.adapter_mode.value,
            "execution_constraints": {
                "max_batch_operations": self.execution_constraints.max_batch_operations
            },
            "resource_classes": [value.value for value in self.resource_classes],
            "model_spec_digest": self.model_spec_digest,
            "weight_digest": self.weight_digest,
            "restored_sessions": list(self.restored_sessions),
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


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


__all__ = [name for name in globals() if not name.startswith("_")]
