"""Worker startup information shared with the scheduler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from ..execution.batch import OpKind
from ..foundation.errors import invalid_descriptor


class RequestKind(StrEnum):
    INFO = "info"
    SUBMIT = "submit"
    POLL = "poll"
    CLOSE = "close"


class ResponseKind(StrEnum):
    INFO = "info"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"


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
class KvCacheConfig:
    block_size: int
    num_blocks: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_token: int
    groups: tuple[KvGroup, ...]
    dtype: str

    def __post_init__(self) -> None:
        if min(
            self.block_size,
            self.num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.bytes_per_token,
        ) < 1 or not self.groups or not self.dtype:
            raise invalid_descriptor("worker info declares incomplete KV geometry")
        if any(group.num_blocks < 1 for group in self.groups):
            raise invalid_descriptor("worker info KV groups must be physical page partitions")
        if sum(group.num_blocks for group in self.groups) != self.num_blocks:
            raise invalid_descriptor("worker info KV groups must cover the physical page pool")

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KvCacheConfig:
        data = _map(value, where)
        return cls(
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            num_kv_heads=_uint(data.get("num_kv_heads"), f"{where}.num_kv_heads"),
            head_dim=_uint(data.get("head_dim"), f"{where}.head_dim"),
            bytes_per_token=_uint(data.get("bytes_per_token"), f"{where}.bytes_per_token"),
            groups=tuple(
                KvGroup.from_mapping(item, f"{where}.groups[{index}]")
                for index, item in enumerate(_seq(data.get("groups"), f"{where}.groups"))
            ),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "bytes_per_token": self.bytes_per_token,
            "groups": [group.to_mapping() for group in self.groups],
            "dtype": self.dtype,
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
class WorkerInfo:
    model_name: str
    weight_version: int
    rank: RankInfo
    supported_ops: tuple[OpKind, ...]
    queue_depth: int
    max_batch_ops: int
    max_batch_tokens: int
    request_slots: int
    kv_cache: KvCacheConfig | None
    latent_page_units: int
    latent_pages: int
    buffer_pool_bytes: int
    max_unresolved_ops: int

    @property
    def latent_capacity_units(self) -> int:
        return max(0, self.latent_pages - 1) * self.latent_page_units

    @property
    def uses_kv(self) -> bool:
        return self.kv_cache is not None

    def __post_init__(self) -> None:
        for name in (
            "queue_depth",
            "max_batch_ops",
            "max_batch_tokens",
            "request_slots",
            "max_unresolved_ops",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"worker info.{name} must be positive")
        requires_kv = any(
            variant in {OpKind.AR_EXTEND, OpKind.AR_DECODE, OpKind.AR_VERIFY}
            for variant in self.supported_ops
        )
        if requires_kv and self.kv_cache is None:
            raise invalid_descriptor("worker advertises AR work without a KV cache")
        for name in (
            "latent_page_units",
            "latent_pages",
            "buffer_pool_bytes",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(f"worker info.{name} must not be negative")
        if not self.supported_ops:
            raise invalid_descriptor("worker info must support a work variant")
        if len(set(self.supported_ops)) != len(self.supported_ops):
            raise invalid_descriptor("worker info repeats a work variant")
        has_latent_geometry = bool(self.latent_page_units or self.latent_pages)
        if has_latent_geometry:
            if self.latent_page_units < 1 or self.latent_pages < 2:
                raise invalid_descriptor(
                    "worker info declares incomplete latent pool capacity"
                )
        addresses_latent = any(
            variant
            in {
                OpKind.DIFFUSION_PREPARE,
                OpKind.DIFFUSION_STEP,
            }
            for variant in self.supported_ops
        )
        if addresses_latent and not has_latent_geometry:
            raise invalid_descriptor("worker info advertise latent work without a latent page pool")
        if not self.model_name or self.weight_version < 0:
            raise invalid_descriptor("worker model name and weight version are invalid")

    @classmethod
    def from_mapping(cls, value: object, where: str = "info") -> WorkerInfo:
        data = _map(value, where)
        return cls(
            model_name=_str(data.get("model_name", ""), f"{where}.model_name"),
            weight_version=_uint(data.get("weight_version", 0), f"{where}.weight_version"),
            rank=RankInfo.from_mapping(data.get("rank"), f"{where}.rank"),
            supported_ops=tuple(
                _enum(OpKind, item, f"{where}.supported_ops[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_ops"), f"{where}.supported_ops")
                )
            ),
            queue_depth=_uint(data.get("queue_depth"), f"{where}.queue_depth"),
            max_batch_ops=_uint(data.get("max_batch_ops"), f"{where}.max_batch_ops"),
            max_batch_tokens=_uint(data.get("max_batch_tokens"), f"{where}.max_batch_tokens"),
            request_slots=_uint(data.get("request_slots"), f"{where}.request_slots"),
            kv_cache=(
                None
                if data.get("kv_cache") is None
                else KvCacheConfig.from_mapping(data.get("kv_cache"), f"{where}.kv_cache")
            ),
            latent_page_units=_uint(data.get("latent_page_units"), f"{where}.latent_page_units"),
            latent_pages=_uint(data.get("latent_pages"), f"{where}.latent_pages"),
            buffer_pool_bytes=_uint(
                data.get("buffer_pool_bytes"), f"{where}.buffer_pool_bytes"
            ),
            max_unresolved_ops=_uint(
                data.get("max_unresolved_ops"), f"{where}.max_unresolved_ops"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "weight_version": self.weight_version,
            "rank": self.rank.to_mapping(),
            "supported_ops": [value.value for value in self.supported_ops],
            "queue_depth": self.queue_depth,
            "max_batch_ops": self.max_batch_ops,
            "max_batch_tokens": self.max_batch_tokens,
            "request_slots": self.request_slots,
            "kv_cache": None if self.kv_cache is None else self.kv_cache.to_mapping(),
            "latent_page_units": self.latent_page_units,
            "latent_pages": self.latent_pages,
            "buffer_pool_bytes": self.buffer_pool_bytes,
            "max_unresolved_ops": self.max_unresolved_ops,
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


__all__ = [
    "KvCacheConfig",
    "KvGroup",
    "KvGroupKind",
    "RankInfo",
    "RequestKind",
    "ResponseKind",
    "WorkerInfo",
]
