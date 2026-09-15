"""Stable identities shared by worker protocol messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..foundation.errors import invalid_descriptor
from .validation import _map, _nonnegative, _uint


@dataclass(frozen=True, slots=True, order=True)
class ComputationId:
    """Logical batch and selection ordinal, independent of physical worker packing."""

    batch_id: int
    request_index: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the uint64 batch, uint32 request ordinal, and admission-identity rule."""

        if not 0 <= self.batch_id <= 0xFFFFFFFFFFFFFFFF:
            raise invalid_descriptor("computation batch id is outside uint64")
        if not 0 <= self.request_index <= 0xFFFFFFFF:
            raise invalid_descriptor("computation request index is outside uint32")
        if self.batch_id == 0 and self.request_index != 0:
            raise invalid_descriptor("admission identity requires request index zero")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.batch_id, self.request_index))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "computation_id") -> ComputationId:
        """Parse a batch and request-ordinal pair, passing through live instances."""

        if isinstance(value, cls):
            return value
        data = _map(value, where)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            request_index=_uint(data.get("request_index"), f"{where}.request_index"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the computation identity for IPC."""

        return {"batch_id": self.batch_id, "request_index": self.request_index}


@dataclass(frozen=True, slots=True)
class RequestKey:
    """Identifies one request epoch within an engine instance."""

    engine_id: int
    request_id: int
    request_epoch: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the non-negative request identifier and epoch."""

        _nonnegative(self.engine_id, "request_key.engine_id")
        _nonnegative(self.request_id, "request_key.request_id")
        _nonnegative(self.request_epoch, "request_key.request_epoch")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.engine_id, self.request_id, self.request_epoch))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "request_key") -> RequestKey:
        """Parse and validate an engine instance, request id, and admission epoch."""

        key = _fast_request_key(value)
        if key is not None:
            return key
        data = _map(value, where)
        return cls(
            engine_id=_uint(data.get("engine_id"), f"{where}.engine_id"),
            request_id=_uint(data.get("request_id"), f"{where}.request_id"),
            request_epoch=_uint(data.get("request_epoch"), f"{where}.request_epoch"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete request-generation identity for IPC."""

        return {
            "engine_id": self.engine_id,
            "request_id": self.request_id,
            "request_epoch": self.request_epoch,
        }


@dataclass(frozen=True, slots=True)
class BufferId:
    """Identifies a versioned operation output buffer and its owning request."""

    owner: RequestKey
    producer_op_id: ComputationId
    output_index: int
    generation: int

    _hash_value: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the owning request, operation identifier, and version."""

        if self.generation < 1:
            raise invalid_descriptor("buffer id has no logical generation")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""

        value = self._hash_value
        if value is None:
            value = hash((self.owner, self.producer_op_id, self.output_index, self.generation))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "buffer_id") -> BufferId:
        """Parse a versioned persistent-buffer identity from the wire schema."""

        data = _map(value, where)
        return cls(
            owner=RequestKey.from_mapping(data.get("owner"), f"{where}.owner"),
            producer_op_id=ComputationId.from_mapping(
                data.get("producer_op_id"), f"{where}.producer_op_id"
            ),
            output_index=_uint(data.get("output_index"), f"{where}.output_index"),
            generation=_uint(data.get("generation"), f"{where}.generation"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer ownership and generation fields for IPC."""

        return {
            "owner": self.owner.to_mapping(),
            "producer_op_id": self.producer_op_id.to_mapping(),
            "output_index": self.output_index,
            "generation": self.generation,
        }


@lru_cache(maxsize=8192)
def _interned_request_key(engine_id: int, request_id: int, request_epoch: int) -> RequestKey:
    """Reuse an immutable request key for identical engine, request, and epoch values."""

    # Callers have already validated the coordinates, so __post_init__ is skipped
    # to keep interning a pure allocation on the hot decode path.
    key = object.__new__(RequestKey)
    object.__setattr__(key, "engine_id", engine_id)
    object.__setattr__(key, "request_id", request_id)
    object.__setattr__(key, "request_epoch", request_epoch)
    object.__setattr__(key, "_hash_value", None)
    return key


def _fast_request_key(value: object) -> RequestKey | None:
    """Decode a trusted compact request-key mapping."""

    if type(value) is not dict:
        return None
    engine_id = value.get("engine_id")
    request_id = value.get("request_id")
    request_epoch = value.get("request_epoch")
    if (
        type(engine_id) is int
        and engine_id >= 0
        and type(request_id) is int
        and request_id >= 0
        and type(request_epoch) is int
        and request_epoch >= 0
    ):
        return _interned_request_key(engine_id, request_id, request_epoch)
    return None
