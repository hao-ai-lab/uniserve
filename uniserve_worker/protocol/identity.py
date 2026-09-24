"""Stable identities shared by worker protocol messages.

`RequestKey` names one admitted request epoch, `CallId` one call of it, and
`BufferId` one versioned output of that call. Workers use these records as
dictionary keys throughout execution and storage. Each memoizes its hash in a
`_hash_value` slot declared with ``init=False`` and ``compare=False``, which
keeps it out of the constructor, equality, and ordering; `to_mapping` does not
emit it.

The PyO3 transport (`crates/worker-ipc-py`) constructs these records
positionally, so their field order is part of that contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.validation import _map, _nonnegative, _uint


@dataclass(frozen=True, slots=True, order=True)
class CallId:
    """Logical call identity: engine batch id and selection ordinal.

    The request index is the ordinal the engine assigns at selection, before
    physical row packing, so the identity is preserved when a scheduler batch
    is split across workers. `CallId(0, 0)` names a request's admission root,
    which no real call uses because real calls have a positive batch id.
    Ordering compares ``(batch_id, request_index)``; `RequestPool` relies on
    it to keep an out-of-order result from rolling request progress back.
    """

    batch_id: int
    request_index: int

    _hash_value: int | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the uint64 batch id and uint32 request ordinal.

        Batch id zero is reserved for the admission root, so it requires
        request index zero.
        """
        if not 0 <= self.batch_id <= 0xFFFFFFFFFFFFFFFF:
            raise invalid_descriptor("computation batch id is outside uint64")
        if not 0 <= self.request_index <= 0xFFFFFFFF:
            raise invalid_descriptor(
                "computation request index is outside uint32"
            )
        if self.batch_id == 0 and self.request_index != 0:
            raise invalid_descriptor(
                "admission identity requires request index zero"
            )

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""
        value = self._hash_value
        if value is None:
            value = hash((self.batch_id, self.request_index))
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "computation_id"
    ) -> CallId:
        """Parse a batch and request-ordinal pair.

        An existing `CallId` instance is returned unchanged.
        """
        if isinstance(value, cls):
            return value
        data = _map(value, where)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            request_index=_uint(
                data.get("request_index"), f"{where}.request_index"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the computation identity for IPC."""
        return {"batch_id": self.batch_id, "request_index": self.request_index}


@dataclass(frozen=True, slots=True)
class RequestKey:
    """Identifies one request epoch within an engine instance.

    The engine stamps each admission with the next value of an epoch counter,
    so a reused request id yields a distinct key and no call or product
    reference aliases across requests or epochs.
    """

    engine_id: int
    request_id: int
    request_epoch: int

    _hash_value: int | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Require a non-negative engine id, request id, and epoch."""
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
    def from_mapping(
        cls, value: object, where: str = "request_key"
    ) -> RequestKey:
        """Parse an engine instance id, request id, and admission epoch."""
        data = _map(value, where)
        return cls(
            engine_id=_uint(data.get("engine_id"), f"{where}.engine_id"),
            request_id=_uint(data.get("request_id"), f"{where}.request_id"),
            request_epoch=_uint(
                data.get("request_epoch"), f"{where}.request_epoch"
            ),
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
    """Identifies a versioned call output buffer and its owning request.

    A `TensorRef` projects to one through `TensorRef.buffer_id`; KV
    endpoints (`Call.kv_input`, `Call.kv_output`, `KvTransfer.source`) carry
    one directly. `BufferAllocation` and the `Free` batch command name their
    buffer by this identity.
    """

    owner: RequestKey
    producer_call_id: CallId
    output_index: int
    generation: int

    _hash_value: int | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Require a positive allocation generation."""
        if self.generation < 1:
            raise invalid_descriptor("buffer id has no logical generation")

    def __hash__(self) -> int:
        """Reuse the hash of these immutable integer identity coordinates."""
        value = self._hash_value
        if value is None:
            value = hash(
                (
                    self.owner,
                    self.producer_call_id,
                    self.output_index,
                    self.generation,
                )
            )
            object.__setattr__(self, "_hash_value", value)
        return value

    @classmethod
    def from_mapping(cls, value: object, where: str = "buffer_id") -> BufferId:
        """Parse a versioned persistent-buffer identity from the wire schema."""
        data = _map(value, where)
        return cls(
            owner=RequestKey.from_mapping(data.get("owner"), f"{where}.owner"),
            producer_call_id=CallId.from_mapping(
                data.get("producer_call_id"), f"{where}.producer_call_id"
            ),
            output_index=_uint(
                data.get("output_index"), f"{where}.output_index"
            ),
            generation=_uint(data.get("generation"), f"{where}.generation"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize buffer ownership and generation into the wire mapping."""
        return {
            "owner": self.owner.to_mapping(),
            "producer_call_id": self.producer_call_id.to_mapping(),
            "output_index": self.output_index,
            "generation": self.generation,
        }


# Full identity of one call: its request epoch and its call id.
CallIdentity: TypeAlias = tuple[RequestKey, CallId]
