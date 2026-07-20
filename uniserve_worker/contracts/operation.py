"""Versioned operation envelopes shared by scheduler and worker boundaries."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from ..foundation.errors import invalid_descriptor
from ..foundation.wire import wire_int

__all__ = [
    "EXECUTION_PROTOCOL_VERSION",
    "OperationClass",
    "OperationEnvelope",
    "operation_digest",
    "seal_operation",
]

EXECUTION_PROTOCOL_VERSION = 1


class OperationClass(StrEnum):
    SEQUENCE = "sequence"
    FLOW = "flow"
    ENCODE = "encode"
    MATERIALIZE = "materialize"
    TRANSFER = "transfer"


_OPERATION_CLASSES = {
    "prefill_und": OperationClass.SEQUENCE,
    "decode_und": OperationClass.SEQUENCE,
    "target_verify_und": OperationClass.SEQUENCE,
    "sample": OperationClass.SEQUENCE,
    "denoise_gen": OperationClass.FLOW,
    "vae_encode": OperationClass.ENCODE,
    "vit_encode": OperationClass.ENCODE,
    "commit_gen": OperationClass.MATERIALIZE,
    "encode_frame": OperationClass.MATERIALIZE,
    "commit_writeback": OperationClass.TRANSFER,
}
_OP_KIND_CODES = {
    "prefill_und": 0,
    "decode_und": 1,
    "target_verify_und": 2,
    "denoise_gen": 3,
    "commit_gen": 4,
    "commit_writeback": 5,
    "vae_encode": 6,
    "vit_encode": 7,
    "sample": 8,
    "encode_frame": 9,
}
_MODALITY_CODES = {"Und": 0, "und": 0, "Gen": 1, "gen": 1}
_TOKEN_SOURCE_CODES = {"wire": 0, "last_sampled": 1}


class _Digest:
    def __init__(self, protocol_version: int) -> None:
        self._hash = hashlib.sha256()
        self._hash.update(b"uniserve-operation\0")
        self.u16(protocol_version)

    def finish(self) -> str:
        return self._hash.hexdigest()

    def u8(self, value: int) -> None:
        self._hash.update(struct.pack("<B", int(value)))

    def u16(self, value: int) -> None:
        self._hash.update(struct.pack("<H", int(value)))

    def u32(self, value: int) -> None:
        self._hash.update(struct.pack("<I", int(value)))

    def u64(self, value: int) -> None:
        self._hash.update(struct.pack("<Q", int(value)))

    def boolean(self, value: bool) -> None:
        self.u8(1 if value else 0)

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self._hash.update(encoded)

    def u32s(self, values: Iterable[int]) -> None:
        materialized = tuple(int(value) for value in values)
        self.u64(len(materialized))
        for value in materialized:
            self.u32(value)

    def optional(self, value: Any, encode: Any) -> None:
        if value is None:
            self.u8(0)
            return
        self.u8(1)
        encode(value)

    def f32(self, value: Any, where: str) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise invalid_descriptor(f"{where} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise invalid_descriptor(f"{where} must be finite")
        self._hash.update(struct.pack("<f", number))


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return value


def _integer(value: Any, where: str) -> int:
    return wire_int(value, where, minimum=0)


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a bool")
    return value


def _optional_u32s(digest: _Digest, value: Any, where: str) -> None:
    digest.optional(
        value,
        lambda items: digest.u32s(_integer(item, f"{where}[]") for item in _sequence(items, where)),
    )


def operation_digest(operation: Mapping[str, Any], protocol_version: int) -> str:
    """Return the language-independent digest of one typed operation envelope."""
    op = _mapping(operation, "operation")
    digest = _Digest(_integer(protocol_version, "protocol_version"))
    digest.u64(_integer(op.get("req_id"), "operation.req_id"))
    digest.u64(_integer(op.get("epoch"), "operation.epoch"))
    digest.optional(op.get("op_id"), lambda value: digest.u64(_integer(value, "operation.op_id")))
    digest.u64(_integer(op.get("base_version"), "operation.base_version"))
    kind = op.get("kind")
    if kind not in _OP_KIND_CODES:
        raise invalid_descriptor(f"operation.kind has unknown value {kind!r}")
    digest.u8(_OP_KIND_CODES[kind])
    modality = op.get("modality", "Und")
    if modality not in _MODALITY_CODES:
        raise invalid_descriptor(f"operation.modality has unknown value {modality!r}")
    digest.u8(_MODALITY_CODES[modality])
    digest.u32s(
        _integer(value, "operation.new_block_ids[]")
        for value in _sequence(op.get("new_block_ids") or (), "operation.new_block_ids")
    )
    pos_range = _sequence(op.get("pos_range") or (0, 0), "operation.pos_range")
    if len(pos_range) != 2:
        raise invalid_descriptor("operation.pos_range must contain two integers")
    digest.u32(_integer(pos_range[0], "operation.pos_range[0]"))
    digest.u32(_integer(pos_range[1], "operation.pos_range[1]"))
    _optional_u32s(digest, op.get("token_ids"), "operation.token_ids")
    token_source = op.get("token_source", "wire")
    if token_source not in _TOKEN_SOURCE_CODES:
        raise invalid_descriptor(f"operation.token_source has unknown value {token_source!r}")
    digest.u8(_TOKEN_SOURCE_CODES[token_source])
    digest.optional(
        op.get("timestep_idx"),
        lambda value: digest.u16(_integer(value, "operation.timestep_idx")),
    )
    digest.optional(
        op.get("cond_pos"), lambda value: digest.u32(_integer(value, "operation.cond_pos"))
    )

    def encode_cfg(value: Any) -> None:
        cfg = _mapping(value, "operation.cfg")
        digest.u8(_integer(cfg.get("branch_count"), "operation.cfg.branch_count"))
        digest.f32(cfg.get("text_scale"), "operation.cfg.text_scale")
        digest.f32(cfg.get("img_scale"), "operation.cfg.img_scale")
        renorm_type = cfg.get("renorm_type")
        if not isinstance(renorm_type, str):
            raise invalid_descriptor("operation.cfg.renorm_type must be a string")
        digest.string(renorm_type)
        digest.f32(cfg.get("renorm_min"), "operation.cfg.renorm_min")
        interval = _sequence(cfg.get("interval") or (0.0, 0.0), "operation.cfg.interval")
        if len(interval) != 2:
            raise invalid_descriptor("operation.cfg.interval must contain two numbers")
        digest.f32(interval[0], "operation.cfg.interval[0]")
        digest.f32(interval[1], "operation.cfg.interval[1]")

    digest.optional(op.get("cfg"), encode_cfg)
    digest.optional(
        op.get("image_in"), lambda value: digest.u64(_integer(value, "operation.image_in"))
    )

    def optional_string(value: Any, where: str) -> None:
        def encode(item: Any) -> None:
            if not isinstance(item, str):
                raise invalid_descriptor(f"{where} must be a string")
            digest.string(item)

        digest.optional(value, encode)

    optional_string(op.get("image_prompt"), "operation.image_prompt")
    optional_string(op.get("image_b64"), "operation.image_b64")
    digest.u32(_integer(op.get("group_id", 0), "operation.group_id"))
    _optional_u32s(digest, op.get("allowed_tokens"), "operation.allowed_tokens")
    _optional_u32s(digest, op.get("suppress_tokens"), "operation.suppress_tokens")
    _optional_u32s(digest, op.get("recent_tokens"), "operation.recent_tokens")
    digest.optional(
        op.get("mm_hash"), lambda value: digest.u64(_integer(value, "operation.mm_hash"))
    )
    _optional_u32s(digest, op.get("spec_token_ids"), "operation.spec_token_ids")
    digest.optional(
        op.get("denoise_step_count"),
        lambda value: digest.u16(_integer(value, "operation.denoise_step_count")),
    )
    digest.optional(
        op.get("decode_token_count"),
        lambda value: digest.u16(_integer(value, "operation.decode_token_count")),
    )
    _optional_u32s(digest, op.get("decode_stop_token_ids"), "operation.decode_stop_token_ids")
    digest.boolean(
        _boolean(op.get("decode_stop_terminal", False), "operation.decode_stop_terminal")
    )
    digest.boolean(_boolean(op.get("return_all_logits", False), "operation.return_all_logits"))
    digest.optional(
        op.get("logits_handle"),
        lambda value: digest.u64(_integer(value, "operation.logits_handle")),
    )
    optional_string(op.get("locator"), "operation.locator")
    return digest.finish()


def seal_operation(
    operation: Mapping[str, Any],
    *,
    epoch: int,
    op_id: int,
    base_version: int,
    protocol_version: int = EXECUTION_PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Construct the canonical wire envelope for one typed operation payload."""
    sealed = dict(operation)
    sealed["epoch"] = wire_int(epoch, "operation.epoch", minimum=1)
    sealed["op_id"] = wire_int(op_id, "operation.op_id", minimum=1)
    sealed["base_version"] = wire_int(base_version, "operation.base_version", minimum=0)
    sealed["digest"] = operation_digest(sealed, protocol_version)
    return sealed


@dataclass(frozen=True)
class OperationEnvelope(Mapping[str, Any]):
    """Validated identity wrapped around one member of the operation union."""

    session_id: int
    epoch: int
    op_id: int
    base_version: int
    digest: str
    operation: OperationClass
    payload: Mapping[str, Any]

    @classmethod
    def from_wire(
        cls,
        operation: Mapping[str, Any],
        *,
        protocol_version: int,
        index: int,
    ) -> OperationEnvelope:
        where = f"execute batch.ops[{index}]"
        if not isinstance(operation, Mapping):
            raise invalid_descriptor(f"{where} must be a map")
        session_id = wire_int(operation.get("req_id"), f"{where}.req_id", minimum=0)
        epoch = wire_int(operation.get("epoch"), f"{where}.epoch", minimum=1)
        op_id = wire_int(operation.get("op_id"), f"{where}.op_id", minimum=1)
        base_version = wire_int(operation.get("base_version"), f"{where}.base_version", minimum=0)
        digest = operation.get("digest")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise invalid_descriptor(f"{where}.digest must be a lowercase SHA-256 digest")
        operation_class = _OPERATION_CLASSES.get(operation.get("kind"))
        if operation_class is None:
            raise invalid_descriptor(f"{where}.kind is not in the operation union")
        expected = operation_digest(operation, protocol_version)
        if digest != expected:
            raise invalid_descriptor(f"{where}.digest does not match its typed payload")
        return cls(
            session_id=session_id,
            epoch=epoch,
            op_id=op_id,
            base_version=base_version,
            digest=digest,
            operation=operation_class,
            payload=MappingProxyType(dict(operation)),
        )

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)
