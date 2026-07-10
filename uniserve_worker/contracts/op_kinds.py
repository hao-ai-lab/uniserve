"""Canonical forward op-kind vocabulary and per-result schemas (torch-free).

This is the single source of truth for the forward op vocabulary: the set of
kinds (``OP_KINDS``), each kind's forward mode (consumed by
``contracts.forward_mode.mode_for_op``), and each kind's per-result schema
(consumed by ``validate_forward_result``) are all derived from ``OP_KIND_TABLE``.
Kept model-/torch-neutral so the load-bearing ``ForwardMode`` enum can depend on
it without dragging in torch. The low-level wire-field validators shared with
``contracts.caps`` live here too.

Per-result validation is declarative: each wire-result type is a list of
``FieldSpec`` and ``validate_against_schema`` derives the required/optional,
type, and lower-bound checks from that data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..foundation.errors import invalid_descriptor
from ..foundation.wire import wire_int

__all__ = [
    "OpKindSpec",
    "PREFILL_UND",
    "DECODE_UND",
    "TARGET_VERIFY_UND",
    "DENOISE_GEN",
    "COMMIT_GEN",
    "COMMIT_WRITEBACK",
    "VAE_ENCODE",
    "VIT_ENCODE",
    "ENCODE_OP_KINDS",
    "OP_KIND_TABLE",
    "OP_KINDS",
    "FieldSpec",
    "wire_mapping",
    "wire_int_field",
    "wire_str",
    "wire_str_list",
    "wire_optional_int",
    "validate_seq_result",
    "validate_against_schema",
]


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return value


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    # Caps/result fields are non-negative by default; delegate the shared
    # int/bool predicate and lower-bound check to the canonical validator.
    return wire_int(value, where, minimum=minimum)


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a bool")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _str_list(value: Any, where: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list of strings")
    out = []
    seen = set()
    for i, item in enumerate(value):
        s = _str(item, f"{where}[{i}]")
        if s not in allowed:
            raise invalid_descriptor(f"{where}[{i}] has unknown value {s!r}")
        if s in seen:
            raise invalid_descriptor(f"{where} contains duplicate value {s!r}")
        seen.add(s)
        out.append(s)
    return tuple(out)


def _optional_int(result: Mapping[str, Any], key: str, where: str, *, minimum: int = 0) -> int | None:
    if key not in result or result[key] is None:
        return None
    return _int(result[key], f"{where}.{key}", minimum=minimum)


def _float(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise invalid_descriptor(f"{where} must be finite")
    return result


def _optional_float(result: Mapping[str, Any], key: str, where: str) -> float | None:
    if key not in result or result[key] is None:
        return None
    return _float(result[key], f"{where}.{key}")


def _top_logprobs(value: Any, where: str, *, minimum: int) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    for j, pair in enumerate(value):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes, bytearray))
            or len(pair) != 3
        ):
            raise invalid_descriptor(f"{where}[{j}] must be [token_id, logprob, rank]")
        _int(pair[0], f"{where}[{j}][0]", minimum=minimum)
        _float(pair[1], f"{where}[{j}][1]")
        _int(pair[2], f"{where}[{j}][2]", minimum=1)


def _prompt_logprobs(value: Any, where: str, *, minimum: int) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    for index, position in enumerate(value):
        _top_logprobs(position, f"{where}[{index}]", minimum=minimum)


def _int_list(value: Any, where: str, *, minimum: int) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    for j, item in enumerate(value):
        _int(item, f"{where}[{j}]", minimum=minimum)


def _image_hw(value: Any, where: str, *, minimum: int) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise invalid_descriptor(f"{where} must be [height, width]")
    _int(value[0], f"{where}[0]", minimum=minimum)
    _int(value[1], f"{where}[1]", minimum=minimum)


# Field-type dispatch: each token maps to a ``(value, where, *, minimum) -> None``
# validator. Scalar tokens ignore ``minimum`` where it does not apply.
_FIELD_VALIDATORS: dict[str, Callable[..., object]] = {
    "int": lambda v, w, *, minimum: _int(v, w, minimum=minimum),
    "float": lambda v, w, *, minimum: _float(v, w),
    "str": lambda v, w, *, minimum: _str(v, w),
    "bool": lambda v, w, *, minimum: _bool(v, w),
    "top_logprobs": _top_logprobs,
    "prompt_logprobs": _prompt_logprobs,
    "int_list": _int_list,
    "image_hw": _image_hw,
}


@dataclass(frozen=True)
class FieldSpec:
    """One wire-result field's declarative validation rule.

    ``name`` is the wire key. ``type`` is a token in ``_FIELD_VALIDATORS``
    naming the per-value validator. ``required`` makes a missing or ``None``
    value an error (otherwise it is skipped). ``minimum`` is the inclusive
    lower bound forwarded to validators that honor it (``int``/``top_logprobs``/
    ``image_hw``); ``None`` means no bound, treated as ``0`` for ``int`` to
    match the non-negative default of the underlying wire validator.
    """

    name: str
    type: str
    required: bool = True
    minimum: int | None = None


def validate_against_schema(
    mapping: Mapping[str, Any], schema: Sequence[FieldSpec], where: str
) -> None:
    """Validate ``mapping`` field-by-field against a list of ``FieldSpec``.

    Raises ``invalid_descriptor`` on a missing required field, a wrong type, or
    a value below a field's ``minimum``. Optional fields that are absent or
    ``None`` are skipped.
    """
    for field in schema:
        present = field.name in mapping and mapping[field.name] is not None
        if not present:
            if field.required:
                raise invalid_descriptor(f"{where}.{field.name} must be a {field.type}")
            continue
        minimum = 0 if field.minimum is None else field.minimum
        _FIELD_VALIDATORS[field.type](
            mapping[field.name], f"{where}.{field.name}", minimum=minimum
        )


_TEXT_RESULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("sampled_token_id", "int", required=False),
    FieldSpec("sampled_token_ids", "int_list", required=False),
    FieldSpec("sampled_logprob", "float", required=False),
    FieldSpec("top_logprobs", "top_logprobs", required=False),
    FieldSpec("prompt_logprobs", "prompt_logprobs", required=False),
)

_DENOISE_RESULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("denoise_done", "bool", required=True),
    FieldSpec("num_steps_done", "int", required=False, minimum=1),
)

_COMMIT_RESULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("image_png_b64", "str", required=False),
    FieldSpec("image_hw", "image_hw", required=False, minimum=1),
    FieldSpec("sampled_token_id", "int", required=False),
)

_ENCODE_RESULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("encoder_handle", "int", required=False),
)

# Sampler-stage result: a sampled token + optional logprobs (same shape as an
# inline decode result when sampling is peeled to a separate worker).
_SAMPLE_RESULT_SCHEMA = _TEXT_RESULT_SCHEMA

# PostProcess-stage result: cumulative frame count + optional encoded shard /
# status string, carried on existing wire fields.
_FRAME_RESULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("num_tokens", "int", required=False),
    FieldSpec("image_png_b64", "str", required=False),
)


def _schema_validator(
    schema: Sequence[FieldSpec],
) -> Callable[[Mapping[str, Any], str], None]:
    def validate(sr: Mapping[str, Any], where: str) -> None:
        validate_against_schema(sr, schema, where)

    return validate


@dataclass(frozen=True)
class OpKindSpec:
    """Canonical per-op-kind facts: wire name, forward mode, result validator.

    ``mode`` is the ``ForwardMode`` *wire value* (a plain string) rather than the
    enum, so this module stays the single, model-/torch-neutral source of truth:
    ``contracts.forward_mode.mode_for_op`` rehydrates the enum from this string,
    which avoids importing ``ForwardMode`` (and its ``torch`` dependency) into the
    contract boundary. ``validate`` receives ``(seq_result_mapping, where)`` and
    raises ``invalid_descriptor`` on a malformed per-kind result. It is derived
    from a declarative ``FieldSpec`` schema via ``validate_against_schema``.
    """

    wire: str
    mode: str
    validate: Callable[[Mapping[str, Any], str], None]


# Image decode is ``commit_gen`` (decode_image), not a standalone forward op.
PREFILL_UND = "prefill_und"
DECODE_UND = "decode_und"
TARGET_VERIFY_UND = "target_verify_und"
DENOISE_GEN = "denoise_gen"
COMMIT_GEN = "commit_gen"
COMMIT_WRITEBACK = "commit_writeback"
VAE_ENCODE = "vae_encode"
VIT_ENCODE = "vit_encode"
ENCODE_OP_KINDS = frozenset({VAE_ENCODE, VIT_ENCODE})

OP_KIND_TABLE: dict[str, OpKindSpec] = {
    spec.wire: spec
    for spec in (
        OpKindSpec(PREFILL_UND, "extend", _schema_validator(_TEXT_RESULT_SCHEMA)),
        OpKindSpec(DECODE_UND, "decode", _schema_validator(_TEXT_RESULT_SCHEMA)),
        OpKindSpec(TARGET_VERIFY_UND, "target_verify", _schema_validator(_TEXT_RESULT_SCHEMA)),
        OpKindSpec(DENOISE_GEN, "denoise", _schema_validator(_DENOISE_RESULT_SCHEMA)),
        OpKindSpec(COMMIT_GEN, "commit", _schema_validator(_COMMIT_RESULT_SCHEMA)),
        OpKindSpec(COMMIT_WRITEBACK, "commit", _schema_validator(_COMMIT_RESULT_SCHEMA)),
        OpKindSpec(VAE_ENCODE, "encode", _schema_validator(_ENCODE_RESULT_SCHEMA)),
        OpKindSpec(VIT_ENCODE, "encode", _schema_validator(_ENCODE_RESULT_SCHEMA)),
        OpKindSpec("sample", "sample", _schema_validator(_SAMPLE_RESULT_SCHEMA)),
        OpKindSpec("encode_frame", "encode_frame", _schema_validator(_FRAME_RESULT_SCHEMA)),
    )
}

OP_KINDS = frozenset(OP_KIND_TABLE)

# Self-consistency: ``OP_KINDS`` is derived from the table, and every spec keys
# itself by its own ``wire`` name, so the set and the table cannot drift.
assert OP_KINDS == set(OP_KIND_TABLE)
assert all(wire == spec.wire for wire, spec in OP_KIND_TABLE.items())


_SEQ_RESULT_TAIL_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("num_tokens", "int", required=False),
    FieldSpec("num_accepted_tokens", "int", required=False),
    FieldSpec("op_id", "int", required=False),
    FieldSpec("logits_handle", "int", required=False),
    FieldSpec("locator", "str", required=False),
)


def _validate_seq_result(result: Any, op: Mapping[str, Any], index: int) -> None:
    where = f"result.per_seq[{index}]"
    sr = _mapping(result, where)
    req_id = _int(sr.get("req_id"), f"{where}.req_id")
    op_req_id = _int(op.get("req_id"), f"batch.ops[{index}].req_id")
    if req_id != op_req_id:
        raise invalid_descriptor(f"{where}.req_id {req_id} does not match op req_id {op_req_id}")

    kind = _str(op.get("kind"), f"batch.ops[{index}].kind")
    spec = OP_KIND_TABLE.get(kind)
    if spec is None:
        raise invalid_descriptor(f"batch.ops[{index}].kind has unknown value {kind!r}")

    spec.validate(sr, where)

    validate_against_schema(sr, _SEQ_RESULT_TAIL_SCHEMA, where)


wire_mapping = _mapping
wire_int_field = _int
wire_str = _str
wire_str_list = _str_list
wire_optional_int = _optional_int
validate_seq_result = _validate_seq_result
