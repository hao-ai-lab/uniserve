"""Model-neutral parsed forward batches (data + parsers only).

Scheduler wire arrives as decoded Python dicts. The runner parses them into typed
views once; models consume only those views. Device-tensor staging into the
unified :class:`~uniserve_worker.contracts.forward_batch.ForwardBatch` lives in
``runtime.forward_batch_builder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..foundation.errors import invalid_descriptor
from ..foundation.wire import wire_int as _int
from .forward_mode import ForwardMode
from .op_kinds import ENCODE_OP_KINDS
from .operation import (
    EXECUTION_PROTOCOL_VERSION,
    OperationClass,
    OperationEnvelope,
    seal_operation,
)

__all__ = [
    "Batch",
    "OperationClass",
    "OperationEnvelope",
    "seal_operation",
    "seal_batch",
    "BatchBase",
    "CfgBatch",
    "TextBatch",
    "DenoiseBatch",
    "CommitBatch",
    "EncodeBatch",
    "MixedBatch",
    "ParsedBatch",
    "parse_batch",
]


def seal_batch(
    step_id: int,
    operations: Sequence[Mapping[str, Any]],
    *,
    new_reqs: Sequence[Mapping[str, Any]] = (),
    epoch: int = 1,
    base_version: int = 0,
) -> dict[str, Any]:
    """Construct a versioned batch for an in-memory scheduler boundary."""
    sealed = []
    for index, operation in enumerate(operations):
        payload = dict(operation)
        operation_epoch = int(payload.pop("epoch", epoch))
        operation_id = int(payload.pop("op_id", (int(step_id) << 32) + index + 1))
        operation_base = int(payload.pop("base_version", base_version))
        payload.pop("digest", None)
        sealed.append(
            seal_operation(
                payload,
                epoch=operation_epoch,
                op_id=operation_id,
                base_version=operation_base,
            )
        )
    return {
        "protocol_version": EXECUTION_PROTOCOL_VERSION,
        "step_id": int(step_id),
        "new_reqs": [dict(request) for request in new_reqs],
        "ops": sealed,
    }


@dataclass(frozen=True)
class Batch:
    """One execute() wire payload, parsed and validated once at the boundary.

    The scheduler control plane decodes to a ``Mapping``; this parses it into
    trusted typed fields so the runner hot loop never re-runs ``isinstance``
    checks on the raw wire dict (parse-don't-validate). ``new_reqs``/``ops`` stay
    as mappings (their per-field shape is validated by the request-state table
    and the per-modality op parsers downstream), but the top-level batch shape is
    guaranteed here with diagnostic context.
    """

    step_id: int
    new_reqs: tuple[Mapping[str, Any], ...]
    protocol_version: int
    ops: tuple[OperationEnvelope, ...]

    @classmethod
    def from_wire(cls, batch: Mapping[str, Any]) -> "Batch":
        if not isinstance(batch, Mapping):
            raise invalid_descriptor("execute batch must be a map")
        protocol_version = _int(
            batch.get("protocol_version"),
            "execute batch.protocol_version",
            minimum=1,
        )
        if protocol_version != EXECUTION_PROTOCOL_VERSION:
            raise invalid_descriptor(f"unsupported execution protocol version {protocol_version}")
        step_id = _int(batch.get("step_id"), "execute batch.step_id")
        raw_new = batch.get("new_reqs") or []
        if not isinstance(raw_new, (list, tuple)):
            raise invalid_descriptor("execute batch.new_reqs must be a list")
        new_reqs: list[Mapping[str, Any]] = []
        for nr in raw_new:
            if not isinstance(nr, Mapping):
                raise invalid_descriptor("execute batch.new_reqs entries must be maps")
            _int(nr.get("req_id"), "execute batch.new_reqs[].req_id")
            new_reqs.append(nr)
        raw_ops = batch.get("ops")
        if not isinstance(raw_ops, list):
            raise invalid_descriptor("execute batch.ops must be a list")
        ops = tuple(
            OperationEnvelope.from_wire(
                operation,
                protocol_version=protocol_version,
                index=index,
            )
            for index, operation in enumerate(raw_ops)
        )
        if not ops:
            raise invalid_descriptor("execute batch must contain an operation")
        sessions = {operation.session_id for operation in ops}
        if len(sessions) != len(ops):
            raise invalid_descriptor("execute batch contains multiple operations for one session")
        new_sessions = {int(request["req_id"]) for request in new_reqs}
        if len(new_sessions) != len(new_reqs):
            raise invalid_descriptor("execute batch contains duplicate admission descriptors")
        if not new_sessions.issubset(sessions):
            raise invalid_descriptor("execute batch admits a session without an operation")
        return cls(
            protocol_version=protocol_version,
            step_id=step_id,
            new_reqs=tuple(new_reqs),
            ops=ops,
        )


@dataclass(frozen=True)
class CfgBatch:
    """Per-op CFG descriptor parsed from ``op['cfg']``.

    Production wire carries ``branch_count`` (denoise driver iterates
    ``range(branch_count)``). Optional geometry fields (``seg_offsets``,
    ``branch_positions``, ``branch_kv_offsets``, ``branch_kv_lens``) are parsed
    when present on richer in-process ops; they default to ``None`` on wire.
    """

    branch_count: int
    seg_offsets: Any
    branch_positions: Any
    branch_kv_offsets: Any
    branch_kv_lens: Any


_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


@dataclass(frozen=True)
class BatchBase:
    """Shared CPU-side parsed batch fields."""

    req_ids: tuple[int, ...]
    ops: tuple[Mapping[str, Any], ...]

    @staticmethod
    def req_ids_from_ops(ops: tuple[Mapping[str, Any], ...]) -> tuple[int, ...]:
        return tuple(_int(op.get("req_id"), f"ops[{i}].req_id") for i, op in enumerate(ops))


@dataclass(frozen=True)
class TextBatch(BatchBase):
    """CPU-side text op group parsed from wire ops."""

    mode: ForwardMode
    token_ids: tuple[tuple[int, ...], ...]
    spec_token_ids: tuple[tuple[int, ...], ...]
    pos_ranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_ops(
        cls,
        mode: ForwardMode,
        ops: tuple[Mapping[str, Any], ...],
        *,
        op_modes: tuple[ForwardMode, ...] = (),
        allow_mixed_text: bool = False,
    ) -> "TextBatch":
        if mode not in _TEXT_MODES:
            if not (
                allow_mixed_text
                and mode == ForwardMode.MIXED
                and all(m in _TEXT_MODES for m in op_modes)
            ):
                raise invalid_descriptor(f"batch mode {mode.value} is not text")
        if mode == ForwardMode.MIXED and not allow_mixed_text:
            raise invalid_descriptor(f"batch mode {mode.value} is not text")
        req_ids = []
        token_ids = []
        spec_token_ids = []
        pos_ranges = []
        for i, op in enumerate(ops):
            req_ids.append(_int(op.get("req_id"), f"ops[{i}].req_id"))
            toks = op.get("token_ids") or []
            if not isinstance(toks, Sequence) or isinstance(toks, (str, bytes, bytearray)):
                raise invalid_descriptor(f"ops[{i}].token_ids must be a list")
            token_ids.append(tuple(_int(t, f"ops[{i}].token_ids") for t in toks))
            spec = op.get("spec_token_ids") or []
            if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes, bytearray)):
                raise invalid_descriptor(f"ops[{i}].spec_token_ids must be a list")
            spec_token_ids.append(tuple(_int(t, f"ops[{i}].spec_token_ids") for t in spec))
            pos = op.get("pos_range") or (0, 0)
            if not isinstance(pos, Sequence) or len(pos) != 2:
                raise invalid_descriptor(f"ops[{i}].pos_range must be [start, end]")
            pos_ranges.append(
                (_int(pos[0], f"ops[{i}].pos_range[0]"), _int(pos[1], f"ops[{i}].pos_range[1]"))
            )
        return cls(
            mode=mode,
            req_ids=tuple(req_ids),
            token_ids=tuple(token_ids),
            spec_token_ids=tuple(spec_token_ids),
            pos_ranges=tuple(pos_ranges),
            ops=ops,
        )


@dataclass(frozen=True)
class DenoiseBatch(BatchBase):
    """CPU-side denoise op group parsed from wire ops."""

    timestep_indices: tuple[int, ...]
    cfg_batches: tuple[CfgBatch, ...]

    @classmethod
    def from_ops(cls, ops: tuple[Mapping[str, Any], ...]) -> "DenoiseBatch":
        req_ids = []
        steps = []
        cfg_batches = []
        for i, op in enumerate(ops):
            req_ids.append(_int(op.get("req_id"), f"ops[{i}].req_id"))
            steps.append(_int(op.get("timestep_idx") or 0, f"ops[{i}].timestep_idx"))
            cfg_batches.append(_cfg_batch(op.get("cfg"), f"ops[{i}].cfg"))
        return cls(
            req_ids=tuple(req_ids),
            timestep_indices=tuple(steps),
            cfg_batches=tuple(cfg_batches),
            ops=ops,
        )


@dataclass(frozen=True)
class CommitBatch(BatchBase):
    """CPU-side commit (image decode) op group."""

    @classmethod
    def from_ops(cls, ops: tuple[Mapping[str, Any], ...]) -> "CommitBatch":
        return cls(req_ids=cls.req_ids_from_ops(ops), ops=ops)


@dataclass(frozen=True)
class EncodeBatch(BatchBase):
    """CPU-side encode op group (vit_encode / vae_encode)."""

    kinds: tuple[str, ...]
    image_b64: tuple[str | None, ...]
    mm_hashes: tuple[int | None, ...]

    @classmethod
    def from_ops(cls, ops: tuple[Mapping[str, Any], ...]) -> "EncodeBatch":
        req_ids = []
        kinds = []
        image_b64 = []
        hashes = []
        for i, op in enumerate(ops):
            req_ids.append(_int(op.get("req_id"), f"ops[{i}].req_id"))
            kind = op.get("kind")
            if kind not in ENCODE_OP_KINDS:
                raise invalid_descriptor(f"ops[{i}].kind is not an encode op")
            kinds.append(kind)
            raw = op.get("image_b64")
            image_b64.append(raw if isinstance(raw, str) else None)
            mm_hash = op.get("mm_hash")
            hashes.append(_int(mm_hash, f"ops[{i}].mm_hash") if mm_hash is not None else None)
        return cls(
            req_ids=tuple(req_ids),
            kinds=tuple(kinds),
            image_b64=tuple(image_b64),
            mm_hashes=tuple(hashes),
            ops=ops,
        )


@dataclass(frozen=True)
class MixedBatch(BatchBase):
    """Heterogeneous op group with per-op forward modes."""

    modes: tuple[ForwardMode, ...]
    kinds: tuple[str, ...]

    @classmethod
    def from_ops(
        cls,
        ops: tuple[Mapping[str, Any], ...],
        *,
        op_modes: tuple[ForwardMode, ...],
    ) -> "MixedBatch":
        # Caller guarantees MIXED mode and populated ``op_modes``.
        return cls(
            modes=tuple(op_modes),
            req_ids=cls.req_ids_from_ops(ops),
            kinds=tuple(op["kind"] for op in ops),
            ops=ops,
        )


ParsedBatch = TextBatch | DenoiseBatch | CommitBatch | EncodeBatch | MixedBatch


def parse_batch(
    mode: ForwardMode,
    ops: tuple[Mapping[str, Any], ...],
    *,
    op_modes: tuple[ForwardMode, ...] = (),
    allow_mixed_text: bool = False,
) -> ParsedBatch:
    """Parse one single-mode op group into its typed view by ``ForwardMode``.

    The match is exhaustive over :class:`ForwardMode`; modes that have no parsed
    batch dataclass (peeled-stage ``SAMPLE``/``ENCODE_FRAME`` consume raw ops via
    their own drivers) and the ``case _`` guard raise ``invalid_descriptor``.
    """
    match mode:
        case ForwardMode.EXTEND | ForwardMode.DECODE | ForwardMode.VERIFY_DRAFT:
            return TextBatch.from_ops(
                mode, ops, op_modes=op_modes, allow_mixed_text=allow_mixed_text
            )
        case ForwardMode.DENOISE:
            return DenoiseBatch.from_ops(ops)
        case ForwardMode.COMMIT:
            return CommitBatch.from_ops(ops)
        case ForwardMode.ENCODE:
            return EncodeBatch.from_ops(ops)
        case ForwardMode.MIXED:
            if allow_mixed_text and all(m in _TEXT_MODES for m in op_modes):
                return TextBatch.from_ops(mode, ops, op_modes=op_modes, allow_mixed_text=True)
            return MixedBatch.from_ops(ops, op_modes=op_modes)
        case ForwardMode.EMIT_TOKEN | ForwardMode.EMIT_FRAME:
            raise invalid_descriptor(f"mode {mode.value} has no parsed batch view")
        case _:
            raise invalid_descriptor(f"unhandled forward mode {mode!r}")


def _cfg_batch(raw: Any, where: str) -> CfgBatch:
    if raw is None:
        return CfgBatch(
            branch_count=1,
            seg_offsets=None,
            branch_positions=None,
            branch_kv_offsets=None,
            branch_kv_lens=None,
        )
    if not isinstance(raw, Mapping):
        raise invalid_descriptor(f"{where} must be a map when provided")
    branch_count = raw.get("branch_count", 1)
    if not isinstance(branch_count, int) or isinstance(branch_count, bool) or branch_count <= 0:
        raise invalid_descriptor(f"{where}.branch_count must be a positive integer")
    return CfgBatch(
        branch_count=branch_count,
        seg_offsets=raw.get("seg_offsets"),
        branch_positions=raw.get("branch_positions"),
        branch_kv_offsets=raw.get("branch_kv_offsets"),
        branch_kv_lens=raw.get("branch_kv_lens"),
    )
