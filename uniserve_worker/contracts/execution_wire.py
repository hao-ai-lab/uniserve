"""Canonical wire envelope for the target execution protocol (dormant).

Stage 10 slice of ``specs/unified_forward_execution.md``: the new-major
worker schema's *values and version semantics*, mirrored byte-for-byte with
`crates/foundation/core/src/execution_identity.rs` (`ExecutionWireEnvelope`).
The canonical encoding is compact JSON in struct declaration order with
fail-closed unknown fields and schema majors; the FlatBuffers transport
replaces the byte carrier at the Stage 10 cutover without changing these
values. Shared vectors in ``crates/protocol/vocab/execution_wire.json`` pin
the encoding from both languages.

Nothing on the production wire consumes this module.
"""
from __future__ import annotations

import json
from typing import Any

from .execution import (
    CacheLease,
    CandidateVerification,
    EncodeStep,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    ExecutionContractError,
    FlowStep,
    MaterializeStep,
    NewSession,
    Operation,
    ProductLease,
    ProductLifetime,
    RepresentationKind,
    SamplingSpec,
    SequenceStep,
    SessionRef,
    TransferKind,
)

__all__ = [
    "EXECUTION_WIRE_SCHEMA_MAJOR",
    "WireDecodeError",
    "decode_execute_batch",
    "encode_execute_batch",
]

EXECUTION_WIRE_SCHEMA_MAJOR = 1


class WireDecodeError(ValueError):
    """The wire payload violates the closed schema (fail closed)."""


# --------------------------------------------------------------------------- #
# Encode (canonical: compact JSON, struct declaration order).
# --------------------------------------------------------------------------- #


def _session_wire(session: SessionRef) -> dict[str, Any]:
    return {
        "engine": {
            "deployment_id": session.engine.deployment_id,
            "engine_epoch": session.engine.engine_epoch,
        },
        "request_id": session.request_id,
        "incarnation": session.incarnation,
        "session_version": session.session_version,
    }


def _operation_wire(operation: Operation) -> dict[str, Any]:
    if isinstance(operation, SequenceStep):
        verification = operation.verification
        return {
            "sequence": {
                "input_tokens": list(operation.input_tokens),
                "history_length": operation.history_length,
                "position_begin": operation.position_begin,
                "requested_outputs": operation.requested_outputs,
                "verification": (
                    None
                    if verification is None
                    else {
                        "candidate_tokens": list(verification.candidate_tokens),
                        "candidate_positions": list(
                            verification.candidate_positions
                        ),
                    }
                ),
            }
        }
    if isinstance(operation, FlowStep):
        return {
            "flow": {
                "schedule_id": operation.schedule_id,
                "step_index": operation.step_index,
                "total_steps": operation.total_steps,
                "input_product": operation.input_product,
                "branch_coefficients": list(operation.branch_coefficients),
                "conditioning_products": list(operation.conditioning_products),
                "output_schema": operation.output_schema,
            }
        }
    if isinstance(operation, EncodeStep):
        return {
            "encode": {
                "kind": int(operation.kind),
                "input_product": operation.input_product,
                "output_schema": operation.output_schema,
                "grid": list(operation.grid),
            }
        }
    if isinstance(operation, MaterializeStep):
        return {
            "materialize": {
                "input_product": operation.input_product,
                "output_schema": operation.output_schema,
            }
        }
    raise ExecutionContractError(
        f"{type(operation).__name__} is not a sealed operation variant"
    )


def encode_execute_batch(batch: ExecuteBatch) -> str:
    rows = []
    for row in batch.rows:
        admission = row.admission
        rows.append(
            {
                "row_id": row.row_id,
                "session": _session_wire(row.session),
                "operation": _operation_wire(row.operation),
                "admission": (
                    None
                    if admission is None
                    else {
                        "request_id": admission.request_id,
                        "incarnation": admission.incarnation,
                        "sampling": {
                            "temperature": admission.sampling.temperature,
                            "top_p": admission.sampling.top_p,
                            "top_k": admission.sampling.top_k,
                            "min_p": admission.sampling.min_p,
                            "repetition_penalty": admission.sampling.repetition_penalty,
                            "frequency_penalty": admission.sampling.frequency_penalty,
                            "presence_penalty": admission.sampling.presence_penalty,
                        },
                        "base_seed": admission.base_seed,
                        "max_history_tokens": admission.max_history_tokens,
                    }
                ),
                "cache_leases": [
                    {
                        "lease_id": lease.lease_id,
                        "engine_epoch": lease.engine_epoch,
                        "identity_digest": list(lease.identity_digest),
                        "identity_schema": lease.identity_schema,
                        "charge": lease.charge,
                        "version": lease.version,
                        "residency_handle": lease.residency_handle,
                    }
                    for lease in row.cache_leases
                ],
                "product_leases": [
                    {
                        "lease_id": lease.lease_id,
                        "schema_id": lease.schema_id,
                        "producer": _session_wire(lease.producer),
                        "product_version": lease.product_version,
                        "extent_rows": lease.extent_rows,
                        "lifetime": _LIFETIME_TO_WIRE[lease.lifetime],
                        "transfer": _TRANSFER_TO_WIRE[lease.transfer],
                    }
                    for lease in row.product_leases
                ],
                "scheduler_op_id": row.scheduler_op_id,
            }
        )
    envelope = {
        "schema_major": EXECUTION_WIRE_SCHEMA_MAJOR,
        "batch": {
            "engine_epoch": batch.engine_epoch,
            "step_id": batch.step_id,
            "acknowledged_through": batch.acknowledged_through,
            "rows": rows,
        },
    }
    return json.dumps(envelope, separators=(",", ":"))


_TRANSFER_TO_WIRE = {
    TransferKind.LOCAL_RESIDENCY: "LocalResidency",
    TransferKind.CUDA_IPC: "CudaIpc",
    TransferKind.SHARED_MEMORY: "SharedMemory",
    TransferKind.MOONCAKE: "Mooncake",
}
_TRANSFER_FROM_WIRE = {wire: kind for kind, wire in _TRANSFER_TO_WIRE.items()}
_LIFETIME_TO_WIRE = {
    ProductLifetime.TRANSITION: "Transition",
    ProductLifetime.REQUEST: "Request",
    ProductLifetime.SESSION: "Session",
    ProductLifetime.CACHE: "Cache",
    ProductLifetime.ACKNOWLEDGED_OUTPUT: "AcknowledgedOutput",
}
_LIFETIME_FROM_WIRE = {wire: kind for kind, wire in _LIFETIME_TO_WIRE.items()}


# --------------------------------------------------------------------------- #
# Decode (fail closed).
# --------------------------------------------------------------------------- #


def _take(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> list[Any]:
    unknown = set(mapping) - set(keys)
    if unknown:
        raise WireDecodeError(f"unknown fields {sorted(unknown)} in {where}")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise WireDecodeError(f"missing fields {missing} in {where}")
    return [mapping[key] for key in keys]


def _decode_session(value: dict[str, Any], where: str) -> SessionRef:
    engine_value, request_id, incarnation, version = _take(
        value, ("engine", "request_id", "incarnation", "session_version"), where
    )
    deployment_id, engine_epoch = _take(
        engine_value, ("deployment_id", "engine_epoch"), f"{where}.engine"
    )
    return SessionRef(
        engine=EngineRef(deployment_id, engine_epoch),
        request_id=request_id,
        incarnation=incarnation,
        session_version=version,
    )


def _decode_operation(value: dict[str, Any], where: str) -> Operation:
    if len(value) != 1:
        raise WireDecodeError(f"{where} must carry exactly one sealed variant")
    (variant, payload), = value.items()
    if variant == "sequence":
        tokens, history, begin, outputs, verification = _take(
            payload,
            (
                "input_tokens",
                "history_length",
                "position_begin",
                "requested_outputs",
                "verification",
            ),
            where,
        )
        decoded = None
        if verification is not None:
            candidate_tokens, candidate_positions = _take(
                verification,
                ("candidate_tokens", "candidate_positions"),
                f"{where}.verification",
            )
            decoded = CandidateVerification(
                tuple(candidate_tokens), tuple(candidate_positions)
            )
        return SequenceStep(tuple(tokens), history, begin, outputs, decoded)
    if variant == "flow":
        schedule, index, total, product, coefficients, conditioning, schema = _take(
            payload,
            (
                "schedule_id",
                "step_index",
                "total_steps",
                "input_product",
                "branch_coefficients",
                "conditioning_products",
                "output_schema",
            ),
            where,
        )
        return FlowStep(
            schedule, index, total, product,
            tuple(float(c) for c in coefficients),
            tuple(conditioning), schema,
        )
    if variant == "encode":
        kind, product, schema, grid = _take(
            payload, ("kind", "input_product", "output_schema", "grid"), where
        )
        if len(grid) != 3:
            raise WireDecodeError(f"{where}.grid must have three axes")
        return EncodeStep(
            RepresentationKind(kind), product, schema, (grid[0], grid[1], grid[2])
        )
    if variant == "materialize":
        product, schema = _take(payload, ("input_product", "output_schema"), where)
        return MaterializeStep(product, schema)
    raise WireDecodeError(f"unknown operation variant {variant!r} in {where}")


def decode_execute_batch(wire: str) -> ExecuteBatch:
    try:
        envelope = json.loads(wire)
    except json.JSONDecodeError as error:
        raise WireDecodeError(f"malformed wire payload: {error}") from error
    schema_major, batch_value = _take(
        envelope, ("schema_major", "batch"), "envelope"
    )
    if schema_major != EXECUTION_WIRE_SCHEMA_MAJOR:
        raise WireDecodeError(
            f"unsupported execution wire schema {schema_major}; this build "
            f"supports {EXECUTION_WIRE_SCHEMA_MAJOR}"
        )
    engine_epoch, step_id, acknowledged, row_values = _take(
        batch_value,
        ("engine_epoch", "step_id", "acknowledged_through", "rows"),
        "batch",
    )
    rows = []
    for index, row_value in enumerate(row_values):
        where = f"rows[{index}]"
        (
            row_id,
            session_value,
            operation_value,
            admission_value,
            cache_values,
            product_values,
            scheduler_op_id,
        ) = _take(
            row_value,
            (
                "row_id",
                "session",
                "operation",
                "admission",
                "cache_leases",
                "product_leases",
                "scheduler_op_id",
            ),
            where,
        )
        admission = None
        if admission_value is not None:
            request_id, incarnation, sampling_value, seed, max_history = _take(
                admission_value,
                (
                    "request_id",
                    "incarnation",
                    "sampling",
                    "base_seed",
                    "max_history_tokens",
                ),
                f"{where}.admission",
            )
            sampling_fields = _take(
                sampling_value,
                (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "frequency_penalty",
                    "presence_penalty",
                ),
                f"{where}.admission.sampling",
            )
            admission = NewSession(
                request_id,
                incarnation,
                SamplingSpec(*sampling_fields),
                seed,
                max_history,
            )
        cache_leases = []
        for lease_index, lease_value in enumerate(cache_values):
            fields = _take(
                lease_value,
                (
                    "lease_id",
                    "engine_epoch",
                    "identity_digest",
                    "identity_schema",
                    "charge",
                    "version",
                    "residency_handle",
                ),
                f"{where}.cache_leases[{lease_index}]",
            )
            fields[2] = bytes(fields[2])
            cache_leases.append(CacheLease(*fields))
        product_leases = []
        for lease_index, lease_value in enumerate(product_values):
            (
                lease_id,
                schema_id,
                producer_value,
                product_version,
                extent_rows,
                lifetime,
                transfer,
            ) = _take(
                lease_value,
                (
                    "lease_id",
                    "schema_id",
                    "producer",
                    "product_version",
                    "extent_rows",
                    "lifetime",
                    "transfer",
                ),
                f"{where}.product_leases[{lease_index}]",
            )
            if lifetime not in _LIFETIME_FROM_WIRE:
                raise WireDecodeError(f"unknown product lifetime {lifetime!r}")
            if transfer not in _TRANSFER_FROM_WIRE:
                raise WireDecodeError(f"unknown transfer kind {transfer!r}")
            product_leases.append(
                ProductLease(
                    lease_id=lease_id,
                    schema_id=schema_id,
                    producer=_decode_session(
                        producer_value, f"{where}.product_leases[{lease_index}]"
                    ),
                    product_version=product_version,
                    extent_rows=extent_rows,
                    lifetime=_LIFETIME_FROM_WIRE[lifetime],
                    transfer=_TRANSFER_FROM_WIRE[transfer],
                )
            )
        rows.append(
            ExecuteRow(
                row_id=row_id,
                session=_decode_session(session_value, f"{where}.session"),
                operation=_decode_operation(operation_value, f"{where}.operation"),
                admission=admission,
                cache_leases=tuple(cache_leases),
                product_leases=tuple(product_leases),
                scheduler_op_id=scheduler_op_id,
            )
        )
    return ExecuteBatch(
        engine_epoch=engine_epoch,
        step_id=step_id,
        acknowledged_through=acknowledged,
        rows=tuple(rows),
    )
