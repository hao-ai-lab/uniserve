"""Typed worker caps/result contracts.

The Rust worker protocol is FlatBuffers at the IPC boundary, and Python drivers
naturally deal in dicts. This module is the model-neutral type boundary:
drivers may keep returning plain dicts, but the runtime validates them against
one canonical schema before the host sees them. The op-kind vocabulary and the
shared wire-field validators live in ``contracts.op_kinds``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..foundation.errors import invalid_descriptor
from .op_kinds import (
    OP_KINDS,
    validate_seq_result,
    wire_int_field,
    wire_mapping,
    wire_optional_int,
    wire_str,
    wire_str_list,
)

__all__ = [
    "Caps",
    "ExecutionConstraints",
    "CONTROL_KINDS",
    "ADAPTER_MODES",
    "RESOURCE_CLASSES",
    "validate_caps",
    "validate_forward_result",
]

CONTROL_KINDS = frozenset(
    {
        "copy_blocks",
        "load_lora",
        "unload_lora",
        "free_encoder",
        "reset_prefix_cache",
        "sleep",
        "wake_up",
    }
)
ADAPTER_MODES = frozenset({"none", "engine_wide", "per_request", "multi_adapter"})
RESOURCE_CLASSES = frozenset(
    {"kv_block", "encoder_output", "image_latent", "scratch", "adapter"}
)
_REQUIRED_CAP_KEYS = (
    "block_size",
    "num_blocks",
    "num_layers",
    "scratch_capacity_tokens",
    "supported_ops",
    "max_latent_size",
    "latent_downsample",
    "bytes_per_token",
    "supported_controls",
    "adapter_mode",
    "execution_constraints",
    "resource_classes",
)


@dataclass(frozen=True)
class ExecutionConstraints:
    """Scheduler-facing batch limits advertised in worker caps."""

    # The scheduler owns lane formation; workers advertise only scalar limits,
    # not a separate mixed-op capability flag.
    max_batch_ops: int


@dataclass(frozen=True)
class Caps:
    """Validated worker capability snapshot returned by ``get_caps``."""

    block_size: int
    num_blocks: int
    num_layers: int
    scratch_capacity_tokens: int
    supported_ops: tuple[str, ...]
    max_latent_size: int
    latent_downsample: int
    bytes_per_token: int
    supported_controls: tuple[str, ...]
    adapter_mode: str
    execution_constraints: ExecutionConstraints
    resource_classes: tuple[str, ...]
    attention_backend: str | None = None
    kv_dtype: str | None = None
    quantization: str | None = None
    pipeline_depth: int | None = None
    encoder_cache_budget: int | None = None
    max_vae_grid_tokens: int = 0
    max_vit_grid_tokens: int = 0
    commit_marker_tokens: int = 2
    gen_rope_advance: int = 2
    max_cfg_branches: int = 3

    def to_wire(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "scratch_capacity_tokens": self.scratch_capacity_tokens,
            "supported_ops": list(self.supported_ops),
            "max_latent_size": self.max_latent_size,
            "latent_downsample": self.latent_downsample,
            "bytes_per_token": self.bytes_per_token,
            "groups": [],
            "kv_dtype": self.kv_dtype or "bf16",
            "attention_backend": self.attention_backend or "auto",
            "quantization": self.quantization,
            "rank": {
                "tp_rank": 0,
                "tp_size": 1,
                "pp_rank": 0,
                "pp_size": 1,
                "dp_rank": 0,
                "dp_size": 1,
            },
            "pipeline_depth": self.pipeline_depth or 1,
            "encoder_cache_budget": self.encoder_cache_budget or 0,
            "max_vae_grid_tokens": self.max_vae_grid_tokens,
            "max_vit_grid_tokens": self.max_vit_grid_tokens,
            "commit_marker_tokens": self.commit_marker_tokens,
            "gen_rope_advance": self.gen_rope_advance,
            "max_cfg_branches": self.max_cfg_branches,
            "supported_controls": list(self.supported_controls),
            "adapter_mode": self.adapter_mode,
            "execution_constraints": {
                "max_batch_ops": self.execution_constraints.max_batch_ops,
            },
            "resource_classes": list(self.resource_classes),
        }


def validate_caps(raw: Mapping[str, Any], *, owner: str = "driver") -> Caps:
    if isinstance(raw, Caps):
        return raw
    caps = wire_mapping(raw, f"{owner}.caps")
    _require_cap_keys(caps, owner)
    supported_ops = _validate_supported_ops(caps, owner)
    supported_controls, adapter_mode = _validate_controls_and_adapters(caps, owner)
    resource_classes = _validate_resource_classes(caps, owner)
    execution_constraints = _validate_execution_constraints(caps, owner)

    return Caps(
        block_size=wire_int_field(caps["block_size"], f"{owner}.caps.block_size", minimum=1),
        num_blocks=wire_int_field(caps["num_blocks"], f"{owner}.caps.num_blocks", minimum=1),
        num_layers=wire_int_field(caps["num_layers"], f"{owner}.caps.num_layers", minimum=1),
        scratch_capacity_tokens=wire_int_field(
            caps["scratch_capacity_tokens"],
            f"{owner}.caps.scratch_capacity_tokens",
        ),
        supported_ops=supported_ops,
        max_latent_size=wire_int_field(caps["max_latent_size"], f"{owner}.caps.max_latent_size"),
        latent_downsample=wire_int_field(caps["latent_downsample"], f"{owner}.caps.latent_downsample", minimum=1),
        bytes_per_token=wire_int_field(caps["bytes_per_token"], f"{owner}.caps.bytes_per_token", minimum=1),
        supported_controls=supported_controls,
        adapter_mode=adapter_mode,
        execution_constraints=execution_constraints,
        resource_classes=resource_classes,
        attention_backend=(
            wire_str(caps["attention_backend"], f"{owner}.caps.attention_backend")
            if "attention_backend" in caps
            else None
        ),
        kv_dtype=wire_str(caps["kv_dtype"], f"{owner}.caps.kv_dtype") if "kv_dtype" in caps else None,
        quantization=(
            wire_str(caps["quantization"], f"{owner}.caps.quantization")
            if caps.get("quantization") is not None
            else None
        ),
        pipeline_depth=(
            wire_int_field(caps["pipeline_depth"], f"{owner}.caps.pipeline_depth", minimum=1)
            if "pipeline_depth" in caps
            else None
        ),
        encoder_cache_budget=(
            wire_int_field(caps["encoder_cache_budget"], f"{owner}.caps.encoder_cache_budget")
            if "encoder_cache_budget" in caps
            else None
        ),
        max_vae_grid_tokens=wire_int_field(
            caps.get("max_vae_grid_tokens", caps.get("max_latent_size", 0)),
            f"{owner}.caps.max_vae_grid_tokens",
        ),
        max_vit_grid_tokens=wire_int_field(
            caps.get("max_vit_grid_tokens", 0),
            f"{owner}.caps.max_vit_grid_tokens",
        ),
        commit_marker_tokens=wire_int_field(
            caps.get("commit_marker_tokens", 2),
            f"{owner}.caps.commit_marker_tokens",
            minimum=1,
        ),
        gen_rope_advance=wire_int_field(
            caps.get("gen_rope_advance", 2),
            f"{owner}.caps.gen_rope_advance",
            minimum=1,
        ),
        max_cfg_branches=wire_int_field(
            caps.get("max_cfg_branches", 3),
            f"{owner}.caps.max_cfg_branches",
            minimum=1,
        ),
    )


def _require_cap_keys(caps: Mapping[str, Any], owner: str) -> None:
    for key in _REQUIRED_CAP_KEYS:
        if key not in caps:
            raise invalid_descriptor(f"{owner}.caps missing required key {key!r}")


def _validate_supported_ops(caps: Mapping[str, Any], owner: str) -> tuple[str, ...]:
    supported_ops = wire_str_list(caps["supported_ops"], f"{owner}.caps.supported_ops", OP_KINDS)
    if not supported_ops:
        raise invalid_descriptor(f"{owner}.caps.supported_ops must not be empty")
    return supported_ops


def _validate_controls_and_adapters(
    caps: Mapping[str, Any], owner: str
) -> tuple[tuple[str, ...], str]:
    supported_controls = wire_str_list(
        caps["supported_controls"], f"{owner}.caps.supported_controls", CONTROL_KINDS
    )
    adapter_mode = wire_str(caps["adapter_mode"], f"{owner}.caps.adapter_mode")
    if adapter_mode not in ADAPTER_MODES:
        raise invalid_descriptor(f"{owner}.caps.adapter_mode has unknown value {adapter_mode!r}")
    if adapter_mode != "none" and not {"load_lora", "unload_lora"} <= set(supported_controls):
        raise invalid_descriptor(
            f"{owner}.caps adapter_mode={adapter_mode!r} requires load_lora/unload_lora"
        )
    return supported_controls, adapter_mode


def _validate_resource_classes(caps: Mapping[str, Any], owner: str) -> tuple[str, ...]:
    resource_classes = wire_str_list(
        caps["resource_classes"], f"{owner}.caps.resource_classes", RESOURCE_CLASSES
    )
    if "kv_block" not in resource_classes:
        raise invalid_descriptor(f"{owner}.caps.resource_classes must include 'kv_block'")
    return resource_classes


def _validate_execution_constraints(
    caps: Mapping[str, Any], owner: str
) -> ExecutionConstraints:
    ec_raw = wire_mapping(caps["execution_constraints"], f"{owner}.caps.execution_constraints")
    if "max_batch_ops" not in ec_raw:
        raise invalid_descriptor(f"{owner}.caps.execution_constraints requires max_batch_ops")
    return ExecutionConstraints(
        max_batch_ops=wire_int_field(
            ec_raw["max_batch_ops"], f"{owner}.caps.execution_constraints.max_batch_ops"
        ),
    )


def validate_forward_result(
    raw: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    owner: str = "driver",
) -> Mapping[str, Any]:
    result = wire_mapping(raw, f"{owner}.execute result")
    batch_map = wire_mapping(batch, "execute batch")
    step_id = wire_int_field(result.get("step_id"), f"{owner}.execute result.step_id")
    expected_step_id = wire_int_field(batch_map.get("step_id"), "execute batch.step_id")
    if step_id != expected_step_id:
        raise invalid_descriptor(
            f"{owner}.execute result.step_id {step_id} does not match batch.step_id {expected_step_id}"
        )

    ops = batch_map.get("ops")
    if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes, bytearray)):
        raise invalid_descriptor("execute batch.ops must be a list")
    per_seq = result.get("per_seq")
    if not isinstance(per_seq, Sequence) or isinstance(per_seq, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{owner}.execute result.per_seq must be a list")
    if len(per_seq) != len(ops):
        raise invalid_descriptor(
            f"{owner}.execute returned {len(per_seq)} per_seq results for {len(ops)} ops"
        )
    for i, (sr, op) in enumerate(zip(per_seq, ops)):
        validate_seq_result(sr, wire_mapping(op, f"batch.ops[{i}]"), i)
    wire_optional_int(result, "worker_exec_us", f"{owner}.execute result")
    return raw
