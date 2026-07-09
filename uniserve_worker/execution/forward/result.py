"""Typed neural results for unified forward execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ...foundation.errors import invalid_descriptor
from .plan import ForwardOutputKind

if TYPE_CHECKING:
    from .plan import ForwardPlan

__all__ = [
    "CommitNeuralResult",
    "DenoiseBranchKey",
    "DenoiseVelocityResult",
    "EncodeResult",
    "ForwardGraphExecutionInfo",
    "ForwardResult",
    "TextLogitsResult",
    "coerce_forward_result",
]


@dataclass(frozen=True)
class DenoiseBranchKey:
    row_index: int
    branch_id: int


@dataclass(frozen=True)
class ForwardGraphExecutionInfo:
    program: str
    shape_key: Any
    captured: bool = False
    replayed: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class TextLogitsResult:
    logits: torch.Tensor
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class DenoiseVelocityResult:
    velocities: Mapping[DenoiseBranchKey, torch.Tensor]


@dataclass(frozen=True)
class EncodeResult:
    outputs: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class CommitNeuralResult:
    outputs: Mapping[int, torch.Tensor]


@dataclass
class ForwardResult:
    text_logits: torch.Tensor | None = None
    denoise_velocities: Mapping[DenoiseBranchKey, torch.Tensor] | None = None
    encode_outputs: Mapping[int, torch.Tensor] | None = None
    commit_outputs: Mapping[int, torch.Tensor] | None = None
    hidden: torch.Tensor | None = None
    graph: ForwardGraphExecutionInfo | None = None
    runtime_outputs: tuple[Any, ...] | None = None

    def validate_for_plan(self, plan: "ForwardPlan") -> None:
        plan.validate()
        if self.runtime_outputs is not None:
            if len(self.runtime_outputs) != len(plan.output_slots):
                raise invalid_descriptor("forward runtime output count must match output slots")
            return
        text_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.TEXT_TOKEN]
        if text_slots:
            if not isinstance(self.text_logits, torch.Tensor):
                raise invalid_descriptor("forward result is missing text logits")
            if self.text_logits.ndim == 1:
                if len(text_slots) != 1:
                    raise invalid_descriptor("single text logits row cannot satisfy multiple output slots")
            elif self.text_logits.ndim < 2:
                raise invalid_descriptor("text logits must include a vocabulary dimension")
            elif int(self.text_logits.reshape(-1, self.text_logits.shape[-1]).shape[0]) < len(text_slots):
                raise invalid_descriptor("text logits row count is smaller than text output slots")
        denoise_slots = [
            slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.DENOISE_STEP
        ]
        if denoise_slots:
            velocities = self.denoise_velocities or {}
            for slot in denoise_slots:
                row = plan.rows[slot.row_index]
                branch_count = int(row.denoise.branch_count if row.denoise is not None else 0)
                for branch_id in range(branch_count):
                    key = DenoiseBranchKey(slot.row_index, branch_id)
                    if key not in velocities:
                        raise invalid_descriptor("forward result is missing a denoise branch velocity")
        encode_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.ENCODE]
        if encode_slots and self.encode_outputs is None:
            raise invalid_descriptor("forward result is missing encode outputs")
        commit_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.COMMIT]
        if commit_slots and self.commit_outputs is None:
            raise invalid_descriptor("forward result is missing commit outputs")


def coerce_forward_result(value: Any) -> ForwardResult:
    if isinstance(value, ForwardResult):
        return value
    if isinstance(value, torch.Tensor):
        return ForwardResult(text_logits=value)
    if isinstance(value, (list, tuple)):
        return ForwardResult(runtime_outputs=tuple(value))
    if isinstance(value, Mapping):
        if "text_logits" in value or "hidden" in value:
            return ForwardResult(
                text_logits=value.get("text_logits"),
                hidden=value.get("hidden"),
            )
        return ForwardResult(runtime_outputs=(dict(value),))
    raise invalid_descriptor(f"unsupported forward result type {type(value).__name__}")
