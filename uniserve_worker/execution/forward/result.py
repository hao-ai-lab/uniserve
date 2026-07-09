"""Typed neural results for unified forward execution."""
from __future__ import annotations

from collections.abc import Callable
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
    "DenoisePostprocessEntry",
    "DenoiseVelocityResult",
    "EncodeResult",
    "ForwardGraphExecutionInfo",
    "ForwardResult",
    "TextLogitsResult",
    "TextPostprocessEntry",
    "coerce_forward_result",
]


@dataclass(frozen=True)
class DenoiseBranchKey:
    row_index: int
    branch_id: int


@dataclass(frozen=True)
class DenoisePostprocessEntry:
    row_index: int
    req_id: int
    step_index: int
    total_steps: int
    branch_names: tuple[Any, ...]
    latent: torch.Tensor
    t: torch.Tensor
    t_next: torch.Tensor
    combine_velocity: Callable[[Mapping[Any, torch.Tensor]], torch.Tensor]
    accept_update: Callable[[torch.Tensor], None]


@dataclass(frozen=True)
class TextPostprocessEntry:
    row_index: int
    req_id: int
    logits_index: int
    position_id: int
    kv_new_length: int
    last_input_token: int
    interleaved_state: Any | None = None
    persistent_cache: Any | None = None
    staged_cache: Any | None = None
    kv_promotion: Any | None = None
    num_layers: int = 0
    mark_staging_advanced: Callable[[Any, Any, int], None] | None = None


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
    text_postprocess: tuple[TextPostprocessEntry, ...] | None = None
    denoise_velocities: Mapping[DenoiseBranchKey, torch.Tensor] | None = None
    denoise_updates: Mapping[int, DenoisePostprocessEntry] | None = None
    encode_outputs: Mapping[int, Any] | None = None
    commit_outputs: Mapping[int, Any] | None = None
    hidden: torch.Tensor | None = None
    graph: ForwardGraphExecutionInfo | None = None
    text_cuda_ready_start_event: Any | None = None
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
            logits_rows = self.text_logits.reshape(-1, self.text_logits.shape[-1])
            if self.text_logits.ndim == 1:
                if len(text_slots) != 1:
                    raise invalid_descriptor("single text logits row cannot satisfy multiple output slots")
            elif self.text_logits.ndim < 2:
                raise invalid_descriptor("text logits must include a vocabulary dimension")
            elif int(logits_rows.shape[0]) < len(text_slots):
                raise invalid_descriptor("text logits row count is smaller than text output slots")
            if self.text_postprocess is not None:
                if len(self.text_postprocess) != len(text_slots):
                    raise invalid_descriptor("text postprocess entry count must match text output slots")
                text_rows = {int(slot.row_index): slot for slot in text_slots}
                for entry in self.text_postprocess:
                    slot = text_rows.get(int(entry.row_index))
                    if slot is None or int(slot.req_id) != int(entry.req_id):
                        raise invalid_descriptor("text postprocess entry does not align with output slot")
                    if int(entry.logits_index) < 0 or int(entry.logits_index) >= int(logits_rows.shape[0]):
                        raise invalid_descriptor("text postprocess entry references an unknown logits row")
        denoise_slots = [
            slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.DENOISE_STEP
        ]
        if denoise_slots:
            velocities = self.denoise_velocities or {}
            updates = self.denoise_updates or {}
            for slot in denoise_slots:
                update = updates.get(int(slot.row_index))
                if update is not None:
                    if int(update.row_index) != int(slot.row_index) or int(update.req_id) != int(slot.req_id):
                        raise invalid_descriptor("denoise update entry does not align with output slot")
                    if not update.branch_names:
                        raise invalid_descriptor("denoise update entry must name at least one branch")
                    branch_count = len(update.branch_names)
                else:
                    row = plan.rows[slot.row_index]
                    branch_count = int(row.denoise.branch_count if row.denoise is not None else 0)
                for branch_id in range(branch_count):
                    key = DenoiseBranchKey(slot.row_index, branch_id)
                    if key not in velocities:
                        raise invalid_descriptor("forward result is missing a denoise branch velocity")
        encode_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.ENCODE]
        if encode_slots and self.encode_outputs is None:
            raise invalid_descriptor("forward result is missing encode outputs")
        for slot in encode_slots:
            if self.encode_outputs is not None and int(slot.row_index) not in self.encode_outputs:
                raise invalid_descriptor("forward result is missing an encode row output")
        commit_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.COMMIT]
        if commit_slots and self.commit_outputs is None:
            raise invalid_descriptor("forward result is missing commit outputs")
        for slot in commit_slots:
            if self.commit_outputs is not None and int(slot.row_index) not in self.commit_outputs:
                raise invalid_descriptor("forward result is missing a commit row output")


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
