"""Forward result postprocessing and request-state side effects."""
from __future__ import annotations

from typing import Any

import torch

from ...contracts.outputs import CommitOutput, DenoiseOutput, EncodeOutput
from ...foundation.errors import invalid_descriptor
from ..text_driver import sample_logits_result
from .plan import ForwardOutputKind, ForwardPlan
from .result import DenoiseBranchKey, ForwardResult

__all__ = ["ForwardPostprocessor"]


class ForwardPostprocessor:
    def apply(self, plan: ForwardPlan, result: ForwardResult) -> list[Any]:
        result.validate_for_plan(plan)
        if result.runtime_outputs is not None:
            outputs = self._normalize_runtime_outputs(plan, result.runtime_outputs)
            side_effects = plan.runtime_handles.get("postprocess_side_effects")
            if callable(side_effects):
                side_effects(outputs)
            return outputs
        outputs: list[Any] = []
        text_row = 0
        for slot in plan.output_slots:
            row = plan.rows[slot.row_index]
            if slot.kind is ForwardOutputKind.TEXT_TOKEN:
                if result.text_logits is None:
                    raise invalid_descriptor("text output slot requires logits")
                logits_rows = result.text_logits.reshape(-1, result.text_logits.shape[-1])
                outputs.append(self._sample_text(plan, row.req_id, row.op, logits_rows[text_row]))
                text_row += 1
            elif slot.kind is ForwardOutputKind.DENOISE_STEP:
                outputs.append(self._apply_denoise(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.COMMIT:
                outputs.append(CommitOutput(req_id=row.req_id))
            elif slot.kind is ForwardOutputKind.ENCODE:
                handle = 0
                if result.encode_outputs:
                    handle = int(slot.row_index)
                outputs.append(EncodeOutput(req_id=row.req_id, encoder_handle=handle))
            else:
                outputs.append({"req_id": row.req_id})
        side_effects = plan.runtime_handles.get("postprocess_side_effects")
        if callable(side_effects):
            side_effects(outputs)
        return outputs

    def _normalize_runtime_outputs(
        self,
        plan: ForwardPlan,
        runtime_outputs: tuple[Any, ...],
    ) -> list[Any]:
        normalizer = plan.runtime_handles.get("output_normalizer")
        if callable(normalizer):
            return [normalizer(output) for output in runtime_outputs]
        return list(runtime_outputs)

    @staticmethod
    def _sample_text(plan: ForwardPlan, req_id: int, op: Any, logits: torch.Tensor) -> dict[str, Any]:
        request_states = plan.runtime_handles.request_states
        if request_states is not None:
            state = request_states.get(int(req_id))
            return sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        token = int(torch.argmax(logits.float()).item())
        return {"req_id": int(req_id), "sampled_token_id": token}

    @staticmethod
    def _apply_denoise(plan: ForwardPlan, row_index: int, result: ForwardResult) -> DenoiseOutput:
        row = plan.rows[row_index]
        denoise = row.denoise
        if denoise is None:
            raise invalid_descriptor("denoise output slot references a non-denoise row")
        velocities = result.denoise_velocities or {}
        for branch_id in range(denoise.branch_count):
            key = DenoiseBranchKey(row_index, branch_id)
            if key not in velocities:
                raise invalid_descriptor("denoise output is missing branch velocity")
        done = denoise.step_index + 1 >= denoise.total_steps
        return DenoiseOutput(
            req_id=row.req_id,
            denoise_done=done,
            num_steps_done=denoise.step_index + 1,
        )
