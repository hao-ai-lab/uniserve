"""Forward result postprocessing and request-state side effects."""

from __future__ import annotations

import base64
from typing import Any, Mapping

import torch

from ...contracts.forward_context import (
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
)
from ...contracts.forward_mode import ForwardMode
from ...contracts.outputs import CommitOutput, DenoiseOutput, EncodeOutput, TextTokenOutput
from ...foundation.errors import invalid_descriptor
from ...nn.diffusion import euler_step
from ...nn.sampler import (
    apply_sampling_batched_with_device_tokens,
    finalize_sampling_result,
    is_deferred_sampling_result,
)
from ...runtime.image_utils import pil_image_to_png_b64, to_uint8_image
from ...runtime.paged_text_cache import copy_paged_text_cache_spans
from ..text_decode_relay import TextDecodeRelay
from ..text_driver import sample_logits_result
from .deferred_text import DeferredTextSeqResult
from .plan import ForwardOutputKind, ForwardPlan
from .result import DenoiseBranchKey, ForwardResult, TextPostprocessEntry

__all__ = ["ForwardPostprocessor"]

_DECODE_RELAY = TextDecodeRelay()
_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY})


class ForwardPostprocessor:
    def apply(self, plan: ForwardPlan, result: ForwardResult) -> list[Any]:
        result.validate_for_plan(plan)
        if result.runtime_outputs is not None:
            runtime_outputs = self._normalize_outputs(plan, result.runtime_outputs)
            side_effects = plan.runtime_handles.get("postprocess_side_effects")
            if callable(side_effects):
                side_effects(runtime_outputs)
            return runtime_outputs
        if self._can_apply_text_batch(plan, result):
            batch_outputs = self._normalize_outputs(plan, self._apply_text_batch(plan, result))
            side_effects = plan.runtime_handles.get("postprocess_side_effects")
            if callable(side_effects):
                side_effects(batch_outputs)
            return batch_outputs
        text_outputs = self._apply_text_entries(plan, result) if result.text_postprocess else {}
        outputs: list[Any] = []
        text_row = 0
        for slot in plan.output_slots:
            row = plan.rows[slot.row_index]
            if slot.kind is ForwardOutputKind.TEXT_TOKEN:
                if slot.row_index in text_outputs:
                    outputs.append(text_outputs[slot.row_index])
                    continue
                if result.text_logits is None:
                    raise invalid_descriptor("text output slot requires logits")
                logits_rows = result.text_logits.reshape(-1, result.text_logits.shape[-1])
                outputs.append(self._sample_text(plan, row.req_id, row.op, logits_rows[text_row]))
                text_row += 1
            elif slot.kind is ForwardOutputKind.DENOISE_STEP:
                outputs.append(self._apply_denoise(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.COMMIT:
                outputs.append(self._apply_commit(plan, slot.row_index, result))
            elif slot.kind is ForwardOutputKind.ENCODE:
                outputs.append(self._apply_encode(plan, slot.row_index, result))
            else:
                outputs.append({"req_id": row.req_id})
        outputs = self._normalize_outputs(plan, outputs)
        side_effects = plan.runtime_handles.get("postprocess_side_effects")
        if callable(side_effects):
            side_effects(outputs)
        return outputs

    def _normalize_outputs(
        self,
        plan: ForwardPlan,
        outputs: tuple[Any, ...] | list[Any],
    ) -> list[Any]:
        normalizer = plan.runtime_handles.get("output_normalizer")
        if callable(normalizer):
            return [normalizer(output) for output in outputs]
        return list(outputs)

    @staticmethod
    def _can_apply_text_batch(plan: ForwardPlan, result: ForwardResult) -> bool:
        return (
            isinstance(result.text_logits, torch.Tensor)
            and bool(plan.output_slots)
            and all(slot.kind is ForwardOutputKind.TEXT_TOKEN for slot in plan.output_slots)
            and plan.runtime_handles.get("dispatch_batch") is not None
        )

    def _apply_text_batch(self, plan: ForwardPlan, result: ForwardResult) -> list[Any]:
        text = plan.runtime_handles.get("dispatch_batch").as_text(
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED
        )
        req_ids = [int(row.req_id) for row in plan.rows]
        if result.text_logits is None:
            raise invalid_descriptor("text batch postprocess requires logits")
        logits_batch = self._text_logits_rows(result.text_logits, len(req_ids))
        if (
            plan.runtime_handles.get("defer_sampling")
            and plan.runtime_handles.tensor_store is not None
        ):
            published_outputs = self._publish_logits(
                list(text.ops),
                req_ids,
                logits_batch,
                plan.runtime_handles.tensor_store,
            )
            self._publish_decode_position_relays(text, logits_batch.device, plan)
            self._advance_text_kv_lengths(text, plan)
            return published_outputs
        sampled_outputs = self._sample_text_logits_batch(
            plan,
            list(text.ops),
            req_ids,
            logits_batch,
            defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results")),
            cuda_ready_start_event=result.text_cuda_ready_start_event,
        )
        self._publish_decode_position_relays(text, logits_batch.device, plan)
        self._advance_text_kv_lengths(text, plan)
        return sampled_outputs

    @staticmethod
    def _text_logits_rows(logits: torch.Tensor, row_count: int) -> torch.Tensor:
        if logits.ndim != 2:
            logits = logits.reshape(-1, logits.shape[-1])
        if int(logits.shape[0]) < int(row_count):
            raise invalid_descriptor("text logits row count is smaller than text output slots")
        return logits[: int(row_count)]

    def _sample_text_logits_batch(
        self,
        plan: ForwardPlan,
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        *,
        defer_cpu_results: bool,
        cuda_ready_start_event: torch.cuda.Event | None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return [
                TextTokenOutput(
                    req_id=req_id,
                    sampled_token_id=int(torch.argmax(logits_batch[row].float()).item()),
                )
                for row, req_id in enumerate(req_ids)
            ]
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator] = []
        for op, req_id in zip(ops, req_ids, strict=True):
            state = request_states.get(int(req_id))
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
            generators.append(state.device_rng(logits_batch.device, stream="text_sampling"))
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_batch,
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=defer_cpu_results,
            enable_cuda_timing=cuda_ready_start_event is not None,
        )
        if is_deferred_sampling_result(sampling_result):
            sampling_result.set_ready_start_event(cuda_ready_start_event)
            outputs: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(int(req_id))
                relay_token_tensor = sampling_result.device_tokens[row : row + 1]
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None,
                    token_tensor=relay_token_tensor,
                )
                outputs.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=relay_token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return outputs

        immediate_result = finalize_sampling_result(sampling_result)
        outputs = []
        for row, (req_id, (tok, lp, top)) in enumerate(
            zip(req_ids, immediate_result.samples, strict=True)
        ):
            _DECODE_RELAY.publish_sample(
                request_states.get(int(req_id)),
                token_id=int(tok),
                token_tensor=immediate_result.device_tokens[row : row + 1],
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(tok),
                    sampled_logprob=lp,
                    top_logprobs=(
                        [(int(item[0]), float(item[1]), int(item[2])) for item in top]
                        if top
                        else None
                    ),
                )
            )
        record_component_elapsed(stats, "text_sample", start)
        return outputs

    @staticmethod
    def _publish_logits(
        ops: list[Mapping[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        tensor_store: Any,
    ) -> list[dict[str, Any]]:
        if logits_batch.ndim != 2 or int(logits_batch.shape[0]) != len(ops):
            raise invalid_descriptor("deferred-sampler logits must be shaped [ops, vocab]")
        if logits_batch.is_cuda:
            torch.cuda.synchronize(logits_batch.device)
        results: list[dict[str, Any]] = []
        for row, op in enumerate(ops):
            handle = tensor_store.publish(logits_batch[row].contiguous(), "logits")
            result: dict[str, Any] = {"req_id": int(op["req_id"]), "logits_handle": int(handle)}
            locator = tensor_store.locator_of(handle)
            if locator is not None:
                result["locator"] = base64.b64encode(locator).decode("ascii")
            results.append(result)
        return results

    @staticmethod
    def _publish_decode_position_relays(text: Any, device: torch.device, plan: ForwardPlan) -> None:
        if text.mode is not ForwardMode.DECODE:
            return
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        stats = get_forward_context().stats
        start = component_timer_start(stats)
        states = [request_states.get(int(req_id)) for req_id in text.req_ids]
        positions = [int(pos_range[1]) for pos_range in text.pos_ranges]
        _DECODE_RELAY.publish_positions(states, position_ids=positions, device=device)
        record_component_elapsed(stats, "text_decode_position_store", start)

    @staticmethod
    def _advance_text_kv_lengths(text: Any, plan: ForwardPlan) -> None:
        if text.mode not in _TEXT_MODES:
            return
        request_states = plan.runtime_handles.request_states
        if request_states is None:
            return
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges, strict=True):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane="text")

    @staticmethod
    def _sample_text(
        plan: ForwardPlan, req_id: int, op: Any, logits: torch.Tensor
    ) -> dict[str, Any]:
        request_states = plan.runtime_handles.request_states
        if request_states is not None:
            state = request_states.get(int(req_id))
            return sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        token = int(torch.argmax(logits.float()).item())
        return {"req_id": int(req_id), "sampled_token_id": token}

    def _apply_text_entries(self, plan: ForwardPlan, result: ForwardResult) -> dict[int, Any]:
        entries = tuple(result.text_postprocess or ())
        if not entries:
            return {}
        if result.text_logits is None:
            raise invalid_descriptor("text postprocess entries require text logits")
        logits_rows = self._text_logits_rows(result.text_logits, len(entries))
        entries_by_index = sorted(entries, key=lambda entry: int(entry.logits_index))
        if [int(entry.logits_index) for entry in entries_by_index] != list(
            range(len(entries_by_index))
        ):
            raise invalid_descriptor("text postprocess logits indices must be contiguous")
        request_states = plan.runtime_handles.request_states
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        generators: list[torch.Generator | None] = []
        for entry in entries_by_index:
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            params.append(dict(getattr(state, "sampling", {}) or {}))
            recent.append(row.op.get("recent_tokens") or [])
            allowed.append(row.op.get("allowed_tokens"))
            suppress.append(row.op.get("suppress_tokens"))
            generators.append(
                None
                if state is None
                else state.device_rng(logits_rows.device, stream="text_sampling")
            )
        sampling_result = apply_sampling_batched_with_device_tokens(
            logits_rows[: len(entries_by_index)],
            params,
            recent,
            allowed,
            suppress,
            generators=generators,
            defer_cpu=bool(plan.runtime_handles.get("defer_text_cpu_results")),
        )
        deferred: Any | None
        if is_deferred_sampling_result(sampling_result):
            deferred = sampling_result
            samples = []
            device_tokens = sampling_result.device_tokens
        else:
            deferred = None
            immediate_result = finalize_sampling_result(sampling_result)
            samples = immediate_result.samples
            device_tokens = immediate_result.device_tokens
        promotions = [
            entry.kv_promotion for entry in entries_by_index if entry.kv_promotion is not None
        ]
        if promotions:
            num_layers = max(int(entry.num_layers) for entry in entries_by_index)
            copy_paged_text_cache_spans(
                promotions,
                num_layers=num_layers,
                missing_message="forward text K/V span is missing from staged cache",
            )
        outputs: dict[int, Any] = {}
        for sample_index, entry in enumerate(entries_by_index):
            row = plan.rows[int(entry.row_index)]
            state = request_states.get(int(entry.req_id)) if request_states is not None else None
            logits = logits_rows[int(entry.logits_index) : int(entry.logits_index) + 1].unsqueeze(0)
            self._publish_text_entry_state(entry, logits)
            token_tensor = device_tokens[sample_index : sample_index + 1]
            position_tensor = torch.tensor(
                [int(entry.position_id)],
                dtype=torch.long,
                device=logits_rows.device,
            )
            if state is not None:
                _DECODE_RELAY.publish_sample(
                    state,
                    token_id=None if deferred is not None else int(samples[sample_index].token_id),
                    token_tensor=token_tensor,
                )
                _DECODE_RELAY.publish_position(
                    state,
                    position_id=int(entry.position_id),
                    position_tensor=position_tensor,
                )
            output: Any
            if deferred is not None:
                if state is None:
                    raise invalid_descriptor("deferred text postprocess requires request state")
                output = DeferredTextSeqResult(
                    req_id=int(entry.req_id),
                    row=sample_index,
                    state=state,
                    sampling_result=deferred,
                    relay_token_tensor=token_tensor,
                )
            else:
                sample = samples[sample_index]
                top_logprobs = (
                    [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                    if sample.top_logprobs is not None
                    else None
                )
                output = TextTokenOutput(
                    req_id=int(entry.req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=top_logprobs,
                )
            outputs[int(row.row_index)] = output
        return outputs

    @staticmethod
    def _publish_text_entry_state(entry: TextPostprocessEntry, logits: torch.Tensor) -> None:
        state = entry.interleaved_state
        cond = getattr(state, "cond", None)
        if cond is not None:
            cond.t_index = int(entry.position_id) - 1
            cond.last_logits = logits
            cond.last_token_id = int(entry.last_input_token)
        persistent_cache = entry.persistent_cache
        if persistent_cache is not None:
            persistent_cache.length = int(entry.kv_new_length)
        if (
            entry.staged_cache is not None
            and persistent_cache is not None
            and entry.staged_cache is not persistent_cache
            and entry.mark_staging_advanced is not None
        ):
            entry.mark_staging_advanced(
                entry.staged_cache, persistent_cache, int(entry.kv_new_length)
            )

    @staticmethod
    def _apply_denoise(plan: ForwardPlan, row_index: int, result: ForwardResult) -> DenoiseOutput:
        row = plan.rows[row_index]
        denoise = row.denoise
        if denoise is None:
            raise invalid_descriptor("denoise output slot references a non-denoise row")
        velocities = result.denoise_velocities or {}
        update = (result.denoise_updates or {}).get(int(row_index))
        if update is not None:
            branch_velocities: dict[Any, torch.Tensor] = {}
            for branch_id, branch_name in enumerate(update.branch_names):
                key = DenoiseBranchKey(row_index, branch_id)
                velocity = velocities.get(key)
                if velocity is None:
                    raise invalid_descriptor("denoise output is missing branch velocity")
                branch_velocities[branch_name] = velocity
            combined = update.combine_velocity(branch_velocities)
            if not isinstance(combined, torch.Tensor):
                raise invalid_descriptor("denoise combined velocity must be a tensor")
            if tuple(combined.shape) != tuple(update.latent.shape):
                raise invalid_descriptor("denoise combined velocity shape must match latent shape")
            updated = euler_step(update.latent, combined, update.t, update.t_next)
            update.accept_update(updated)
            done = update.step_index + 1 >= update.total_steps
            return DenoiseOutput(
                req_id=update.req_id,
                denoise_done=done,
                num_steps_done=update.step_index + 1,
            )
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

    @staticmethod
    def _apply_commit(plan: ForwardPlan, row_index: int, result: ForwardResult) -> CommitOutput:
        row = plan.rows[row_index]
        if row.commit is None:
            raise invalid_descriptor("commit output slot references a non-commit row")
        if result.commit_outputs is None:
            return CommitOutput(req_id=row.req_id)
        return _commit_output_from_value(
            int(row.req_id),
            plan.runtime_handles.request_states.get(int(row.req_id))
            if plan.runtime_handles.request_states is not None
            else None,
            row.op,
            result.commit_outputs[int(row_index)],
        )

    @staticmethod
    def _apply_encode(plan: ForwardPlan, row_index: int, result: ForwardResult) -> EncodeOutput:
        row = plan.rows[row_index]
        if row.encode is None:
            raise invalid_descriptor("encode output slot references a non-encode row")
        if result.encode_outputs is None:
            return EncodeOutput(req_id=row.req_id, encoder_handle=0)
        return _coerce_encode_output(result.encode_outputs[int(row_index)])


def _commit_output_from_value(
    req_id: int,
    state: Any,
    op: Mapping[str, Any],
    value: Any,
) -> CommitOutput:
    out = dict(value) if isinstance(value, Mapping) else _image_to_result(req_id, value)
    logits = out.pop("logits", None)
    if logits is not None:
        if state is None:
            raise invalid_descriptor("commit logits require request state for sampling")
        sampled = sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
        sampled.pop("req_id", None)
        out.update(sampled)
    return _commit_output_from_dict(req_id, out)


def _commit_output_from_dict(req_id: int, out: Mapping[str, Any]) -> CommitOutput:
    image_hw = out.get("image_hw")
    return CommitOutput(
        req_id=req_id,
        image_png_b64=out.get("image_png_b64"),
        image_hw=(int(image_hw[0]), int(image_hw[1])) if image_hw is not None else None,
        sampled_token_id=out.get("sampled_token_id"),
        sampled_logprob=out.get("sampled_logprob"),
        top_logprobs=out.get("top_logprobs"),
        num_tokens=out.get("num_tokens"),
        locator=out.get("locator"),
    )


def _image_to_result(req_id: int, image: Any) -> dict[str, Any]:
    save = getattr(image, "save", None)
    if callable(save):
        width, height = getattr(image, "size", (None, None))
        out = {"req_id": req_id, "image_png_b64": pil_image_to_png_b64(image)}
        if width is not None and height is not None:
            out["image_hw"] = [int(height), int(width)]
        return out
    if isinstance(image, torch.Tensor):
        if image.ndim not in (3, 4):
            raise invalid_descriptor("decode_image tensor output must be CHW or NCHW")
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency failure is environment-specific.
            raise invalid_descriptor("PIL is required to encode tensor image outputs") from exc
        pil = Image.fromarray(to_uint8_image(image, value_range=(0.0, 1.0)))
        return _image_to_result(req_id, pil)
    raise invalid_descriptor("decode_image must return a mapping, PIL image, or image tensor")


def _coerce_encode_output(output: Mapping[str, Any] | EncodeOutput) -> EncodeOutput:
    if isinstance(output, EncodeOutput):
        return output
    if not isinstance(output, Mapping):
        raise invalid_descriptor("encode adapter outputs must be mappings")
    req_id = output.get("req_id")
    handle = output.get("encoder_handle")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise invalid_descriptor("encode output req_id must be an integer")
    if not isinstance(handle, int) or isinstance(handle, bool):
        raise invalid_descriptor("encode output encoder_handle must be an integer")
    return EncodeOutput(
        req_id=int(req_id),
        encoder_handle=int(handle),
        num_tokens=_optional_int(output.get("num_tokens")),
        image_hw=_image_hw(output.get("image_hw")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor("encode output num_tokens must be an integer")
    return int(value)


def _image_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise invalid_descriptor("encode output image_hw must be [height, width]")
    return (int(value[0]), int(value[1]))
