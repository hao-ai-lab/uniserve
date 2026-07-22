"""Autoregressive preparation and projection internals for ``ModelExecutor``."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from uniserve_worker.contracts.batches import TextBatch
from uniserve_worker.contracts.forward_batch import (
    ForwardBatch,
    ForwardExecutionOptions,
    ForwardPlan,
    ForwardResult,
)
from uniserve_worker.contracts.forward_context import (
    ForwardStats,
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.outputs import (
    TextTokenOutput,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.foundation.profiling import profile_range
from uniserve_worker.nn.sampler import (
    sample_one_from_logits,
    score_prompt_token_logprobs,
)
from uniserve_worker.runtime.forward_batch_builder import ForwardBatchBuilder
from uniserve_worker.runtime.request_state import RequestState, RequestStateTable

from .sampling import (
    _DECODE_RELAY,
    _GRAPH_RUNNER_UNSET,
    _KV_LANE,
    _RELAY_PLACEHOLDER_TOKEN_ID,
    DecodeBurstExecutor,
    DeferredTextSeqResult,
    TextForwardLogits,
    _DecodeBurstGraphMiss,
    _positive_int,
    _text_model_forward,
    sample_text_rows_batched,
    sampled_token_position,
    sampling_draw_generator,
    text_input_id_replacements_from_relays,
    verify_speculative_tokens,
)

if TYPE_CHECKING:
    from uniserve_worker.backends.attention.text_dispatch import TextBackendGate
    from uniserve_worker.contracts.forward_batch import ForwardBatch
    from uniserve_worker.runtime.kv_pool import PagedKVPool

_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


class _AutoregressiveRuntime:
    """Run the system-managed text forward and own the post-model sampler.

    Constructed by the runner with the system collaborators it orchestrates:
    the ``ForwardBatchBuilder`` (GPU snapshot + residency + plan), the
    ``TextBackendGate`` (batched-vs-per-op), and the system-owned ``kv_pool``.
    The optional ``graph_runner`` captures/replays the decode/prefill graphs
    around the graph-unaware model.
    """

    def __init__(
        self,
        *,
        builder: "ForwardBatchBuilder | None" = None,
        gate: "TextBackendGate | None" = None,
        kv_pool: "PagedKVPool | None" = None,
        graph_runner: Any | None = None,
    ) -> None:
        self.builder = builder
        self.gate = gate
        self.kv_pool = kv_pool
        self.graph_runner = graph_runner

    def _system_forward_runtime(self) -> tuple["ForwardBatchBuilder", "PagedKVPool"]:
        if self.builder is None or self.kv_pool is None:
            raise invalid_descriptor("system-managed text forward requires a builder and KV pool")
        return self.builder, self.kv_pool

    def prepare_batch(
        self,
        plan: ForwardPlan,
        request_states: RequestStateTable,
        *,
        device: torch.device,
    ) -> ForwardBatch | None:
        """Build the reusable text snapshot consumed by the public model forward."""

        if self.builder is None or self.kv_pool is None:
            return None
        if not plan.rows or any(row.mode not in _TEXT_MODES for row in plan.rows):
            return None
        text = TextBatch.from_ops(
            plan.forward_mode,
            plan.ops,
            op_modes=plan.op_modes,
            allow_mixed_text=plan.forward_mode is ForwardMode.MIXED,
        )
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode is ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(
                text,
                request_states,
                device,
            )
        elif text.mode is ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(
                text,
                request_states,
                device,
            )
        padded = self._graph_padded_num_tokens(text, get_forward_context())
        batch = self.builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
            request_states=request_states,
            input_ids_override=relay_input_ids,
            positions_override=relay_positions,
            input_ids_replacements=relay_replacements,
            padded_num_tokens=padded,
        )
        batch.op_modes = plan.op_modes
        return batch

    def forward_result(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        options: ForwardExecutionOptions = ForwardExecutionOptions(),
        tensor_store: Any | None = None,
    ) -> ForwardResult:
        """Execute one complete text batch behind the model's public forward boundary."""

        text_result = self.forward_logits(
            fb,
            request_states,
            model,
            defer_cpu_results=options.defer_text_cpu_results,
            defer_sampling=options.defer_sampling,
        )
        if text_result is not None:
            expected_req_ids = tuple(int(op["req_id"]) for op in fb.ops)
            if tuple(int(req_id) for req_id in text_result.req_ids) != expected_req_ids:
                raise invalid_descriptor("text logits result req_ids must align with forward ops")
            return ForwardResult(
                text_logits=text_result.logits,
                text_cuda_ready_start_event=text_result.cuda_ready_start_event,
            )
        outputs = self.step(
            fb,
            request_states,
            model,
            defer_cpu_results=options.defer_text_cpu_results,
            defer_sampling=options.defer_sampling,
            tensor_store=tensor_store,
        )
        return ForwardResult(runtime_outputs=tuple(outputs))

    @torch.inference_mode()
    def step(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            with profile_range("uniserve.text.prompt_logprobs"):
                return self._step_prompt_prefill(
                    text,
                    ops,
                    request_states,
                    model,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
        if any(text.spec_token_ids):
            pass

            with profile_range("uniserve.text.speculative_verify"):
                return verify_speculative_tokens(self, model, text, request_states)
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            with profile_range("uniserve.text.decode_burst"):
                return DecodeBurstExecutor(
                    self._step_once,
                    relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
                ).run(
                    ops,
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                )
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            # Compute in an order that keeps graph token-bucket padding legal
            # for the final row, but return results in the wire op order (the
            # response finalizer matches per-seq results to ops positionally).
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                results = self._step_once(
                    reordered,
                    list(reordered.ops),
                    request_states,
                    model,
                    defer_cpu_results=defer_cpu_results,
                    defer_sampling=defer_sampling,
                    tensor_store=tensor_store,
                )
                return [results[row_by_op[id(op)]] for op in ops]
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

    def forward_logits(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution and leave postprocessing to the forward stack."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            prepared = (
                self._forward_prepared(fb, text, model)
                if fb.device is not None and fb.device.type == "cuda" and fb.attn_plan is not None
                else None
            )
            logits_batch, req_ids = (
                prepared
                if prepared is not None
                else self._forward_with_optional_padding_reorder(
                    text,
                    ops,
                    request_states,
                    model,
                    store_position_relays=False,
                )
            )
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def _forward_prepared(
        self,
        batch: ForwardBatch,
        text: "TextBatch",
        model: Any,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if self.kv_pool is None:
            return None
        ctx = get_forward_context()
        input_ids, positions = self._reshape_inputs(batch, text)
        with use_forward_context(
            replace(ctx, attention_plan=batch.attn_plan)
        ):
            logits = self._run_model_forward(
                model,
                input_ids,
                positions,
                batch,
                ctx,
            )
        if logits is None:
            return None
        return logits, [int(req_id) for req_id in text.req_ids]

    def forward_logits_graph(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> TextForwardLogits | None:
        """Run text neural execution only when a CUDA graph handles the batch."""

        if graph_runner is None and self.builder is not None:
            return None
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(bool(op.get("return_all_logits")) for op in ops):
            return None
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            return None
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            if graph_runner is None:
                graph_result = self._forward(
                    model,
                    text,
                    request_states,
                    store_position_relays=False,
                    require_graph=True,
                )
            else:
                graph_result = self._forward_graph_with_optional_padding_reorder(
                    text,
                    ops,
                    request_states,
                    model,
                    graph_runner=graph_runner,
                    store_position_relays=False,
                )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        return TextForwardLogits(
            logits=logits_batch,
            req_ids=tuple(int(req_id) for req_id in req_ids),
            cuda_ready_start_event=cuda_ready_start_event,
        )

    def forward_graph_result(
        self,
        fb: ForwardBatch,
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> ForwardResult | None:
        """Run a text batch only when graph-backed execution can cover it."""

        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(text.spec_token_ids):
            return None
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            outputs = self._decode_burst_graph_many(
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
            )
            if outputs is None:
                return None
            return ForwardResult(runtime_outputs=tuple(outputs))
        text_result = self.forward_logits_graph(
            fb,
            request_states,
            model,
            graph_runner=graph_runner,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
        )
        if text_result is None:
            return None
        expected_req_ids = tuple(int(op["req_id"]) for op in ops)
        req_ids = tuple(int(req_id) for req_id in text_result.req_ids)
        if req_ids != expected_req_ids:
            raise invalid_descriptor("text graph result req_ids must align with forward ops")
        return ForwardResult(
            text_logits=text_result.logits,
            text_cuda_ready_start_event=text_result.cuda_ready_start_event,
        )

    def _step_once(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward"):
            forward_result = self._forward(model, text, request_states)
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        logits_batch, req_ids = forward_result
        # KV-length advance is system-owned now (derived from seq_lens), not the
        # model's job.
        self._advance_kv_lengths(text, request_states)
        if defer_sampling and tensor_store is not None:
            with profile_range("uniserve.text.publish_logits"):
                return self._publish_logits(ops, req_ids, logits_batch, tensor_store)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    def _decode_burst(
        self,
        first_op: dict[str, Any],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            [first_op],
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )[0]

    def _decode_burst_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]]:
        return DecodeBurstExecutor(
            self._step_once,
            relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
        ).run(
            list(first_ops),
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
        )

    def _decode_burst_graph_many(
        self,
        first_ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
    ) -> list[dict[str, Any]] | None:
        def step_once_graph(
            text: "TextBatch",
            ops: list[Mapping[str, Any]],
            request_states: RequestStateTable,
            model: Any,
            *,
            defer_cpu_results: bool = False,
            defer_sampling: bool = False,
            tensor_store: Any | None = None,
        ) -> list[Any]:
            del tensor_store
            outputs = self._step_once_graph(
                text,
                ops,
                request_states,
                model,
                graph_runner=graph_runner,
                defer_cpu_results=defer_cpu_results,
                defer_sampling=defer_sampling,
            )
            if outputs is None:
                raise _DecodeBurstGraphMiss
            return outputs

        try:
            return DecodeBurstExecutor(
                step_once_graph,
                relay_placeholder_token_id=_RELAY_PLACEHOLDER_TOKEN_ID,
            ).run(
                list(first_ops),
                request_states,
                model,
                defer_cpu_results=defer_cpu_results,
            )
        except _DecodeBurstGraphMiss:
            return None

    def _step_once_graph(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
    ) -> list[Any] | None:
        stats = get_forward_context().stats
        cuda_ready_start_event = _record_cuda_ready_start_event(
            self.kv_pool,
            stats=stats,
            defer_cpu_results=defer_cpu_results,
        )
        with profile_range("uniserve.text.forward_graph"):
            graph_result = self._forward(
                model,
                text,
                request_states,
                graph_runner=graph_runner,
                require_graph=True,
            )
        if graph_result is None:
            return None
        logits_batch, req_ids = graph_result
        self._advance_kv_lengths(text, request_states)
        if defer_sampling:
            return None
        start = component_timer_start(stats)
        with profile_range("uniserve.text.sample"):
            return self._sample_logits_batch(
                ops,
                req_ids,
                logits_batch,
                request_states,
                stats,
                start,
                defer_cpu_results=defer_cpu_results,
                cuda_ready_start_event=cuda_ready_start_event,
            )

    # ---- forward ---------------------------------------------------------

    def _forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if self.builder is None or self.kv_pool is None:
            # Self-managing text models (the HF day-zero fallback and the
            # composed multimodal programs whose KV is intrinsically coupled to
            # their modality FSM) declare no ``kv_cache_spec``; the system owns no
            # pool for them. They expose their own per-op text logits and the
            # driver still owns the post-model sampler.
            if require_graph:
                return self._model_owned_kv_forward_graph(model, text, request_states)
            return self._model_owned_kv_forward(model, text, request_states)
        ctx = get_forward_context()
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        batched = self.gate is not None and self.gate.batched_capable(
            text, attention_preference=ctx.attention_preference
        )
        if batched:
            return self._forward_batched(
                model,
                text,
                request_states,
                ctx,
                device,
                store_position_relays=store_position_relays,
                graph_runner=graph_runner,
                require_graph=require_graph,
            )
        if require_graph:
            return None
        return self._forward_per_op(
            model,
            text,
            request_states,
            ctx,
            device,
            store_position_relays=store_position_relays,
        )

    def _forward_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]]:
        if text.mode == ForwardMode.MIXED and self.graph_runner is not None:
            reordered = self.graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                forward_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                )
                if forward_result is None:
                    raise invalid_descriptor("reordered eager text forward did not produce logits")
                logits, req_ids = forward_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        forward_result = self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
        )
        if forward_result is None:
            raise invalid_descriptor("eager text forward did not produce logits")
        return forward_result

    def _forward_graph_with_optional_padding_reorder(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        graph_runner: Any,
        store_position_relays: bool,
    ) -> tuple[torch.Tensor, list[int]] | None:
        if text.mode == ForwardMode.MIXED:
            reordered = graph_runner.reorder_mixed_for_padding(text)
            if reordered is not text:
                row_by_op = {id(op): row for row, op in enumerate(reordered.ops)}
                graph_result = self._forward(
                    model,
                    reordered,
                    request_states,
                    store_position_relays=store_position_relays,
                    graph_runner=graph_runner,
                    require_graph=True,
                )
                if graph_result is None:
                    return None
                logits, req_ids = graph_result
                rows = [row_by_op[id(op)] for op in ops]
                order = torch.tensor(rows, dtype=torch.long, device=logits.device)
                return logits.index_select(0, order), [int(req_ids[row]) for row in rows]
        return self._forward(
            model,
            text,
            request_states,
            store_position_relays=store_position_relays,
            graph_runner=graph_runner,
            require_graph=True,
        )

    def _model_owned_kv_forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run a self-managing model's per-op text logits and stack them.

        Accepts a batched ``run_text_logits_batch`` or a per-op
        ``run_text_logits``; both return a raw logits tensor per op, coerced to
        one ``[vocab]`` row. Req ids come from the op order, not the tensors.

        The driver owns the decode-relay lookup: ``last_sampled`` ops get the
        device relay tensor attached as ``op['token_tensor']`` (and the resolved
        id when the CPU copy has landed) so the model side can consume the
        sampled token without a GPU synchronize and without reaching into
        system request state.
        """

        ops = self._model_owned_ops(text, request_states)
        outputs = list(model.run_text_logits_batch(ops))
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    def _model_owned_kv_forward_graph(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]] | None:
        graph_logits = getattr(model, "try_run_graph_logits_batch", None)
        if not callable(graph_logits):
            return None
        ops = self._model_owned_ops(text, request_states)
        outputs = graph_logits(ops)
        if outputs is None:
            return None
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

    @staticmethod
    def _model_owned_ops(
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> list[dict[str, Any]]:
        ops = [dict(op) for op in text.ops]
        for op in ops:
            if str(op.get("token_source") or "wire") != "last_sampled":
                continue
            _DECODE_RELAY.attach_last_sampled_to_op(op, request_states.get(int(op["req_id"])))
        return ops

    @staticmethod
    def _coerce_logits_row(logits: Any) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise invalid_descriptor("self-managing text model must return a logits tensor")
        return logits.reshape(-1, logits.shape[-1])[-1]

    def _forward_batched(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | None:
        stats = ctx.stats
        builder, kv_pool = self._system_forward_runtime()
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        start = component_timer_start(stats)
        # Pure decode uses the contiguous-relay override (fast); a mixed
        # extend+decode batch replaces only its last_sampled decode rows by index.
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode == ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(
                text, request_states, device
            )
        elif text.mode == ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(
                text, request_states, device
            )
        record_component_elapsed(stats, "text_decode_relay", start)
        start = component_timer_start(stats)
        padded = self._graph_padded_num_tokens(text, ctx, graph_runner=active_graph_runner)
        fb = builder.build_text(
            text,
            device=device,
            kv_pool=kv_pool,
            request_states=request_states,
            input_ids_override=relay_input_ids,
            positions_override=relay_positions,
            input_ids_replacements=relay_replacements,
            padded_num_tokens=padded,
        )
        record_component_elapsed(stats, "text_build_batch", start)
        input_ids, positions = self._reshape_inputs(fb, text)
        start = component_timer_start(stats)
        with profile_range("uniserve.text.model_forward"):
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan)):
                logits = self._run_model_forward(
                    model,
                    input_ids,
                    positions,
                    fb,
                    ctx,
                    graph_runner=active_graph_runner,
                    require_graph=require_graph,
                )
        if logits is None:
            return None
        record_component_elapsed(stats, "text_model_forward", start)
        if store_position_relays:
            start = component_timer_start(stats)
            self._store_decode_position_relays(text, fb, request_states)
            record_component_elapsed(stats, "text_decode_position_store", start)
        return logits, [int(req_id) for req_id in text.req_ids]

    def _forward_per_op(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
        ctx: Any,
        device: torch.device,
        *,
        store_position_relays: bool = True,
    ) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        next_positions: list[tuple[int, int]] = []
        builder, kv_pool = self._system_forward_runtime()
        for req_id, tokens, pos_range, op in zip(
            text.req_ids, text.token_ids, text.pos_ranges, text.ops
        ):
            state = request_states.get(int(req_id))
            relay = self._per_op_relay_input(op, tokens, state, device)
            fb = builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=int(req_id),
                mode=text.mode,
                device=device,
                kv_pool=kv_pool,
                request_states=request_states,
                input_ids_override=relay,
            )
            with use_forward_context(replace(ctx, attention_plan=fb.attn_plan)):
                logits = _text_model_forward(model, fb)
            rows.append(logits.reshape(-1, logits.shape[-1])[-1])
            next_positions.append((int(req_id), int(pos_range[1])))
        if store_position_relays and text.mode == ForwardMode.DECODE:
            for (req_id, position), op_positions in zip(next_positions, text.pos_ranges):
                tensor = torch.tensor([position], dtype=torch.long, device=device)
                self._store_position_relay(
                    request_states.get(int(req_id)),
                    position_id=position,
                    position_tensor=tensor,
                )
        return torch.stack(rows, dim=0), [int(req_id) for req_id in text.req_ids]

    def _run_model_forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        fb: "ForwardBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
        require_graph: bool = False,
    ) -> torch.Tensor | None:
        # System-owned CUDA graphs capture/replay around the graph-unaware model;
        # a miss (or graphs disabled) falls through to the eager forward.
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is not None:
            logits = active_graph_runner.maybe_run(model, input_ids, positions, fb, ctx)
            if logits is not None:
                return logits
        if require_graph:
            return None
        return _text_model_forward(
            model,
            fb,
            input_ids=input_ids,
            positions=positions,
        )

    def _reshape_inputs(
        self, fb: "ForwardBatch", text: "TextBatch"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the model input geometry (flat varlen vs rectangular) for the mode.

        Decode is rectangular ``[batch, 1]`` (the batched-decode kernels); ragged
        or padded extend is flat ``[total]`` (gathered logits via
        ``last_token_indices``); equal-length extend is rectangular ``[batch, L]``;
        mixed stays flat varlen.
        """

        mode = text.mode
        if fb.input_ids is None or fb.positions is None:
            raise invalid_descriptor("text forward batch is missing input ids or positions")
        if mode == ForwardMode.DECODE:
            return fb.input_ids.reshape(fb.batch_size, 1), fb.positions.reshape(fb.batch_size, 1)
        if mode == ForwardMode.EXTEND:
            return fb.input_ids, fb.positions
        return fb.input_ids, fb.positions

    def _graph_padded_num_tokens(
        self,
        text: "TextBatch",
        ctx: Any,
        *,
        graph_runner: Any = _GRAPH_RUNNER_UNSET,
    ) -> int | None:
        active_graph_runner = (
            self.graph_runner if graph_runner is _GRAPH_RUNNER_UNSET else graph_runner
        )
        if active_graph_runner is None:
            return None
        return active_graph_runner.padded_num_tokens(
            text, attention_preference=ctx.attention_preference
        )

    def _advance_kv_lengths(self, text: "TextBatch", request_states: RequestStateTable) -> None:
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane=_KV_LANE)

    # ---- deferred sampling -----------------------------------------------

    def _publish_logits(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
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

    # ---- sampling --------------------------------------------------------

    def _step_prompt_prefill(
        self,
        text: "TextBatch",
        ops: list[Mapping[str, Any]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> list[Any]:
        if any(str(op.get("kind")) != "prefill_und" for op in ops):
            raise invalid_descriptor("prompt scoring is valid only for prefill_und operations")

        final_logits: list[torch.Tensor] = []
        prompt_scores: list[list[list[tuple[int, float, int]]] | None] = []
        for row, (op, req_id, tokens, pos_range) in enumerate(
            zip(ops, text.req_ids, text.token_ids, text.pos_ranges, strict=True)
        ):
            del row
            state = request_states.get(int(req_id))
            logits = self._forward_prompt_op(
                model,
                op,
                tokens,
                pos_range,
                int(req_id),
                request_states,
            )
            rows = logits.reshape(-1, logits.shape[-1])
            if bool(op.get("return_all_logits")) and int(rows.shape[0]) != len(tokens):
                raise invalid_descriptor(
                    "prompt-scoring model output must contain one logits row per input token"
                )
            chunk_scores = self._score_prompt_chunk(state, rows, tokens)
            prompt_scores.append(chunk_scores or None)
            final_logits.append(rows[-1])
            state.set_kv_length(int(pos_range[1]), lane=_KV_LANE)

        logits_batch = torch.stack(final_logits, dim=0)
        if defer_sampling:
            if tensor_store is None:
                raise invalid_descriptor("deferred prompt sampling requires a tensor store")
            results = self._publish_logits(ops, text.req_ids, logits_batch, tensor_store)
            for result, prompt_score in zip(results, prompt_scores, strict=True):
                if prompt_score is not None:
                    result["prompt_logprobs"] = prompt_score
            return results

        outputs: list[TextTokenOutput] = []
        for row, (op, req_id, prompt_score) in enumerate(
            zip(ops, text.req_ids, prompt_scores, strict=True)
        ):
            state = request_states.get(int(req_id))
            sample = sample_one_from_logits(
                logits_batch[row],
                dict(state.sampling or {}),
                recent=op.get("recent_tokens") or [],
                allowed=op.get("allowed_tokens"),
                suppress=op.get("suppress_tokens"),
                n_logprobs=int(state.sampling.get("n_logprobs", 0) or 0),
                generator=sampling_draw_generator(
                    state,
                    logits_batch.device,
                    position=sampled_token_position(op),
                ),
            )
            self._store_sampled_token_relay(
                state,
                token_id=int(sample.token_id),
                token_tensor=torch.tensor(
                    [int(sample.token_id)], dtype=torch.long, device=logits_batch.device
                ),
            )
            outputs.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=int(sample.token_id),
                    sampled_logprob=sample.logprob,
                    top_logprobs=(
                        [
                            (int(item[0]), float(item[1]), int(item[2]))
                            for item in sample.top_logprobs
                        ]
                        if sample.top_logprobs
                        else None
                    ),
                    prompt_logprobs=prompt_score,
                )
            )
        return outputs

    def _forward_prompt_op(
        self,
        model: Any,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        pos_range: tuple[int, int],
        req_id: int,
        request_states: RequestStateTable,
    ) -> torch.Tensor:
        if not tokens:
            raise invalid_descriptor("prompt-scoring prefill operation has no token ids")
        if self.builder is None or self.kv_pool is None:
            predecessor = getattr(model, "prompt_predecessor_logits", None)
            if bool(op.get("return_all_logits")) and callable(predecessor):
                previous_logits = predecessor(req_id)
                if isinstance(previous_logits, torch.Tensor) and previous_logits.ndim > 0:
                    request_states.get(req_id).prompt_last_logits = previous_logits.reshape(
                        -1, previous_logits.shape[-1]
                    )[-1].detach()
            logits = model.run_text_logits(dict(op))
        else:
            ctx = get_forward_context()
            device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
            batch = self.builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=req_id,
                mode=ForwardMode.EXTEND,
                device=device,
                kv_pool=self.kv_pool,
                request_states=request_states,
            )
            batch.return_all_logits = bool(op.get("return_all_logits"))
            with use_forward_context(
                replace(ctx, attention_plan=batch.attn_plan)
            ):
                logits = _text_model_forward(model, batch)
        if not isinstance(logits, torch.Tensor) or logits.ndim == 0:
            raise invalid_descriptor("prompt-scoring model output must be a logits tensor")
        return logits

    @staticmethod
    def _score_prompt_chunk(
        state: RequestState,
        logits: torch.Tensor,
        tokens: tuple[int, ...],
    ) -> list[list[tuple[int, float, int]]]:
        sampling = dict(state.sampling or {})
        if not (
            bool(sampling.get("return_prompt_logprobs"))
            or int(sampling.get("n_prompt_logprobs", 0) or 0) > 0
        ):
            return []
        predictors: list[torch.Tensor] = []
        targets: list[int] = []
        if state.prompt_last_logits is not None:
            predictors.append(state.prompt_last_logits.reshape(1, -1))
            targets.append(int(tokens[0]))
        if len(tokens) > 1:
            predictors.append(logits[:-1])
            targets.extend(int(token_id) for token_id in tokens[1:])
        state.prompt_last_logits = logits[-1].detach()
        if not predictors:
            return []
        return score_prompt_token_logprobs(
            torch.cat(predictors, dim=0),
            targets,
            n_logprobs=int(sampling.get("n_prompt_logprobs", 0) or 0),
            logprob_token_ids=sampling.get("logprob_token_ids") or (),
        )

    def _sample_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
        req_ids: Sequence[int],
        logits_batch: torch.Tensor,
        request_states: RequestStateTable,
        stats: ForwardStats | None,
        start: int,
        *,
        defer_cpu_results: bool = False,
        cuda_ready_start_event: torch.cuda.Event | None = None,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        out = sample_text_rows_batched(
            ops,
            req_ids,
            logits_batch,
            request_states,
            defer_cpu_results=defer_cpu_results,
            cuda_ready_start_event=cuda_ready_start_event,
        )
        record_component_elapsed(stats, "text_sample", start)
        return out

    # ---- decode relays ---------------------------------------------------

    def _per_op_relay_input(
        self,
        op: Mapping[str, Any],
        tokens: tuple[int, ...],
        state: RequestState,
        device: torch.device,
    ) -> torch.Tensor | None:
        source = str(op.get("token_source") or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        if source != "last_sampled":
            return None
        if len(tokens) != 1:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but does not have exactly one token"
            )
        return _DECODE_RELAY.consume_token(
            state,
            expected_token_id=None,
            device=device,
            token_source=source,
            require=True,
        )

    def _decode_relay_tensors(
        self,
        text: "TextBatch",
        request_states: RequestStateTable,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if text.mode != ForwardMode.DECODE:
            return None, None
        return _DECODE_RELAY.resolve_decode_batch(
            text,
            request_states,
            device,
            stats=get_forward_context().stats,
        )

    def _store_decode_position_relays(
        self,
        text: "TextBatch",
        fb: "ForwardBatch",
        request_states: RequestStateTable,
    ) -> None:
        if text.mode != ForwardMode.DECODE:
            return
        if any(len(tokens) != 1 for tokens in text.token_ids):
            return
        if fb.positions is None:
            raise invalid_descriptor("decode position relay requires position ids")
        next_positions = fb.positions.reshape(-1) + 1
        for row, (req_id, pos_range) in enumerate(zip(text.req_ids, text.pos_ranges)):
            self._store_position_relay(
                request_states.get(int(req_id)),
                position_id=int(pos_range[1]),
                position_tensor=next_positions[row : row + 1],
            )

    @staticmethod
    def _store_sampled_token_relay(
        state: RequestState,
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_sample(state, token_id=token_id, token_tensor=token_tensor)

    @staticmethod
    def _store_position_relay(
        state: RequestState,
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> None:
        _DECODE_RELAY.publish_position(
            state,
            position_id=position_id,
            position_tensor=position_tensor,
        )


def _can_decode_burst(
    text: "TextBatch", ops: list[Mapping[str, Any]], *, defer_sampling: bool
) -> bool:
    if defer_sampling or text.mode != ForwardMode.DECODE:
        return False
    if any(text.spec_token_ids):
        return False
    try:
        return any(
            _positive_int(op.get("decode_token_count") or 1, "decode_token_count") > 1 for op in ops
        )
    except Exception:
        raise


def _record_cuda_ready_start_event(
    kv_pool: "PagedKVPool | None",
    *,
    stats: ForwardStats | None,
    defer_cpu_results: bool,
) -> torch.cuda.Event | None:
    if stats is None or not defer_cpu_results:
        return None
    tensor = getattr(kv_pool, "k", None)
    device = getattr(tensor, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event
