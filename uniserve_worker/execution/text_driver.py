"""System text execution driver.

The driver *owns* the text forward end to end (SGLang's ``ModelRunner`` role,
minus the god object): it builds the GPU ``ForwardBatch`` through the
``ForwardBatchBuilder`` (staging + system-owned KV residency + the attention
plan), decides batched-paged vs per-op-dense via the ``TextBackendGate``,
publishes the ``ForwardContext`` (backend + plan + pool), runs the thin
``model.forward(input_ids, positions, forward_batch)``, advances the KV length,
and samples. The model contributes only the network forward; pools, metadata,
graphs, KV-length, and sampling are all system-side here.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ..contracts.forward_context import (
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)
from ..contracts.forward_mode import ForwardMode
from ..contracts.forward_stats import ForwardStats
from ..contracts.outputs import ForwardOutputBase, TextTokenOutput
from ..foundation.errors import invalid_descriptor
from ..foundation.profiling import profile_range
from ..nn.sampler import (
    DeferredBatchedSamplingResult,
    apply_sampling_batched_with_device_tokens,
    sample_one_from_logits,
)
from ..runtime.request_state import RequestState, RequestStateTable

if TYPE_CHECKING:
    from ..backends.attention.text_dispatch import TextBackendGate
    from ..contracts.batches import TextBatch, UniForwardBatch
    from ..contracts.forward_batch import ForwardBatch
    from ..runtime.forward_batch_builder import ForwardBatchBuilder
    from ..runtime.kv_pool import PagedKVPool

__all__ = [
    'DeferredTextSeqResult',
    'sample_logits_result',
    'text_input_id_replacements_from_relays',
    'TextDriver',
]

_KV_LANE = "text"


def text_input_id_replacements_from_relays(
    text: "TextBatch",
    request_states: RequestStateTable,
    device: torch.device,
) -> dict[int, torch.Tensor] | None:
    """Return flat input-token replacements for ``last_sampled`` decode rows.

    The async decode fast path keeps the last sampled token resident on device;
    a mixed extend+decode forward replaces only the decode rows' placeholder
    tokens by flat index (the contiguous-relay override is the pure-decode case).
    """

    replacements: dict[int, torch.Tensor] = {}
    flat_idx = 0
    for req_id, tokens, op in zip(text.req_ids, text.token_ids, text.ops):
        source = str(op.get("token_source") or "wire")
        if source not in {"wire", "last_sampled"}:
            raise invalid_descriptor(f"unsupported text token_source {source!r}")
        if source == "last_sampled":
            if len(tokens) != 1:
                raise invalid_descriptor(
                    "decode op requested token_source='last_sampled' but does not have exactly one token"
                )
            state = request_states.get(int(req_id))
            relay = state.decode_relay.token_tensor
            if not isinstance(relay, torch.Tensor) or relay.dtype != torch.long or relay.device != device:
                raise invalid_descriptor(
                    "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
                )
            replacements[flat_idx] = relay.reshape(1)
        flat_idx += len(tokens)
    return replacements or None


class DeferredTextSeqResult(ForwardOutputBase):
    """One text seq-result whose CPU token id is finalized at response time."""

    def __init__(
        self,
        *,
        req_id: int,
        row: int,
        state: RequestState,
        sampling_result: DeferredBatchedSamplingResult,
        relay_token_tensor: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "req_id", int(req_id))
        object.__setattr__(self, "_row", int(row))
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_sampling_result", sampling_result)
        object.__setattr__(self, "_relay_token_tensor", relay_token_tensor)
        object.__setattr__(self, "_finalized", None)

    def to_seq_result(self) -> "DeferredTextSeqResult":
        return self

    def finalize(self) -> dict[str, Any]:
        if self._finalized is None:
            sample = self._sampling_result.finalize().samples[self._row]
            tok, lp, top = sample
            current = self._state.decode_relay.token_tensor
            if _same_tensor(current, self._relay_token_tensor):
                self._state.decode_relay.token_id = int(tok)
            result: dict[str, Any] = {
                "req_id": self.req_id,
                "sampled_token_id": int(tok),
            }
            if lp is not None:
                result["sampled_logprob"] = lp
            if top:
                result["top_logprobs"] = top
            object.__setattr__(self, "_finalized", result)
        return dict(self._finalized)


def sample_logits_result(
    *,
    req_id: int,
    state: RequestState,
    logits: torch.Tensor,
    op: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one token from model-produced logits using the canonical pipeline."""

    if not isinstance(logits, torch.Tensor):
        raise invalid_descriptor("text logits output must contain a tensor")
    if logits.ndim == 0:
        raise invalid_descriptor("text logits tensor must have a vocabulary dimension")
    vocab_logits = logits.float()
    if vocab_logits.ndim > 1:
        vocab_logits = vocab_logits.reshape(-1, vocab_logits.shape[-1])[-1]
    sp = dict(state.sampling or {})
    tok, lp, top = sample_one_from_logits(
        vocab_logits,
        sp,
        recent=op.get("recent_tokens") or [],
        allowed=op.get("allowed_tokens"),
        suppress=op.get("suppress_tokens"),
        n_logprobs=int(sp.get("n_logprobs", 0) or 0),
    )
    result: dict[str, Any] = {"req_id": int(req_id), "sampled_token_id": tok}
    if lp is not None:
        result["sampled_logprob"] = lp
    if top:
        result["top_logprobs"] = top
    return result


class TextDriver:
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

    @torch.inference_mode()
    def step(
        self,
        fb: "UniForwardBatch",
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ) -> list[Any]:
        text = fb.as_text(allow_mixed_text=fb.mode == ForwardMode.MIXED)
        ops = list(text.ops)
        if any(text.spec_token_ids):
            from .spec_verify import verify_speculative_tokens

            with profile_range("uniserve.text.speculative_verify"):
                return verify_speculative_tokens(
                    self, model, text, request_states
                )
        if _can_decode_burst(text, ops, defer_sampling=defer_sampling):
            with profile_range("uniserve.text.decode_burst"):
                return [
                    self._decode_burst(
                        dict(ops[0]),
                        request_states,
                        model,
                        defer_cpu_results=defer_cpu_results,
                    )
                ]
        return self._step_once(
            text,
            ops,
            request_states,
            model,
            defer_cpu_results=defer_cpu_results,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
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
        with profile_range("uniserve.text.forward"):
            logits_batch, req_ids = self._forward(model, text, request_states)
        # KV-length advance is system-owned now (derived from seq_lens), not the
        # model's job.
        self._advance_kv_lengths(text, request_states)
        if defer_sampling and tensor_store is not None:
            with profile_range("uniserve.text.publish_logits"):
                return self._publish_logits(ops, req_ids, logits_batch, tensor_store)
        stats = get_forward_context().stats
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
            )

    def _decode_burst(
        self,
        first_op: dict[str, Any],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_cpu_results: bool = False,
    ) -> dict[str, Any]:
        from ..contracts.batches import UniForwardBatch

        count = _positive_int(first_op.get("decode_token_count") or 1, "decode_token_count")
        stop_ids = set(_int_list(first_op.get("decode_stop_token_ids") or []))
        tokens: list[int] = []
        last: dict[str, Any] = {}
        op = dict(first_op)
        op["decode_token_count"] = 1
        op["decode_stop_token_ids"] = []
        for _ in range(count):
            fb = UniForwardBatch.from_ops([op])
            text = fb.as_text()
            out = self._step_once(
                text,
                [op],
                request_states,
                model,
                defer_cpu_results=False,
                defer_sampling=False,
                tensor_store=None,
            )[0]
            result = _seq_result_dict(out)
            tok = _positive_int(result.get("sampled_token_id"), "sampled_token_id", minimum=0)
            tokens.append(tok)
            last = result
            if tok in stop_ids:
                break
            next_pos = _next_decode_position(op)
            op = dict(first_op)
            op["new_block_ids"] = []
            op["token_ids"] = [tok]
            op["token_source"] = "last_sampled"
            op["pos_range"] = [next_pos, next_pos + 1]
            op["decode_token_count"] = 1
            op["decode_stop_token_ids"] = []

        result = dict(last)
        result["sampled_token_id"] = tokens[-1]
        result["sampled_token_ids"] = tokens
        return result

    # ---- forward ---------------------------------------------------------

    def _forward(
        self,
        model: Any,
        text: "TextBatch",
        request_states: RequestStateTable,
    ) -> tuple[torch.Tensor, list[int]]:
        if self.builder is None or self.kv_pool is None:
            # Self-managing text models (the HF day-zero fallback and the
            # multimodal/interleaved models whose KV is intrinsically coupled to
            # their modality FSM) declare no ``kv_cache_spec``; the system owns no
            # pool for them. They expose their own per-op text logits and the
            # driver still owns the post-model sampler.
            return self._model_owned_kv_forward(model, text)
        ctx = get_forward_context()
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        batched = self.gate is not None and self.gate.batched_capable(
            text, attention_backend_name=ctx.attention_backend_name
        )
        if batched:
            return self._forward_batched(model, text, request_states, ctx, device)
        return self._forward_per_op(model, text, request_states, ctx, device)

    def _model_owned_kv_forward(
        self,
        model: Any,
        text: "TextBatch",
    ) -> tuple[torch.Tensor, list[int]]:
        """Run a self-managing model's per-op text logits and stack them.

        Accepts a batched ``run_text_logits_batch`` or a per-op
        ``run_text_logits``; both return a raw logits tensor per op, coerced to
        one ``[vocab]`` row. Req ids come from the op order, not the tensors.
        """

        ops = list(text.ops)
        outputs = list(model.run_text_logits_batch(ops))
        rows = [self._coerce_logits_row(out) for out in outputs]
        return torch.stack(rows, dim=0), [int(op["req_id"]) for op in ops]

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
    ) -> tuple[torch.Tensor, list[int]]:
        stats = ctx.stats
        start = component_timer_start(stats)
        # Pure decode uses the contiguous-relay override (fast); a mixed
        # extend+decode batch replaces only its last_sampled decode rows by index.
        relay_input_ids = relay_positions = None
        relay_replacements = None
        if text.mode == ForwardMode.DECODE:
            relay_input_ids, relay_positions = self._decode_relay_tensors(text, request_states, device)
        elif text.mode == ForwardMode.MIXED:
            relay_replacements = text_input_id_replacements_from_relays(text, request_states, device)
        record_component_elapsed(stats, "text_decode_relay", start)
        start = component_timer_start(stats)
        padded = self._graph_padded_num_tokens(text, ctx)
        fb = self.builder.build_text(
            text,
            device=device,
            kv_pool=self.kv_pool,
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
            with use_forward_context(
                replace(ctx, attention_metadata=fb.attn_metadata, kv_pool=self.kv_pool)
            ):
                logits = self._run_model_forward(model, input_ids, positions, fb, ctx)
        record_component_elapsed(stats, "text_model_forward", start)
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
    ) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        next_positions: list[tuple[int, int]] = []
        for req_id, tokens, pos_range, op in zip(
            text.req_ids, text.token_ids, text.pos_ranges, text.ops
        ):
            state = request_states.get(int(req_id))
            relay = self._per_op_relay_input(op, tokens, state, device)
            fb = self.builder.build_text_op(
                op=op,
                token_ids=tokens,
                pos_range=pos_range,
                req_id=int(req_id),
                mode=text.mode,
                device=device,
                kv_pool=self.kv_pool,
                request_states=request_states,
                input_ids_override=relay,
            )
            with use_forward_context(
                replace(ctx, attention_metadata=fb.attn_metadata, kv_pool=self.kv_pool)
            ):
                logits = model.forward(fb.input_ids, fb.positions, fb)
            rows.append(logits.reshape(-1, logits.shape[-1])[-1])
            next_positions.append((int(req_id), int(pos_range[1])))
        if text.mode == ForwardMode.DECODE:
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
    ) -> torch.Tensor:
        # System-owned CUDA graphs capture/replay around the graph-unaware model;
        # a miss (or graphs disabled) falls through to the eager forward.
        if self.graph_runner is not None:
            logits = self.graph_runner.maybe_run(model, input_ids, positions, fb, ctx)
            if logits is not None:
                return logits
        return model.forward(input_ids, positions, fb)

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
        if mode == ForwardMode.DECODE:
            return fb.input_ids.reshape(fb.batch_size, 1), fb.positions.reshape(fb.batch_size, 1)
        if mode == ForwardMode.EXTEND:
            lengths = [len(tokens) for tokens in text.token_ids]
            ragged = len(set(lengths)) != 1
            if ragged or fb.has_padding:
                return fb.input_ids, fb.positions
            length = lengths[0]
            return (
                fb.input_ids.reshape(fb.batch_size, length),
                fb.positions.reshape(fb.batch_size, length),
            )
        return fb.input_ids, fb.positions

    def _graph_padded_num_tokens(self, text: "TextBatch", ctx: Any) -> int | None:
        if self.graph_runner is None:
            return None
        return self.graph_runner.padded_num_tokens(text, attention_backend_name=ctx.attention_backend_name)

    def _advance_kv_lengths(self, text: "TextBatch", request_states: RequestStateTable) -> None:
        for req_id, pos_range in zip(text.req_ids, text.pos_ranges):
            request_states.get(int(req_id)).set_kv_length(int(pos_range[1]), lane=_KV_LANE)

    # ---- deferred sampling -----------------------------------------------

    def _publish_logits(
        self,
        ops: list[dict[str, Any]],
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

    # ---- sampling --------------------------------------------------------

    def _sample_logits_batch(
        self,
        ops: list[dict[str, Any]],
        req_ids: list[int],
        logits_batch: torch.Tensor,
        request_states: RequestStateTable,
        stats: ForwardStats | None,
        start: int,
        *,
        defer_cpu_results: bool = False,
    ) -> list[TextTokenOutput | DeferredTextSeqResult]:
        if logits_batch.ndim != 2:
            raise invalid_descriptor("batched text logits rows must form a [batch, vocab] tensor")
        if int(logits_batch.shape[0]) != len(req_ids):
            raise invalid_descriptor("batched text logits row count must match req_ids")
        params: list[dict[str, Any]] = []
        recent: list[list[int] | tuple[int, ...]] = []
        allowed: list[list[int] | tuple[int, ...] | None] = []
        suppress: list[list[int] | tuple[int, ...] | None] = []
        for op, req_id in zip(ops, req_ids):
            state = request_states.get(req_id)
            params.append(dict(state.sampling or {}))
            recent.append(op.get("recent_tokens") or [])
            allowed.append(op.get("allowed_tokens"))
            suppress.append(op.get("suppress_tokens"))
        with profile_range("uniserve.text.apply_sampling"):
            sampling_result = apply_sampling_batched_with_device_tokens(
                logits_batch,
                params,
                recent,
                allowed,
                suppress,
                defer_cpu=defer_cpu_results,
            )
        if isinstance(sampling_result, DeferredBatchedSamplingResult):
            out: list[TextTokenOutput | DeferredTextSeqResult] = []
            for row, req_id in enumerate(req_ids):
                state = request_states.get(req_id)
                relay_token_tensor = sampling_result.device_tokens[row:row + 1]
                self._store_sampled_token_relay(state, token_id=None, token_tensor=relay_token_tensor)
                out.append(
                    DeferredTextSeqResult(
                        req_id=req_id,
                        row=row,
                        state=state,
                        sampling_result=sampling_result,
                        relay_token_tensor=state.decode_relay.token_tensor,
                    )
                )
            record_component_elapsed(stats, "text_sample", start)
            return out

        samples = sampling_result.samples
        out = []
        for row, (req_id, (tok, lp, top)) in enumerate(zip(req_ids, samples)):
            self._store_sampled_token_relay(
                request_states.get(req_id),
                token_id=int(tok),
                token_tensor=sampling_result.device_tokens[row:row + 1],
            )
            out.append(
                TextTokenOutput(
                    req_id=int(req_id),
                    sampled_token_id=tok,
                    sampled_logprob=lp,
                    top_logprobs=top or None,
                )
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
        relay = state.decode_relay.token_tensor
        if not isinstance(relay, torch.Tensor) or relay.dtype != torch.long or relay.device != device:
            raise invalid_descriptor(
                "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
            )
        return relay.reshape(1)

    def _decode_relay_tensors(
        self,
        text: "TextBatch",
        request_states: RequestStateTable,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if text.mode != ForwardMode.DECODE:
            return None, None
        stats = get_forward_context().stats
        relay_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        position_complete = True
        for req_id, tokens, pos_range, op in zip(
            text.req_ids,
            text.token_ids,
            text.pos_ranges,
            text.ops,
        ):
            if len(tokens) != 1:
                _bump_stat(stats, "text_decode_token_relay_misses")
                return None, None
            state = request_states.get(int(req_id))
            expected = int(tokens[0])
            source = str(op.get("token_source") or "wire")
            from_last_sampled = source == "last_sampled"
            if source not in {"wire", "last_sampled"}:
                raise invalid_descriptor(f"unsupported text token_source {source!r}")
            relay_token_id = state.decode_relay.token_id
            relay_token_tensor = state.decode_relay.token_tensor
            if relay_token_id is None and not from_last_sampled:
                _bump_stat(stats, "text_decode_token_relay_misses")
                return None, None
            if not from_last_sampled and int(relay_token_id) != expected:
                _bump_stat(stats, "text_decode_token_relay_misses")
                return None, None
            if not isinstance(relay_token_tensor, torch.Tensor):
                _bump_stat(stats, "text_decode_token_relay_misses")
                if from_last_sampled:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but the relay tensor is unavailable"
                    )
                return None, None
            if relay_token_tensor.dtype != torch.long or relay_token_tensor.device != device:
                _bump_stat(stats, "text_decode_token_relay_misses")
                if from_last_sampled:
                    raise invalid_descriptor(
                        "decode op requested token_source='last_sampled' but the relay tensor is on the wrong device"
                    )
                return None, None
            relay_rows.append(relay_token_tensor.reshape(1))
            expected_position = int(pos_range[0])
            relay_position_id = state.decode_relay.position_id
            relay_position_tensor = state.decode_relay.position_tensor
            if (
                relay_position_id is None
                or int(relay_position_id) != expected_position
                or not isinstance(relay_position_tensor, torch.Tensor)
                or relay_position_tensor.dtype != torch.long
                or relay_position_tensor.device != device
            ):
                position_complete = False
                continue
            position_rows.append(relay_position_tensor.reshape(1))
        _bump_stat(stats, "text_decode_token_relay_hits", len(relay_rows))
        if position_complete and len(position_rows) == len(relay_rows):
            _bump_stat(stats, "text_decode_position_relay_hits", len(position_rows))
            relay_positions = _coalesce_relay_rows(position_rows)
        else:
            _bump_stat(stats, "text_decode_position_relay_misses")
            relay_positions = None
        relay_input_ids = _coalesce_relay_rows(relay_rows)
        return relay_input_ids, relay_positions

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
        next_positions = fb.positions.reshape(-1) + 1
        for row, (req_id, pos_range) in enumerate(zip(text.req_ids, text.pos_ranges)):
            self._store_position_relay(
                request_states.get(int(req_id)),
                position_id=int(pos_range[1]),
                position_tensor=next_positions[row:row + 1],
            )

    @staticmethod
    def _store_sampled_token_relay(
        state: RequestState,
        *,
        token_id: int | None,
        token_tensor: torch.Tensor,
    ) -> None:
        device_token = token_tensor.detach().reshape(1)
        if device_token.device.type == "cuda":
            device_token.record_stream(torch.cuda.current_stream(device_token.device))
        state.decode_relay.token_id = None if token_id is None else int(token_id)
        state.decode_relay.token_tensor = device_token

    @staticmethod
    def _store_position_relay(
        state: RequestState,
        *,
        position_id: int,
        position_tensor: torch.Tensor,
    ) -> None:
        device_position = position_tensor.detach().reshape(1)
        if device_position.device.type == "cuda":
            device_position.record_stream(torch.cuda.current_stream(device_position.device))
        state.decode_relay.position_id = int(position_id)
        state.decode_relay.position_tensor = device_position


def _coalesce_relay_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise invalid_descriptor("decode relay rows must not be empty")
    if len(rows) == 1:
        return rows[0].reshape(-1)
    first = rows[0].reshape(-1)
    if int(first.numel()) != 1:
        return torch.cat([row.reshape(-1) for row in rows], dim=0)
    elem_size = int(first.element_size())
    base_ptr = int(first.data_ptr())
    for idx, row in enumerate(rows):
        flat = row.reshape(-1)
        if (
            int(flat.numel()) != 1
            or flat.dtype != first.dtype
            or flat.device != first.device
            or int(flat.data_ptr()) != base_ptr + idx * elem_size
        ):
            return torch.cat([candidate.reshape(-1) for candidate in rows], dim=0)
    try:
        return first.as_strided((len(rows),), (1,))
    except RuntimeError:
        return torch.cat([candidate.reshape(-1) for candidate in rows], dim=0)


def _can_decode_burst(text: "TextBatch", ops: list[Mapping[str, Any]], *, defer_sampling: bool) -> bool:
    if defer_sampling or text.mode != ForwardMode.DECODE or len(ops) != 1:
        return False
    if any(text.spec_token_ids):
        return False
    try:
        return _positive_int(ops[0].get("decode_token_count") or 1, "decode_token_count") > 1
    except Exception:
        raise


def _seq_result_dict(output: Any) -> dict[str, Any]:
    if hasattr(output, "finalize") and callable(output.finalize):
        finalized = output.finalize()
        if isinstance(finalized, Mapping):
            return dict(finalized)
    if isinstance(output, ForwardOutputBase):
        return dict(output.to_seq_result())
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported text burst output type {type(output).__name__}")


def _positive_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise invalid_descriptor(f"{where} must be an integer >= {minimum}")
    return int(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise invalid_descriptor("decode_stop_token_ids must be a list")
    out: list[int] = []
    for idx, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise invalid_descriptor(f"decode_stop_token_ids[{idx}] must be a non-negative integer")
        out.append(int(item))
    return out


def _next_decode_position(op: Mapping[str, Any]) -> int:
    pos = op.get("pos_range") or [0, 0]
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise invalid_descriptor("decode burst op.pos_range must be [start, end]")
    return _positive_int(pos[1], "decode burst op.pos_range[1]", minimum=0)


def _same_tensor(lhs: Any, rhs: torch.Tensor) -> bool:
    if not isinstance(lhs, torch.Tensor):
        return False
    if lhs.device != rhs.device or lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
        return False
    return int(lhs.data_ptr()) == int(rhs.data_ptr())


def _bump_stat(stats: ForwardStats | None, attr: str, delta: int = 1) -> None:
    if stats is None:
        return
    setattr(stats, attr, int(getattr(stats, attr)) + int(delta))
