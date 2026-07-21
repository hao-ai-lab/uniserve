"""System-owned sequence execution over paged recurrent state.

``SequenceExecutor`` advances sequence operations for any family adapter, while
``Span`` and ``Step`` lower eligible prefill and decode shapes onto the shared
graph primitives. Sequence execution is independent of the workload profile
that produced the operation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch

from uniserve_worker.contracts.attention_plan import PagedVarlenPlan
from uniserve_worker.contracts.forward_context import get_forward_context
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.execution.graph.bucket import padding_blocks
from uniserve_worker.execution.graph.span import Runner as SpanCapture
from uniserve_worker.execution.graph.span import State as SpanState
from uniserve_worker.execution.graph.span import resolve_prepare as resolve_span
from uniserve_worker.execution.graph.step import Inputs as StepInputs
from uniserve_worker.execution.graph.step import Runner as StepCapture
from uniserve_worker.execution.graph.step import State as StepState
from uniserve_worker.execution.graph.step import resolve_prepare as resolve_step
from uniserve_worker.foundation.errors import invalid_descriptor, model_execution_error
from uniserve_worker.foundation.runtime_config import get_execution_config
from uniserve_worker.nn.logits import forced_eos_logits
from uniserve_worker.runtime.forward_stream import build_text_position_indexes
from uniserve_worker.runtime.masks import create_causal_mask
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache, PagedTextCache
from uniserve_worker.runtime.request_state import append_new_block_ids
from uniserve_worker.runtime.tensor_staging import TextTensorStager

if TYPE_CHECKING:
    from uniserve_worker.runtime.kv_pool import PagedKVPool

# ---------------------
# Text caches and token stepping
# ---------------------


def resolve_op_token_ids(op: Mapping[str, Any]) -> list[int]:
    """Resolve one op's input token ids, synchronizing a pending relay if needed.

    ``last_sampled`` ops from the pipelined decode burst carry the sampled
    token only as a device relay tensor (``op['token_tensor']``); the eager
    path materializes it here (the graph path consumes the tensor directly and
    never synchronizes).
    """

    source = str(op.get("token_source") or "wire")
    tokens = list(op.get("token_ids") or [])
    if source != "last_sampled":
        return tokens
    if len(tokens) != 1:
        raise model_execution_error(
            "decode op requested token_source='last_sampled' but does not have exactly one token"
        )
    relay = op.get("token_tensor")
    if isinstance(relay, torch.Tensor):
        return [int(relay.reshape(-1)[0].item())]
    if int(tokens[0]) < 0:
        raise model_execution_error(
            "decode op requested token_source='last_sampled' but carries neither "
            "a relay tensor nor a resolved token id"
        )
    return [int(tokens[0])]


def hydrate_cached_prefix_from_op(cache: Any, op: Mapping[str, Any]) -> None:
    """Adopt host prefix-cache hits into a fresh local paged text cache.

    Prefix-cache reuse crosses the worker boundary as block IDs plus an op
    ``pos_range``.  A newly-created model cache has the right block table but a
    zero logical length, so its first suffix prefill/decode must advance the
    local length to the cached prefix before writing new K/V.
    """

    past = getattr(cache, "past", None)
    if past is None:
        return
    pos = op.get("pos_range")
    if not isinstance(pos, Sequence) or len(pos) < 1:
        return
    start = int(pos[0])
    if start <= 0 or start <= int(getattr(past, "length", 0)):
        return
    if int(getattr(past, "length", 0)) != 0 or int(getattr(cache, "t_index", -1)) >= 0:
        return
    past.ensure_capacity(start)
    past.length = start
    cache.t_index = start - 1


class SequenceCache:
    """Paged recurrent cache and decode state for one sequence branch.

    ``last_token_id`` may be committed either as a resolved CPU int or as a
    pending device relay tensor (the pipelined decode burst commits the tensor
    so the graph replay loop never synchronizes on the token value). The
    property materializes a pending tensor on first read; readers only consult
    it at modality transitions, long after the producing kernel has finished.
    """

    def __init__(self) -> None:
        self.past: Any = None
        self.block_ids: list[int] = []
        self.t_index: int = -1
        self.last_logits: torch.Tensor | None = None
        self._last_token_id: int | None = None
        self._last_token_tensor: torch.Tensor | None = None

    @property
    def last_token_id(self) -> int | None:
        if self._last_token_tensor is not None:
            self._last_token_id = int(self._last_token_tensor.reshape(-1)[0].item())
            self._last_token_tensor = None
        return self._last_token_id

    @last_token_id.setter
    def last_token_id(self, value: int | None) -> None:
        self._last_token_tensor = None
        self._last_token_id = None if value is None else int(value)

    def set_last_token_tensor(self, tensor: torch.Tensor) -> None:
        self._last_token_tensor = tensor


class SequenceAdapter(Protocol):
    """Family boundary required by system sequence and product execution.

    ``SequenceExecutor`` owns cache mutation and operation progression while
    the adapter supplies family-specific embeddings and neural computation.
    """

    @property
    def device(self) -> Any: ...

    @property
    def kv_pool(self) -> "PagedKVPool | None": ...

    @property
    def scratch_pool(self) -> "PagedKVPool | None": ...

    @property
    def residency(self) -> Any: ...

    @property
    def num_layers(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def img_start_id(self) -> int: ...

    # Collaborator methods. Model-specific state types (the request state / image
    # state) are kept ``Any`` here: this system component is duck-typed against the
    # concrete model and must not name model-layer types.
    def sequence_forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        text_only_rope: bool = False,
        causal_paged_update: bool = False,
        return_all_logits: bool = False,
    ) -> Any: ...
    def sequence_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def program_state(self, req_id: int) -> Any: ...


class SequenceExecutor:
    """Own text prefill/decode cache appends for native decoder models."""

    def __init__(
        self,
        owner: SequenceAdapter,
        *,
        image_start_token: str,
    ) -> None:
        self.owner = owner
        self.image_start_token = image_start_token
        self._step_runner: Any | None = None
        self._span_runner: Any | None = None

    def state(self, op: Mapping[str, Any]) -> Any:
        return self.owner.program_state(int(op["req_id"]))

    def run_text_logits_batch(self, ops: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
        """Return one logits row per op, graphing the eligible one-token decode batch.

        The system-owned decode graph adapter owns capture/replay for the eligible
        one-token host-KV decode rows; a ``None`` result (ineligible batch, graph
        miss, or non-fatal fallback) hands the whole batch to the scalar eager
        path so no non-decode / ineligible text semantics ever change.
        """

        op_list = [dict(op) for op in ops]
        if not op_list:
            return []
        graphed = self.try_run_graph_logits_batch(op_list)
        if graphed is not None:
            return graphed
        return [self._run_text_logits_one(op) for op in op_list]

    def try_run_graph_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor] | None:
        """Return graph-produced logits for eligible text rows without eager fallback."""

        op_list = [dict(op) for op in ops]
        if not op_list:
            return []
        if any(bool(op.get("return_all_logits")) for op in op_list):
            return None
        graphed = self._span().maybe_run_batch(self, op_list)
        if graphed is not None:
            return graphed
        return self.try_run_decode_graph_logits_batch(op_list)

    def try_run_decode_graph_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor] | None:
        """Return graph-produced logits for eligible one-token decode rows.

        Unlike :meth:`run_text_logits_batch`, this method does not fall back to
        eager execution. Callers that must account for graph misses explicitly can
        use this seam and keep the existing eager path authoritative.
        """

        op_list = [dict(op) for op in ops]
        if not op_list:
            return []
        return self._step().maybe_run_batch(self, op_list)

    def run_text_logits(self, op: dict[str, Any]) -> torch.Tensor:
        return self._run_text_logits_one(dict(op))

    def _run_text_logits_one(self, op: dict[str, Any]) -> torch.Tensor:
        st = self.state(op)
        self.extend_cache_blocks(st.cond, op)
        tokens = resolve_op_token_ids(op)
        if not tokens:
            return forced_eos_logits(
                int(self.owner.eos_id or 0), device=self.owner.device, batch_shape=(1,)
            )

        if st.cond.past is None:
            self.ensure_host_cache(st.cond)
            hydrate_cached_prefix_from_op(st.cond, op)
            self.prefix_forward_ids(
                st.cond,
                tokens,
                int(op["pos_range"][0]),
                return_all_logits=bool(op.get("return_all_logits")),
            )
        elif len(tokens) == 1:
            hydrate_cached_prefix_from_op(st.cond, op)
            self.append_one(st.cond, int(tokens[0]))
        else:
            hydrate_cached_prefix_from_op(st.cond, op)
            self.append_ids(
                st.cond,
                tokens,
                return_all_logits=bool(op.get("return_all_logits")),
            )
        if bool(op.get("return_all_logits")):
            return st.cond.last_logits
        return st.cond.last_logits[:, -1, :]

    def _step(self) -> Any:
        runner = self._step_runner
        if runner is None:
            runner = Step()
            self._step_runner = runner
        return runner

    def _span(self) -> Any:
        runner = self._span_runner
        if runner is None:
            runner = Span()
            self._span_runner = runner
        return runner

    def extend_cache_blocks(self, cache: SequenceCache, op: Mapping[str, Any]) -> None:
        # Host-issued KV block ids belong only to host-KV caches. Scratch caches
        # use worker-local block ids and must not ingest host ids.
        if cache.past is None or getattr(cache.past, "pool", None) is self.owner.kv_pool:
            append_new_block_ids(cache.block_ids, op.get("new_block_ids"))
        if cache.past is not None and getattr(cache.past, "pool", None) is self.owner.kv_pool:
            cache.past.set_blocks(cache.block_ids)

    def ensure_host_cache(self, cache: SequenceCache) -> None:
        if cache.past is not None:
            return
        if self.owner.kv_pool is None:
            raise model_execution_error("paged KV pool is not initialized")
        cache.past = PagedTextCache(
            self.owner.kv_pool,
            cache.block_ids,
            num_layers=self.owner.num_layers,
        )

    def ensure_scratch_cache(self, cache: SequenceCache) -> None:
        if cache.past is not None:
            return
        if self.owner.scratch_pool is None:
            raise model_execution_error("scratch KV pool is not initialized")
        allocate_blocks = self.owner.residency.require_allocator_for_pool(
            self.owner.scratch_pool,
            label="scratch KV pool",
        )
        cache.past = PagedTextCache(
            self.owner.scratch_pool,
            [],
            num_layers=self.owner.num_layers,
            allocate_blocks=allocate_blocks,
        )

    def prefix_forward_ids(
        self,
        cache: SequenceCache,
        tokens: list[int],
        start: int = 0,
        *,
        return_all_logits: bool = False,
    ) -> None:
        if cache.past is None:
            raise model_execution_error("text prefix requires an initialized paged cache")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        indexes = self.text_indexes(start, len(tokens))
        seq_len = input_ids.shape[1]
        past_len = cache.past.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.owner.device)
        mask[:, :, :, past_len:] = create_causal_mask(seq_len, device=self.owner.device)
        outputs = self.owner.sequence_forward(
            input_ids=input_ids,
            indexes=indexes,
            text_only_rope=True,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
            return_all_logits=return_all_logits,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = outputs.logits
        cache.last_token_id = int(tokens[-1])

    def prefix_from_query(self, query: str) -> SequenceCache:
        cache = SequenceCache()
        self.ensure_scratch_cache(cache)
        build_inputs = getattr(self.owner, "sequence_inputs", None)
        if not callable(build_inputs):
            raise model_execution_error("model does not support worker-side text tokenization")
        ids, indexes, attn = build_inputs(query)
        outputs = self.owner.sequence_forward(
            input_ids=ids,
            indexes=indexes,
            attention_mask=attn,
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = outputs.logits
        cache.last_token_id = int(ids[0, -1].item())
        return cache

    def append_ids(
        self,
        cache: SequenceCache,
        tokens: list[int],
        *,
        return_all_logits: bool = False,
    ) -> None:
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        seq_len = input_ids.shape[1]
        embeds = self.owner.sequence_embeddings(input_ids)
        indexes = self.text_indexes(cache.t_index + 1, seq_len)
        past_len = cache.past.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.owner.device)
        mask[:, :, :, past_len:] = create_causal_mask(seq_len, device=self.owner.device)
        outputs = self.owner.sequence_forward(
            inputs_embeds=embeds,
            indexes=indexes,
            text_only_rope=True,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
            return_all_logits=return_all_logits,
        )
        cache.past = outputs.past_key_values
        cache.t_index += seq_len
        cache.last_logits = outputs.logits
        cache.last_token_id = int(tokens[-1])

    def append_one(self, cache: SequenceCache, token_id: int) -> None:
        ids = torch.tensor([token_id], dtype=torch.long, device=self.owner.device)
        indexes = self.text_indexes(cache.t_index + 1, 1)
        outputs = self.owner.sequence_forward(
            input_ids=ids.unsqueeze(0),
            indexes=indexes,
            past_key_values=cache.past,
            use_cache=True,
            text_only_rope=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index += 1
        cache.last_logits = outputs.logits
        cache.last_token_id = int(token_id)

    def ensure_img_start(self, cache: SequenceCache | None) -> None:
        if cache is None or cache.past is None or cache.last_token_id == self.owner.img_start_id:
            return
        self.append_one(cache, int(self.owner.img_start_id))

    def empty_img_start_prefix(self) -> SequenceCache:
        build_query = getattr(self.owner, "empty_image_start_query", None)
        if not callable(build_query):
            raise model_execution_error("model does not support an empty image-start prefix")
        query = str(build_query(self.image_start_token))
        return self.prefix_from_query(query)

    def text_indexes(self, start: int, seq_len: int) -> torch.Tensor:
        return build_text_position_indexes(int(start), int(seq_len), self.owner.device)


# ---------------------
# Sequence decode/prefill graph runners
# ---------------------

logger = logging.getLogger(__name__)


def _owner_max_context_len(owner: Any, pool: Any) -> int:
    config = getattr(owner, "config", None)
    candidates = (
        getattr(config, "max_position_embeddings", None),
        getattr(config, "model_max_length", None),
        getattr(owner, "max_context_len", None),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return max(
        0, int(getattr(pool, "num_blocks", 0) or 0) * int(getattr(pool, "block_size", 0) or 0)
    )


class _SequenceDecodeGraphPast:
    """Native-language-model ``past_key_values`` bound to a shared graph cache.

    Presents the paged-update protocol the native decoder attention expects, but
    every layer's ``request_cache_for_update`` returns the *same*
    ``BatchedPagedRequestCache`` the captured graph's paged decode plan names as
    its residency cache. RadixAttention's identity check
    (``ctx.attention_plan.residency_cache is kv_cache``) then routes attention
    through the static graph tensors. The finish/cancel hooks are no-ops: the
    graph cache is a transient write target, and persistent request lengths are
    advanced by the adapter only after a successful replay.
    """

    supports_batched_paged = True

    def __init__(self, cache: BatchedPagedRequestCache) -> None:
        self.cache = cache
        self.pool = cache.pool

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.cache.base_len)

    def request_cache_for_update(self, layer_idx: int, n_tokens: int) -> BatchedPagedRequestCache:
        if int(n_tokens) != 1:
            raise invalid_descriptor("sequence decode graph requires exactly one token per row")
        return self.cache

    def finish_layer_update(self, layer_idx: int, n_tokens: int) -> None:
        return None

    def cancel_layer_update(self, layer_idx: int) -> None:
        return None


@dataclass
class _Sidecar:
    """Per-graph-state stable adapter read by the sequence closure."""

    past: _SequenceDecodeGraphPast


@dataclass
class _Row:
    """One eligible one-token decode row's real state + graph replay inputs.

    ``token_id`` is the resolved wire token; pipelined-burst rows instead carry
    the sampled token as ``token_tensor`` (device relay) so replay never
    synchronizes on the token value.
    """

    text_cache: "SequenceCache"
    past_cache: PagedTextCache
    token_id: int | None
    pos: int
    base_len: int
    block_ids: list[int] = field(default_factory=list)
    token_tensor: torch.Tensor | None = None


class _SequencePrefillGraphPast:
    """Native-language-model ``past_key_values`` bound to a prefill graph cache."""

    supports_batched_paged = True

    def __init__(self, cache: BatchedPagedRequestCache) -> None:
        self.cache = cache
        self.pool = cache.pool

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self.cache.base_len)

    def request_cache_for_update(self, layer_idx: int, n_tokens: int) -> BatchedPagedRequestCache:
        if int(n_tokens) <= 0:
            raise invalid_descriptor("sequence prefill graph requires positive token count")
        return self.cache

    def finish_layer_update(self, layer_idx: int, n_tokens: int) -> None:
        return None

    def cancel_layer_update(self, layer_idx: int) -> None:
        self.cache.invalidate_append_plan()


@dataclass
class _PrefillSidecar:
    """Per-graph-state stable adapter for sequence prefill replay."""

    past: _SequencePrefillGraphPast


@dataclass
class _PrefillRow:
    text_cache: "SequenceCache"
    past_cache: PagedTextCache
    tokens: list[int]
    base_len: int
    raw_len: int
    block_ids: list[int]


class Span:
    """Route sequence prefill through the shared prefill graph."""

    def __init__(self) -> None:
        runtime = get_execution_config()
        self._runner = SpanCapture(
            name="sequence",
            default_enabled=runtime.prefill_cuda_graph,
            default_warmup=False,
            default_warmup_token_buckets=runtime.prefill_cuda_graph_warmup_tokens,
            default_warmup_batch_sizes=runtime.cuda_graph_warmup_batches,
            metric_prefix="owner_",
            logger=logger,
        )
        self._sidecars: dict[int, _PrefillSidecar] = {}
        self._graphed_steps = 0

    def maybe_run_batch(
        self,
        driver: "SequenceExecutor",
        ops: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor] | None:
        """Capture/replay a single sequence extend row, or return ``None``."""

        prep = self._prepare(driver, ops)
        if prep is None:
            return None
        rows, inputs, prepare_backend = prep
        logits = self._runner.maybe_run(
            kv_pool=rows[0].past_cache.pool,
            num_blocks=int(rows[0].past_cache.pool.num_blocks),
            num_tokens=int(inputs["num_tokens"]),
            max_kv_tokens=int(inputs["max_kv_tokens"]),
            batch_size=len(rows),
            input_ids=inputs["input_ids"],
            positions=inputs["positions"],
            attention_plan=inputs["attention_plan"],
            last_token_indices=inputs["last_token_indices"],
            raw_num_tokens=sum(row.raw_len for row in rows),
            ctx=get_forward_context(),
            forward_fn=lambda state: self._forward(driver, state),
            prepare_backend=prepare_backend,
        )
        if logits is None:
            return None
        self._graphed_steps += 1
        if self._graphed_steps == 1:
            logger.info(
                "sequence prefill CUDA graph active: captured bucket(s)=%s (shared Span)",
                sorted(self._runner.states),
            )
        return self._commit(rows, logits)

    def _prepare(
        self,
        driver: "SequenceExecutor",
        ops: Sequence[Mapping[str, Any]],
    ) -> tuple[list[_PrefillRow], dict[str, Any], Any] | None:
        if not self._runner.enabled() or not torch.cuda.is_available():
            return None
        if not ops:
            return None
        owner = driver.owner
        pool = getattr(owner, "kv_pool", None)
        model = getattr(owner, "model", None)
        if pool is None or model is None:
            return None
        device = torch.device(str(getattr(owner, "device", "cpu") or "cpu"))
        if device.type != "cuda":
            return None
        if getattr(pool, "is_quantized", False) or not bool(
            getattr(pool, "supports_paged_attention_storage", True)
        ):
            return None
        prepare_backend = resolve_span(
            owner=owner,
            kv_pool=pool,
            attention_preference=getattr(get_forward_context(), "attention_preference", None),
        )
        if prepare_backend is None:
            return None

        validated: list[tuple[dict[str, Any], list[int], Any]] = []
        for raw_op in ops:
            op = dict(raw_op)
            if mode_for_op(str(op.get("kind"))) is not ForwardMode.EXTEND:
                return None
            tokens = resolve_op_token_ids(op)
            if not tokens:
                return None
            cache = driver.state(op).cond
            if cache.past is not None and getattr(cache.past, "pool", None) is not pool:
                return None
            validated.append((op, list(tokens), cache))

        rows: list[_PrefillRow] = []
        for op, tokens, cache in validated:
            driver.extend_cache_blocks(cache, op)
            driver.ensure_host_cache(cache)
            if cache.past is None:
                return None
            hydrate_cached_prefix_from_op(cache, op)
            base_len = int(cache.past.length)
            raw_len = len(tokens)
            cache.past.ensure_capacity(base_len + raw_len)
            rows.append(
                _PrefillRow(
                    text_cache=cache,
                    past_cache=cache.past,
                    tokens=tokens,
                    base_len=base_len,
                    raw_len=raw_len,
                    block_ids=list(cache.past.block_ids),
                )
            )

        raw_num_tokens = sum(row.raw_len for row in rows)
        num_tokens = self._runner.bucket_num_tokens(raw_num_tokens)
        padding_tokens = num_tokens - raw_num_tokens
        max_kv_tokens = self._runner.bucket_kv_tokens(
            max(
                row.base_len + row.raw_len + (padding_tokens if index == len(rows) - 1 else 0)
                for index, row in enumerate(rows)
            ),
            max_context_len=_owner_max_context_len(owner, pool),
        )
        graph_cache = BatchedPagedRequestCache(
            pool,
            [row.block_ids for row in rows],
            [row.base_len for row in rows],
        )
        input_ids = torch.tensor(
            [token for row in rows for token in row.tokens],
            dtype=torch.long,
            device=device,
        )
        positions = torch.cat(
            [
                torch.arange(
                    row.base_len,
                    row.base_len + row.raw_len,
                    dtype=torch.long,
                    device=device,
                )
                for row in rows
            ]
        )
        query_lens_cpu = tuple(row.raw_len for row in rows)
        cache_seqlens_cpu = tuple(row.base_len for row in rows)
        kv_seqlens_cpu = tuple(row.base_len + row.raw_len for row in rows)
        query_lens = torch.tensor(query_lens_cpu, dtype=torch.int32, device=device)
        cache_seqlens = graph_cache.cache_seqlens(device=device)
        kv_seqlens = torch.tensor(kv_seqlens_cpu, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.zeros(len(rows) + 1, dtype=torch.int32, device=device)
        torch.cumsum(query_lens, dim=0, out=cu_seqlens_q[1:])
        cu_seqlens_k = torch.zeros(len(rows) + 1, dtype=torch.int32, device=device)
        torch.cumsum(kv_seqlens, dim=0, out=cu_seqlens_k[1:])
        offsets = []
        offset = 0
        for row in rows:
            offsets.append(offset + row.raw_len - 1)
            offset += row.raw_len
        inputs = {
            "num_tokens": num_tokens,
            "max_kv_tokens": max_kv_tokens,
            "input_ids": input_ids,
            "positions": positions,
            "last_token_indices": torch.tensor(offsets, dtype=torch.long, device=device),
            "attention_plan": PagedVarlenPlan(
                residency_cache=graph_cache,
                block_table=graph_cache.block_table(device=device),
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=cache_seqlens_cpu,
                query_lens=query_lens,
                query_lens_cpu=query_lens_cpu,
                kv_seqlens=kv_seqlens,
                kv_seqlens_cpu=kv_seqlens_cpu,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max(query_lens_cpu),
                max_seqlen_k=max(kv_seqlens_cpu),
                max_context_len=_owner_max_context_len(owner, pool),
                mode=ForwardMode.EXTEND,
            ),
        }
        return rows, inputs, prepare_backend

    def _sidecar_for(self, state: SpanState) -> _PrefillSidecar:
        sidecar = self._sidecars.get(id(state))
        if sidecar is None:
            sidecar = _PrefillSidecar(past=_SequencePrefillGraphPast(state.cache))
            self._sidecars[id(state)] = sidecar
        return sidecar

    def _forward(
        self,
        driver: "SequenceExecutor",
        state: SpanState,
    ) -> torch.Tensor:
        sidecar = self._sidecar_for(state)
        outputs = driver.owner.sequence_forward(
            input_ids=state.input_ids.reshape(1, int(state.num_tokens)),
            cache_position=state.positions,
            past_key_values=sidecar.past,
            use_cache=True,
            text_only_rope=True,
            causal_paged_update=True,
            return_all_logits=True,
        )
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise invalid_descriptor("sequence prefill graph must return batched logits")
        batch = int(state.batch_size)
        indices = state.last_token_indices[:batch].to(device=logits.device, dtype=torch.long)
        return logits.reshape(-1, int(logits.shape[-1])).index_select(0, indices)

    @staticmethod
    def _commit(rows: list[_PrefillRow], logits: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for row_index, row in enumerate(rows):
            row_logits = logits[row_index : row_index + 1]
            row.past_cache.length = row.base_len + row.raw_len
            row.text_cache.t_index = row.base_len + row.raw_len - 1
            row.text_cache.last_token_id = int(row.tokens[-1])
            row.text_cache.last_logits = row_logits.unsqueeze(1)
            outputs.append(row.text_cache.last_logits[:, -1, :])
        return outputs


class Step:
    """Route one-token sequence decode through the shared decode graph."""

    def __init__(self) -> None:
        runtime = get_execution_config()
        self._runner = StepCapture(
            name="sequence",
            default_enabled=runtime.cuda_graph,
            # Lazy capture on the first eligible decode batch: request
            # state is only well-formed at request time, so there is nothing safe
            # to warm up ahead of serving.
            default_warmup=False,
            default_warmup_batch_sizes=(),
            metric_prefix="owner_",
            logger=logger,
        )
        self._sidecars: dict[int, _Sidecar] = {}
        self._graphed_steps = 0
        # Pinned host + reusable device staging for the tiny per-replay input
        # tensors. Building them with pageable ``torch.tensor(..., device=)``
        # would issue an implicit cudaStreamSynchronize per copy, blocking the
        # CPU behind the in-flight replay and serializing decode. Completion
        # events keep each ring slot alive until every copy and replay using it
        # has completed, including wrap-around inside a decode burst.
        self._stager = TextTensorStager(ring_depth=8)

    # -- public entry ---------------------------------------------------------

    def maybe_run_batch(
        self,
        driver: "SequenceExecutor",
        ops: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor] | None:
        """Replay (or capture) the shared decode graph for a one-token decode batch.

        Returns one ``[1, vocab]`` logits row per op on success, or ``None`` when
        the batch is not graph-eligible or the shared runner reports a
        miss/fallback — in which case the driver's eager path owns the batch.
        """

        prep = self._prepare(driver, ops)
        if prep is None:
            return None
        rows, prepare_backend = prep
        batch = len(rows)
        graph_batch = self._runner.resolve_bucket(batch)
        device = rows[0].past_cache.pool.k.device
        pool = rows[0].past_cache.pool
        graph_rows = self._pad_rows(driver, rows, graph_batch, pool)
        slot = self._stager.acquire_slot(device=device)
        host_inputs = StepInputs(
            input_ids=tuple(r.token_id if r.token_id is not None else 0 for r in graph_rows),
            positions=tuple(r.pos for r in graph_rows),
            block_ids_by_row=tuple(tuple(r.block_ids) for r in graph_rows),
            cache_seqlens_cpu=tuple(r.base_len for r in graph_rows),
            kv_seqlens_cpu=tuple(r.base_len + 1 for r in graph_rows),
            token_replacements=tuple(
                (row_idx, r.token_tensor)
                for row_idx, r in enumerate(graph_rows)
                if r.token_tensor is not None
            ),
            max_context_len=_owner_max_context_len(driver.owner, pool),
        )

        try:
            logits = self._runner.maybe_run_host_inputs(
                kv_pool=pool,
                num_blocks=int(pool.num_blocks),
                device=device,
                host_inputs=host_inputs,
                ctx=get_forward_context(),
                forward_fn=lambda state: self._forward(driver, state),
                prepare_backend=prepare_backend,
                staging_slot=slot,
            )
        finally:
            self._stager.mark_slot_submitted(slot, device=device)
        if logits is None:
            return None
        self._graphed_steps += 1
        if self._graphed_steps == 1:
            logger.info(
                "sequence decode CUDA graph active: captured bucket(s)=%s (shared Step)",
                sorted(self._runner.states),
            )
        return self._commit(rows, logits)

    def _pad_rows(
        self,
        driver: "SequenceExecutor",
        rows: list[_Row],
        graph_batch: int,
        pool: Any,
    ) -> list[_Row]:
        graph_batch = int(graph_batch)
        if graph_batch <= len(rows):
            return rows
        padding_block_ids = padding_blocks(pool)
        if not padding_block_ids:
            raise invalid_descriptor(
                "sequence decode graph padded replay requires reserved KV padding blocks"
            )
        block_size = int(getattr(pool, "block_size", 0) or 0)
        needed = graph_batch - len(rows)
        if block_size <= 0 or needed > len(padding_block_ids) * block_size:
            raise invalid_descriptor(
                "sequence decode graph padding exceeds the reserved KV padding blocks"
            )
        base = rows[0]
        padded = list(rows)
        for offset in range(needed):
            padded.append(
                _Row(
                    text_cache=base.text_cache,
                    past_cache=base.past_cache,
                    token_id=0,
                    pos=offset,
                    base_len=offset,
                    block_ids=list(padding_block_ids),
                    token_tensor=None,
                )
            )
        return padded

    # -- eligibility + mutation ----------------------------------------------

    def _prepare(
        self,
        driver: "SequenceExecutor",
        ops: Sequence[Mapping[str, Any]],
    ) -> tuple[list[_Row], Any] | None:
        if not self._runner.enabled() or not torch.cuda.is_available():
            return None
        owner = driver.owner
        model = getattr(owner, "model", None)
        pool = getattr(owner, "kv_pool", None)
        if model is None or pool is None:
            return None
        device = torch.device(str(getattr(owner, "device", "cpu") or "cpu"))
        if device.type != "cuda":
            return None
        # Paged attention backends read current K/V straight out of the pool; a
        # quantized store cannot be consumed through ``layer_cache`` and must not
        # be discovered mid-replay after cache mutation.
        if getattr(pool, "is_quantized", False) or not bool(
            getattr(pool, "supports_paged_attention_storage", True)
        ):
            return None
        prepare_backend = resolve_step(
            owner=owner,
            kv_pool=pool,
            num_blocks=int(pool.num_blocks),
            attention_preference=getattr(get_forward_context(), "attention_preference", None),
        )
        if prepare_backend is None:
            # No re-plannable paged-decode backend or no query-geometry hook: a
            # captured graph would bake its plan, so the eager path stays
            # authoritative.
            return None

        # Validation phase: prove every row is a one-token host-KV decode before
        # mutating any cache block ids or lengths.
        validated: list[
            tuple[Mapping[str, Any], "SequenceCache", int | None, torch.Tensor | None]
        ] = []
        padding_block_ids = set(padding_blocks(pool))
        for op in ops:
            tokens = list(op.get("token_ids") or [])
            if len(tokens) != 1:
                return None
            if mode_for_op(str(op.get("kind"))) != ForwardMode.DECODE:
                return None
            cache = driver.state(op).cond
            if cache.past is None or not isinstance(cache.past, PagedTextCache):
                return None
            if getattr(cache.past, "pool", None) is not pool:
                return None
            token_id: int | None = int(tokens[0])
            token_tensor: torch.Tensor | None = None
            if str(op.get("token_source") or "wire") == "last_sampled":
                relay = op.get("token_tensor")
                if (
                    isinstance(relay, torch.Tensor)
                    and relay.dtype == torch.long
                    and relay.numel() == 1
                    and relay.device.type == "cuda"
                ):
                    # Pipelined-burst row: consume the device relay directly and
                    # commit the token lazily; no synchronize on the token value.
                    token_id, token_tensor = None, relay
                elif token_id is not None and token_id < 0:
                    # Neither a relay tensor nor a resolved id; the eager path
                    # owns the descriptive failure.
                    return None
            validated.append((op, cache, token_id, token_tensor))
        if not validated:
            return None

        # Mutation phase: extend host block ids and ensure one-token capacity. This
        # matches the eager scalar path; the real logical length is advanced only
        # by ``_commit`` after a successful replay.
        rows: list[_Row] = []
        for op, cache, token_id, token_tensor in validated:
            driver.extend_cache_blocks(cache, op)
            driver.ensure_host_cache(cache)
            if cache.past is None:
                return None
            hydrate_cached_prefix_from_op(cache, op)
            base_len = int(cache.past.length)
            cache.past.ensure_capacity(base_len + 1)
            block_ids = list(cache.past.block_ids)
            if padding_block_ids.intersection(block_ids):
                raise invalid_descriptor(
                    "scheduler assigned a reserved sequence decode graph padding block"
                )
            rows.append(
                _Row(
                    text_cache=cache,
                    past_cache=cache.past,
                    token_id=token_id,
                    pos=int(cache.t_index) + 1,
                    base_len=base_len,
                    block_ids=block_ids,
                    token_tensor=token_tensor,
                )
            )
        return rows, prepare_backend

    # -- forward closure + commit --------------------------------------------

    def _sidecar_for(self, state: StepState) -> _Sidecar:
        sidecar = self._sidecars.get(id(state))
        if sidecar is None:
            sidecar = _Sidecar(past=_SequenceDecodeGraphPast(state.cache))
            self._sidecars[id(state)] = sidecar
        return sidecar

    def _forward(self, driver: "SequenceExecutor", state: StepState) -> torch.Tensor:
        sidecar = self._sidecar_for(state)
        outputs = driver.owner.sequence_forward(
            input_ids=state.input_ids,
            cache_position=state.positions.reshape(-1),
            past_key_values=sidecar.past,
            use_cache=True,
            text_only_rope=True,
        )
        # Return ``[batch, vocab]``; Step slices dim 0 back to the
        # actual batch and the commit restores the scalar ``[1, 1, vocab]`` shape.
        return outputs.logits[:, -1, :]

    @staticmethod
    def _commit(rows: list[_Row], logits: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for row_idx, row in enumerate(rows):
            row_logits = logits[row_idx : row_idx + 1]
            row.past_cache.length = row.base_len + 1
            row.text_cache.t_index = row.pos
            if row.token_tensor is not None:
                row.text_cache.set_last_token_tensor(row.token_tensor)
            else:
                row.text_cache.last_token_id = row.token_id
            row.text_cache.last_logits = row_logits.unsqueeze(1)
            out.append(row.text_cache.last_logits[:, -1, :])
        return out
