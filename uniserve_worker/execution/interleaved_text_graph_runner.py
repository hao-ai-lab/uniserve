"""System-owned CUDA-graph adapter for interleaved one-token text decode.

Native interleaved text+image decoders drive text decode through
:class:`InterleavedTextCacheDriver` rather than the thin ``TextGraphRunner``
stack, because their KV is coupled to a modality FSM. This
adapter lets that one-token decode use the **shared** decode CUDA graph instead
of a model-local one: it translates the driver's per-op decode into
:class:`DecodeCudaGraphRunner`, and owns *no* graph lifecycle itself. Capture,
replay, the capture pool, static input buffers, graph stats, misses, fallbacks,
and bucket disabling all stay in ``DecodeCudaGraphRunner`` / ``_GraphRunnerBase``.

The adapter contributes only the interleaved-specific glue:

* eligible one-token host-KV decode rows resolved from the op batch,
* a tiny "past" adapter bridging the native language model's paged-cache update
  protocol onto the shared :class:`BatchedPagedRequestCache`, and
* cache-position inputs that the native language model resolves into pure-text
  three-axis RoPE indexes inside the captured forward.

Correctness across sequence growth and across requests rests on graph-aware
FlashInfer decode planning: the prepare hook refills the decode wrapper's static
plan buffers from the freshly-copied static block table + cache lengths before
every capture/replay, so a single captured graph adapts to growing lengths and
to entirely different block ids — with no per-length recapture and no baked page
ids. When that graph-capable backend is unavailable the adapter returns ``None``
and the driver's eager path stays authoritative (it never captures a graph whose
plan would silently go stale).

Model-neutral: the concrete model is only touched through the duck-typed
``InterleavedTextCacheDriver`` / ``InterleavedModelOwner`` surface plus an
optional ``text_decode_graph_query_geometry`` owner hook.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..contracts.forward_context import get_forward_context
from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..foundation.errors import invalid_descriptor
from ..foundation.runtime_config import get_worker_config
from ..runtime.paged_text_cache import BatchedPagedRequestCache, PagedTextCache
from ..runtime.tensor_staging import TextTensorStager
from .decode_cuda_graph import (
    DecodeCudaGraphRunner,
    TextDecodeGraphHostInputs,
    TextDecodeGraphState,
    resolve_paged_decode_graph_prepare,
)
from .interleaved_text_stepper import hydrate_cached_prefix_from_op

if TYPE_CHECKING:
    from .interleaved_text_stepper import InterleavedTextCacheDriver, TextCache

logger = logging.getLogger(__name__)

__all__ = ["InterleavedTextDecodeGraphRunner"]


def _padding_block_id(owner: Any, pool: Any) -> int | None:
    hook = getattr(owner, "interleaved_decode_graph_padding_block_id", None)
    if not callable(hook):
        return None
    raw = hook(pool)
    if raw is None:
        return None
    block_id = int(raw)
    num_blocks = int(getattr(pool, "num_blocks", 0) or 0)
    if block_id < 0 or block_id >= num_blocks:
        raise invalid_descriptor("reserved interleaved decode graph padding block is out of range")
    return block_id


def _owner_max_context_len(owner: Any, pool: Any) -> int:
    config = getattr(owner, "config", None)
    candidates = (
        getattr(config, "max_position_embeddings", None),
        getattr(config, "model_max_length", None),
        getattr(owner, "max_context_len", None),
    )
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return max(0, int(getattr(pool, "num_blocks", 0) or 0) * int(getattr(pool, "block_size", 0) or 0))


class _InterleavedDecodeGraphPast:
    """Native-language-model ``past_key_values`` bound to a shared graph cache.

    Presents the paged-update protocol the native decoder attention expects, but
    every layer's ``request_cache_for_update`` returns the *same*
    ``BatchedPagedRequestCache`` the captured graph's ``TextAttentionMetadata``
    already points at. RadixAttention's identity check
    (``ctx.attention_metadata.cache is kv_cache``) then routes attention through
    the static graph tensors. The finish/cancel hooks are no-ops: the graph cache
    is a transient write target, and persistent request lengths are advanced by
    the adapter only after a successful replay.
    """

    supports_batched_paged = True

    def __init__(self, cache: BatchedPagedRequestCache) -> None:
        self.cache = cache
        self.pool = cache.pool

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.cache.base_len)

    def request_cache_for_update(self, layer_idx: int, n_tokens: int) -> BatchedPagedRequestCache:
        if int(n_tokens) != 1:
            raise invalid_descriptor("interleaved decode graph supports one-token decode only")
        return self.cache

    def finish_layer_update(self, layer_idx: int, n_tokens: int) -> None:
        return None

    def cancel_layer_update(self, layer_idx: int) -> None:
        return None


@dataclass
class _Sidecar:
    """Per-graph-state stable adapter the interleaved language-model closure reads."""

    past: _InterleavedDecodeGraphPast


@dataclass
class _Row:
    """One eligible one-token decode row's real state + graph replay inputs.

    ``token_id`` is the resolved wire token; pipelined-burst rows instead carry
    the sampled token as ``token_tensor`` (device relay) so replay never
    synchronizes on the token value.
    """

    text_cache: "TextCache"
    past_cache: PagedTextCache
    token_id: int | None
    pos: int
    base_len: int
    block_ids: list[int] = field(default_factory=list)
    token_tensor: torch.Tensor | None = None


class InterleavedTextDecodeGraphRunner:
    """Route interleaved one-token text decode through the shared decode graph."""

    def __init__(self) -> None:
        runtime = get_worker_config()
        self._decode = DecodeCudaGraphRunner(
            name="interleaved_text",
            default_enabled=runtime.cuda_graph,
            # Lazy capture on the first eligible decode batch: interleaved request
            # state is only well-formed at request time, so there is nothing safe
            # to warm up ahead of serving.
            default_warmup=False,
            default_warmup_batch_sizes=(),
            metric_prefix="text_",
            logger=logger,
        )
        self._sidecars: dict[int, _Sidecar] = {}
        self._graphed_steps = 0
        # Pinned host + reusable device staging for the tiny per-replay input
        # tensors. Building them with pageable ``torch.tensor(..., device=)``
        # would issue an implicit cudaStreamSynchronize per copy, blocking the
        # CPU behind the in-flight replay and serializing decode. The ring
        # keeps a slot's buffers alive across the one-step CPU/GPU overlap the
        # pipelined burst runs at.
        self._stager = TextTensorStager(ring_depth=3)

    # -- public entry ---------------------------------------------------------

    def maybe_run_batch(
        self,
        driver: "InterleavedTextCacheDriver",
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
        graph_batch = self._decode.resolve_bucket(batch)
        device = rows[0].past_cache.pool.k.device
        pool = rows[0].past_cache.pool
        graph_rows = self._pad_rows(driver, rows, graph_batch, pool)
        slot = self._stager.next_slot()
        host_inputs = TextDecodeGraphHostInputs(
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

        logits = self._decode.maybe_run_host_inputs(
            kv_pool=pool,
            num_blocks=int(pool.num_blocks),
            device=device,
            host_inputs=host_inputs,
            ctx=get_forward_context(),
            forward_fn=lambda state: self._forward(driver, state),
            prepare_backend=prepare_backend,
            staging_slot=slot,
        )
        if logits is None:
            return None
        self._graphed_steps += 1
        if self._graphed_steps == 1:
            logger.info(
                "interleaved text decode CUDA graph active: captured bucket(s)=%s (shared DecodeCudaGraphRunner)",
                sorted(self._decode.states),
            )
        return self._commit(rows, logits)

    def _pad_rows(
        self,
        driver: "InterleavedTextCacheDriver",
        rows: list[_Row],
        graph_batch: int,
        pool: Any,
    ) -> list[_Row]:
        graph_batch = int(graph_batch)
        if graph_batch <= len(rows):
            return rows
        padding_block_id = _padding_block_id(driver.owner, pool)
        if padding_block_id is None:
            raise invalid_descriptor(
                "interleaved decode graph padded replay requires a reserved KV padding block"
            )
        block_size = int(getattr(pool, "block_size", 0) or 0)
        needed = graph_batch - len(rows)
        if block_size <= 0 or needed > block_size:
            raise invalid_descriptor(
                "interleaved decode graph padding exceeds the reserved KV padding block"
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
                    block_ids=[padding_block_id],
                    token_tensor=None,
                )
            )
        return padded

    # -- eligibility + mutation ----------------------------------------------

    def _prepare(
        self,
        driver: "InterleavedTextCacheDriver",
        ops: Sequence[Mapping[str, Any]],
    ) -> tuple[list[_Row], Any] | None:
        if not self._decode.enabled() or not torch.cuda.is_available():
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
        prepare_backend = resolve_paged_decode_graph_prepare(
            owner=owner,
            kv_pool=pool,
            num_blocks=int(pool.num_blocks),
            attention_backend_name=getattr(
                get_forward_context(), "attention_backend_name", None
            ),
        )
        if prepare_backend is None:
            # No re-plannable paged-decode backend or no query-geometry hook: a
            # captured graph would bake its plan, so the eager path stays
            # authoritative.
            return None

        # Validation phase: prove every row is a one-token host-KV decode before
        # mutating any cache block ids or lengths.
        validated: list[tuple[Mapping[str, Any], "TextCache", int | None, torch.Tensor | None]] = []
        padding_block_id = _padding_block_id(owner, pool)
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
                elif token_id < 0:
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
            if padding_block_id is not None and padding_block_id in block_ids:
                raise invalid_descriptor(
                    "scheduler assigned the reserved interleaved decode graph padding block"
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

    def _sidecar_for(self, state: TextDecodeGraphState) -> _Sidecar:
        sidecar = self._sidecars.get(id(state))
        if sidecar is None:
            sidecar = _Sidecar(past=_InterleavedDecodeGraphPast(state.cache))
            self._sidecars[id(state)] = sidecar
        return sidecar

    def _forward(self, driver: "InterleavedTextCacheDriver", state: TextDecodeGraphState) -> torch.Tensor:
        sidecar = self._sidecar_for(state)
        outputs = driver.owner.interleaved_text_forward(
            input_ids=state.input_ids,
            cache_position=state.positions.reshape(-1),
            past_key_values=sidecar.past,
            use_cache=True,
            text_only_rope=True,
        )
        # Return ``[batch, vocab]``; DecodeCudaGraphRunner slices dim 0 back to the
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
