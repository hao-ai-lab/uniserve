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
* a stable three-axis ``indexes`` sidecar the native language model reads.

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

from ..contracts.forward_context import TextAttentionMetadata, get_forward_context
from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..foundation.errors import invalid_descriptor
from ..foundation.runtime_config import get_worker_config
from ..runtime.paged_text_cache import BatchedPagedRequestCache, PagedTextCache
from .decode_cuda_graph import (
    DecodeCudaGraphRunner,
    TextDecodeGraphState,
    prepare_paged_decode_graph_backend,
    resolve_paged_decode_graph_backend,
)

if TYPE_CHECKING:
    from .interleaved_text_stepper import InterleavedTextCacheDriver, TextCache

logger = logging.getLogger(__name__)

__all__ = ["InterleavedTextDecodeGraphRunner"]


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
    """Per-graph-state stable tensors the interleaved language-model closure reads."""

    indexes: torch.Tensor
    past: _InterleavedDecodeGraphPast


@dataclass
class _Row:
    """One eligible one-token decode row's real state + graph replay inputs."""

    text_cache: "TextCache"
    past_cache: PagedTextCache
    token_id: int
    pos: int
    base_len: int
    block_ids: list[int] = field(default_factory=list)


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
        rows, backend, geometry = prep
        batch = len(rows)
        # Exact-batch only. Paged decode writes current K/V for every graph row,
        # so replaying a larger captured bucket for a smaller batch would write a
        # padded row into physical block 0 (no reserved padding block exists).
        # Only run when the shared runner would use an exact-size graph; otherwise
        # fall back to eager. The single-seq interleaved workload is always batch
        # 1, so this never blocks it — it forecloses a latent hazard for batched
        # interleaved models until the shared runner grows a reserved padding row.
        if self._decode.resolve_bucket(batch) != batch:
            return None
        device = rows[0].past_cache.pool.k.device
        pool = rows[0].past_cache.pool

        input_ids = torch.tensor([[r.token_id] for r in rows], dtype=torch.long, device=device)
        positions = torch.tensor([[r.pos] for r in rows], dtype=torch.long, device=device)
        source_cache = BatchedPagedRequestCache(
            pool,
            [r.block_ids for r in rows],
            [r.base_len for r in rows],
        )
        source_metadata = TextAttentionMetadata(
            cache=source_cache,
            block_table=source_cache.block_table(device=device),
            cache_seqlens=source_cache.cache_seqlens(device=device),
            cache_seqlens_cpu=tuple(r.base_len for r in rows),
            query_lens=torch.ones(batch, dtype=torch.int32, device=device),
            query_lens_cpu=tuple(1 for _ in rows),
            kv_seqlens_cpu=tuple(r.base_len + 1 for r in rows),
            mode=ForwardMode.DECODE,
        )

        num_blocks = int(pool.num_blocks)
        num_q_heads, scale, q_dtype = geometry
        max_indices = num_blocks * batch

        def prepare_backend(state: TextDecodeGraphState, _ctx: Any) -> None:
            sidecar = self._sidecar_for(state)
            # Three-axis text indexes: temporal position per row, spatial rows zero.
            sidecar.indexes[0, :, 0].copy_(state.positions[:, 0], non_blocking=True)
            sidecar.indexes[1:, :, :].zero_()
            prepare_paged_decode_graph_backend(
                state,
                backend=backend,
                num_q_heads=num_q_heads,
                num_kv_heads=int(pool.n_kv),
                head_dim=int(pool.head_dim),
                page_size=int(pool.block_size),
                q_dtype=q_dtype,
                kv_dtype=pool.k.dtype,
                scale=scale,
                max_indices=max_indices,
            )

        logits = self._decode.maybe_run(
            kv_pool=pool,
            num_blocks=num_blocks,
            batch_size=batch,
            input_ids=input_ids,
            positions=positions,
            attention_metadata=source_metadata,
            ctx=get_forward_context(),
            forward_fn=lambda state: self._forward(driver, state),
            prepare_backend=prepare_backend,
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

    # -- eligibility + mutation ----------------------------------------------

    def _prepare(
        self,
        driver: "InterleavedTextCacheDriver",
        ops: Sequence[Mapping[str, Any]],
    ) -> tuple[list[_Row], Any, tuple[int, float, torch.dtype]] | None:
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
        geometry_hook = getattr(owner, "text_decode_graph_query_geometry", None)
        if not callable(geometry_hook):
            return None
        backend = resolve_paged_decode_graph_backend(
            getattr(get_forward_context(), "attention_backend_name", None)
        )
        if backend is None:
            return None

        # Validation phase: prove every row is a one-token host-KV decode before
        # mutating any cache block ids or lengths.
        validated: list[tuple[Mapping[str, Any], "TextCache", int]] = []
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
            validated.append((op, cache, int(tokens[0])))
        if not validated:
            return None

        geometry = geometry_hook()

        # Mutation phase: extend host block ids and ensure one-token capacity. This
        # matches the eager scalar path; the real logical length is advanced only
        # by ``_commit`` after a successful replay.
        rows: list[_Row] = []
        for op, cache, token_id in validated:
            driver.extend_cache_blocks(cache, op)
            driver.ensure_host_cache(cache)
            if cache.past is None:
                return None
            base_len = int(cache.past.length)
            cache.past.ensure_capacity(base_len + 1)
            rows.append(
                _Row(
                    text_cache=cache,
                    past_cache=cache.past,
                    token_id=token_id,
                    pos=int(cache.t_index) + 1,
                    base_len=base_len,
                    block_ids=list(cache.past.block_ids),
                )
            )
        return rows, backend, geometry

    # -- forward closure + commit --------------------------------------------

    def _sidecar_for(self, state: TextDecodeGraphState) -> _Sidecar:
        sidecar = self._sidecars.get(id(state))
        if sidecar is None:
            indexes = torch.zeros(
                (3, int(state.batch_size), 1),
                dtype=torch.long,
                device=state.input_ids.device,
            )
            sidecar = _Sidecar(indexes=indexes, past=_InterleavedDecodeGraphPast(state.cache))
            self._sidecars[id(state)] = sidecar
        return sidecar

    def _forward(self, driver: "InterleavedTextCacheDriver", state: TextDecodeGraphState) -> torch.Tensor:
        sidecar = self._sidecar_for(state)
        outputs = driver.owner.model.language_model(
            input_ids=state.input_ids,
            indexes=sidecar.indexes,
            past_key_values=sidecar.past,
            use_cache=True,
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
            row.text_cache.last_token_id = row.token_id
            row.text_cache.last_logits = row_logits.unsqueeze(1)
            out.append(row.text_cache.last_logits[:, -1, :])
        return out
