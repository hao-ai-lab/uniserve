"""Geometry-keyed CUDA-graph replay for batched text-image denoise steps.

The batched denoise velocity prediction (``TextImageDenoiseOps._predict_v_batched``)
launches the full generation-tower forward — thousands of kernels plus, under
tensor parallelism, two NCCL all-reduces per layer — for every one of ~50 flow
steps. Image embeds, timestep, latent, positions, and paged-KV side tables are
graph inputs; model weights and tensor geometry are stable. The runner captures
one graph per bounded geometry and rebinds it to live request caches before each
replay.

Correctness rests on three pillars:

* **Graph key.** Device, dtype, row count, block-table bucket, token geometry,
  latent geometry, and output mode define graph identity. Cache IDs, block IDs,
  and prefix lengths are copied into stable side-table buffers per replay.
* **Bounded graph microbatches.** Large mixed batches are partitioned into
  graph-sized row groups and concatenated in input order. This bounds resident
  activation pools without changing request concurrency or branch semantics.
* **Graph-capable attention backend.** The dispatcher-winner probe guarantees
  capture never swaps in a backend the eager path would not use. Backends with
  mutable wrapper plans bind a graph-scoped exclusive wrapper (see
  ``bind_paged_prefill_graph_wrapper``); direct paged-varlen kernels declare
  graph safety through their attention capabilities.
* **Private capture pool.** Geometry graphs can replay in any order, so each
  capture owns an isolated allocation pool rather than aliasing static graph
  storage across shapes.

The production runner is always enabled. Unsupported shapes or capture failures
are returned to the strict graph executor as misses and fail closed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

import torch

import uniserve_worker.ops as ops

from ....contracts.forward_context import get_forward_context, use_forward_context
from ....contracts.forward_mode import ForwardMode
from ....nn.attention import RadixAttention
from ....runtime.paged_text_cache import BatchedPagedTextCache
from .base import GraphEvent, _GraphRunnerBase, record_graph_stats

if TYPE_CHECKING:
    from ...interleaved_image_denoise import DenoiseRow

logger = logging.getLogger(__name__)

__all__ = [
    "DenoiseStepGraphRunner",
    "maybe_run_denoise_step_graph",
]

# Consecutive capture/replay failures before the runner hard-disables itself.
_MAX_FAILURES = 2

_RUNNER_ATTR = "_denoise_step_graph_runner"

# Keep graph-resident activation pools bounded while retaining useful GEMM
# batching: one graph microbatch covers two full three-branch CFG requests.
_MAX_GRAPH_ROWS = 6


class _DenoiseGraphMetadata:
    """Per-graph identity sentinel published as ``ctx.attention_metadata``.

    It intentionally omits ``cache`` and ``mode`` so attention stays on the
    transient paged-varlen path. Stable plan tensors are refreshed before replay,
    while object identity routes FlashInfer to the graph-scoped wrapper.
    """

    __slots__ = ("block_table", "cu_seqlens_k", "cu_seqlens_q", "__weakref__")

    def __init__(
        self,
        *,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> None:
        self.block_table = block_table
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k


class _GraphBackendUnplanned(RuntimeError):
    """Capture completed but the exclusive wrapper was never planned."""


@dataclass
class DenoiseStepGraphState:
    """Static buffers and one captured denoise geometry graph."""

    key: tuple[Any, ...]
    graph: torch.cuda.CUDAGraph
    image_embeds: torch.Tensor          # [rows, tokens, hidden] static input
    t: torch.Tensor                     # timestep static input (shape of step.t)
    z: torch.Tensor                     # [rows, tokens, latent] static input
    indexes: torch.Tensor               # [3, rows, tokens] static input
    cache: BatchedPagedTextCache        # batched view over the rows' caches
    request_cache: Any                  # stable graph-owned paged side tables
    metadata: _DenoiseGraphMetadata     # wrapper-routing sentinel (kept alive here)
    backend: Any
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    scale: float
    image_token_num: int
    image_size: tuple[int, int]
    release_backend: Any = None         # callable dropping the exclusive wrapper
    logits: Any = None                  # (velocity, hidden|None) written by capture/replay


class DenoiseStepGraphRunner(_GraphRunnerBase):
    """Own bounded denoise-step CUDA graphs and refreshable request inputs."""

    def __init__(
        self,
        *,
        name: str = "denoise_step",
        default_enabled: bool | None = None,
        logger: Any = logger,
    ) -> None:
        self.name = str(name)
        self.default_enabled = True if default_enabled is None else bool(default_enabled)
        self.default_warmup = False
        self.metric_prefix = "denoise_"
        self.logger = logger
        self.states: dict[tuple[Any, ...], DenoiseStepGraphState] = {}
        self.disabled: set[tuple[Any, ...]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}
        self._failures = 0
        self._hard_disabled = False
        self._backend_ineligible = False
        self._replays = 0

    # -- gating ----------------------------------------------------------------

    def enabled(self) -> bool:
        return self.default_enabled and not self._hard_disabled and not self._backend_ineligible

    def capture_pool(self) -> Any:
        # Independent geometry graphs can replay in any order. Keep their static
        # allocations isolated instead of sharing addresses across graph shapes.
        return None

    # -- public entry ------------------------------------------------------------

    def maybe_run_rows(
        self,
        owner: Any,
        rows: "Sequence[DenoiseRow]",
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Capture or replay the denoise step for ``rows``; ``None`` is a miss.

        Outputs are cloned per replay so nothing the caller retains (velocity
        slices, TeaCache hidden records) aliases the graph's static output
        buffers, which the next replay overwrites.
        """

        if not self.enabled() or not torch.cuda.is_available():
            return None
        if len(rows) > _MAX_GRAPH_ROWS:
            chunks = []
            for start in range(0, len(rows), _MAX_GRAPH_ROWS):
                chunk = self.maybe_run_rows(
                    owner,
                    rows[start : start + _MAX_GRAPH_ROWS],
                    return_hidden=return_hidden,
                )
                if chunk is None:
                    return None
                chunks.append(chunk)
            if return_hidden:
                velocities, hidden = zip(*chunks, strict=True)
                return torch.cat(velocities, dim=0), torch.cat(hidden, dim=0)
            return torch.cat(chunks, dim=0)
        if not self._ensure_transient_capacity(rows):
            return None
        key = self._rows_key(rows, return_hidden)
        ctx = get_forward_context()
        tokens = len(rows) * int(rows[0].img.token_h) * int(rows[0].img.token_w)
        if key is None or key in self.disabled:
            self._record(ctx, GraphEvent.MISS, tokens)
            return None
        if key not in self.states:  # capture path resolves the winner backend
            backend = self._resolve_graph_backend(ctx, rows)
            if backend is None:
                # Dispatcher-winner probe failed: the backend eager would pick
                # cannot host the captured graph path. Config-wide, so disable
                # the runner rather than accumulate geometry-specific miss keys.
                self._backend_ineligible = True
                if self.logger is not None:
                    self.logger.info(
                        "%s CUDA graph disabled: eager attention winner cannot host a "
                        "graph-scoped prefill plan",
                        self.name,
                    )
                self._record(ctx, GraphEvent.MISS, tokens)
                return None
        else:
            backend = None  # replay path never touches the backend

        out = self._capture_or_replay(
            key=key,
            ctx=ctx,
            capture=lambda: self._capture(owner, rows, key, ctx, backend, return_hidden),
            copy_inputs=lambda state: self._copy_inputs(state, rows),
            replay=self._replay,
            record=lambda event: self._record(ctx, event, tokens),
            disable=lambda exc: self._disable(key, exc),
            capture_metric=f"{self.metric_prefix}step_graph_capture",
            input_copy_metric=f"{self.metric_prefix}step_graph_input_copy",
            replay_metric=f"{self.metric_prefix}step_graph_replay_launch",
            after_copy=self._prepare_backend,
            after_copy_metric=f"{self.metric_prefix}step_graph_attention_prepare",
        )
        if out is None:
            return None
        self._replays += 1
        if self._replays == 1 and self.logger is not None:
            self.logger.info(
                "%s CUDA graph active: captured batched denoise step (rows=%d, tokens=%d)",
                self.name,
                len(rows),
                tokens,
            )
        velocity, hidden = out
        if return_hidden:
            assert hidden is not None
            return velocity.clone(), hidden.clone()
        return velocity.clone()

    def clear(self) -> None:
        """Release every resident graph and graph-scoped backend binding."""

        if self.states and torch.cuda.is_available():
            device = next(iter(self.states.values())).image_embeds.device
            torch.cuda.synchronize(device)
        for state in self.states.values():
            if callable(state.release_backend):
                state.release_backend()
        self.states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- keying / stats ------------------------------------------------------------

    @staticmethod
    def _ensure_transient_capacity(rows: "Sequence[DenoiseRow]") -> bool:
        """Extend each row cache for its transient tokens *before* keying.

        ``request_cache_for_transient`` grows ``block_ids`` on first use; keying
        beforehand would snapshot the pre-extension blocks on the capture step
        and then miss (and leak a recapture) on every later step. This mirrors
        the extension the forward performs, idempotently.
        """

        try:
            for row in rows:
                n_tokens = int(row.img.token_h) * int(row.img.token_w)
                row.cache.ensure_capacity(int(row.cache.length) + n_tokens)
        except Exception:
            return False
        return True

    @staticmethod
    def _rows_key(rows: "Sequence[DenoiseRow]", return_hidden: bool) -> tuple[Any, ...] | None:
        first = rows[0]
        img = first.img
        embeds = first.step.extra["image_embeds"]
        latent = first.step.latent
        t = first.step.t
        if embeds.device.type != "cuda":
            return None
        pool = getattr(first.cache, "pool", None)
        if pool is None:
            return None
        max_blocks = 0
        for row in rows:
            cache = row.cache
            if getattr(cache, "pool", None) is not pool:
                return None
            row_embeds = row.step.extra["image_embeds"]
            row_latent = row.step.latent
            if (
                row_embeds.shape != embeds.shape
                or row_embeds.dtype != embeds.dtype
                or row_embeds.device != embeds.device
                or row_latent.shape != latent.shape
                or row_latent.dtype != latent.dtype
                or row.indexes.shape != first.indexes.shape
                or row.indexes.dtype != first.indexes.dtype
            ):
                return None
            max_blocks = max(max_blocks, len(cache.block_ids))
        block_width = 1 << max(0, int(max_blocks - 1).bit_length())
        return (
            id(pool),
            len(rows),
            block_width,
            str(embeds.device),
            str(embeds.dtype),
            tuple(int(dim) for dim in embeds.shape),
            str(latent.dtype),
            tuple(int(dim) for dim in latent.shape),
            str(t.dtype),
            tuple(int(dim) for dim in t.shape),
            str(first.indexes.dtype),
            tuple(int(dim) for dim in first.indexes.shape),
            int(img.token_h),
            int(img.token_w),
            int(img.height),
            int(img.width),
            bool(return_hidden),
        )

    def _record(self, ctx: Any, event: GraphEvent, tokens: int) -> None:
        record_graph_stats(
            ctx,
            event,
            mode=ForwardMode.DENOISE,
            unpadded_tokens=int(tokens),
            padded_tokens=int(tokens),
        )

    def _disable(self, key: tuple[Any, ...], exc: BaseException) -> None:
        self.disabled.add(key)
        state = self.states.pop(key, None)
        if state is not None and callable(state.release_backend):
            try:
                state.release_backend()
            except Exception:  # pragma: no cover - defensive release
                pass
        self._failures += 1
        if self._failures >= _MAX_FAILURES:
            self._hard_disabled = True
        if self.logger is not None:
            self.logger.warning(
                "disabling %s CUDA graph (%s failure(s)%s): %s",
                self.name,
                self._failures,
                "; runner hard-disabled" if self._hard_disabled else "",
                exc,
            )

    # -- backend probe ---------------------------------------------------------

    def _resolve_graph_backend(self, ctx: Any, rows: "Sequence[DenoiseRow]") -> Any | None:
        """First eligible attention provider for the transient varlen denoise step.

        Mirrors the transient paged-varlen probe the eager forward runs, but
        resolves the *winner* rather than "any provider": capture must never
        swap in a backend the eager path would not use, and the winner must
        expose the graph-scoped prefill wrapper surface so the captured plan
        cannot be invalidated by other requests.
        """

        try:
            first = rows[0]
            img = first.img
            embeds = first.step.extra["image_embeds"]
            pool = first.cache.pool
            n_tokens = int(img.token_h) * int(img.token_w)
            cache = BatchedPagedTextCache([row.cache for row in rows])
            view = cache.request_cache_for_transient(0, n_tokens)
            head_dim = int(pool.head_dim)
            q_shape_probe = embeds.new_empty((len(rows), 1, n_tokens, head_dim))
            transient = RadixAttention._transient_varlen_metadata(view, q_shape_probe)
            if transient is None:
                return None
            (_query_lens, block_table, _seqlens, cu_q, cu_k, max_q, max_k) = transient
            k_cache, v_cache = pool.layer_cache(0)
            q_probe = embeds.new_empty((1, 1, head_dim))
            preferred = getattr(ctx, "attention_backend_name", None) or "torch_sdpa"
            override = RadixAttention._attention_override(ctx, preferred)
            req = ops.AttentionReq(
                q=q_probe,
                k=k_cache,
                v=v_cache,
                regime=ops.AttentionRegime.EXTEND,
                causal=True,
                scale=1.0,
                kv_cache=view,
                ctx=ctx,
                block_table=block_table,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
            )
            for provider in ops.attention_dispatcher().ordered(override):
                try:
                    if not provider.can_run(req):
                        continue
                except Exception:
                    continue
                backend = getattr(provider, "backend", None)
                if backend is None:
                    backend = getattr(req, "backend", None) or getattr(ctx, "attention_backend", None)
                if backend is not None and self._backend_can_host_graph(backend):
                    return backend
                return None
        except Exception:
            if self.logger is not None:
                self.logger.debug("%s graph backend probe failed", self.name, exc_info=True)
        return None

    @staticmethod
    def _backend_can_host_graph(backend: Any) -> bool:
        bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
        release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
        if callable(bind) and callable(release):
            return True
        try:
            caps = backend.capabilities()
        except Exception:
            return False
        return bool(getattr(caps, "paged_varlen_cuda_graph", False))

    # -- capture / replay ------------------------------------------------------

    def _capture(
        self,
        owner: Any,
        rows: "Sequence[DenoiseRow]",
        key: tuple[Any, ...],
        ctx: Any,
        backend: Any,
        return_hidden: bool,
    ) -> DenoiseStepGraphState:
        first = rows[0]
        img = first.img
        device = first.step.extra["image_embeds"].device
        cache = BatchedPagedTextCache(
            [row.cache for row in rows],
            block_table_width=int(key[2]),
        )
        request_cache = cache.request_cache_for_transient(
            0, int(img.token_h) * int(img.token_w)
        )
        pool = request_cache.pool
        head_dim = int(pool.head_dim)
        q_shape_probe = first.step.extra["image_embeds"].new_empty(
            (len(rows), 1, int(img.token_h) * int(img.token_w), head_dim)
        )
        transient = RadixAttention._transient_varlen_metadata(request_cache, q_shape_probe)
        if transient is None:
            raise RuntimeError("denoise graph could not build paged side-table inputs")
        (_query_lens, block_table, _cache_lens, cu_q, cu_k, _max_q, _max_k) = transient
        metadata = _DenoiseGraphMetadata(
            block_table=block_table,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )
        geometry = getattr(owner, "text_decode_graph_query_geometry", None)
        if callable(geometry):
            num_q_heads, scale, _q_dtype = geometry()
        else:
            first_attention = getattr(owner, "attn", None)
            if first_attention is None:
                raise RuntimeError("denoise graph owner does not expose query geometry")
            num_q_heads = int(first_attention.num_heads)
            scale = float(first_attention.scale)
        state = DenoiseStepGraphState(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            image_embeds=torch.cat(
                [row.step.extra["image_embeds"] for row in rows], dim=0
            ).contiguous(),
            t=first.step.t.detach().clone(),
            z=torch.cat([row.step.latent for row in rows], dim=0).contiguous(),
            indexes=torch.stack([row.indexes for row in rows], dim=1).contiguous(),
            cache=cache,
            request_cache=request_cache,
            metadata=metadata,
            backend=backend,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(pool.n_kv),
            head_dim=head_dim,
            page_size=int(pool.block_size),
            scale=float(scale),
            image_token_num=int(img.token_h) * int(img.token_w),
            image_size=(int(img.width), int(img.height)),
        )
        bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
        release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
        if callable(bind) and callable(release):
            bind(metadata, device=device)
            state.release_backend = lambda: release(metadata)

        # Capture under a context whose ``attention_metadata`` is this graph's
        # sentinel (routes FlashInfer to the exclusive wrapper) and with stats
        # detached, mirroring the shared decode/prefill graph capture discipline.
        graph_ctx = replace(ctx, attention_metadata=metadata, stats=None)

        def run() -> tuple[torch.Tensor, torch.Tensor | None]:
            # Capture with the *requested* ``return_hidden`` (it is part of the
            # graph key): requesting the pre-norm hidden stream switches the
            # decoder's final-norm tail from the fused in-place add+rmsnorm to
            # an unfused add-then-norm, which is not bit-identical — the graph
            # must take exactly the path the eager forward takes. All outputs
            # are cloned inside the capture so the returned buffers are
            # dedicated graph outputs, not views of reusable intermediates.
            with use_forward_context(graph_ctx):
                out = owner.interleaved_image_predict_velocity(
                    state.image_embeds,
                    state.indexes,
                    {"full_attention": None},
                    state.cache,
                    state.t,
                    state.z,
                    image_token_num=state.image_token_num,
                    image_size=state.image_size,
                    return_hidden=return_hidden,
                )
                if return_hidden:
                    velocity, hidden = out
                    return velocity.clone(), hidden.clone()
                return out.clone(), None

        try:
            self._capture_graph_state(
                device=device,
                state=state,
                run=run,
                copy_inputs=lambda capture_state: self._copy_inputs(capture_state, rows),
            )
            planned = getattr(backend, "paged_prefill_graph_wrapper_planned", None)
            if callable(planned) and not planned(metadata):
                raise _GraphBackendUnplanned(
                    "captured denoise forward did not plan the graph-scoped prefill "
                    "wrapper; the dispatcher routed attention elsewhere"
                )
        except BaseException:
            release = state.release_backend
            state.release_backend = None
            if callable(release):
                release()
            raise
        return state

    @staticmethod
    def _copy_inputs(state: DenoiseStepGraphState, rows: "Sequence[DenoiseRow]") -> None:
        state.cache.refresh_caches(
            [row.cache for row in rows],
            state.image_token_num,
        )
        for row_index, row in enumerate(rows):
            state.image_embeds[row_index].copy_(
                row.step.extra["image_embeds"][0], non_blocking=True
            )
            state.z[row_index].copy_(row.step.latent[0], non_blocking=True)
            state.indexes[:, row_index].copy_(row.indexes, non_blocking=True)
        state.t.copy_(rows[0].step.t, non_blocking=True)

    @staticmethod
    def _prepare_backend(state: DenoiseStepGraphState) -> None:
        prepare = getattr(state.backend, "prepare_paged_prefill_cuda_graph", None)
        if not callable(prepare):
            return
        prepare(
            state.metadata,
            num_q_heads=state.num_q_heads,
            num_kv_heads=state.num_kv_heads,
            head_dim=state.head_dim,
            page_size=state.page_size,
            q_dtype=state.image_embeds.dtype,
            kv_dtype=state.request_cache.pool.k.dtype,
            causal=False,
            scale=state.scale,
        )

    @staticmethod
    def _replay(state: DenoiseStepGraphState) -> tuple[torch.Tensor, torch.Tensor]:
        state.graph.replay()
        return state.logits


def denoise_step_graph_runner(owner: Any) -> DenoiseStepGraphRunner:
    """The per-owner runner (graphs close over the owner's weights/caches)."""

    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = DenoiseStepGraphRunner()
        setattr(owner, _RUNNER_ATTR, runner)
    return runner


def maybe_run_denoise_step_graph(
    owner: Any,
    rows: "Sequence[DenoiseRow]",
    *,
    return_hidden: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Try the denoise-step graph for ``rows``; ``None`` means run eager."""

    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        runner = denoise_step_graph_runner(owner)
    return runner.maybe_run_rows(owner, rows, return_hidden=return_hidden)
