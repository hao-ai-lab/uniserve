"""Per-image CUDA-graph capture/replay for the batched text-image denoise step.

The batched denoise velocity prediction (``TextImageDenoiseOps._predict_v_batched``)
launches the full generation-tower forward — thousands of kernels plus, under
tensor parallelism, two NCCL all-reduces per layer — for every one of ~50 flow
steps. Per step only the *inputs* change (image embeds, timestep, latent);
the paged-KV geometry (scratch blocks, base lengths, rope indexes, shapes) is
frozen for the lifetime of one image. That makes each step a textbook CUDA-graph
replay: this runner captures the forward once per image and replays it with a
handful of device-to-device input copies, eliminating the CPU launch overhead
that dominates the TP4 denoise wall clock.

Correctness rests on three pillars:

* **Graph key.** A graph is only replayed while its identity holds: the same
  KV pool, the same per-branch cache objects, the same base lengths, and the
  *same block-id contents* (``request_cache_for_transient`` reuses stable
  scratch blocks per image, so the captured page writes stay valid). Any drift
  changes the key and the step falls back to eager (or captures anew).
* **Exclusive attention plan.** The captured FlashInfer prefill ``wrapper.run``
  bakes its plan; the runner binds a graph-scoped exclusive wrapper (see
  ``bind_paged_prefill_graph_wrapper``) to a per-graph metadata sentinel so no
  other request can re-plan the captured wrapper. The dispatcher-winner probe
  guarantees capture never swaps in a backend the eager path would not use.
* **Private capture pool.** Per-image graphs are freed independently at image
  commit; sharing one ``graph_pool_handle`` across graphs aborts with a
  ``use_count > 0`` internal assert (CUDACachingAllocator.cpp:2291) once the
  last graph in the shared pool is destroyed. ``capture_pool`` therefore
  returns ``None`` so each capture owns a private pool.

Env gate: ``UNISERVE_DENOISE_STEP_GRAPH`` (default **off**). Replay is a net
win when the step is CPU-launch-bound (TP4); at TP1 the eager path is already
GPU-bound and the gate should stay off.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

import torch

import uniserve_worker.ops as ops
from ..contracts.forward_context import get_forward_context, use_forward_context
from ..contracts.forward_mode import ForwardMode
from ..foundation.env import env_flag
from ..nn.attention import RadixAttention
from ..runtime.paged_text_cache import BatchedPagedTextCache
from .cuda_graph_base import GraphEvent, _GraphRunnerBase, record_graph_stats

if TYPE_CHECKING:
    from .interleaved_image_denoise import DenoiseRow

logger = logging.getLogger(__name__)

__all__ = [
    "DENOISE_STEP_GRAPH_ENV",
    "DenoiseStepGraphRunner",
    "maybe_run_denoise_step_graph",
    "release_denoise_step_graphs",
]

DENOISE_STEP_GRAPH_ENV = "UNISERVE_DENOISE_STEP_GRAPH"

# Consecutive capture/replay failures before the runner hard-disables itself.
_MAX_FAILURES = 2

_RUNNER_ATTR = "_denoise_step_graph_runner"


class _DenoiseGraphMetadata:
    """Per-graph identity sentinel published as ``ctx.attention_metadata``.

    Deliberately attribute-free: the varlen/paged eligibility probes read
    ``metadata.cache`` / ``metadata.mode`` via ``getattr`` defaults, so an empty
    sentinel keeps the captured forward on the exact transient paged-varlen path
    the eager forward takes, while its *identity* routes FlashInfer's
    ``forward_varlen`` to the graph-scoped exclusive prefill wrapper.
    """

    __slots__ = ("__weakref__",)


class _GraphBackendUnplanned(RuntimeError):
    """Capture completed but the exclusive wrapper was never planned."""


@dataclass
class DenoiseStepGraphState:
    """Static buffers + captured graph for one image's batched denoise step."""

    key: tuple[Any, ...]
    cache_ids: frozenset[int]
    graph: torch.cuda.CUDAGraph
    image_embeds: torch.Tensor          # [rows, tokens, hidden] static input
    t: torch.Tensor                     # timestep static input (shape of step.t)
    z: torch.Tensor                     # [rows, tokens, latent] static input
    indexes: torch.Tensor               # [3, rows, tokens] static input
    cache: BatchedPagedTextCache        # batched view over the rows' caches
    metadata: _DenoiseGraphMetadata     # wrapper-routing sentinel (kept alive here)
    image_token_num: int
    image_size: tuple[int, int]
    release_backend: Any = None         # callable dropping the exclusive wrapper
    logits: Any = None                  # (velocity, hidden|None) written by capture/replay


class DenoiseStepGraphRunner(_GraphRunnerBase):
    """Own per-image denoise-step CUDA graphs and their capture/replay lifecycle."""

    def __init__(
        self,
        *,
        name: str = "denoise_step",
        default_enabled: bool | None = None,
        logger: Any = logger,
    ) -> None:
        self.name = str(name)
        self.default_enabled = (
            env_flag(DENOISE_STEP_GRAPH_ENV, default=False)
            if default_enabled is None
            else bool(default_enabled)
        )
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
        # CRITICAL: per-image graphs are released independently at image commit.
        # A shared ``torch.cuda.graph_pool_handle`` aborts with an internal
        # ``use_count > 0`` assert (CUDACachingAllocator.cpp:2291) once the last
        # graph sharing the pool is freed — every capture gets a private pool.
        return None

    # -- public entry ------------------------------------------------------------

    def maybe_run_rows(
        self,
        owner: Any,
        rows: "Sequence[DenoiseRow]",
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Capture-or-replay the denoise step for ``rows``; ``None`` -> eager.

        Outputs are cloned per replay so nothing the caller retains (velocity
        slices, TeaCache hidden records) aliases the graph's static output
        buffers, which the next replay overwrites.
        """

        if not self.enabled() or not torch.cuda.is_available():
            return None
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
                # cannot host a baked-plan graph. Config-wide, so disable the
                # runner rather than accumulate per-image miss keys.
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

    # -- release -----------------------------------------------------------------

    def release_image(self, image_state: Any) -> int:
        """Free every graph keyed on ``image_state``'s denoise caches.

        Called from the owner's image-state release at commit/drop. Graphs use
        private capture pools, so dropping the last reference here frees the
        graph memory without touching any other image's capture.
        """

        cache_ids = {
            id(cache)
            for cache in (
                getattr(image_state, "cond_cache", None),
                getattr(image_state, "tu_cache", None),
                getattr(image_state, "iu_cache", None),
            )
            if cache is not None
        }
        released = 0
        for key, state in list(self.states.items()):
            if not (state.cache_ids & cache_ids):
                continue
            self.states.pop(key, None)
            self.disabled.discard(key)
            if callable(state.release_backend):
                try:
                    state.release_backend()
                except Exception:  # pragma: no cover - defensive release
                    if self.logger is not None:
                        self.logger.warning(
                            "%s failed to release graph attention wrapper", self.name, exc_info=True
                        )
            released += 1
        return released

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
        cache_ids: list[int] = []
        base_lens: list[int] = []
        block_ids: list[tuple[int, ...]] = []
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
            cache_ids.append(id(cache))
            base_lens.append(int(cache.length))
            block_ids.append(tuple(int(block) for block in cache.block_ids))
        return (
            id(pool),
            tuple(cache_ids),
            tuple(base_lens),
            tuple(block_ids),
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
                if backend is not None and callable(
                    getattr(backend, "bind_paged_prefill_graph_wrapper", None)
                ):
                    return backend
                return None
        except Exception:
            if self.logger is not None:
                self.logger.debug("%s graph backend probe failed", self.name, exc_info=True)
        return None

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
        metadata = _DenoiseGraphMetadata()
        state = DenoiseStepGraphState(
            key=key,
            cache_ids=frozenset(id(row.cache) for row in rows),
            graph=torch.cuda.CUDAGraph(),
            image_embeds=torch.cat(
                [row.step.extra["image_embeds"] for row in rows], dim=0
            ).contiguous(),
            t=first.step.t.detach().clone(),
            z=torch.cat([row.step.latent for row in rows], dim=0).contiguous(),
            indexes=torch.stack([row.indexes for row in rows], dim=1).contiguous(),
            cache=BatchedPagedTextCache([row.cache for row in rows]),
            metadata=metadata,
            image_token_num=int(img.token_h) * int(img.token_w),
            image_size=(int(img.width), int(img.height)),
        )
        backend.bind_paged_prefill_graph_wrapper(metadata, device=device)
        state.release_backend = lambda: backend.release_paged_prefill_graph_wrapper(metadata)

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
        for row_index, row in enumerate(rows):
            state.image_embeds[row_index].copy_(
                row.step.extra["image_embeds"][0], non_blocking=True
            )
            state.z[row_index].copy_(row.step.latent[0], non_blocking=True)
            state.indexes[:, row_index].copy_(row.indexes, non_blocking=True)
        state.t.copy_(rows[0].step.t, non_blocking=True)

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
        if not env_flag(DENOISE_STEP_GRAPH_ENV, default=False):
            return None
        runner = denoise_step_graph_runner(owner)
    return runner.maybe_run_rows(owner, rows, return_hidden=return_hidden)


def release_denoise_step_graphs(owner: Any, image_state: Any) -> None:
    """Free the graphs captured for ``image_state`` (image commit/drop hook)."""

    if image_state is None:
        return
    runner = getattr(owner, _RUNNER_ATTR, None)
    if runner is None:
        return
    runner.release_image(image_state)
