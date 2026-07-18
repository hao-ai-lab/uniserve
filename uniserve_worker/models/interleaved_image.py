"""Interleaved image family runtime: denoise state machine, residual reuse, graphs.

Family-owned machinery for image generation inside interleaved requests:
per-request latent/image state, batched denoise rows with flow-match
schedules and CFG, the residual-reuse (TeaCache-style) adapter, and the
denoise-step CUDA-graph runner built on ``execution.cuda_graph`` primitives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

import uniserve_worker.ops as ops
from uniserve_worker.contracts.attention_plan import GraphBinding, PagedVarlenPlan
from uniserve_worker.contracts.forward_context import get_forward_context, use_forward_context
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution.cuda_graph import GraphEvent, _GraphRunnerBase, record_graph_stats
from uniserve_worker.execution.engine import TextImageDenoiseStep, text_image_cfg_branch_count
from uniserve_worker.foundation.env import env_flag, env_str
from uniserve_worker.foundation.errors import invalid_descriptor, model_execution_error
from uniserve_worker.models.interleaved_text import TextCache
from uniserve_worker.nn.attention import RadixAttention
from uniserve_worker.nn.diffusion import (
    FlowMatchSchedule,
    ScheduleDirection,
    ScheduleShiftDomain,
    init_latent,
)
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_text_image_cfg_plan
from uniserve_worker.nn.vision import patchify_batch, unpatchify_batch
from uniserve_worker.runtime.image_params import (
    TextImageGenerationParams as _ImageParams,
)
from uniserve_worker.runtime.image_params import (
    parse_text_image_generation_params,
)
from uniserve_worker.runtime.paged_denoise import can_run_paged_denoise_attention
from uniserve_worker.runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache

# ---------------------
# Residual-reuse cache
# ---------------------

@dataclass(frozen=True)
class DenoiseResidualCacheBinding:
    """Model-supplied hooks: decision embedding + calibrated rescale polynomial.

    The cached residual lives in the *pre-final-norm* residual stream (where
    its magnitude dwarfs the input-embedding delta, so replaying
    ``embeds' + residual`` is a faithful approximation); ``finalize_hidden``
    re-applies whatever the model's backbone does after its block stack
    (typically the final RMSNorm of the generation branch) before the
    hidden→velocity head consumes the replayed stream.
    """

    # Maps the step's input embeddings [B, N, C] to the decision embedding the
    # distance metric runs on (typically a cheap norm of the first row).
    decision_embedding: Callable[[torch.Tensor], torch.Tensor]
    # Polynomial coefficients (highest degree first) rescaling the raw
    # relative-L1 distance into accumulated skip budget, calibrated per model.
    rescale_coefficients: tuple[float, ...]
    # Post-block-stack finalization applied to a replayed pre-norm stream.
    finalize_hidden: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DenoiseResidualCachePolicy:
    enabled: bool
    threshold: float

    def active(self, adapter: DenoiseResidualCacheBinding | None) -> bool:
        return self.enabled and adapter is not None


def resolve_denoise_residual_cache_policy() -> DenoiseResidualCachePolicy:
    enabled = env_flag("UNISERVE_DENOISE_RESIDUAL_CACHE", default=False)
    raw = env_str("UNISERVE_DENOISE_RESIDUAL_CACHE_THRESHOLD", default="0.2")
    try:
        threshold = float(raw)
    except ValueError:
        threshold = 0.2
    return DenoiseResidualCachePolicy(enabled=enabled, threshold=max(0.0, threshold))


def _poly_eval(coefficients: tuple[float, ...], x: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


@dataclass
class ImageResidualCacheState:
    """Per-image reuse state: decision history + per-branch residuals.

    One instance rides on the :class:`ImageState` for the image being
    denoised, so its lifetime (and memory) ends with the image commit.
    """

    threshold: float
    coefficients: tuple[float, ...]
    accumulated: float = 0.0
    previous_decision: torch.Tensor | None = None
    residuals: dict[str, torch.Tensor] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def decide_reuse(self, decision: torch.Tensor, branches: tuple[str, ...]) -> bool:
        """Advance the decision stream; True when every branch can be replayed."""
        previous = self.previous_decision
        self.previous_decision = decision
        if previous is None or previous.shape != decision.shape:
            self.accumulated = 0.0
            self.misses += 1
            return False
        denom = previous.abs().mean()
        if float(denom) == 0.0:
            self.misses += 1
            return False
        rel_l1 = float((decision - previous).abs().mean() / denom)
        self.accumulated += _poly_eval(self.coefficients, rel_l1)
        if self.accumulated >= self.threshold:
            self.accumulated = 0.0
            self.misses += 1
            return False
        for branch in branches:
            if branch not in self.residuals:
                self.misses += 1
                return False
        self.hits += 1
        return True

    def replay(self, branch: str, input_embeds: torch.Tensor) -> torch.Tensor:
        return input_embeds + self.residuals[branch]

    def record(self, branch: str, input_embeds: torch.Tensor, hidden: torch.Tensor) -> None:
        self.residuals[branch] = (hidden - input_embeds).detach()

    def invalidate(self) -> None:
        """Drop replay state (e.g. the step ran outside this policy's view)."""
        self.previous_decision = None
        self.accumulated = 0.0
        self.residuals.clear()

    def stats(self) -> Mapping[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# ---------------------
# Denoise-step graph runner
# ---------------------

logger = logging.getLogger(__name__)


# Consecutive capture/replay failures before the runner hard-disables itself.
_MAX_FAILURES = 2

_RUNNER_ATTR = "_denoise_step_graph_runner"

# Keep graph-resident activation pools bounded while retaining useful GEMM
# batching: one graph microbatch covers two full three-branch CFG requests.
_MAX_GRAPH_ROWS = 6


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
    plan: PagedVarlenPlan               # stable transient paged-varlen plan
    graph_binding: GraphBinding         # wrapper-routing identity (kept alive here)
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
            preferred = getattr(ctx, "attention_preference", None) or "torch_sdpa"
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
        (_query_lens, block_table, _cache_lens, cu_q, cu_k, max_q, max_k) = transient
        # The directly-passed denoise cache remains authoritative. An unset
        # residency cache keeps RadixAttention on its transient paged-varlen path.
        plan = PagedVarlenPlan(
            block_table=block_table,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            residency_cache=None,
        )
        graph_binding = GraphBinding()
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
            plan=plan,
            graph_binding=graph_binding,
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
            bind(graph_binding, plan, device=device)
            state.release_backend = lambda: release(graph_binding)

        # Capture with this graph's stable plan and wrapper-routing binding while
        # stats are detached, mirroring the shared decode/prefill graph discipline.
        graph_ctx = replace(
            ctx,
            attention_plan=plan,
            graph_binding=graph_binding,
            stats=None,
        )

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
            if callable(planned) and not planned(graph_binding):
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
            state.graph_binding,
            state.plan,
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


# ---------------------
# Interleaved image denoise state machine
# ---------------------

_RESIDUAL_CACHE_POLICY = None


def _denoise_residual_cache_policy():
    """Process-wide policy, resolved from the environment once on first use."""
    global _RESIDUAL_CACHE_POLICY
    if _RESIDUAL_CACHE_POLICY is None:
        _RESIDUAL_CACHE_POLICY = resolve_denoise_residual_cache_policy()
    return _RESIDUAL_CACHE_POLICY


@dataclass
class ImageState:
    """Mutable denoise state for one in-flight image generation request.

    The latent trajectory ``x_t`` is not stored on the model state — it lives in
    the system-owned :class:`~uniserve_worker.runtime.residency.LatentPool`
    as a leased buffer addressed by ``latent_handle``. ``x_t`` here is a property
    reading/writing that system buffer.
    """

    latent_pool: Any           # system LatentPool (residency.latent)
    latent_handle: int         # request-scoped handle into the LatentPool
    schedule: FlowMatchSchedule
    timesteps: torch.Tensor
    token_h: int
    token_w: int
    grid_h: int
    grid_w: int
    grid_hw: torch.Tensor
    indexes_cond: torch.Tensor
    indexes_tu: torch.Tensor | None
    indexes_iu: torch.Tensor | None
    cond_cache: Any
    tu_cache: Any
    iu_cache: Any
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_interval: tuple[float, float]
    cfg_norm: str
    cfg_renorm_min: float
    noise_scale: float
    height: int
    width: int
    noise_scale_embedding: torch.Tensor | None = None
    # Timestep-aware residual-reuse state (see denoise_residual_cache); engaged
    # only when the policy is enabled and the owner supplies an adapter. Rides
    # the image state so its memory ends with the image commit.
    residual_cache: ImageResidualCacheState | None = None

    @property
    def x_t(self) -> torch.Tensor:
        return self.latent_pool.get(self.latent_handle)

    @x_t.setter
    def x_t(self, value: torch.Tensor) -> None:
        self.latent_pool.set(self.latent_handle, value)


@dataclass
class DenoiseRow:
    """One CFG branch of one denoise step queued for batched velocity prediction."""

    step_index: int
    step: TextImageDenoiseStep
    branch: str
    img: ImageState
    indexes: torch.Tensor
    cache: PagedTextCache


@dataclass
class InterleavedImageRequestState:
    """Per-request interleaved text/image caches and generation state."""

    sampling: dict = field(default_factory=dict)
    image: dict = field(default_factory=dict)
    neg_token_ids: list[int] = field(default_factory=list)
    cond: TextCache = field(default_factory=TextCache)
    tu: TextCache = field(default_factory=TextCache)
    iu: TextCache = field(default_factory=TextCache)
    image_state: ImageState | None = None
    rng: torch.Generator | None = None


class TextImageDenoiseOwner(Protocol):
    """Collaborator surface a concrete model must provide to TextImageDenoiseOps.

    The mixin owns the denoise step/branch/commit flow but delegates model- and
    cache-specific work back to the concrete owner. Every member declared below
    is part of the mixin's contract and is called directly: the concrete owner
    must define all of them, including the ``_denoise`` / ``_wait`` cache and
    stream hooks.
    """

    # Collaborator attributes.
    device: Any
    gen_device: Any
    latent_downsample: int
    merge_size: int
    _img_start_token: str
    residency: Any                 # ResidencyManager; .latent backs ImageState.x_t
    attention_backend: str         # preferred attention provider ("auto" allowed)
    _dataplane_handoff: Any | None  # Mode A tower handoff; None on single-device owners

    # Owner-supplied denoise configuration (models differ only by these enums).
    denoise_schedule_direction: ScheduleDirection
    denoise_schedule_shift_domain: ScheduleShiftDomain
    denoise_cfg_recipe: CfgRecipe

    # Collaborator methods.
    def _state(self, op: dict[str, Any]) -> "InterleavedImageRequestState": ...
    def _maybe_stage_conditioning_from_op(
        self, st: "InterleavedImageRequestState", op: dict[str, Any]
    ) -> None: ...
    def _extend_cache_blocks(self, cache: "TextCache", op: dict[str, Any]) -> None: ...
    def _ensure_img_start(self, cache: "TextCache | None") -> None: ...
    def _prefix_from_query(self, query: str) -> "TextCache": ...
    def _empty_img_start_prefix(self) -> "TextCache": ...
    def _denoise_cache(self, cache: Any) -> Any: ...
    def _wait_gen_cache_ready(self, cache: Any) -> None: ...
    def interleaved_image_query(self, text: str, *, append_text: str) -> str: ...
    def interleaved_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: Any,
    ) -> torch.Tensor: ...
    def interleaved_image_predict_velocity(
        self,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: Any,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int],
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def packed_hidden_to_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int] | None,
    ) -> torch.Tensor: ...
    def denoise_residual_cache_adapter(self) -> DenoiseResidualCacheBinding | None: ...
    def interleaved_image_patch_size(self) -> int: ...
    def interleaved_image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor: ...
    def interleaved_image_gen_feature_dtype(self) -> torch.dtype: ...
    def interleaved_image_noise_scale(self, grid_h: int, grid_w: int) -> float: ...
    def interleaved_image_noise_scale_embedding(
        self,
        noise_scale: float,
        token_count: int,
        *,
        dtype: torch.dtype,
        device: Any,
    ) -> torch.Tensor | None: ...
    def interleaved_image_timestep_embeddings(self, t_values: torch.Tensor) -> torch.Tensor: ...

    # Mixin methods (from TextImageDenoiseOps) reached through ``self``.
    def _init_image_state(
        self, st: "InterleavedImageRequestState", op: dict | None = ...
    ) -> "ImageState": ...
    def _parse_image_params(self, ip: dict) -> "_ImageParams": ...
    def _setup_cfg_caches(
        self, st: "InterleavedImageRequestState", op: dict | None, params: "_ImageParams"
    ) -> "TextCache": ...
    def _build_indexes(
        self,
        st: "InterleavedImageRequestState",
        cond: "TextCache",
        token_h: int,
        token_w: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...
    def _compute_noise_scale(self, grid_h: int, grid_w: int) -> float: ...
    def _init_latent(
        self,
        st: "InterleavedImageRequestState",
        params: "_ImageParams",
        device: Any,
        noise_scale: float,
    ) -> torch.Tensor: ...
    def predict_denoise_velocity(
        self, step: "TextImageDenoiseStep", branch: str, *, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def _denoise_branch_inputs(
        self, img: "ImageState", branch: str
    ) -> tuple[torch.Tensor, Any]: ...
    def _batched_denoise_row_key(self, row: "DenoiseRow") -> "tuple[Any, ...] | None": ...
    def _batched_paged_denoise_available(
        self, image_embeds: torch.Tensor, cache: "PagedTextCache"
    ) -> bool: ...
    def _predict_v_batched(
        self,
        rows: "Sequence[DenoiseRow]",
        *,
        return_hidden: bool = False,
        graph_mode: str = "auto",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: ...
    def _predict_v(
        self,
        img: "ImageState",
        image_embeds: torch.Tensor,
        indexes: torch.Tensor | None,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def _denoise_residual_state(
        self, img: "ImageState"
    ) -> ImageResidualCacheState | None: ...
    def _predict_row_recorded(
        self,
        row: "DenoiseRow",
        state: ImageResidualCacheState | None,
    ) -> torch.Tensor: ...


class TextImageDenoiseOps:
    """Mixin implementing text/image denoise setup, batching, and velocity prediction."""

    def _init_image_state(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        op: dict | None = None,
    ) -> ImageState:
        ip = st.image or {}
        params = self._parse_image_params(ip)
        cond = self._setup_cfg_caches(st, op, params)

        token_h = params.height // self.latent_downsample
        token_w = params.width // self.latent_downsample
        patch_size = self.interleaved_image_patch_size()
        grid_h = params.height // patch_size
        grid_w = params.width // patch_size
        device = getattr(self, "gen_device", self.device)
        indexes_cond, indexes_tu, indexes_iu = self._build_indexes(
            st, cond, token_h, token_w, device
        )

        schedule = FlowMatchSchedule(
            num_steps=params.steps,
            shift=params.timestep_shift,
            direction=self.denoise_schedule_direction,
            shift_domain=self.denoise_schedule_shift_domain,
        )
        timesteps = schedule.timesteps(device=device)
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        noise_scale = self._compute_noise_scale(grid_h, grid_w)
        noise_scale_embedding = self.interleaved_image_noise_scale_embedding(
            noise_scale,
            token_h * token_w,
            dtype=timesteps.dtype,
            device=device,
        )

        x_t = self._init_latent(st, params, device, noise_scale)
        cond_cache = self._denoise_cache(cond.past)
        tu_cache = self._denoise_cache(st.tu.past)
        iu_cache = self._denoise_cache(st.iu.past)
        # The latent lives in the system-owned LatentPool, keyed by the request
        # handle; ``ImageState.x_t`` reads/writes that buffer.
        latent_handle = int(op["req_id"]) if op and "req_id" in op else id(st)
        image_state = ImageState(
            latent_pool=self.residency.latent,
            latent_handle=latent_handle,
            schedule=schedule,
            timesteps=timesteps,
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            grid_hw=grid_hw,
            indexes_cond=indexes_cond,
            indexes_tu=indexes_tu,
            indexes_iu=indexes_iu,
            cond_cache=cond_cache,
            tu_cache=tu_cache,
            iu_cache=iu_cache,
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            cfg_interval=(float(params.cfg_interval[0]), float(params.cfg_interval[1])),
            cfg_norm=params.cfg_norm,
            cfg_renorm_min=params.cfg_renorm_min,
            noise_scale=float(noise_scale),
            height=params.height,
            width=params.width,
            noise_scale_embedding=noise_scale_embedding,
        )
        image_state.x_t = x_t  # store the initial noise into the system LatentPool
        return image_state

    def _parse_image_params(self: TextImageDenoiseOwner, ip: dict) -> _ImageParams:
        return parse_text_image_generation_params(ip)

    def _setup_cfg_caches(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        op: dict | None,
        params: _ImageParams,
    ) -> TextCache:
        """Prepare the cond/text-uncond/img-uncond text caches for denoising.

        Returns the conditioning :class:`TextCache` to use (a fresh image-prompt
        prefix when ``op`` supplies one, otherwise the request's own ``st.cond``).
        Mutates ``st.tu``/``st.iu`` in place as required by the CFG scales.
        """
        if params.retain_images:
            self._ensure_img_start(st.cond)
        cond = st.cond
        image_prompt = (op or {}).get("image_prompt")
        if isinstance(image_prompt, str) and image_prompt.strip():
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor(
                    "Mode A cuda_ipc tower split does not support per-op image_prompt overrides yet"
                )
            query = self.interleaved_image_query(
                image_prompt.strip(),
                append_text=self._img_start_token,
            )
            cond = self._prefix_from_query(query)
        elif not params.retain_images:
            self._ensure_img_start(st.cond)
        cfg_plan = build_text_image_cfg_plan(
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            recipe=self.denoise_cfg_recipe,
            renorm=params.cfg_norm,
            renorm_min=params.cfg_renorm_min,
        )
        needs_text_uncond = Branch.TEXT_UNCOND in cfg_plan.branches
        if needs_text_uncond and st.tu.past is None:
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor("Mode A denoise is missing pre-staged text-unconditional CFG KV")
            st.tu = self._empty_img_start_prefix()
        elif needs_text_uncond:
            self._ensure_img_start(st.tu)
        needs_img_uncond = Branch.IMG_UNCOND in cfg_plan.branches
        if needs_img_uncond and st.iu.past is None:
            if getattr(self, "_dataplane_handoff", None) is not None:
                raise invalid_descriptor("Mode A denoise is missing pre-staged image-unconditional CFG KV")
            st.iu = self._empty_img_start_prefix()
        elif needs_img_uncond:
            self._ensure_img_start(st.iu)
        return cond

    def _build_indexes(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        cond: TextCache,
        token_h: int,
        token_w: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        indexes_cond = self.interleaved_image_indexes(
            token_h, token_w, cond.t_index + 1, device=device
        )
        indexes_tu = (
            self.interleaved_image_indexes(token_h, token_w, st.tu.t_index + 1, device=device)
            if st.tu.past is not None
            else None
        )
        indexes_iu = (
            self.interleaved_image_indexes(token_h, token_w, st.iu.t_index + 1, device=device)
            if st.iu.past is not None
            else None
        )
        return indexes_cond, indexes_tu, indexes_iu

    def _compute_noise_scale(self: TextImageDenoiseOwner, grid_h: int, grid_w: int) -> float:
        return self.interleaved_image_noise_scale(grid_h, grid_w)

    def _init_latent(
        self: TextImageDenoiseOwner,
        st: InterleavedImageRequestState,
        params: _ImageParams,
        device: Any,
        noise_scale: float,
    ) -> torch.Tensor:
        if st.rng is None:
            seed = params.seed
            st.rng = torch.Generator(device=device).manual_seed(int(seed if seed is not None else 0))
        if st.cond.last_logits is None:
            raise model_execution_error("image denoise requires conditional text logits")
        dtype = st.cond.last_logits.dtype
        return init_latent(
            (1, 3, params.height, params.width),
            rng=st.rng,
            device=device,
            dtype=dtype,
            scale=noise_scale,
        )

    def prepare_denoise_step(
        self: TextImageDenoiseOwner, req_id: int, state: Any, op: dict
    ) -> TextImageDenoiseStep:
        ctx = get_forward_context()
        start = ctx.component_timer_start()
        st = self._state(dict(op))
        # Mode A (tower disaggregation): the gen pool never ran the und text, so
        # rebuild st.cond from the conditioning KV the und pool published (carried
        # on op["locator"]). A no-op in Mode C / single-device.
        self._maybe_stage_conditioning_from_op(st, op)
        self._extend_cache_blocks(st.cond, op)
        if st.image_state is None:
            st.image_state = self._init_image_state(st, op)
        img = st.image_state
        step_i = int(op.get("timestep_idx") or 0)
        ctx.record_component_elapsed("interleaved_denoise_prepare_state", start)
        # Gen-tower feature extraction and timestep embedding run on the gen
        # coordinate's device (the gen modules are Pinned there); the tower
        # transport, not a dedicated stream, orders the und->gen handoff.
        device = getattr(self, "gen_device", self.device)
        t, t_next = img.schedule.pair(step_i, device=device, dtype=img.timesteps.dtype)
        start = ctx.component_timer_start()
        z = patchify_batch(img.x_t, self.latent_downsample)
        image_input = patchify_batch(
            img.x_t,
            self.interleaved_image_patch_size(),
            channel_first=True,
        )
        image_input = image_input.to(
            device=device,
            dtype=self.interleaved_image_gen_feature_dtype(),
        )
        ctx.record_component_elapsed("interleaved_denoise_patchify", start)
        start = ctx.component_timer_start()
        image_embeds = self.interleaved_image_features(
            image_input.view(1 * img.grid_h * img.grid_w, -1),
            gen_model=True,
            grid_hw=img.grid_hw,
        ).view(1, img.token_h * img.token_w, -1)
        ctx.record_component_elapsed("interleaved_denoise_vision_feature", start)
        start = ctx.component_timer_start()
        t_expanded = t.expand(img.token_h * img.token_w)
        timestep_embeddings = self.interleaved_image_timestep_embeddings(t_expanded).view(
            1, img.token_h * img.token_w, -1
        )
        if img.noise_scale_embedding is None:
            img.noise_scale_embedding = self.interleaved_image_noise_scale_embedding(
                img.noise_scale,
                img.token_h * img.token_w,
                dtype=t_expanded.dtype,
                device=device,
            )
        if img.noise_scale_embedding is not None:
            timestep_embeddings += img.noise_scale_embedding
        image_embeds = image_embeds + timestep_embeddings
        ctx.record_component_elapsed("interleaved_denoise_timestep_embed", start)
        total = int(img.schedule.num_steps)
        return TextImageDenoiseStep(
            req_id=int(req_id),
            state=state,
            op=op,
            latent=z,
            t=t,
            t_next=t_next,
            step_index=step_i,
            total_steps=total,
            cfg_text_scale=img.cfg_text_scale,
            cfg_img_scale=img.cfg_img_scale,
            cfg_interval=img.cfg_interval,
            cfg_renorm_type=img.cfg_norm,
            cfg_renorm_min=img.cfg_renorm_min,
            cfg_branch_count=text_image_cfg_branch_count(op),
            image_scale_applies_to_text=self.denoise_cfg_recipe,
            extra={
                "img": img,
                "image_embeds": image_embeds,
            },
        )

    def predict_denoise_velocity(
        self: TextImageDenoiseOwner,
        step: TextImageDenoiseStep,
        branch: str,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        img = step.extra["img"]
        indexes, cache = self._denoise_branch_inputs(img, branch)
        return self._predict_v(
            img,
            step.extra["image_embeds"],
            indexes,
            cache,
            step.t,
            step.latent,
            return_hidden=return_hidden,
        )

    def _denoise_residual_state(
        self: TextImageDenoiseOwner, img: ImageState
    ) -> ImageResidualCacheState | None:
        policy = _denoise_residual_cache_policy()
        adapter = self.denoise_residual_cache_adapter()
        if adapter is None or not policy.active(adapter):
            return None
        state = img.residual_cache
        if state is None:
            state = ImageResidualCacheState(
                threshold=policy.threshold,
                coefficients=adapter.rescale_coefficients,
            )
            img.residual_cache = state
        return state

    def denoise_residual_cache_adapter(
        self: TextImageDenoiseOwner,
    ) -> DenoiseResidualCacheBinding | None:
        """Default: no residual-reuse adapter — the cache never engages."""
        return None

    def _predict_row_recorded(
        self: TextImageDenoiseOwner,
        row: DenoiseRow,
        state: ImageResidualCacheState | None,
    ) -> torch.Tensor:
        if state is None:
            velocity = self.predict_denoise_velocity(row.step, row.branch)
            assert isinstance(velocity, torch.Tensor)
            return velocity
        velocity, hidden = self.predict_denoise_velocity(
            row.step, row.branch, return_hidden=True
        )
        state.record(str(row.branch), row.step.extra["image_embeds"], hidden)
        return velocity

    def predict_text_image_velocity_batch(
        self: TextImageDenoiseOwner,
        steps: Sequence[TextImageDenoiseStep],
        branches_by_step: Sequence[Sequence[str]],
        *,
        graph_mode: str = "auto",
    ) -> list[dict[str, torch.Tensor]] | None:
        results: list[dict[str, torch.Tensor]] = [dict() for _ in steps]
        rows: list[DenoiseRow] = []
        states: dict[int, ImageResidualCacheState | None] = {}
        for step_index, (step, branches) in enumerate(zip(steps, branches_by_step)):
            img = step.extra["img"]
            state = self._denoise_residual_state(img)
            states[step_index] = state
            if state is not None:
                adapter = self.denoise_residual_cache_adapter()
                assert adapter is not None
                image_embeds = step.extra["image_embeds"]
                decision = adapter.decision_embedding(image_embeds)
                if state.decide_reuse(decision, tuple(str(b) for b in branches)):
                    if graph_mode == "require":
                        return None
                    # Replay: pre-norm hidden ≈ input embeds + previous
                    # residual, re-finalized (final norm), then the ordinary
                    # hidden→velocity head. No backbone forward.
                    for branch in branches:
                        hidden = adapter.finalize_hidden(
                            state.replay(str(branch), image_embeds)
                        )
                        results[step_index][branch] = self.packed_hidden_to_velocity(
                            hidden,
                            step.t,
                            step.latent,
                            image_token_num=img.token_h * img.token_w,
                            image_size=(img.width, img.height),
                        )
                    continue
            for branch in branches:
                indexes, cache = self._denoise_branch_inputs(img, branch)
                if not isinstance(cache, PagedTextCache):
                    if graph_mode == "require":
                        return None
                    results[step_index][branch] = self._predict_row_recorded(
                        DenoiseRow(step_index, step, branch, img, indexes, cache), state
                    )
                    continue
                rows.append(DenoiseRow(step_index, step, branch, img, indexes, cache))

        def predict_group(group: Sequence[DenoiseRow]) -> bool:
            record = any(states.get(row.step_index) is not None for row in group)
            if record:
                predicted = self._predict_v_batched(
                    group,
                    return_hidden=True,
                    graph_mode=graph_mode,
                )
                if predicted is None:
                    if graph_mode == "require":
                        return False
                    for row in group:
                        results[row.step_index][row.branch] = self._predict_row_recorded(
                            row,
                            states.get(row.step_index),
                        )
                    return True
                batched, hidden = predicted
            else:
                batched = self._predict_v_batched(group, graph_mode=graph_mode)
                if batched is None:
                    if graph_mode == "require":
                        return False
                    for row in group:
                        results[row.step_index][row.branch] = self._predict_row_recorded(
                            row,
                            states.get(row.step_index),
                        )
                    return True
                hidden = None
            for row_index, row in enumerate(group):
                results[row.step_index][row.branch] = batched[
                    row_index : row_index + 1
                ].contiguous()
                state = states.get(row.step_index)
                if state is not None and hidden is not None:
                    state.record(
                        str(row.branch),
                        row.step.extra["image_embeds"],
                        hidden[row_index : row_index + 1].contiguous(),
                    )
            return True

        if len(rows) == 1:
            if not predict_group(rows):
                return None
            return results

        grouped: dict[tuple[Any, ...], list[DenoiseRow]] = {}
        for row in rows:
            key = self._batched_denoise_row_key(row)
            if key is None:
                if graph_mode == "require":
                    return None
                results[row.step_index][row.branch] = self._predict_row_recorded(
                    row, states.get(row.step_index)
                )
                continue
            grouped.setdefault(key, []).append(row)

        for group in grouped.values():
            if not predict_group(group):
                return None
        return results

    def _denoise_branch_inputs(self, img: ImageState, branch: str) -> tuple[torch.Tensor, Any]:
        # ``Branch`` is a ``str`` Enum, so this mapping resolves both ``Branch``
        # members and the equivalent bare strings ("cond"/"text_uncond"/
        # "img_uncond") to the same entry.
        branch_inputs: dict[Branch, tuple[torch.Tensor | None, Any]] = {
            Branch.COND: (img.indexes_cond, img.cond_cache),
            Branch.TEXT_UNCOND: (img.indexes_tu, img.tu_cache),
            Branch.IMG_UNCOND: (img.indexes_iu, img.iu_cache),
        }
        try:
            indexes, cache = branch_inputs[branch]  # type: ignore[index]
        except KeyError:
            raise model_execution_error(f"unknown denoise branch {branch!r}") from None
        if indexes is None or cache is None:
            raise model_execution_error("required CFG cache is not initialized")
        return indexes, cache

    def _batched_denoise_row_key(self, row: DenoiseRow) -> tuple[Any, ...] | None:
        step = row.step
        img = row.img
        indexes = row.indexes
        cache = row.cache
        pool = getattr(cache, "pool", None)
        image_embeds = step.extra["image_embeds"]
        if pool is None or indexes.ndim != 2 or indexes.shape[0] != 3:
            return None
        if not self._batched_paged_denoise_available(image_embeds, cache):
            return None
        if image_embeds.ndim != 3 or step.latent.ndim != 3:
            return None
        t_values = tuple(float(v) for v in step.t.detach().float().reshape(-1).cpu().tolist())
        return (
            id(pool),
            str(image_embeds.device),
            str(image_embeds.dtype),
            tuple(image_embeds.shape[1:]),
            str(step.latent.device),
            str(step.latent.dtype),
            tuple(step.latent.shape[1:]),
            tuple(indexes.shape),
            int(img.token_h),
            int(img.token_w),
            int(img.height),
            int(img.width),
            t_values,
        )

    def _batched_paged_denoise_available(
        self,
        image_embeds: torch.Tensor,
        cache: PagedTextCache,
    ) -> bool:
        return can_run_paged_denoise_attention(
            cache,
            prototype=image_embeds,
            attention_backend=getattr(self, "attention_backend", "auto"),
        )

    def _predict_v_batched(
        self: TextImageDenoiseOwner,
        rows: Sequence[DenoiseRow],
        *,
        return_hidden: bool = False,
        graph_mode: str = "auto",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        first = rows[0]
        img = first.img
        for row in rows:
            self._wait_gen_cache_ready(row.cache)
        # Capture or replay the whole compatible batched step as one CUDA graph.
        # Strict unified-forward callers use ``graph_mode="require"`` and reject
        # the batch when its geometry is not covered.
        if graph_mode != "eager":
            graphed = maybe_run_denoise_step_graph(self, rows, return_hidden=return_hidden)
            if graphed is not None:
                return graphed
            if graph_mode == "require":
                return None
        image_embeds = torch.cat([row.step.extra["image_embeds"] for row in rows], dim=0)
        indexes = torch.stack([row.indexes for row in rows], dim=1).contiguous()
        cache = BatchedPagedTextCache([row.cache for row in rows])
        z = torch.cat([row.step.latent for row in rows], dim=0)
        return self._predict_v(
            img,
            image_embeds,
            indexes,
            cache,
            first.step.t,
            z,
            return_hidden=return_hidden,
        )

    def apply_denoise_update(
        self: TextImageDenoiseOwner, step: TextImageDenoiseStep, latent: torch.Tensor
    ) -> None:
        img = step.extra["img"]
        img.x_t = unpatchify_batch(
            latent,
            self.latent_downsample,
            height=img.height,
            width=img.width,
        )

    def _predict_v(
        self: TextImageDenoiseOwner,
        img: ImageState,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor | None,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if indexes is None or cache is None:
            raise model_execution_error("required CFG cache is not initialized")
        # B2: wait the snapshot's readiness before the gen tower reads the replica.
        self._wait_gen_cache_ready(cache)
        return self.interleaved_image_predict_velocity(
            image_embeds,
            indexes,
            {"full_attention": None},
            cache,
            t,
            z,
            image_token_num=img.token_h * img.token_w,
            image_size=(img.width, img.height),
            return_hidden=return_hidden,
        )
