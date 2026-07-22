"""System-owned continuous-flow operation execution.

The module owns scheduled state updates, branch composition, and graph lowering.
Family adapters supply neural features and velocity prediction without owning
the operation lifecycle.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

import torch

import uniserve_worker.ops as ops
from uniserve_worker.contracts.attention_plan import GraphBinding, PagedVarlenPlan
from uniserve_worker.contracts.forward_context import get_forward_context, use_forward_context
from uniserve_worker.execution.graph.capture import Event, FailureManagedRunner
from uniserve_worker.execution.sequence import SequenceCache
from uniserve_worker.foundation.errors import invalid_descriptor, model_execution_error
from uniserve_worker.nn.attention import RadixAttention
from uniserve_worker.nn.diffusion import (
    FlowMatchSchedule,
    ScheduleDirection,
    ScheduleShiftDomain,
    init_latent,
)
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.cfg import CfgPlan as DiffusionCfgPlan
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
# Flow-step graph runner
# ---------------------

logger = logging.getLogger(__name__)


# Keep graph-resident activation pools bounded while retaining useful GEMM
# batching: one graph microbatch covers two full three-branch CFG requests.
_MAX_GRAPH_ROWS = 6


@dataclass(frozen=True)
class PreparedFlowStep:
    """Model/runtime boundary for one scheduled denoise update."""

    req_id: int
    state: Any
    op: Mapping[str, Any]
    latent: torch.Tensor
    t: torch.Tensor
    t_next: torch.Tensor
    step_index: int
    total_steps: int
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_interval: tuple[float, float]
    cfg_renorm_type: str
    cfg_renorm_min: float
    cfg_branch_count: int | None = None
    image_scale_applies_to_text: CfgRecipe = CfgRecipe.ADDITIVE_DELTAS
    extra: Any = None

    def __post_init__(self) -> None:
        recipe = CfgRecipe.coerce(self.image_scale_applies_to_text)
        if recipe is not self.image_scale_applies_to_text:
            object.__setattr__(self, "image_scale_applies_to_text", recipe)
        if self.cfg_branch_count is not None:
            branch_count = int(self.cfg_branch_count)
            if branch_count < 1:
                raise ValueError("cfg_branch_count must be >= 1")
            object.__setattr__(self, "cfg_branch_count", branch_count)


def flow_cfg_plan(step: PreparedFlowStep) -> DiffusionCfgPlan:
    """Resolve the branches and weights for one prepared denoise step."""

    if step.cfg_branch_count == 1:
        return DiffusionCfgPlan(branches=(Branch.COND,))
    t_value = float(step.t.detach().float().item())
    lo, hi = step.cfg_interval
    return build_flow_cfg_plan(
        cfg_text_scale=step.cfg_text_scale,
        cfg_img_scale=step.cfg_img_scale,
        recipe=step.image_scale_applies_to_text,
        renorm=step.cfg_renorm_type,
        renorm_min=step.cfg_renorm_min,
        use_cfg=lo <= t_value <= hi,
    )


def flow_branches(step: PreparedFlowStep) -> tuple[str, ...]:
    return flow_cfg_plan(step).branches


def flow_cfg_branch_count(op: Mapping[str, Any]) -> int | None:
    cfg = op.get("cfg")
    if not isinstance(cfg, Mapping) or cfg.get("branch_count") is None:
        return None
    try:
        branch_count = int(cfg["branch_count"])
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("cfg.branch_count must be a positive integer") from exc
    if branch_count < 1:
        raise invalid_descriptor("cfg.branch_count must be a positive integer")
    return branch_count


def combine_flow_velocity(
    step: PreparedFlowStep,
    outputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return flow_cfg_plan(step).combine(outputs)


class _GraphBackendUnplanned(RuntimeError):
    """Capture completed but the exclusive wrapper was never planned."""


@dataclass
class FlowGraphState:
    """Static buffers and one captured denoise geometry graph."""

    key: tuple[Any, ...]
    graph: torch.cuda.CUDAGraph
    image_embeds: torch.Tensor  # [rows, tokens, hidden] static input
    t: torch.Tensor  # timestep static input (shape of step.t)
    z: torch.Tensor  # [rows, tokens, latent] static input
    indexes: torch.Tensor  # [3, rows, tokens] static input
    cache: BatchedPagedTextCache  # batched view over the rows' caches
    request_cache: Any  # stable graph-owned paged side tables
    plan: PagedVarlenPlan  # stable transient paged-varlen plan
    graph_binding: GraphBinding  # wrapper-routing identity (kept alive here)
    backend: Any
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    scale: float
    image_token_num: int
    image_size: tuple[int, int]
    release_backend: Any = None  # callable dropping the exclusive wrapper
    logits: Any = None  # (velocity, hidden|None) written by capture/replay


class FlowGraphRunner(FailureManagedRunner):
    """Own bounded denoise-step CUDA graphs and refreshable request inputs."""

    def __init__(
        self,
        *,
        name: str = "denoise_step",
        default_enabled: bool | None = None,
        logger: Any = logger,
    ) -> None:
        super().__init__(
            name=name,
            default_enabled=True if default_enabled is None else bool(default_enabled),
            default_warmup=False,
            metric_prefix="denoise_",
            logger=logger,
        )

    def capture_pool(self) -> Any:
        # Independent geometry graphs can replay in any order. Keep their static
        # allocations isolated instead of sharing addresses across graph shapes.
        return None

    # -- public entry ------------------------------------------------------------

    def maybe_run_rows(
        self,
        owner: Any,
        rows: "Sequence[FlowRow]",
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Capture or replay the denoise step for ``rows``; ``None`` is a miss.

        Outputs are cloned per replay so nothing retained by the caller aliases
        the graph's static output buffers, which the next replay overwrites.
        """

        if not self.enabled() or not torch.cuda.is_available():
            return None
        if len(rows) > _MAX_GRAPH_ROWS:
            if return_hidden:
                paired_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
                for start in range(0, len(rows), _MAX_GRAPH_ROWS):
                    chunk = self.maybe_run_rows(
                        owner,
                        rows[start : start + _MAX_GRAPH_ROWS],
                        return_hidden=True,
                    )
                    if chunk is None:
                        return None
                    assert isinstance(chunk, tuple)
                    paired_chunks.append(chunk)
                velocity_chunks, hidden_chunks = zip(*paired_chunks, strict=True)
                return torch.cat(velocity_chunks, dim=0), torch.cat(hidden_chunks, dim=0)
            tensor_chunks: list[torch.Tensor] = []
            for start in range(0, len(rows), _MAX_GRAPH_ROWS):
                chunk = self.maybe_run_rows(
                    owner,
                    rows[start : start + _MAX_GRAPH_ROWS],
                    return_hidden=False,
                )
                if chunk is None:
                    return None
                assert isinstance(chunk, torch.Tensor)
                tensor_chunks.append(chunk)
            return torch.cat(tensor_chunks, dim=0)
        if not self._ensure_transient_capacity(rows):
            return None
        key = self._rows_key(rows, return_hidden)
        ctx = get_forward_context()
        tokens = len(rows) * int(rows[0].img.token_h) * int(rows[0].img.token_w)
        if key is None or key in self.disabled:
            self.record_event(ctx, Event.MISS, tokens)
            return None
        if key not in self.states and not ctx.allow_capture:
            self.record_event(ctx, Event.MISS, tokens)
            return None
        if key not in self.states:  # capture path resolves the winner backend
            backend = self._resolve_graph_backend(ctx, rows)
            if backend is None:
                # Dispatcher-winner probe failed: the backend eager would pick
                # cannot host the captured graph path. Config-wide, so disable
                # the runner rather than accumulate geometry-specific miss keys.
                self.reject_backend()
                if self.logger is not None:
                    self.logger.info(
                        "%s CUDA graph disabled: eager attention winner cannot host a "
                        "graph-scoped prefill plan",
                        self.name,
                    )
                self.record_event(ctx, Event.MISS, tokens)
                return None
        else:
            backend = None  # replay path never touches the backend

        out = self._capture_or_replay(
            key=key,
            device=rows[0].step.extra["image_embeds"].device,
            ctx=ctx,
            capture=lambda: self._capture(owner, rows, key, ctx, backend, return_hidden),
            copy_inputs=lambda state: self._copy_inputs(state, rows),
            replay=self._replay,
            record=lambda event: self.record_event(ctx, event, tokens),
            disable=lambda exc: self.disable_state(key, exc),
            capture_metric=f"{self.metric_prefix}step_graph_capture",
            input_copy_metric=f"{self.metric_prefix}step_graph_input_copy",
            replay_metric=f"{self.metric_prefix}step_graph_replay_launch",
            after_copy=self._prepare_backend,
            after_copy_metric=f"{self.metric_prefix}step_graph_attention_prepare",
        )
        if out is None:
            return None
        if self.replay_completed() and self.logger is not None:
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
    def _ensure_transient_capacity(rows: "Sequence[FlowRow]") -> bool:
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
    def _rows_key(rows: "Sequence[FlowRow]", return_hidden: bool) -> tuple[Any, ...] | None:
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

    # -- backend probe ---------------------------------------------------------

    def _resolve_graph_backend(self, ctx: Any, rows: "Sequence[FlowRow]") -> Any | None:
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
                    backend = getattr(req, "backend", None) or getattr(
                        ctx, "attention_backend", None
                    )
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
        rows: "Sequence[FlowRow]",
        key: tuple[Any, ...],
        ctx: Any,
        backend: Any,
        return_hidden: bool,
    ) -> FlowGraphState:
        first = rows[0]
        img = first.img
        device = first.step.extra["image_embeds"].device
        cache = BatchedPagedTextCache(
            [row.cache for row in rows],
            block_table_width=int(key[2]),
        )
        request_cache = cache.request_cache_for_transient(0, int(img.token_h) * int(img.token_w))
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
        geometry = getattr(owner, "query_geometry", None)
        if callable(geometry):
            num_q_heads, scale, _q_dtype = geometry()
        else:
            first_attention = getattr(owner, "attn", None)
            if first_attention is None:
                raise RuntimeError("denoise graph owner does not expose query geometry")
            num_q_heads = int(first_attention.num_heads)
            scale = float(first_attention.scale)
        state = FlowGraphState(
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
                out = owner.flow_predict_velocity(
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
    def _copy_inputs(state: FlowGraphState, rows: "Sequence[FlowRow]") -> None:
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
    def _prepare_backend(state: FlowGraphState) -> None:
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
    def _replay(state: FlowGraphState) -> tuple[torch.Tensor, torch.Tensor | None]:
        state.graph.replay()
        return state.logits


def run_flow_graph(
    adapter: Any,
    rows: "Sequence[FlowRow]",
    *,
    return_hidden: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run flow rows through the executor-owned graph view.

    A forward without a flow graph runner in its view is a graph miss: the
    caller falls back to the equivalent eager execution.
    """

    runner = getattr(get_forward_context().graph_view, "flow", None)
    if runner is None:
        return None
    return runner.maybe_run_rows(adapter, rows, return_hidden=return_hidden)


# ---------------------
# Flow state machine
# ---------------------


@dataclass
class FlowState:
    """Mutable denoise state for one in-flight image generation request.

    The latent trajectory ``x_t`` is not stored on the model state — it lives in
    the system-owned :class:`~uniserve_worker.runtime.residency.LatentStore`
    as a leased buffer addressed by ``latent_handle``. ``x_t`` here is a property
    reading/writing that system buffer.
    """

    latent_pool: Any  # system LatentStore (residency.latent)
    latent_handle: int  # request-scoped handle into the LatentStore
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

    @property
    def x_t(self) -> torch.Tensor:
        return self.latent_pool.get(self.latent_handle)

    @x_t.setter
    def x_t(self, value: torch.Tensor) -> None:
        self.latent_pool.set(self.latent_handle, value)


@dataclass
class FlowRow:
    """One CFG branch of one denoise step queued for batched velocity prediction."""

    step_index: int
    step: PreparedFlowStep
    branch: str
    img: FlowState
    indexes: torch.Tensor
    cache: PagedTextCache


@dataclass
class ProgramState:
    """Per-request state shared by composed sequence, flow, and product operations."""

    sampling: dict = field(default_factory=dict)
    image: dict = field(default_factory=dict)
    neg_token_ids: list[int] = field(default_factory=list)
    cond: SequenceCache = field(default_factory=SequenceCache)
    tu: SequenceCache = field(default_factory=SequenceCache)
    iu: SequenceCache = field(default_factory=SequenceCache)
    rng: torch.Generator | None = None
    _image_state: FlowState | None = field(default=None, repr=False)
    _latent_store: Any = field(default=None, init=False, repr=False)
    _request_id: int | None = field(default=None, init=False, repr=False)

    def bind_latent(self, store: Any, request_id: int) -> None:
        self._latent_store = store
        self._request_id = int(request_id)

    @property
    def image_state(self) -> FlowState | None:
        if self._latent_store is None or self._request_id is None:
            return self._image_state
        return self._latent_store.state(self._request_id)

    @image_state.setter
    def image_state(self, value: FlowState | None) -> None:
        if self._latent_store is None or self._request_id is None:
            self._image_state = value
            return
        if value is None:
            self._latent_store.pop_state(self._request_id)
        else:
            self._latent_store.set_state(self._request_id, value)


@dataclass(frozen=True)
class _ProgramSnapshot:
    """One request's committed branch state captured for a step transaction.

    ``caches`` records each live branch cache with its committed length and
    block table; ``cond_blocks_shared`` records whether the conditional branch
    shares its block-id list with the request session, so a restore rebuilds
    the same single block-table list.
    """

    program: ProgramState
    caches: tuple[tuple[Any, int, tuple[int, ...]], ...]
    cond_blocks_shared: bool


class FlowBranchStore:
    """Own per-request conditional and CFG sequence branches.

    The branch caches hold the committed sequence-KV lengths and block tables
    for interleaved execution; ``snapshot_requests``/``restore_requests`` make
    them transactional, so a failed step leaves the committed length and block
    table of every touched request unchanged.
    """

    def __init__(self, sessions: Any, latents: Any, *, rng_device: Any) -> None:
        self.sessions = sessions
        self.latents = latents
        self.rng_device = rng_device
        self._programs: dict[int, ProgramState] = {}

    def program(self, request_id: int) -> ProgramState:
        request_id = int(request_id)
        session = self.sessions.get(request_id)
        program = self._programs.get(request_id)
        if program is None:
            program = ProgramState(
                sampling=dict(session.sampling or {}),
                image=dict(session.image or {}),
                neg_token_ids=list(session.neg_token_ids or []),
                rng=session.device_rng(self.rng_device),
            )
            program.cond.block_ids = session.block_ids
            program.bind_latent(self.latents, request_id)
            self._programs[request_id] = program
            return program
        if session.sampling:
            program.sampling = dict(session.sampling)
        if session.image:
            program.image = dict(session.image)
        if session.neg_token_ids:
            program.neg_token_ids = list(session.neg_token_ids)
        if not program.cond.block_ids and session.block_ids:
            program.cond.block_ids = session.block_ids
            set_blocks = getattr(program.cond.past, "set_blocks", None)
            if callable(set_blocks):
                set_blocks(program.cond.block_ids)
        return program

    def view(self, request_ids: Iterable[int]) -> "KvView":
        return KvView(self, frozenset(int(value) for value in request_ids))

    def live_cache_ids(self) -> set[int]:
        return {
            id(cache)
            for program in self._programs.values()
            for cache in (program.cond.past, program.tu.past, program.iu.past)
            if cache is not None
        }

    def drop(self, request_id: int, *, residency: Any, segment_executor: Any) -> None:
        program = self._programs.pop(int(request_id), None)
        if program is None:
            return
        seen: set[int] = set()
        for branch in (program.cond, program.tu, program.iu):
            cache = branch.past
            if cache is None or id(cache) in seen:
                continue
            seen.add(id(cache))
            if segment_executor is not None:
                segment_executor.release_staging(cache)
            residency.release_scratch_cache(cache)

    def snapshot_requests(self, request_ids: set[int]) -> dict[int, _ProgramSnapshot]:
        return {
            request_id: self._snapshot(request_id, program)
            for request_id in {int(value) for value in request_ids}
            if (program := self._programs.get(request_id)) is not None
        }

    def restore_requests(
        self,
        request_ids: set[int],
        snapshot: dict[int, _ProgramSnapshot],
    ) -> None:
        for request_id in {int(value) for value in request_ids}:
            self._programs.pop(request_id, None)
        for request_id, snap in snapshot.items():
            program = snap.program
            if snap.cond_blocks_shared:
                program.cond.block_ids = self.sessions.get(request_id).block_ids
            for cache, length, block_ids in snap.caches:
                cache.restore_committed(length, block_ids)
            self._programs[request_id] = program

    def _snapshot(self, request_id: int, program: ProgramState) -> _ProgramSnapshot:
        cloned = copy.copy(program)
        cloned.sampling = dict(program.sampling)
        cloned.image = dict(program.image)
        cloned.neg_token_ids = list(program.neg_token_ids)
        caches: list[tuple[Any, int, tuple[int, ...]]] = []
        seen: set[int] = set()
        for name in ("cond", "tu", "iu"):
            branch = copy.copy(getattr(program, name))
            branch.block_ids = list(branch.block_ids)
            setattr(cloned, name, branch)
            past = branch.past
            if (
                past is not None
                and id(past) not in seen
                and callable(getattr(past, "restore_committed", None))
            ):
                seen.add(id(past))
                caches.append(
                    (past, int(past.length), tuple(int(block) for block in past.block_ids))
                )
        return _ProgramSnapshot(
            program=cloned,
            caches=tuple(caches),
            cond_blocks_shared=(
                program.cond.block_ids is self.sessions.get(request_id).block_ids
            ),
        )


class KvView:
    """Batch-bounded access to request KV branch state."""

    def __init__(self, store: FlowBranchStore, request_ids: frozenset[int]) -> None:
        self._store = store
        self._request_ids = request_ids

    def program(self, request_id: int) -> ProgramState:
        request_id = int(request_id)
        if request_id not in self._request_ids:
            raise invalid_descriptor(f"request {request_id} is outside the KV view")
        return self._store.program(request_id)


class LatentView:
    """Batch-bounded access to active flow or generation state."""

    def __init__(
        self,
        store: Any,
        request_ids: Iterable[int],
        *,
        kv_store: FlowBranchStore,
        residency: Any,
        segment_executor: Any,
    ) -> None:
        self._store = store
        self._request_ids = frozenset(int(value) for value in request_ids)
        self._kv_store = kv_store
        self._residency = residency
        self._segment_executor = segment_executor

    def state(self, request_id: int) -> Any | None:
        self._require(request_id)
        return self._store.state(int(request_id))

    def set_state(self, request_id: int, state: Any) -> None:
        self._require(request_id)
        self._store.set_state(int(request_id), state)

    def pop_state(self, request_id: int) -> Any | None:
        self._require(request_id)
        state = self._store.pop_state(int(request_id))
        self.release_state(state)
        return state

    def release_state(self, state: Any | None) -> None:
        if state is None:
            return
        latent_handle = getattr(state, "latent_handle", None)
        if latent_handle is not None:
            self._store.free(int(latent_handle))
        branches = getattr(state, "paged_branches", None)
        if branches is not None:
            branches.release(self._residency)
            state.paged_branches = None
        if hasattr(state, "graph_image"):
            state.graph_image = None
        live_cache_ids = self._kv_store.live_cache_ids()
        seen: set[int] = set()
        for cache in (
            getattr(state, "cond_cache", None),
            getattr(state, "tu_cache", None),
            getattr(state, "iu_cache", None),
        ):
            cache_id = id(cache)
            if cache is None or cache_id in seen or cache_id in live_cache_ids:
                continue
            seen.add(cache_id)
            if self._segment_executor is not None:
                self._segment_executor.release_staging(cache)
            self._residency.release_scratch_cache(cache)

    def _require(self, request_id: int) -> None:
        request_id = int(request_id)
        if request_id not in self._request_ids:
            raise invalid_descriptor(f"request {request_id} is outside the latent view")


class ProductTransferState(Protocol):
    """Transfer state consulted while constructing a flow operation."""

    @property
    def distributed(self) -> bool: ...


class FlowAdapter(Protocol):
    """Family boundary required by system flow execution.

    ``FlowExecution`` owns operation state, branch planning, scheduling, and
    updates while the adapter supplies family-specific neural computation.
    """

    # Collaborator attributes.
    device: Any
    gen_device: Any
    latent_downsample: int
    merge_size: int
    _img_start_token: str
    residency: Any  # ResidencyManager; .latent backs FlowState.x_t
    attention_backend: str  # preferred attention provider ("auto" allowed)

    # Owner-supplied denoise configuration (models differ only by these enums).
    denoise_schedule_direction: ScheduleDirection
    denoise_schedule_shift_domain: ScheduleShiftDomain
    denoise_cfg_recipe: CfgRecipe

    # Collaborator methods.
    def _state(self, op: dict[str, Any]) -> "ProgramState": ...
    def _maybe_stage_conditioning_from_op(self, st: "ProgramState", op: dict[str, Any]) -> None: ...
    def _extend_cache_blocks(self, cache: "SequenceCache", op: dict[str, Any]) -> None: ...
    def _ensure_img_start(self, cache: "SequenceCache | None") -> None: ...
    def _prefix_from_query(self, query: str) -> "SequenceCache": ...
    def _empty_img_start_prefix(self) -> "SequenceCache": ...
    def _denoise_cache(self, cache: Any) -> Any: ...
    def _wait_gen_cache_ready(self, cache: Any) -> None: ...
    def flow_query(self, text: str, *, append_text: str) -> str: ...
    def flow_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: Any,
    ) -> torch.Tensor: ...
    def flow_predict_velocity(
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
    def image_patch_size(self) -> int: ...
    def image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor: ...
    def flow_feature_dtype(self) -> torch.dtype: ...
    def flow_noise_scale(self, grid_h: int, grid_w: int) -> float: ...
    def flow_noise_scale_embedding(
        self,
        noise_scale: float,
        token_count: int,
        *,
        dtype: torch.dtype,
        device: Any,
    ) -> torch.Tensor | None: ...
    def flow_timestep_embeddings(self, t_values: torch.Tensor) -> torch.Tensor: ...


class FlowExecution:
    """Implement flow setup, branch batching, velocity prediction, and updates."""

    def __init__(
        self,
        adapter: FlowAdapter,
        *,
        transfer: ProductTransferState | None = None,
    ) -> None:
        self.adapter = adapter
        self.transfer = transfer

    @property
    def distributed(self) -> bool:
        return self.transfer is not None and self.transfer.distributed

    def _init_image_state(
        self,
        st: ProgramState,
        op: dict | None = None,
    ) -> FlowState:
        ip = st.image or {}
        params = self._parse_image_params(ip)
        cond = self._setup_cfg_caches(st, op, params)

        adapter = self.adapter
        token_h = params.height // adapter.latent_downsample
        token_w = params.width // adapter.latent_downsample
        patch_size = adapter.image_patch_size()
        grid_h = params.height // patch_size
        grid_w = params.width // patch_size
        device = getattr(adapter, "gen_device", adapter.device)
        indexes_cond, indexes_tu, indexes_iu = self._build_indexes(
            st, cond, token_h, token_w, device
        )

        schedule = FlowMatchSchedule(
            num_steps=params.steps,
            shift=params.timestep_shift,
            direction=adapter.denoise_schedule_direction,
            shift_domain=adapter.denoise_schedule_shift_domain,
        )
        timesteps = schedule.timesteps(device=device)
        grid_hw = torch.tensor([[grid_h, grid_w]], device=device)
        noise_scale = self._compute_noise_scale(grid_h, grid_w)
        noise_scale_embedding = adapter.flow_noise_scale_embedding(
            noise_scale,
            token_h * token_w,
            dtype=timesteps.dtype,
            device=device,
        )

        x_t = self._init_latent(st, params, device, noise_scale)
        cond_cache = adapter._denoise_cache(cond.past)
        tu_cache = adapter._denoise_cache(st.tu.past)
        iu_cache = adapter._denoise_cache(st.iu.past)
        # The latent lives in the system-owned LatentStore, keyed by the request
        # handle; ``FlowState.x_t`` reads/writes that buffer.
        latent_handle = int(op["req_id"]) if op and "req_id" in op else id(st)
        image_state = FlowState(
            latent_pool=adapter.residency.latent,
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
        image_state.x_t = x_t  # store the initial noise into the system LatentStore
        return image_state

    def _parse_image_params(self, ip: dict) -> _ImageParams:
        return parse_text_image_generation_params(ip)

    def _setup_cfg_caches(
        self,
        st: ProgramState,
        op: dict | None,
        params: _ImageParams,
    ) -> SequenceCache:
        """Prepare the cond/text-uncond/img-uncond text caches for denoising.

        Returns the conditioning :class:`SequenceCache` to use (a fresh image-prompt
        prefix when ``op`` supplies one, otherwise the request's own ``st.cond``).
        Mutates ``st.tu``/``st.iu`` in place as required by the CFG scales.
        """
        if params.retain_images:
            self.adapter._ensure_img_start(st.cond)
        cond = st.cond
        image_prompt = (op or {}).get("image_prompt")
        if isinstance(image_prompt, str) and image_prompt.strip():
            if self.distributed:
                raise invalid_descriptor(
                    "Mode A cuda_ipc tower split does not support per-op image_prompt overrides yet"
                )
            query = self.adapter.flow_query(
                image_prompt.strip(),
                append_text=self.adapter._img_start_token,
            )
            cond = self.adapter._prefix_from_query(query)
        elif not params.retain_images:
            self.adapter._ensure_img_start(st.cond)
        cfg_plan = build_flow_cfg_plan(
            cfg_text_scale=params.cfg_text,
            cfg_img_scale=params.cfg_img,
            recipe=self.adapter.denoise_cfg_recipe,
            renorm=params.cfg_norm,
            renorm_min=params.cfg_renorm_min,
        )
        needs_text_uncond = Branch.TEXT_UNCOND in cfg_plan.branches
        if needs_text_uncond and st.tu.past is None:
            if self.distributed:
                raise invalid_descriptor(
                    "Mode A denoise is missing pre-staged text-unconditional CFG KV"
                )
            st.tu = self.adapter._empty_img_start_prefix()
        elif needs_text_uncond:
            self.adapter._ensure_img_start(st.tu)
        needs_img_uncond = Branch.IMG_UNCOND in cfg_plan.branches
        if needs_img_uncond and st.iu.past is None:
            if self.distributed:
                raise invalid_descriptor(
                    "Mode A denoise is missing pre-staged image-unconditional CFG KV"
                )
            st.iu = self.adapter._empty_img_start_prefix()
        elif needs_img_uncond:
            self.adapter._ensure_img_start(st.iu)
        return cond

    def _build_indexes(
        self,
        st: ProgramState,
        cond: SequenceCache,
        token_h: int,
        token_w: int,
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        indexes_cond = self.adapter.flow_indexes(token_h, token_w, cond.t_index + 1, device=device)
        indexes_tu = (
            self.adapter.flow_indexes(token_h, token_w, st.tu.t_index + 1, device=device)
            if st.tu.past is not None
            else None
        )
        indexes_iu = (
            self.adapter.flow_indexes(token_h, token_w, st.iu.t_index + 1, device=device)
            if st.iu.past is not None
            else None
        )
        return indexes_cond, indexes_tu, indexes_iu

    def _compute_noise_scale(self, grid_h: int, grid_w: int) -> float:
        return self.adapter.flow_noise_scale(grid_h, grid_w)

    def _init_latent(
        self,
        st: ProgramState,
        params: _ImageParams,
        device: Any,
        noise_scale: float,
    ) -> torch.Tensor:
        if st.rng is None:
            seed = params.seed
            st.rng = torch.Generator(device=device).manual_seed(
                int(seed if seed is not None else 0)
            )
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

    def prepare_flow_step(self, req_id: int, state: Any, op: dict) -> PreparedFlowStep:
        ctx = get_forward_context()
        start = ctx.component_timer_start()
        adapter = self.adapter
        st = adapter._state(dict(op))
        # Mode A (tower disaggregation): the gen pool never ran the und text, so
        # rebuild st.cond from the conditioning KV the und pool published (carried
        # on op["locator"]). A no-op in Mode C / single-device.
        adapter._maybe_stage_conditioning_from_op(st, op)
        adapter._extend_cache_blocks(st.cond, op)
        if st.image_state is None:
            st.image_state = self._init_image_state(st, op)
        img = st.image_state
        step_i = int(op.get("timestep_idx") or 0)
        ctx.record_component_elapsed("flow_prepare_state", start)
        # Gen-tower feature extraction and timestep embedding run on the gen
        # coordinate's device (the gen modules are Pinned there); the tower
        # transport, not a dedicated stream, orders the und->gen handoff.
        device = getattr(adapter, "gen_device", adapter.device)
        t, t_next = img.schedule.pair(step_i, device=device, dtype=img.timesteps.dtype)
        start = ctx.component_timer_start()
        z = patchify_batch(img.x_t, adapter.latent_downsample)
        image_input = patchify_batch(
            img.x_t,
            adapter.image_patch_size(),
            channel_first=True,
        )
        image_input = image_input.to(
            device=device,
            dtype=adapter.flow_feature_dtype(),
        )
        ctx.record_component_elapsed("flow_patchify", start)
        start = ctx.component_timer_start()
        image_embeds = adapter.image_features(
            image_input.view(1 * img.grid_h * img.grid_w, -1),
            gen_model=True,
            grid_hw=img.grid_hw,
        ).view(1, img.token_h * img.token_w, -1)
        ctx.record_component_elapsed("flow_vision_feature", start)
        start = ctx.component_timer_start()
        t_expanded = t.expand(img.token_h * img.token_w)
        timestep_embeddings = adapter.flow_timestep_embeddings(t_expanded).view(
            1, img.token_h * img.token_w, -1
        )
        if img.noise_scale_embedding is None:
            img.noise_scale_embedding = adapter.flow_noise_scale_embedding(
                img.noise_scale,
                img.token_h * img.token_w,
                dtype=t_expanded.dtype,
                device=device,
            )
        if img.noise_scale_embedding is not None:
            timestep_embeddings += img.noise_scale_embedding
        image_embeds = image_embeds + timestep_embeddings
        ctx.record_component_elapsed("flow_timestep_embed", start)
        total = int(img.schedule.num_steps)
        return PreparedFlowStep(
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
            cfg_branch_count=flow_cfg_branch_count(op),
            image_scale_applies_to_text=adapter.denoise_cfg_recipe,
            extra={
                "img": img,
                "image_embeds": image_embeds,
            },
        )

    def predict_flow_velocity(
        self,
        step: PreparedFlowStep,
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

    def _predict_row(self, row: FlowRow) -> torch.Tensor:
        velocity = self.predict_flow_velocity(row.step, row.branch)
        assert isinstance(velocity, torch.Tensor)
        return velocity

    def predict_flow_velocity_batch(
        self,
        steps: Sequence[PreparedFlowStep],
        branches_by_step: Sequence[Sequence[str]],
        *,
        graph_mode: str = "auto",
    ) -> list[dict[str, torch.Tensor]] | None:
        results: list[dict[str, torch.Tensor]] = [dict() for _ in steps]
        rows: list[FlowRow] = []
        for step_index, (step, branches) in enumerate(zip(steps, branches_by_step)):
            img = step.extra["img"]
            for branch in branches:
                indexes, cache = self._denoise_branch_inputs(img, branch)
                if not isinstance(cache, PagedTextCache):
                    if graph_mode == "require":
                        return None
                    results[step_index][branch] = self._predict_row(
                        FlowRow(step_index, step, branch, img, indexes, cache)
                    )
                    continue
                rows.append(FlowRow(step_index, step, branch, img, indexes, cache))

        def predict_group(group: Sequence[FlowRow]) -> bool:
            batched = self._predict_v_batched(group, graph_mode=graph_mode)
            if batched is None:
                if graph_mode == "require":
                    return False
                for row in group:
                    results[row.step_index][row.branch] = self._predict_row(row)
                return True
            assert isinstance(batched, torch.Tensor)
            for row_index, row in enumerate(group):
                results[row.step_index][row.branch] = batched[
                    row_index : row_index + 1
                ].contiguous()
            return True

        if len(rows) == 1:
            if not predict_group(rows):
                return None
            return results

        grouped: dict[tuple[Any, ...], list[FlowRow]] = {}
        for row in rows:
            key = self._batched_denoise_row_key(row)
            if key is None:
                if graph_mode == "require":
                    return None
                results[row.step_index][row.branch] = self._predict_row(row)
                continue
            grouped.setdefault(key, []).append(row)

        for group in grouped.values():
            if not predict_group(group):
                return None
        return results

    def _denoise_branch_inputs(self, img: FlowState, branch: str) -> tuple[torch.Tensor, Any]:
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

    def _batched_denoise_row_key(self, row: FlowRow) -> tuple[Any, ...] | None:
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
            attention_backend=getattr(self.adapter, "attention_backend", "auto"),
        )

    def _predict_v_batched(
        self,
        rows: Sequence[FlowRow],
        *,
        return_hidden: bool = False,
        graph_mode: str = "auto",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        first = rows[0]
        img = first.img
        for row in rows:
            self.adapter._wait_gen_cache_ready(row.cache)
        # Capture or replay the whole compatible batched step as one CUDA graph.
        # Strict unified-forward callers use ``graph_mode="require"`` and reject
        # the batch when its geometry is not covered.
        if graph_mode != "eager":
            graphed = run_flow_graph(self.adapter, rows, return_hidden=return_hidden)
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

    def apply_flow_update(self, step: PreparedFlowStep, latent: torch.Tensor) -> None:
        img = step.extra["img"]
        img.x_t = unpatchify_batch(
            latent,
            self.adapter.latent_downsample,
            height=img.height,
            width=img.width,
        )

    def _predict_v(
        self,
        img: FlowState,
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
        self.adapter._wait_gen_cache_ready(cache)
        return self.adapter.flow_predict_velocity(
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
