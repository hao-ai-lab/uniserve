"""Shared model runner.

Owns request-state creation, wire-op parsing into ``UniForwardBatch``, model
``forward`` dispatch, and per-op result reassembly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ..backends.attention import (
    get_attention_backend,
    normalize_attention_backend_name,
)
from ..contracts.batch_policy import BatchPolicy
from ..contracts.batches import ExecuteBatch, UniForwardBatch
from ..contracts.caps import Caps
from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..contracts.forward_stats import ForwardStats
from ..contracts.model_protocols import ModelHooks, UniModel
from ..contracts.resource_plan import LatentTokens, ResourcePlan
from ..foundation.env import env_flag
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import get_worker_config
from ..runtime.forward_batch_builder import ForwardBatchBuilder
from ..runtime.request_session import RequestSessionTable
from ..runtime.residency_manager import ResidencyLeaseManager
from ..runtime.resources import ResourceRuntime
from .denoise_driver import DenoiseDriver
from .encode_driver import EncodeDriver
from .forward import (
    EagerFallbackRecorder,
    ForwardExecutor,
    ForwardGraphPolicy,
    ForwardGroupPlanner,
    ForwardPlanBuilder,
    ForwardPostprocessor,
    ForwardStepExecutor,
    ForwardStepOptions,
    WorkerForwardAdapter,
)
from .forward import (
    ForwardBatchBuilder as UnifiedForwardBatchBuilder,
)
from .image_decode_driver import ImageDecodeDriver
from .text_driver import TextDriver

if TYPE_CHECKING:
    from ..runtime.residency import ResidencyManager

__all__ = [
    'ModelRunner',
    'RunnerConfig',
    'RunnerDrivers',
]

# Modes routed to the typed text driver.
_TEXT_DRIVER_MODES = frozenset(
    {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY}
)

logger = logging.getLogger(__name__)
_MIXED_PROOF_LOG = logging.getLogger("uniserve.mixed_proof")
# Per-step mixed-forward trace (opt-in). Fallback warnings are always emitted.
_MIXED_PROOF_ENABLED = env_flag("UNISERVE_MIXED_PROOF_LOG")


def _model_max_context_len(model: Any) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "max_position_embeddings", None)
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


@dataclass
class RunnerDrivers:
    """Optional per-modality driver overrides for ``ModelRunner``.

    Each is constructed with a default when left ``None``; the text driver is
    additionally wired to the system text-execution stack built in ``__init__``.
    """

    denoise_driver: DenoiseDriver | None = None
    encode_driver: EncodeDriver | None = None
    image_decode_driver: ImageDecodeDriver | None = None
    text_driver: TextDriver | None = None


@dataclass
class RunnerConfig:
    """Non-driver ``ModelRunner`` configuration knobs.

    Groups the sampler-stage split (``defer_sampling``/``tensor_store``),
    multimodal processor, batch policy, and attention-backend selection.
    """

    batch_policy: BatchPolicy | None = None
    attention_backend: Any | None = None
    multimodal_processor: Any | None = None
    defer_sampling: bool = False
    tensor_store: Any | None = None
    simulation: bool = False


@dataclass
class _ResolvedRunnerDeps:
    batch_policy: BatchPolicy | None
    attention_backend: Any | None
    denoise_driver: DenoiseDriver | None
    encode_driver: EncodeDriver | None
    image_decode_driver: ImageDecodeDriver | None
    text_driver: TextDriver | None
    multimodal_processor: Any | None
    defer_sampling: bool
    tensor_store: Any | None
    simulation: bool


@dataclass(frozen=True)
class _TextExecutionStack:
    builder: ForwardBatchBuilder | None
    gate: Any | None
    graph_runner: Any | None


class ModelRunner:
    """Dispatches wire ops through modality drivers and resource accounting."""

    def __init__(
        self,
        model: UniModel,
        request_states: RequestSessionTable | None = None,
        *,
        drivers: RunnerDrivers | None = None,
        config: RunnerConfig | None = None,
        resource_runtime: ResourceRuntime | None = None,
        residency: "ResidencyManager | None" = None,
        batch_policy: BatchPolicy | None = None,
        attention_backend: Any | None = None,
        denoise_driver: DenoiseDriver | None = None,
        encode_driver: EncodeDriver | None = None,
        image_decode_driver: ImageDecodeDriver | None = None,
        text_driver: TextDriver | None = None,
        multimodal_processor: Any | None = None,
        defer_sampling: bool = False,
        tensor_store: Any | None = None,
    ):
        if not isinstance(model, ModelHooks):
            raise capability_mismatch("runner model must inherit ModelHooks")
        deps = self._resolve_deps(
            drivers=drivers,
            config=config,
            batch_policy=batch_policy,
            attention_backend=attention_backend,
            denoise_driver=denoise_driver,
            encode_driver=encode_driver,
            image_decode_driver=image_decode_driver,
            text_driver=text_driver,
            multimodal_processor=multimodal_processor,
            defer_sampling=defer_sampling,
            tensor_store=tensor_store,
        )

        self.model = model
        self.residency = residency
        self.request_states = request_states or RequestSessionTable()
        # Deferred sampling: text decode/extend ops publish logits to
        # ``tensor_store`` and return handles — a separate Sampler worker samples.
        # Off = sample inline (default).
        self.defer_sampling = bool(deps.defer_sampling) and deps.tensor_store is not None
        self.tensor_store = deps.tensor_store
        self.simulation = bool(deps.simulation)
        self.batch_policy = deps.batch_policy or self._model_batch_policy()
        self.attention_backend, self.attention_backend_name = self._resolve_attention_backend(
            deps.attention_backend
        )
        self.denoise_driver = deps.denoise_driver or DenoiseDriver()
        self.encode_driver = deps.encode_driver or EncodeDriver()
        self.image_decode_driver = deps.image_decode_driver or ImageDecodeDriver()
        self._init_text_execution(model, residency, deps.text_driver)
        self._init_unified_forward_execution(model, residency)
        self.multimodal_processor = deps.multimodal_processor
        self._init_resource_accounting(resource_runtime, residency)
        # CUDA Green Context SM partitioning. ``None`` unless runtime config
        # enables it and the model runs on a CUDA device.
        self.stream_manager = self._maybe_build_stream_manager()
        self._group_planner = ForwardGroupPlanner(
            self.batch_policy,
            log_text_mixed_split=self._log_text_mixed_split,
            can_run_forward=self.forward_adapter.can_run_forward,
        )
        self._step_executor = ForwardStepExecutor(self, group_planner=self._group_planner)

    def _init_unified_forward_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> None:
        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        kv_pool = residency.kv if residency is not None else None
        self.forward_plan_builder = ForwardPlanBuilder()
        self.unified_forward_batch_builder = UnifiedForwardBatchBuilder(
            runtime_builder=self.forward_batch_builder,
            kv_pool=kv_pool,
            request_states=self.request_states,
            default_device=device,
        )
        self.forward_graph_policy = ForwardGraphPolicy(
            prefer_graph=bool(get_worker_config().cuda_graph),
            strict=not self.simulation,
        )
        self.forward_graph_runner = self._build_forward_graph_runner(model)
        self.forward_fallback_recorder = EagerFallbackRecorder()
        self.forward_adapter = WorkerForwardAdapter(
            model=model,
            request_states=self.request_states,
            text_driver=self.text_driver,
            denoise_driver=self.denoise_driver,
            encode_driver=self.encode_driver,
            image_decode_driver=self.image_decode_driver,
            defer_sampling=self.defer_sampling,
            tensor_store=self.tensor_store,
            mixed_proof_callback=self._log_mixed_proof if _MIXED_PROOF_ENABLED else None,
        )
        self.forward_executor = ForwardExecutor(
            model=self.forward_adapter,
            graph_runner=self.forward_graph_runner,
            graph_policy=self.forward_graph_policy,
            fallback_recorder=self.forward_fallback_recorder,
        )
        self.forward_postprocessor = ForwardPostprocessor()

    def _build_forward_graph_runner(self, model: UniModel) -> Any | None:
        from .forward.graph import (
            CudaGraphForwardRunner,
            DenoiseStepGraphProgram,
            ForwardGraphProgram,
            PackedVisibleGraphProgram,
        )
        programs: list[ForwardGraphProgram] = []

        if self.text_graph_runner is not None:
            from .forward.graph import (
                DecodeGraphProgram,
                PrefillGraphProgram,
            )

            programs.extend(
                (
                    DecodeGraphProgram(
                        text_driver=self.text_driver,
                        model=model,
                        request_states=self.request_states,
                        text_graph_runner=self.text_graph_runner,
                    ),
                    PrefillGraphProgram(
                        text_driver=self.text_driver,
                        model=model,
                        request_states=self.request_states,
                        text_graph_runner=self.text_graph_runner,
                    ),
                )
            )
        programs.append(
            PackedVisibleGraphProgram(
                owner=model,
                request_states=self.request_states,
                image_decode_driver=self.image_decode_driver,
            )
        )
        if callable(getattr(model, "try_run_text_graph_logits_batch", None)):
            from .forward.graph import ModelOwnedTextGraphProgram

            programs.append(
                ModelOwnedTextGraphProgram(
                    text_driver=self.text_driver,
                    model=model,
                    request_states=self.request_states,
                )
            )
        programs.append(
            DenoiseStepGraphProgram(
                denoise_driver=self.denoise_driver,
                model=model,
                request_states=self.request_states,
            )
        )
        if programs:
            return CudaGraphForwardRunner(programs=tuple(programs))
        return None

    def _init_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
        text_driver: TextDriver | None,
    ) -> None:
        # System text execution: a thin model declares its KV geometry and the
        # runtime owns builder/gate/graph/sampler around the graph-unaware model.
        text_stack = self._build_text_execution(model, residency)
        self.forward_batch_builder = text_stack.builder
        self.text_gate = text_stack.gate
        self.text_graph_runner = text_stack.graph_runner
        self.text_driver = text_driver or TextDriver(
            builder=self.forward_batch_builder,
            gate=self.text_gate,
            kv_pool=residency.kv if residency is not None else None,
        )

    def _init_resource_accounting(
        self,
        resource_runtime: ResourceRuntime | None,
        residency: "ResidencyManager | None",
    ) -> None:
        resource_plan = self._model_resource_plan()
        classes = resource_plan.classes()
        self.resource_runtime = resource_runtime or ResourceRuntime(
            classes,
            totals=self._model_resource_totals(classes),
        )
        # The accountant holds ``resource_plan`` as the single source of truth;
        # ``ModelRunner.resource_plan`` forwards to it so a runtime reassignment
        # is seen by both.
        self._accountant = ResidencyLeaseManager(
            self.resource_runtime,
            self.request_states,
            resource_plan,
            residency=residency,
        )

    @staticmethod
    def _resolve_deps(
        *,
        drivers: RunnerDrivers | None,
        config: RunnerConfig | None,
        batch_policy: BatchPolicy | None,
        attention_backend: Any | None,
        denoise_driver: DenoiseDriver | None,
        encode_driver: EncodeDriver | None,
        image_decode_driver: ImageDecodeDriver | None,
        text_driver: TextDriver | None,
        multimodal_processor: Any | None,
        defer_sampling: bool,
        tensor_store: Any | None,
    ) -> _ResolvedRunnerDeps:
        drivers = drivers or RunnerDrivers()
        config = config or RunnerConfig()
        return _ResolvedRunnerDeps(
            batch_policy=batch_policy if batch_policy is not None else config.batch_policy,
            attention_backend=(
                attention_backend if attention_backend is not None else config.attention_backend
            ),
            denoise_driver=denoise_driver if denoise_driver is not None else drivers.denoise_driver,
            encode_driver=encode_driver if encode_driver is not None else drivers.encode_driver,
            image_decode_driver=(
                image_decode_driver
                if image_decode_driver is not None
                else drivers.image_decode_driver
            ),
            text_driver=text_driver if text_driver is not None else drivers.text_driver,
            multimodal_processor=(
                multimodal_processor
                if multimodal_processor is not None
                else config.multimodal_processor
            ),
            defer_sampling=defer_sampling or config.defer_sampling,
            tensor_store=tensor_store if tensor_store is not None else config.tensor_store,
            simulation=bool(config.simulation),
        )

    @staticmethod
    def _resolve_attention_backend(attention_backend: Any | None) -> tuple[Any | None, str | None]:
        if isinstance(attention_backend, str) or attention_backend is None:
            attention_backend_name = normalize_attention_backend_name(attention_backend or "auto")
            if attention_backend_name != "auto":
                get_attention_backend(attention_backend_name)
            return None, attention_backend_name
        return attention_backend, getattr(attention_backend, "name", None)

    @property
    def resource_plan(self) -> ResourcePlan:
        return self._accountant.resource_plan

    @resource_plan.setter
    def resource_plan(self, plan: ResourcePlan) -> None:
        self._accountant.resource_plan = plan

    def _build_text_execution(
        self,
        model: UniModel,
        residency: "ResidencyManager | None",
    ) -> _TextExecutionStack:
        """Build the system text builder + backend gate + CUDA-graph runner.

        A thin model declares ``kv_cache_spec`` and the runtime owns its KV pool
        (``residency.kv``); the gate decides batched-paged vs per-op-dense from
        the model geometry + the system pool's storage flag, and the graph runner
        captures/replays decode/prefill graphs around the graph-unaware model.
        Models without ``kv_cache_spec`` (self-managed KV) get none of these.
        """

        if residency is None or residency.kv is None:
            return _TextExecutionStack(None, None, None)
        if self._model_kv_cache_spec(model) is None:
            return _TextExecutionStack(None, None, None)
        import torch

        from ..backends.attention.text_dispatch import TextBackendGate

        device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
        max_context_len = _model_max_context_len(model)
        gate = TextBackendGate(
            head_dim=int(getattr(model, "head_dim", residency.kv.head_dim)),
            block_size=int(residency.kv.block_size),
            device_type=device.type,
            paged_storage_ok=bool(getattr(residency.kv, "supports_paged_attention_storage", True)),
        )
        graph_runner = None
        if device.type == "cuda":
            from .forward.graph.text import TextGraphRunner

            graph_runner = TextGraphRunner(
                kv_pool=residency.kv,
                num_blocks=int(residency.kv.num_blocks),
                block_size=int(residency.kv.block_size),
                device=device,
                attention_backend_name=self.attention_backend_name,
                max_context_len=max_context_len,
            )
            try:
                graph_runner.warmup(model)
            except Exception:  # noqa: BLE001 - a warmup failure must never block serving.
                logger.warning("text CUDA-graph warmup failed; falling back to eager", exc_info=True)
        return _TextExecutionStack(
            ForwardBatchBuilder(max_context_len=max_context_len),
            gate,
            graph_runner,
        )

    @staticmethod
    def _model_kv_cache_spec(model: UniModel) -> Any | None:
        return model.kv_cache_spec()

    def _maybe_build_stream_manager(self):
        if not get_worker_config().green_contexts:
            return None
        import torch

        device = torch.device(str(getattr(self.model, "device", "cpu") or "cpu"))
        if device.type != "cuda":
            return None
        try:
            from ..runtime.stream_manager import StreamManager

            gpu_id = device.index if device.index is not None else torch.cuda.current_device()
            manager = StreamManager(int(gpu_id))
            logger.info(
                "green contexts enabled: %d SMs, %d stream groups (partitioned=%s)",
                manager.total_sms, len(manager.stream_groups), manager.using_green_contexts,
            )
            return manager
        except Exception:  # noqa: BLE001 - never let a stream-setup failure block serving.
            logger.warning("StreamManager init failed; green contexts disabled", exc_info=True)
            return None

    def _forward_stream_context(self, fb: UniForwardBatch):
        """Context manager that runs a single-mode group on its SM partition.

        Prefill/verify groups run on the prefill (large-SM) partition; decode
        groups on the decode partition sized by the running decode batch; mixed
        and non-text groups run full-SM (no partitioning helps a mixed forward).
        A no-op ``nullcontext`` when green contexts are disabled.
        """
        from contextlib import nullcontext

        if self.stream_manager is None:
            return nullcontext()
        mode = fb.mode
        if mode == ForwardMode.EXTEND or mode == ForwardMode.TARGET_VERIFY:
            stream = self.stream_manager.select_streams(0)[0]
        elif mode == ForwardMode.DECODE:
            stream = self.stream_manager.select_streams(len(fb.ops))[1]
        else:
            stream = self.stream_manager.default_stream()
        import torch

        return torch.cuda.stream(stream)

    def drop_request(self, req_id: int) -> None:
        self.model.drop_request(req_id)
        self._accountant.release_request(int(req_id))
        self.request_states.drop(req_id)

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        parsed = ExecuteBatch.from_wire(batch)
        return self._step_executor.execute(
            parsed,
            ForwardStepOptions(defer_text_cpu_results=defer_text_cpu_results),
        )

    def _register_new_reqs(self, new_reqs: tuple[Mapping[str, Any], ...]) -> None:
        """Create/refresh request state and account resident blocks for new reqs.

        A block-accounting failure rolls back any request *this* call freshly
        created (existing requests are left untouched) before re-raising.
        """
        for nr in new_reqs:
            req_id = nr["req_id"]
            existed = req_id in self.request_states
            state = self.request_states.create_or_update(req_id, dict(nr))
            try:
                self._accountant.account_blocks(req_id, state.block_ids, append_to_state=False)
            except Exception:
                if not existed:
                    self._accountant.release_request(int(req_id))
                    self.request_states.drop(req_id)
                raise
            self.model.on_new_request(req_id, state)

    def _stamp_conditioning_locators(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[Any],
    ) -> None:
        """Mode A: stamp the und->gen conditioning locator on a text result that
        begins an image, so the gen pool can fetch the conditioning KV.

        Delegates the decision to the model's ``maybe_publish_conditioning`` hook
        (a no-op unless a data-plane handoff is bound). Only fires for text-decode
        results carrying an inline sampled token (the image-start trigger)."""
        if fb.mode not in _TEXT_DRIVER_MODES:
            return
        for idx, op in group:
            result = results[idx]
            if not isinstance(result, dict):
                continue
            sampled = result.get("sampled_token_id")
            if sampled is None:
                continue
            locator = self.model.maybe_publish_conditioning(int(op["req_id"]), int(sampled))
            if locator:
                result["locator"] = locator

    def _log_mixed_proof(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
    ) -> None:
        # Per-step "mixed forward executed" trace (opt-in via the proof flag).
        n_ext = sum(1 for m in fb.op_modes if m == ForwardMode.EXTEND)
        n_dec = sum(1 for m in fb.op_modes if m == ForwardMode.DECODE)
        n_den = sum(1 for m in fb.op_modes if m == ForwardMode.DENOISE)
        n_tok = sum(len(op.get("token_ids") or []) for _, op in group)
        _MIXED_PROOF_LOG.info(
            "MIXED FORWARD executed: %d ops (%d extend + %d decode + %d denoise), %d tokens",
            len(fb.ops), n_ext, n_dec, n_den, n_tok,
        )

    def _log_text_mixed_split(self, ops: list[Mapping[str, Any]], decision: Any) -> None:
        # Warn when a prefill+decode text mix is split to per-mode groups.
        modes = {mode_for_op(str(op.get("kind"))) for op in ops}
        if {ForwardMode.EXTEND, ForwardMode.DECODE} <= modes:
            _MIXED_PROOF_LOG.warning(
                "MIXED TEXT SPLIT: scheduler co-batched %d ops with both prefill+decode "
                "and runner is splitting per-mode (use_forward=%s)",
                len(ops), decision.use_forward,
            )

    def _model_batch_policy(self) -> BatchPolicy:
        policy = self.model.batch_policy()
        if isinstance(policy, BatchPolicy):
            return policy
        raise invalid_descriptor("model.batch_policy() must return BatchPolicy")

    def _model_resource_plan(self) -> ResourcePlan:
        return self.model.resource_plan

    def _model_resource_totals(self, classes: tuple[str, ...]) -> dict[str, int]:
        caps = self._model_caps()
        totals = {cls: 0 for cls in classes}
        if "kv_block" in totals:
            totals["kv_block"] = int(caps.num_blocks if caps is not None else 0)
        if "scratch" in totals:
            totals["scratch"] = int(caps.scratch_capacity_tokens if caps is not None else 0)
        if "image_latent" in totals:
            totals["image_latent"] = int(caps.max_latent_size if caps is not None else 0)
        if "encoder_output" in totals:
            totals["encoder_output"] = int(
                caps.encoder_cache_budget if caps is not None and caps.encoder_cache_budget is not None else 0
            )
        if "adapter" in totals:
            totals["adapter"] = 0
        return totals

    def _model_caps(self) -> Caps | None:
        return self.model.caps() if isinstance(self.model, ModelHooks) else None

    def _advance_state(self, fb: UniForwardBatch, results: list[Any]) -> None:
        if fb.mode == ForwardMode.MIXED:
            # Advance each op by its own mode; results align with ``fb.op_modes``.
            for mode, result in zip(fb.op_modes, results):
                self._advance_op_state(mode, result)
            return
        for result in results:
            self._advance_op_state(fb.mode, result)

    def _advance_op_state(self, mode: ForwardMode, result: Any) -> None:
        if mode == ForwardMode.DENOISE:
            self.request_states.advance_denoise(
                int(result["req_id"]),
                int(result["num_steps_done"]) if result.get("num_steps_done") is not None else None,
            )
        elif mode == ForwardMode.COMMIT:
            req_id = int(result["req_id"])
            self._accountant.release_generation(req_id, committed=True)

    def _record_group_shape(self, stats: ForwardStats, fb: UniForwardBatch) -> None:
        if fb.mode is ForwardMode.MIXED:
            for mode, op in zip(fb.op_modes, fb.ops):
                stats.record_mode_shape(
                    mode.value,
                    ops=1,
                    tokens=self._op_token_count(op),
                )
            return
        stats.record_mode_shape(
            fb.mode.value,
            ops=len(fb.ops),
            tokens=sum(self._op_token_count(op) for op in fb.ops),
        )

    def _op_token_count(self, op: Mapping[str, Any]) -> int:
        mode = mode_for_op(str(op.get("kind")))
        if mode in _TEXT_DRIVER_MODES:
            if mode == ForwardMode.DECODE:
                try:
                    return max(1, int(op.get("decode_token_count") or 1))
                except (TypeError, ValueError):
                    raise invalid_descriptor("decode_token_count must be a positive integer") from None
            tokens = op.get("token_ids") or []
            return len(tokens) if isinstance(tokens, (list, tuple)) else 0
        if mode == ForwardMode.DENOISE:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            cfg = op.get("cfg")
            # Token throughput count matches denoise_driver branch iterations.
            branch_count = int(cfg.get("branch_count") or 1) if isinstance(cfg, Mapping) else 1
            step_count = max(1, int(op.get("denoise_step_count") or 1))
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return (
                self._accountant.latent_units(op, state.image, latent_rule)
                * max(1, branch_count)
                * step_count
            )
        if mode == ForwardMode.COMMIT:
            req_id = int(op["req_id"])
            state = self.request_states.get(req_id)
            latent_rule = self.resource_plan.image_latent or LatentTokens(downsample=16)
            return self._accountant.latent_units(op, state.image, latent_rule)
        return 0
