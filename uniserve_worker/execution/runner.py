"""Shared model runner.

Owns request-state creation, wire-op parsing into ``UniForwardBatch``, model
``forward`` dispatch, and per-op result reassembly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

from ..backends.attention import get_attention_backend, normalize_attention_backend_name
from ..contracts.batches import ExecuteBatch, UniForwardBatch
from ..contracts.caps import Caps
from ..contracts.forward_context import ForwardContext, use_forward_context
from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..contracts.forward_stats import ForwardStats
from ..contracts.model_protocols import ModelHooks, UniModel
from ..contracts.outputs import ForwardOutput, ForwardOutputBase

if TYPE_CHECKING:
    from ..contracts.model_protocols import DenoiseCapable
from ..contracts.batch_policy import BatchPolicy
from ..contracts.resource_plan import LatentTokens, ResourcePlan
from ..foundation.env import env_flag
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.profiling import profile_range
from ..foundation.runtime_config import get_worker_config
from ..runtime.forward_batch_builder import ForwardBatchBuilder
from ..runtime.request_state import RequestStateTable
from ..runtime.resources import ResourceRuntime
from .denoise_driver import DenoiseDriver
from .encode_driver import EncodeDriver
from .forward_admission import ForwardAdmissionRouter
from .forward_driver import ForwardDriver
from .image_decode_driver import ImageDecodeDriver
from .resource_accountant import ResourceAccountant
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


def _overrides_model_hook(model: Any, name: str) -> bool:
    hook = getattr(type(model), name, None)
    default = getattr(ModelHooks, name, None)
    return hook is not None and hook is not default


@dataclass
class RunnerDrivers:
    """Optional per-modality driver overrides for ``ModelRunner``.

    Each is constructed with a default when left ``None``; the text driver is
    additionally wired to the system text-execution stack built in ``__init__``.
    """

    denoise_driver: DenoiseDriver | None = None
    forward_driver: ForwardDriver | None = None
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


@dataclass
class _ResolvedRunnerDeps:
    batch_policy: BatchPolicy | None
    attention_backend: Any | None
    denoise_driver: DenoiseDriver | None
    forward_driver: ForwardDriver | None
    encode_driver: EncodeDriver | None
    image_decode_driver: ImageDecodeDriver | None
    text_driver: TextDriver | None
    multimodal_processor: Any | None
    defer_sampling: bool
    tensor_store: Any | None


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
        request_states: RequestStateTable | None = None,
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
        self.request_states = request_states or RequestStateTable()
        # Deferred sampling: text decode/extend ops publish logits to
        # ``tensor_store`` and return handles — a separate Sampler worker samples.
        # Off = sample inline (default).
        self.defer_sampling = bool(deps.defer_sampling) and deps.tensor_store is not None
        self.tensor_store = deps.tensor_store
        self.batch_policy = deps.batch_policy or self._model_batch_policy()
        self.attention_backend, self.attention_backend_name = self._resolve_attention_backend(
            deps.attention_backend
        )
        self.denoise_driver = deps.denoise_driver or DenoiseDriver()
        self.forward_driver = deps.forward_driver or ForwardDriver()
        self.encode_driver = deps.encode_driver or EncodeDriver()
        self.image_decode_driver = deps.image_decode_driver or ImageDecodeDriver()
        self._init_text_execution(model, residency, deps.text_driver)
        self.multimodal_processor = deps.multimodal_processor
        self._init_capability_flags(model)
        self._mode_strategies = self._build_mode_strategies()
        self._init_resource_accounting(resource_runtime, residency)
        # CUDA Green Context SM partitioning. ``None`` unless runtime config
        # enables it and the model runs on a CUDA device.
        self.stream_manager = self._maybe_build_stream_manager()

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
            graph_runner=self.text_graph_runner,
        )

    def _init_capability_flags(self, model: UniModel) -> None:
        # Resolve model capability flags once for dispatch.
        self._has_text_forward = hasattr(model, "forward")
        self._has_text_logits_batch = _overrides_model_hook(model, "run_text_logits_batch")
        self._is_text_capable = self._has_text_forward or self._has_text_logits_batch
        self._has_predict_velocity = _overrides_model_hook(model, "predict_velocity")
        self._has_decode_image = _overrides_model_hook(model, "decode_image")
        self._has_encode = _overrides_model_hook(model, "encode_image") or _overrides_model_hook(
            model, "encode_latents"
        )
        self._whole_batch_forward = bool(getattr(model, "whole_batch_forward", False))

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
        self._accountant = ResourceAccountant(
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
            forward_driver=drivers.forward_driver,
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
        gate = TextBackendGate(
            head_dim=int(getattr(model, "head_dim", residency.kv.head_dim)),
            block_size=int(residency.kv.block_size),
            device_type=device.type,
            paged_storage_ok=bool(getattr(residency.kv, "supports_paged_attention_storage", True)),
        )
        graph_runner = None
        if device.type == "cuda":
            from .text_graph_runner import TextGraphRunner

            graph_runner = TextGraphRunner(
                kv_pool=residency.kv,
                num_blocks=int(residency.kv.num_blocks),
                block_size=int(residency.kv.block_size),
                device=device,
            )
            try:
                graph_runner.warmup(model)
            except Exception:  # noqa: BLE001 - a warmup failure must never block serving.
                logger.warning("text CUDA-graph warmup failed; falling back to eager", exc_info=True)
        return _TextExecutionStack(ForwardBatchBuilder(), gate, graph_runner)

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
        self.resource_runtime.release_request(int(req_id))
        self.request_states.drop(req_id)

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        parsed = ExecuteBatch.from_wire(batch)
        forward_stats = ForwardStats() if _forward_metrics_enabled() else None
        self._register_new_reqs(parsed.new_reqs)

        ops = parsed.ops
        results: list[dict[str, Any] | None] = [None] * len(ops)
        with profile_range("uniserve.runner.group_ops"):
            groups = self._groups(ops)
        for group in groups:
            self._run_group(
                group,
                results,
                forward_stats=forward_stats,
                defer_text_cpu_results=defer_text_cpu_results,
            )

        if any(result is None for result in results):
            raise invalid_descriptor("runner missed at least one op result")
        out = {"step_id": parsed.step_id, "per_seq": results}
        if forward_stats is not None:
            out["forward_stats"] = forward_stats.to_wire()
        return out

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
                    self.resource_runtime.release_request(int(req_id))
                    self.request_states.drop(req_id)
                raise
            self.model.on_new_request(req_id, state)

    def _run_group(
        self,
        group: list[tuple[int, Mapping[str, Any]]],
        results: list[dict[str, Any] | None],
        *,
        forward_stats: ForwardStats | None,
        defer_text_cpu_results: bool,
    ) -> None:
        """Account, forward, and post-advance one single-step op group in place.

        Writes each op's per-seq result into ``results`` at its original index.
        """
        indices = [idx for idx, _ in group]
        fb = UniForwardBatch.from_ops([op for _, op in group])
        with profile_range(f"uniserve.runner.group.{fb.mode.value}"):
            self._accountant.account_group(group)
            if forward_stats is not None:
                self._record_group_shape(forward_stats, fb)
            ctx = ForwardContext(
                attention_backend=self.attention_backend,
                attention_backend_name=self.attention_backend_name,
                stats=forward_stats,
            )
            group_start = time.perf_counter_ns() if forward_stats is not None else 0
            stream_ctx = self._forward_stream_context(fb)
            with use_forward_context(ctx), stream_ctx:
                outputs = self._dispatch_by_mode(
                    fb, group, defer_text_cpu_results=defer_text_cpu_results
                )
            if forward_stats is not None:
                forward_stats.record_mode_wall_time(
                    fb.mode.value, time.perf_counter_ns() - group_start
                )
            if len(outputs) != len(group):
                raise invalid_descriptor(
                    f"model returned {len(outputs)} outputs for {len(group)} ops"
                )
            for idx, output in zip(indices, outputs):
                results[idx] = _to_seq_result(output)
            # Every index in this group was just populated above, so the
            # per-op results line up with `fb.op_modes` one-for-one.
            self._advance_state(fb, [results[idx] for idx in indices])
            self._stamp_conditioning_locators(fb, group, results)

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

    def _dispatch_by_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        *,
        defer_text_cpu_results: bool,
    ) -> list[Any]:
        """Route one single-mode group to the forward path implementing its mode.

        ``fb.mode`` comes from the shared ``ForwardMode``/``mode_for_op`` mapping;
        the model-capability flags (``self._has_*``) are resolved once in
        ``__init__``. A group whose mode has no matching implemented path is a
        capability mismatch.
        """
        strategy = self._mode_strategies.get(fb.mode)
        if strategy is not None:
            result = strategy(fb, group, defer_text_cpu_results)
            if result is not None:
                return result
        result = self._run_whole_batch_forward(fb, group, defer_text_cpu_results)
        if result is not None:
            return result
        raise capability_mismatch(
            f"model advertises ops for mode {fb.mode.value!r} but implements no "
            f"matching forward path"
        )

    def _build_mode_strategies(
        self,
    ) -> dict[ForwardMode, Callable[[UniForwardBatch, list[tuple[int, Mapping[str, Any]]], bool], list[Any] | None]]:
        strategies: dict[
            ForwardMode,
            Callable[[UniForwardBatch, list[tuple[int, Mapping[str, Any]]], bool], list[Any] | None],
        ] = {ForwardMode.MIXED: self._run_mixed_mode}
        strategies.update({mode: self._run_text_mode for mode in _TEXT_DRIVER_MODES})
        strategies[ForwardMode.DENOISE] = self._run_denoise_mode
        strategies[ForwardMode.COMMIT] = self._run_commit_mode
        strategies[ForwardMode.ENCODE] = self._run_encode_mode
        return strategies

    def _run_mixed_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        if self._is_decode_denoise_forward_batch(fb):
            if _MIXED_PROOF_ENABLED:
                self._log_mixed_proof(fb, group)
            with profile_range("uniserve.runner.forward"):
                return self.forward_driver.step(fb, group, self.request_states, self.model)
        if not all(m in _TEXT_DRIVER_MODES for m in fb.op_modes):
            return None
        if _MIXED_PROOF_ENABLED:
            self._log_mixed_proof(fb, group)
        return self._run_text_driver(fb, defer_text_cpu_results)

    def _run_text_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del group
        if self._whole_batch_forward:
            return self._run_model_forward(fb)
        if self._is_text_capable:
            return self._run_text_driver(fb, defer_text_cpu_results)
        return None

    def _run_text_driver(
        self,
        fb: UniForwardBatch,
        defer_text_cpu_results: bool,
    ) -> list[Any]:
        with profile_range("uniserve.runner.text_driver"):
            return self.text_driver.step(
                fb,
                self.request_states,
                self.model,
                defer_cpu_results=defer_text_cpu_results,
                defer_sampling=self.defer_sampling,
                tensor_store=self.tensor_store,
            )

    def _run_denoise_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_predict_velocity:
            return None
        with profile_range("uniserve.runner.denoise_driver"):
            return self.denoise_driver.step_many(
                [
                    (int(op["req_id"]), self.request_states.get(int(op["req_id"])), op)
                    for _, op in group
                ],
                cast("DenoiseCapable", self.model),
            )

    def _run_commit_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del fb, defer_text_cpu_results
        if not self._has_decode_image:
            return None
        with profile_range("uniserve.runner.image_decode"):
            return [
                self.image_decode_driver.step(
                    int(op["req_id"]),
                    self.request_states.get(int(op["req_id"])),
                    self.model,
                    op,
                )
                for _, op in group
            ]

    def _run_encode_mode(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del group, defer_text_cpu_results
        if not self._has_encode:
            return None
        with profile_range("uniserve.runner.encode_driver"):
            return self.encode_driver.step(fb, self.model)

    def _run_whole_batch_forward(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
        defer_text_cpu_results: bool,
    ) -> list[Any] | None:
        del group, defer_text_cpu_results
        if not self._whole_batch_forward:
            return None
        return self._run_model_forward(fb)

    def _run_model_forward(self, fb: UniForwardBatch) -> list[Any]:
        with profile_range("uniserve.runner.model_forward"):
            return self.model.forward(fb)

    def _groups(self, ops: list[Mapping[str, Any]]) -> list[list[tuple[int, Mapping[str, Any]]]]:
        """Group ops for one forward step.

        ``supports_mixed_modes`` selects per-mode (not per-position) grouping via
        ``_mode_ordered_groups``. Decode+denoise mixed windows admitted by
        ``ForwardAdmissionRouter`` run as one forward. All-text extend+decode
        windows still require the system ``TextBackendGate``; otherwise the batch
        splits per-mode.
        """
        if self.batch_policy.supports_mixed_modes:
            decision = ForwardAdmissionRouter.from_runtime_config().decide(ops)
            if decision.use_forward and self._accepts_forward_batch(ops, decision):
                return [list(enumerate(ops))]
            self._log_text_mixed_split(ops, decision)
            return self._mode_ordered_groups(ops)
        return self._contiguous_groups(ops)

    def _accepts_forward_batch(self, ops: list[Mapping[str, Any]], decision: Any) -> bool:
        """Whether the admitted mixed window can actually run as one forward.

        Text-only extend+decode windows run through the system ``TextBackendGate``.
        Decode+denoise generation windows are accepted unconditionally once the
        admission router selects them; if the model hook is missing, dispatch
        fails loudly instead of splitting per-mode.
        """

        if not decision.requires_model_acceptance:
            return bool(decision.use_forward)
        if self.text_gate is not None:
            return self.text_gate.mixed_capable(
                ops, attention_backend_name=self.attention_backend_name
            )
        return False

    @staticmethod
    def _is_decode_denoise_forward_batch(fb: UniForwardBatch) -> bool:
        modes = set(fb.op_modes)
        return (
            fb.mode is ForwardMode.MIXED
            and ForwardMode.DECODE in modes
            and ForwardMode.DENOISE in modes
            and modes <= {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.DENOISE}
        )

    def _contiguous_groups(
        self, ops: list[Mapping[str, Any]]
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        for idx, op in enumerate(ops):
            mode = self._validated_mode(idx, op)
            if groups:
                modes = [mode_for_op(item[1]["kind"]) for item in groups[-1]]
                if self.batch_policy.allows_group([*modes, mode]):
                    groups[-1].append((idx, op))
                    continue
            groups.append([(idx, op)])
        return groups

    def _mode_ordered_groups(
        self, ops: list[Mapping[str, Any]]
    ) -> list[list[tuple[int, Mapping[str, Any]]]]:
        buckets: dict[ForwardMode, list[tuple[int, Mapping[str, Any]]]] = {}
        first_seen: list[ForwardMode] = []
        for idx, op in enumerate(ops):
            mode = self._validated_mode(idx, op)
            if mode not in buckets:
                buckets[mode] = []
                first_seen.append(mode)
            buckets[mode].append((idx, op))

        ordered_modes: list[ForwardMode] = []
        for mode in self.batch_policy.mode_order:
            if mode in buckets:
                ordered_modes.append(mode)
        for mode in first_seen:
            if mode not in ordered_modes:
                ordered_modes.append(mode)

        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        max_batch_ops = self.batch_policy.max_batch_ops
        for mode in ordered_modes:
            items = buckets[mode]
            for start in range(0, len(items), max_batch_ops):
                groups.append(items[start:start + max_batch_ops])
        return groups

    def _validated_mode(self, idx: int, op: Mapping[str, Any]) -> ForwardMode:
        if not isinstance(op, Mapping):
            raise invalid_descriptor(f"execute batch.ops[{idx}] must be a map")
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor(f"execute batch.ops[{idx}].kind must be a string")
        return mode_for_op(kind)

    def _log_mixed_proof(
        self,
        fb: UniForwardBatch,
        group: list[tuple[int, Mapping[str, Any]]],
    ) -> None:
        # Per-step "mixed forward executed" trace (opt-in via the proof flag).
        n_ext = sum(1 for m in fb.op_modes if m == ForwardMode.EXTEND)
        n_dec = sum(1 for m in fb.op_modes if m == ForwardMode.DECODE)
        n_tok = sum(len(op.get("token_ids") or []) for _, op in group)
        _MIXED_PROOF_LOG.info(
            "MIXED FORWARD executed: %d ops (%d extend + %d decode), %d tokens",
            len(fb.ops), n_ext, n_dec, n_tok,
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
            totals["encoder_output"] = int(caps.encoder_cache_budget if caps is not None else 0)
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
            self._accountant.release_class_if_managed("image_latent", req_id)
            self._accountant.release_class_if_managed("scratch", req_id)
            # commit() clears the residency flags (image_latent_active/scratch_active).
            self.request_states.commit(req_id)

    def _record_group_shape(self, stats: ForwardStats, fb: UniForwardBatch) -> None:
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


def _to_seq_result(output: ForwardOutput | Mapping[str, Any]) -> Any:
    # Typed drivers return ForwardOutput; whole-batch forward may return raw maps.
    if isinstance(output, ForwardOutputBase):
        return output.to_seq_result()
    if isinstance(output, Mapping):
        return dict(output)
    raise invalid_descriptor(f"unsupported forward output type {type(output).__name__}")


def _forward_metrics_enabled() -> bool:
    return env_flag("UNISERVE_FORWARD_METRICS")
