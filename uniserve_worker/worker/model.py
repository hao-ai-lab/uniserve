"""Model-backed worker built around the shared :class:`ModelExecutor`."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from uniserve_worker.execution.runner import ExecutorConfig, ModelExecutor
from uniserve_worker.execution.segment import SegmentExecutor

from ..backends.attention import (
    get_attention_backend,
    has_attention_backend,
    normalize_attention_backend_name,
)
from ..contracts.caps import Caps, validate_caps
from ..contracts.model_family import ModelFamilyDescriptor, ModelLoadScope
from ..contracts.model_protocols import UniModel
from ..foundation.env import DEFAULT_ATTENTION_BACKEND
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.mesh import get_current_mesh
from ..processors import get_processor_for_descriptor
from ..runtime.adapter_store import AdapterStore
from ..runtime.resources import ResourceRuntime
from .protocol import BaseWorker, ResultPolicy

if TYPE_CHECKING:
    from ..runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)

# Controls the worker serves against system-owned state; a model declares them
# as capability only and implements no method.
ADAPTER_CONTROLS = frozenset({"load_lora", "unload_lora"})
WORKER_SERVED_CONTROLS = ADAPTER_CONTROLS | {
    "copy_blocks",
    "free_encoder",
    "reset_prefix_cache",
}


def free_encoder_handles(model: UniModel, handles: Any) -> None:
    """Release system-owned encoder-output residency for the given handles."""
    residency = model.residency
    if residency is None:
        return
    for handle in handles or []:
        residency.encoder.pop(int(handle))


def bind_model_residency(model: UniModel, residency: "ResidencyManager | None") -> None:
    """Hand the system-built residency to the model surfaces that read it.

    The manager itself lands on the contract-declared ``model.residency``; the
    pool aliases land only on the slots the model declares.
    """
    if residency is None:
        return
    model.residency = residency
    for name, pool in (
        ("kv_pool", residency.kv),
        ("scratch_pool", residency.scratch),
        ("gen_scratch_pool", residency.gen_scratch),
    ):
        if hasattr(model, name):
            setattr(model, name, pool)


def bind_model_segment_execution(model: UniModel) -> None:
    """Build the system-owned segment executor over the model's family adapter.

    A model that lowers heterogeneous operations through segment execution
    declares its adapter surface via ``segment_adapter()``; the executor lands
    on the contract-declared ``model.segment_executor``.
    """
    adapter = model.segment_adapter()
    if adapter is None:
        return
    model.segment_executor = SegmentExecutor(adapter)


def build_model_adapter_store(model: UniModel) -> AdapterStore | None:
    """Build the system-owned adapter store over the model's module graph.

    Adapter state is system-owned: the worker serves the load_lora/unload_lora
    controls against the store, and the store keeps pre-merge copies of only
    the parameters an adapter touches. A model constructed without loaded
    weights has no module graph yet and therefore no store; adapter controls
    against such a worker fail with a capability mismatch.
    """
    if model.adapter_mode == "none":
        return None
    module = getattr(model, "model", None)
    if module is None:
        return None
    return AdapterStore(module)


class ModelWorker(BaseWorker):
    """Owns one loaded model and its request-to-forward execution runner."""

    def __init__(
        self,
        model: UniModel,
        *,
        allowed_ops: frozenset[str] | None = None,
        pipeline_depth: int = 1,
        result_policy: ResultPolicy = ResultPolicy.DEFER_WHEN_AVAILABLE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        defer_sampling: bool = False,
        transfer_backend: str = "local",
        model_scope: ModelLoadScope = ModelLoadScope.WHOLE,
        family_descriptor: ModelFamilyDescriptor | None = None,
        simulation: bool = False,
        spec_digest: str | None = None,
    ) -> None:
        super().__init__(block_size=block_size)
        self.model = model
        # Resolved ModelSpec + DeploymentOverlay identity from bootstrap; the
        # caps wire schema carries no spec field yet, so the digest is logged
        # here and scoped onto the executor's graph store.
        self.spec_digest = spec_digest
        if spec_digest is not None:
            logger.info("model worker serving resolved spec digest=%s", spec_digest)
        self.attention_backend = attention_backend or DEFAULT_ATTENTION_BACKEND
        self.defer_sampling = bool(defer_sampling)
        self.transfer_backend = str(transfer_backend)
        self.model_scope = model_scope
        self._validate_scope_transport()
        self.family_descriptor = family_descriptor or ModelFamilyDescriptor.from_model_class(
            type(model)
        )
        self.block_size = self._adjust_block_size(self.block_size)
        self.kv_token_capacity = kv_token_capacity

        declared_capabilities = self._caps_with_current_rank(
            validate_caps(
                self._build_model_capabilities(),
                owner=f"{type(model).__name__}.ModelWorker",
            )
        )
        self._validate_declared_controls(declared_capabilities)
        self._compile_contract(
            declared_capabilities,
            allowed_ops=(
                allowed_ops
                if allowed_ops is not None
                else frozenset(declared_capabilities.supported_ops)
            ),
            pipeline_depth=pipeline_depth,
            result_policy=result_policy,
        )
        self.model.configure_runtime(
            block_size=self.block_size,
            kv_token_capacity=self.kv_token_capacity,
            caps=self.contract.capabilities.to_wire(),
        )

        tensor_store = None
        if self.defer_sampling:
            from ..runtime.tensor_store import TensorStore
            from ..runtime.transfer import make_transport

            tensor_store = TensorStore(transport=make_transport(self.transfer_backend))
        resource_runtime = self._create_resource_runtime()
        residency = self._create_residency_manager(resource_runtime)
        bind_model_residency(self.model, residency)
        bind_model_segment_execution(self.model)
        self.adapter_store = build_model_adapter_store(self.model)
        self.model_executor = ModelExecutor(
            model,
            config=ExecutorConfig(simulation=bool(simulation), spec_digest=spec_digest),
            attention_backend=self.attention_backend,
            resource_runtime=resource_runtime,
            multimodal_processor=get_processor_for_descriptor(self.family_descriptor),
            defer_sampling=self.defer_sampling,
            tensor_store=tensor_store,
            residency=residency,
        )
        self._bind_tower_transport()

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        return self.model_executor.execute(
            dict(batch),
            defer_text_cpu_results=defer_text_cpu_results,
        )

    def drop_request(self, request_id: int) -> None:
        self.model_executor.drop_request(int(request_id))

    def copy_blocks(self, copies: Any) -> None:
        del copies

    def load_lora(self, lora_id: int, lora_path: str) -> None:
        count = self._require_adapter_store().load(lora_id, lora_path)
        logger.info("merged LoRA adapter %s into %d parameters", lora_id, count)

    def unload_lora(self, lora_id: int) -> None:
        count = self._require_adapter_store().unload(lora_id)
        if count:
            logger.info("unmerged LoRA adapter %s", lora_id)

    def free_encoder(self, handles: Any) -> None:
        free_encoder_handles(self.model, handles)

    def reset_prefix_cache(self) -> None:
        pass

    def sleep(self) -> None:
        sleep = getattr(self.model, "sleep")
        sleep()

    def wake_up(self) -> None:
        wake_up = getattr(self.model, "wake_up")
        wake_up()

    def resource_pressure(self) -> list[dict[str, Any]]:
        return self.model_executor.resource_runtime.pressure()

    def _build_model_capabilities(self) -> Caps:
        arguments = {
            "block_size": self.block_size,
            "kv_token_capacity": self.kv_token_capacity,
        }
        signature = inspect.signature(self.model.caps)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        supported_arguments = (
            arguments
            if accepts_kwargs
            else {key: value for key, value in arguments.items() if key in signature.parameters}
        )
        capabilities = self.model.caps(**supported_arguments)
        if isinstance(capabilities, Caps):
            return capabilities
        raise capability_mismatch(f"{type(self.model).__name__}.caps() must return Caps")

    @staticmethod
    def _caps_with_current_rank(capabilities: Caps) -> Caps:
        mesh = get_current_mesh()
        return replace(
            capabilities,
            tp_rank=int(mesh.tp_rank),
            tp_size=int(mesh.tp_size),
        )

    def _create_residency_manager(
        self, resource_runtime: ResourceRuntime
    ) -> "ResidencyManager | None":
        from ..runtime.residency import ResidencyManager

        generation_spec = self.model.gen_residency_spec()
        if generation_spec is not None:
            return ResidencyManager.build_gen(generation_spec, ledger=resource_runtime)
        specification = self.model.kv_cache_spec()
        if specification is None:
            return None
        capabilities = self.contract.capabilities
        device = str(getattr(self.model, "device", "cuda") or "cuda")
        return ResidencyManager.build(
            specification,
            num_blocks=int(capabilities.num_blocks),
            block_size=int(capabilities.block_size),
            device=device,
            ledger=resource_runtime,
        )

    def _create_resource_runtime(self) -> ResourceRuntime:
        capabilities = self.contract.capabilities
        totals = {
            "kv_block": int(capabilities.num_blocks),
            "scratch": int(capabilities.scratch_capacity_tokens),
            "image_latent": int(capabilities.max_latent_size),
            "encoder_output": int(capabilities.encoder_cache_budget or 0),
            "adapter": 0,
        }
        return ResourceRuntime(
            capabilities.resource_classes or ("kv_block",),
            totals=totals,
        )

    def _require_adapter_store(self) -> AdapterStore:
        if self.adapter_store is None:
            raise capability_mismatch("adapter controls require loaded model weights")
        return self.adapter_store

    def _validate_declared_controls(self, capabilities: Caps) -> None:
        for control in capabilities.supported_controls:
            name = str(control)
            if name in WORKER_SERVED_CONTROLS:
                if name in ADAPTER_CONTROLS and capabilities.adapter_mode == "none":
                    raise capability_mismatch(
                        f"control {name!r} is served by the worker adapter store and "
                        "requires a non-'none' adapter_mode"
                    )
                continue
            if not callable(getattr(self.model, name, None)):
                raise capability_mismatch(
                    f"model declares control {control!r} but does not implement it"
                )

    def _adjust_block_size(self, block_size: int) -> int:
        backend_name = normalize_attention_backend_name(self.attention_backend)
        if backend_name == "auto" or not has_attention_backend(backend_name):
            return block_size
        multiple = get_attention_backend(backend_name).capabilities().paged_block_size_multiple
        if multiple <= 1 or block_size % multiple == 0:
            return block_size
        adjusted = ((block_size + multiple - 1) // multiple) * multiple
        logger.warning(
            "%s requires block_size to be a multiple of %d; adjusting %d to %d",
            backend_name,
            multiple,
            block_size,
            adjusted,
        )
        return adjusted

    def _validate_scope_transport(self) -> None:
        if self.model_scope is ModelLoadScope.WHOLE:
            return
        if self.transfer_backend in {"", "local", "shm"}:
            raise capability_mismatch(
                f"{self.model_scope.value} model scope requires a "
                "cross-process data-plane transport"
            )

    def _bind_tower_transport(self) -> None:
        if self.model_scope is ModelLoadScope.WHOLE:
            return
        from ..runtime.transfer import make_transport

        self.model.bind_data_plane_handoff(make_transport(self.transfer_backend))
