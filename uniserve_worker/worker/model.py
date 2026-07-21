"""Model-backed worker built around the shared :class:`ModelExecutor`."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from uniserve_worker.execution.runner import ExecutorConfig, ModelExecutor

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
from ..runtime.resources import ResourceRuntime
from .protocol import BaseWorker, ResultPolicy

if TYPE_CHECKING:
    from ..runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(block_size=block_size)
        self.model = model
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
        self.model_executor = ModelExecutor(
            model,
            config=ExecutorConfig(simulation=bool(simulation)),
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
        self.model.copy_blocks(copies)

    def load_lora(self, lora_id: int, lora_path: str) -> None:
        self.model.load_lora(lora_id, lora_path)

    def unload_lora(self, lora_id: int) -> None:
        self.model.unload_lora(lora_id)

    def free_encoder(self, handles: Any) -> None:
        self.model.free_encoder(handles)

    def reset_prefix_cache(self) -> None:
        self.model.reset_prefix_cache()

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

    def _validate_declared_controls(self, capabilities: Caps) -> None:
        for control in capabilities.supported_controls:
            if not callable(getattr(self.model, str(control), None)):
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
