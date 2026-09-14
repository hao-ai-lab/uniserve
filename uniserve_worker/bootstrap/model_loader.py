"""Bind numerical loading results to worker placement, inputs and capacity."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from uniserve.loading import load_model
from uniserve.model.limits import ModelLimits
from uniserve.model.model import Model
from uniserve.model.video import VideoMixin
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.layer import LayerConfig
from uniserve_models import ModelSource, resolve_model
from uniserve_models.processing import FlowPrompt, ImageProcessor, stub_processor

from ..config import WorkerConfig
from ..execution.model_entry import ModelEntry
from ..execution.resources import media_state_buffers
from ..foundation.errors import unsupported_setup
from ..runtime.results import resolve_outputs
from .components import bind_components, validate_components
from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerModel:
    """Numerical composition and the worker's resolved execution inputs."""

    model: Model
    config: WorkerConfig
    tokenizer: Any | None = None
    schedule: DiffusionSchedule | None = None
    image_processor: ImageProcessor | None = None
    flow_prompt: FlowPrompt | None = None


def prepare_worker_model(config: WorkerProcessArgs) -> ModelSource | None:
    """Resolve typed architecture and validate placement before process groups."""

    if config.use_stub_model:
        return None
    launch = config.model
    if launch is None:
        raise RuntimeError("validated model worker is missing model configuration")
    components = dict(config.components)
    source = resolve_model(
        launch.path,
        load=config.load,
        components=frozenset(
            name for name, value in components.items() if config.execution.rank in value.ranks
        ),
        quantization=launch.quantization_config,
    )
    paths = dict(source.entry.component_paths)
    validate_components(source.model_class, source.config, components, paths=paths)
    source.model_class.validate_parallel(
        source.config,
        {paths[name]: value.parallel_config for name, value in components.items()},
    )
    capability = source.model_class.minimum_cuda_capability
    device = torch.device(config.execution.device)
    if capability is not None and (
        device.type != "cuda" or torch.cuda.get_device_capability(device) < capability
    ):
        raise unsupported_setup(
            f"{source.entry.architecture} requires CUDA compute capability {capability[0]}.{capability[1]}"
        )
    return source


def load_worker_model(
    config: WorkerProcessArgs,
    bindings: Mapping[str, ModelEntry],
    *,
    source: ModelSource | None,
) -> WorkerModel:
    """Load numerical modules, then bind worker calls and execution capacity."""

    if config.use_stub_model:
        from uniserve_models.stub import StubModel

        model: Model = StubModel()
        bind_components(model, bindings, paths={name: "" for name in bindings})
        return WorkerModel(
            model,
            replace(config.execution, attention_backend="torch_sdpa", encoder_cache_entries=1024),
            image_processor=stub_processor(),
        )
    launch = config.model
    if launch is None or source is None:
        raise RuntimeError("validated model worker is missing model configuration")
    paths = dict(source.entry.component_paths)
    meshes = {
        paths[name]: binding.mesh for name, binding in bindings.items() if binding.mesh is not None
    }
    parallel = {paths[name]: binding.config.parallel_config for name, binding in bindings.items()}
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        config.execution.model_dtype
    ]
    layers = source.configure_layers(
        {
            name: LayerConfig(
                mesh.get_group("tp"),
                None,
                pipeline=mesh.get_group("pp"),
                sequence=mesh.get_group("ulysses"),
                dense_dtype=dtype,
            )
            for name, mesh in meshes.items()
        }
    )
    loaded = load_model(
        source.model_class,
        source.config,
        sources=source.weights,
        load=config.load,
        device=config.execution.device,
        dtype=dtype,
        parallel=parallel,
        meshes=meshes,
        layers=layers,
        limits=ModelLimits(launch.max_text_rows, math.floor(launch.max_video_seconds * 24.0 + 0.5)),
        flow_device=config.execution.generation_device,
    )
    model = loaded.model
    bind_components(model, bindings, paths=paths)
    outputs = resolve_outputs(model)
    for name, binding in bindings.items():
        binding.outputs = outputs.get(name, ())
    worker_config = loaded_worker_config(
        model, config.execution, bindings, config.ipc.pipeline_depth
    )
    create_schedule = source.entry.create_schedule
    schedule = (
        create_schedule(source.config, torch.device(config.execution.device))
        if create_schedule is not None
        and any("forward_diffusion" in binding.methods for binding in bindings.values())
        else None
    )
    logger.info(
        "loaded model architecture=%s precisions=%s", model.architecture, dict(source.precisions)
    )
    return WorkerModel(
        model, worker_config, source.tokenizer, schedule, source.image_processor, source.flow_prompt
    )


def loaded_worker_config(
    model: Model, config: WorkerConfig, bindings: Mapping[str, ModelEntry], pipeline_depth: int
) -> WorkerConfig:
    """Resolve worker storage capacity from resident numerical state buffers."""

    if isinstance(model, VideoMixin) or media_state_buffers(model, bindings):
        # Two unresolved outputs and one further physical position permit
        # publication retirement before a resident request slot is reused.
        state_slots = min(config.max_batch_operations, pipeline_depth // 3)
        if state_slots < 2:
            raise unsupported_setup(
                "resident media execution requires two slots with two unresolved outputs each"
            )
        config = replace(
            config,
            kv_token_capacity=None,
            attention_backend=None,
            max_batch_operations=state_slots,
            max_batch_tokens=state_slots,
            max_request_pool_size=state_slots,
            min_request_pool_size=2,
            generation_device=None,
        )
    return config
