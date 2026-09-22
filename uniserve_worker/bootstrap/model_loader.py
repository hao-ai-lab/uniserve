"""Load public numerical modules and bind worker-owned execution choices."""

from __future__ import annotations

import logging
import socket
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import torch
from torch import nn

from uniserve.loading import weights
from uniserve.model import (
    CausalLM,
    ComponentEntry,
    Denoiser,
    ImageDecoder,
    VideoDecoder,
)
from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    Ulysses,
)
from uniserve.nn.vae.patch import PatchAutoencoder
from uniserve.processing import (
    FlowPrompt,
    ImageProcessor,
    load_tokenizer,
)
from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve_models import loading as models

from ..config import WorkerConfig
from ..execution.component_binding import Call, ComponentBinding
from ..foundation.errors import unsupported_setup
from .components import (
    is_host_component,
    validate_components,
)
from .config import ComponentConfig, WorkerProcessArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerModel:
    """Loaded composition and caller-owned input processing."""

    model: nn.Module
    config: WorkerConfig
    tokenizer: Any | None = None
    image_processor: ImageProcessor | None = None
    flow_prompt: FlowPrompt | None = None
    # Identity of the loaded checkpoint; empty for a model without one.
    checkpoint_identity: str = ""
    entry_points: Mapping[str, ComponentEntry] | None = None


def verify_checkpoint_identity(
    expected: str | None, actual: str, *, rank: int, host: str
) -> None:
    """Refuse a rank whose loaded checkpoint is not the one the head derived.

    ``expected`` is absent when the launching side could not read the
    checkpoint locally; the engine then still requires every rank to report
    the same identity. The refusal names the rank and host so an operator can
    find the divergent copy.
    """
    if expected is None or expected == actual:
        return

    raise unsupported_setup(
        f"rank {rank} on host {host} loaded checkpoint {actual}, but the "
        f"head derived checkpoint {expected}"
    )


def prepare_worker_model(
    config: WorkerProcessArgs,
) -> tuple[models.Config | None, nn.Module, dict[str, tuple[Call, ...]]]:
    """Resolve the resident checkpoint closure.

    The closure is resolved before creating process groups, and its checkpoint
    identity is verified against the launch expectation before any weight
    is read.
    """
    if config.use_stub_model:
        from uniserve_models.stub import Model

        with torch.device("meta"):
            description = Model()
        declarations = validate_components(description, dict(config.components))
        return None, description, declarations

    launch = config.model
    if launch is None:
        raise RuntimeError(
            "validated model worker is missing model configuration"
        )

    # A meta-device skeleton is enough to validate placement and select the
    # modules this rank must read from the checkpoint.
    metadata = models.read_config(
        launch.path, io=config.load, modules=frozenset()
    )
    with torch.device("meta"):
        model = metadata.model_class(metadata.model)
    declared = validate_components(
        model, dict(config.components), entries=metadata.entry_points
    )

    resident = frozenset(
        call.path
        for name, component in config.components
        if config.execution.rank in component.ranks
        for call in declared[name]
    )
    source = models.read_config(launch.path, io=config.load, modules=resident)

    # The host name is the same identity the rank's endpoint reports, so the
    # refusal and the engine's own report name one host.
    verify_checkpoint_identity(
        launch.checkpoint_identity,
        source.checkpoint_identity,
        rank=config.execution.rank,
        host=socket.gethostname(),
    )

    return (
        replace(
            source,
            weights=_weight_config(
                source, launch.quantization_config, config.execution
            ),
        ),
        model,
        declared,
    )


def _weight_config(source, options, execution) -> weights.Config:
    """Translate launch precision selectors into the public loading value."""
    unknown = options.keys() - {
        "mode",
        "quant_method",
        "components",
        "ignored_layers",
        "kv_cache_dtype",
    }
    if unknown:
        raise ValueError(
            f"quantization_config has unknown fields {sorted(unknown)}"
        )
    if "mode" in options and "quant_method" in options:
        raise ValueError(
            "precision mode and quant_method are mutually exclusive"
        )

    checkpoint_format = source.checkpoint_format
    numerical_overrides = options.keys() & {
        "mode",
        "quant_method",
        "components",
        "ignored_layers",
    }
    if checkpoint_format is not None and numerical_overrides:
        raise ValueError(
            f"checkpoint format {checkpoint_format!r} owns its numerical "
            "configuration; remove --quantization-config"
        )

    selected = options.get("mode", options.get("quant_method"))
    components = options.get("components", {})
    if not isinstance(components, Mapping):
        raise TypeError("precision components must be an object")

    package = import_module(source.model_class.__module__.rsplit(".", 1)[0])
    if components:
        factory = getattr(package, "weight_config", None)
        if factory is None:
            raise ValueError(
                "this model exposes complete precision presets "
                "without component selectors"
            )
        result = factory(
            preset="default" if selected is None else selected, **components
        )
    elif selected is None:
        result = source.weights
    elif selected in source.precisions:
        result = source.precisions[selected]
    elif "quant_method" in options and selected in {
        "unquantized",
        "fp8",
        "mxfp8",
        "nvfp4",
    }:
        quantizer = (
            None
            if selected == "unquantized"
            else Quantizer(selected, axis=0 if selected == "fp8" else None)
        )
        result = replace(
            source.weights,
            quantization={
                "": None
                if quantizer is None
                else QuantizationConfig(quantizer, quantizer)
            },
        )
    else:
        raise ValueError(
            f"unknown precision {selected!r}; "
            f"choose from {tuple(source.precisions)}"
        )

    ignored = options.get("ignored_layers", ())
    if not isinstance(ignored, (tuple, list)) or any(
        not isinstance(path, str) for path in ignored
    ):
        raise TypeError("ignored_layers must contain numerical module paths")

    return replace(
        result,
        dtype=getattr(torch, execution.model_dtype),
        quantization={**result.quantization, **dict.fromkeys(ignored)},
    )


def attention_parallel(component: ComponentConfig) -> AttentionParallelConfig:
    """Translate degree declarations into mathematical attention axes."""
    sequence = component.parallel_config.sequence_parallel
    heads = Ulysses() if sequence.kind in {"ulysses", "hybrid"} else None
    context = (
        ContextParallelConfig(gather_axis="cp")
        if sequence.kind in {"allgather", "hybrid"}
        else None
    )
    return AttentionParallelConfig(heads=heads, context=context)


def load_worker_model(
    config: WorkerProcessArgs,
    bindings: Mapping[str, ComponentBinding],
    *,
    source: models.Config | None,
    description: nn.Module,
    declarations: Mapping[str, tuple[Call, ...]],
) -> WorkerModel:
    """Materialize selected modules and attach borrowed capability methods."""
    if config.use_stub_model:
        from uniserve_models.stub import Model, image_processor

        model = Model().to(config.execution.device)
        for path, device in (
            _devices(model, config.execution.generation_device) or {}
        ).items():
            model.get_submodule(path).to(device)
        return WorkerModel(
            model,
            replace(
                config.execution,
                attention_backend="torch",
                encoder_cache_entries=1024,
            ),
            image_processor=image_processor(),
        )
    if config.model is None or source is None:
        raise RuntimeError(
            "validated model worker is missing model configuration"
        )

    if all(
        is_host_component(name) or not declarations.get(name)
        for name in bindings
    ):
        # A host rank holds only host components: it needs the model's
        # declared media geometry, rates and unit division, which the
        # weight-less description carries, and no numerical state.
        worker_config = loaded_worker_config(
            description, config.execution, config.ipc.queue_depth
        )
        logger.info(
            "described numerical model %s for host work",
            type(description).__qualname__,
        )
        return WorkerModel(
            description,
            worker_config,
            None,
            source.image_processor,
            source.flow_prompt,
            source.checkpoint_identity,
            source.entry_points,
        )

    meshes, attention = {}, {}
    for name, binding in bindings.items():
        if binding.mesh is None:
            continue
        paths = {call.path for call in declarations[name]}
        # Contained encoders are bound by their numerical parent's traversal;
        # siblings sharing a backbone retain their independent capability roots.
        roots = {
            path
            for path in paths
            if not any(
                parent != path and (not parent or path.startswith(parent + "."))
                for parent in paths
            )
        }
        for path in sorted(roots):
            meshes[path] = binding.mesh
            attention[path] = attention_parallel(binding.config)

    loaded = models.load_model(
        source,
        device=config.execution.device,
        meshes=meshes,
        attention=attention,
        devices=_devices(description, config.execution.generation_device),
    )
    model = loaded.model

    worker_config = loaded_worker_config(
        model, config.execution, config.ipc.queue_depth
    )
    override = config.model.quantization_config.get("kv_cache_dtype")
    if override is not None:
        worker_config = replace(worker_config, kv_cache_dtype=override)

    logger.info("loaded numerical model %s", type(model).__qualname__)
    return WorkerModel(
        model,
        worker_config,
        None if source.tokenizer is None else load_tokenizer(source.tokenizer),
        source.image_processor,
        source.flow_prompt,
        source.checkpoint_identity,
        source.entry_points,
    )


def _devices(
    model: nn.Module, generation_device: str | None
) -> Mapping[str, str] | None:
    """Place the flow route and denoiser modules on the selected device."""
    if generation_device is None:
        return None
    text_modules = {
        id(child)
        for module in model.modules()
        if isinstance(module, CausalLM)
        for child in module.modules()
    }

    placed = set()
    for module in model.modules():
        if isinstance(module, Denoiser):
            for child in module.children():
                if id(child) not in text_modules:
                    placed.add(id(child))
        if isinstance(module, nn.ModuleDict) and "flow" in module:
            placed.add(id(module["flow"]))
        if isinstance(module, (ImageDecoder, PatchAutoencoder)):
            placed.add(id(module))

    # Shared backbones and codecs have multiple ordinary module paths. Every
    # alias must express the same placement before the loader materializes it.
    paths = {
        path: generation_device
        for path, module in model.named_modules(remove_duplicate=False)
        if id(module) in placed
    }
    if not paths:
        raise unsupported_setup(
            "generation device requires a model with a distinct flow route"
        )
    return paths


def loaded_worker_config(
    model: nn.Module, config: WorkerConfig, queue_depth: int
) -> WorkerConfig:
    """Resolve media request slots from the worker's publication lifetime."""
    if any(isinstance(module, VideoDecoder) for module in model.modules()):
        state_slots = min(config.max_batch_calls, queue_depth // 3)
        if state_slots < 2:
            raise unsupported_setup(
                "resident media execution requires two slots with "
                "two unresolved outputs each"
            )
        config = replace(
            config,
            kv_token_capacity=None,
            attention_backend=None,
            max_batch_calls=state_slots,
            max_batch_tokens=state_slots,
            max_request_pool_size=state_slots,
            min_request_pool_size=2,
            generation_device=None,
        )
    return config
