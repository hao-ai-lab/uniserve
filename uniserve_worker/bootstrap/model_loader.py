"""Load public numerical modules and bind worker-owned execution choices.

Model loading runs in two phases around process-group creation in
``Worker.from_config``. ``prepare_worker_model`` reads the checkpoint's
configuration, builds a meta-device skeleton, validates the component
placement and resolves which module paths this rank must load.
``load_worker_model`` then materializes those modules through
``uniserve_models.loading.load_model`` with the component meshes and attention
partitioning, and resolves the load-time ``WorkerConfig``. The launch's
``quantization_config`` is translated here into the public
``uniserve.loading.weights.Config``.
"""

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
from uniserve_worker.bootstrap.components import (
    is_host_component,
    validate_components,
)
from uniserve_worker.config.deployment import (
    ComponentConfig,
    WorkerProcessArgs,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.model_executor.component_binding import (
    Call,
    ComponentBinding,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerModel:
    """Loaded composition and caller-owned input processing.

    Attributes:
        model: The loaded numerical model. On a host worker, whose
            components have no numerical calls, it is the weight-less
            meta-device description.
        config: The execution configuration. For a checkpoint model it is
            resolved by ``loaded_worker_config``, and a numerical worker also
            applies the launch's ``kv_cache_dtype`` override.
        tokenizer: The loaded tokenizer; ``None`` for the stub model, on a
            host worker, or for a checkpoint without one.
        image_processor: The model's image preprocessing, if any.
        flow_prompt: The checkpoint's flow prompt template, if any.
        checkpoint_identity: Identity of the loaded checkpoint; empty for a
            model without one.
        entry_points: The checkpoint's IPC entry declarations; ``None`` for
            the stub model, whose declarations ``describe_components`` reads
            from the model's module.
    """

    model: nn.Module
    config: WorkerConfig
    tokenizer: Any | None = None
    image_processor: ImageProcessor | None = None
    flow_prompt: FlowPrompt | None = None
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
    """Resolve the checkpoint configuration and this rank's resident modules.

    This runs before process groups exist. The checkpoint identity is
    verified against the launch expectation before ``load_worker_model``
    materializes any weight. A placement that does not fit the model's
    declarations, or a checkpoint identity that differs from the launch
    expectation, raises ``WorkerError`` with ``UNSUPPORTED_SETUP``. Errors
    from reading the checkpoint configuration and from ``_weight_config``
    propagate.

    Returns:
        A tuple of the checkpoint configuration, narrowed to the module paths
        this rank's components call and carrying the launch's weight choices
        (``None`` for the stub model); the meta-device model skeleton; and
        the validated component declarations.
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
    # modules this rank must read from the checkpoint. This first read selects
    # no modules, so it resolves no weight payload source.
    metadata = models.read_config(
        launch.path, io=config.load, modules=frozenset()
    )
    with torch.device("meta"):
        model = metadata.model_class(metadata.model)
    declared = validate_components(
        model, dict(config.components), entries=metadata.entry_points
    )

    # The second read narrows payload downloads, and later the load, to the
    # module paths this rank's components call.
    resident = frozenset(
        call.path
        for name, component in config.components
        if config.execution.rank in component.ranks
        for call in declared[name]
    )
    source = models.read_config(launch.path, io=config.load, modules=resident)

    # The host name is the ``node`` the rank's ``WorkerEndpoint`` reports, so
    # the refusal and the engine's own report name one host.
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
    """Translate launch precision selectors into the public loading value.

    ``options`` is the launch's ``quantization_config``. Exactly one source
    selects the base weight configuration: ``components`` selectors passed to
    the model package's ``weight_config`` factory (with ``mode`` or
    ``quant_method`` as its preset, the destination's default without one);
    a named precision in ``source.precisions``; a ``quant_method`` among the
    generic formats, applied to every module; or the checkpoint's own weights
    when its package has no device policy. ``ignored_layers`` then maps each
    listed module path to no quantization, and the dtype comes from
    ``execution.model_dtype``.
    ``kv_cache_dtype`` is accepted here but applied by ``load_worker_model``.

    Raises:
        ValueError: An unknown key, both ``mode`` and ``quant_method``,
            numerical selectors for a checkpoint whose format owns its
            numerics, component selectors for a model without a
            ``weight_config`` factory, or an unknown precision.
        TypeError: ``components`` is not a mapping or ``ignored_layers`` is
            not a list of strings.
    """
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
    if selected in (None, "default") and source.default_precision is not None:
        selected = source.default_precision(torch.device(execution.device))
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
    # The quantization mapping resolves by longest module-path prefix, so the
    # empty key applies the generic format to every module.
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
    """Translate degree declarations into mathematical attention axes.

    The ``ulysses`` and ``hybrid`` sequence-parallel kinds partition attention
    heads; ``allgather`` and ``hybrid`` gather the context partition over the
    ``cp`` axis. ``initialize_components`` passes the same value to
    ``communication_axes`` to decide which fibers need backend groups, and
    ``load_worker_model`` passes it to the model loader.
    """
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
    """Materialize this rank's selected modules on their meshes.

    ``source``, ``description`` and ``declarations`` are the results of
    ``prepare_worker_model``; ``bindings`` come from
    ``initialize_components``. The stub model is built directly on the worker
    device. When no configured component has numerical calls (a host
    worker), no weights load and the meta-device description is kept.
    Otherwise each outermost declared module path of a meshed component is
    loaded on that component's mesh with its ``attention_parallel``
    partitioning. Capability methods are attached later, by
    ``bind_components`` in ``ModelExecutor``.
    """
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
        # Only the outermost declared paths receive a mesh: a path nested in
        # another declared path is bound by its parent's traversal, and the
        # empty path (the model root) contains every other path. Siblings
        # sharing a backbone remain independent roots.
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
        # Launch precision, dtype and exclusions are already resolved above.
        weights=source.weights,
        meshes=meshes,
        attention=attention,
        devices=_devices(description, config.execution.generation_device),
    )
    model = loaded.model

    worker_config = loaded_worker_config(
        model, config.execution, config.ipc.queue_depth
    )

    # ``kv_cache_dtype`` in the launch's quantization_config overrides the
    # KV storage dtype; ``_weight_config`` accepts the key but ignores it.
    override = config.model.quantization_config.get("kv_cache_dtype")
    if override is not None:
        if not isinstance(override, str):
            raise invalid_descriptor("worker KV dtype is unsupported")
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
    """Place the flow route and denoiser modules on the selected device.

    The selected modules are every ``Denoiser``'s direct children that are not
    part of a ``CausalLM``, every ``flow`` entry of a ``ModuleDict``, and
    every ``ImageDecoder`` and ``PatchAutoencoder``.

    Returns:
        Every module path of a selected module, including aliases, mapped to
        ``generation_device``; ``None`` when no generation device is set.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when a generation device is
            set but the model has no module to place on it.
    """
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
    """Resolve media request slots from the worker's publication lifetime.

    A resident media slot occupies three positions of the worker's batch
    queue, one reserved pipeline position and two unresolved outputs, and a
    media worker keeps at least two slots resident. The same three-position
    bound governs ``request_tensor_window`` and ``resolve_request_capacity``
    in ``uniserve_worker.bootstrap.capacity``.

    Only a model containing a ``VideoDecoder`` is adjusted: its batch bounds
    and maximum request-pool size become the slot count, its minimum
    request-pool size becomes two, and it drops the KV token capacity,
    attention backend and generation device. Any other model's configuration
    is returned unchanged.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when the queue depth or
            ``max_batch_calls`` leaves fewer than two slots.
    """
    if any(isinstance(module, VideoDecoder) for module in model.modules()):
        state_slots = min(config.max_batch_calls, queue_depth // 3)
        if state_slots < 2:
            raise unsupported_setup(
                "resident media execution requires two slots of three queue "
                f"positions each; queue depth {queue_depth} holds "
                f"{queue_depth // 3}"
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
