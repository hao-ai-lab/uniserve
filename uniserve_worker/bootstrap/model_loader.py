"""Model discovery and materialization for one worker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from ..execution.device_transfer import DeviceTransfer
from ..foundation.errors import unsupported_setup
from ..loader import LoadConfig, LoadRequest, WeightSet, get_model_loader
from ..loader.source import read_model_config, resolve_model_root
from ..models.minimax_h3 import MiniMaxH3Runner
from ..models.minimax_h3.placement import H3Placement
from ..models.minimax_h3.precision import H3LinearPrecisionPolicy
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.mesh import DeviceMesh, GroupCoordinator, TensorParallel
from .capacity import DEFAULT_MAX_REQUEST_POOL_SIZE
from .catalog import CatalogEntry, resolve_catalog_entry
from .config import WorkerProcessArgs
from .execution_config import ExecutionConfig
from .plan import ComponentDeployConfig, ModelLoadScope, WorkerPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerModelLoadRequest:
    """Carries checkpoint identity and deployment metadata into model construction.

    The request also fixes load policy, architecture configuration, and rank geometry.
    """

    model_path: str
    device: str
    block_size: int
    max_batch_operations: int
    max_batch_tokens: int
    kv_token_capacity: int | None
    attention_backend: str | None
    execution: ExecutionConfig
    parallel: TensorParallel
    process_rank: int = 0
    process_world_size: int = 1
    max_model_len: int = 8192
    max_video_seconds: float = 15.0
    quantization_config: dict[str, object] = field(default_factory=dict)
    scope: ModelLoadScope = ModelLoadScope.WHOLE
    generation_device: str | None = None
    load: LoadConfig = LoadConfig()


@dataclass(frozen=True)
class LoadedWorkerModel:
    """Bundles a materialized model with its tokenizer, deployment geometry, weights, and sidecar metadata."""

    model: ExecutionModel
    tokenizer: Any | None
    deployment: WorkerDeployment
    architecture: str
    weights: WeightSet
    weight_sidecars: tuple[str, ...]


def load_worker_model(
    request: WorkerModelLoadRequest,
    *,
    mesh: DeviceMesh | None = None,
    pipeline_depth: int | None = None,
    component_meshes: dict[str, DeviceMesh] | None = None,
    component_deployment: dict[str, ComponentDeployConfig] | None = None,
    process_group: GroupCoordinator | None = None,
) -> LoadedWorkerModel:
    """Resolve architecture, sources, deployment geometry, and checkpoint weights for one model."""

    model_root, repository_id = resolve_model_root(request.model_path, request.load)
    config = read_model_config(model_root)
    entry = resolve_catalog_entry(tuple(str(value) for value in config.get("architectures") or ()))
    _require_supported_scope(entry, request.scope)

    if entry.model_class is MiniMaxH3Runner:
        return _load_h3_worker_model(
            request,
            entry,
            mesh=mesh,
            pipeline_depth=pipeline_depth,
            component_meshes=component_meshes,
            component_deployment=component_deployment,
            process_group=process_group,
        )

    if mesh is not None and (mesh.size("sp") != 1 or mesh.size("pp") != 1):
        raise unsupported_setup(
            f"{entry.architecture} requires an executing sequence/pipeline binding for the requested layout"
        )
    if component_meshes is not None and set(component_meshes) != {"model"}:
        raise unsupported_setup(
            f"{entry.architecture} requires an executing placement binding for these component names"
        )

    quantization_config = request.quantization_config or {}
    if "mode" in quantization_config:
        raise unsupported_setup("quantization modes require a componentized model")
    quant_method = str(quantization_config.get("quant_method", "fp8"))
    components = quantization_config.get("components")
    if components:
        raise unsupported_setup("component quantization policies require a componentized model")
    if quant_method != "fp8":
        raise unsupported_setup(
            f"quantization method {quant_method!r} is not supported by this model"
        )

    load_request = LoadRequest(
        model_path=request.model_path,
        device=request.device,
        execution=request.execution,
        parallel=request.parallel,
        tp_group=mesh.get_group("tp") if mesh is not None else None,
        scope=request.scope,
        transfers=DeviceTransfer(
            (torch.device(request.device), torch.device(request.generation_device))
            if request.generation_device is not None
            else ()
        ),
        load=request.load,
        attention_backend=request.attention_backend,
    )
    loaded = get_model_loader(request.load.load_format).load(
        entry,
        config,
        load_request,
        root=model_root,
        repository_id=repository_id,
    )
    model = _check_model_interface(loaded.model)

    _resolve_input_tokens(model, loaded.tokenizer)
    deployment = _deployment(request)
    logger.info(
        "loaded model architecture=%s weight_version=%d", model.architecture, loaded.weights.version
    )
    return LoadedWorkerModel(
        model=model,
        tokenizer=loaded.tokenizer,
        deployment=deployment,
        architecture=model.architecture,
        weights=loaded.weights,
        weight_sidecars=entry.sidecars,
    )


def materialize_worker_model(
    config: WorkerProcessArgs,
    plan: WorkerPlan,
    mesh: DeviceMesh,
    *,
    component_meshes: dict[str, DeviceMesh] | None = None,
    process_group: GroupCoordinator | None = None,
) -> LoadedWorkerModel:
    """Construct the configured model scope or the explicitly enabled deterministic stub."""

    if config.use_stub_model:
        return _stub_worker_model(config, plan)
    return load_worker_model(
        _checkpoint_request(config, plan, mesh),
        mesh=mesh,
        pipeline_depth=config.ipc.pipeline_depth,
        component_meshes=component_meshes,
        component_deployment=dict(config.components),
        process_group=process_group,
    )


def _load_h3_worker_model(
    request: WorkerModelLoadRequest,
    entry: CatalogEntry,
    *,
    mesh: DeviceMesh | None,
    pipeline_depth: int | None,
    component_meshes: dict[str, DeviceMesh] | None,
    component_deployment: dict[str, ComponentDeployConfig] | None,
    process_group: GroupCoordinator | None,
) -> LoadedWorkerModel:
    """Load an SM100 H3 replica and derive capacities from its resident state pool."""

    if (
        pipeline_depth is None
        or component_meshes is None
        or component_deployment is None
        or process_group is None
    ):
        raise unsupported_setup(
            "MiniMax H3 loading requires resolved component placement and process transfers"
        )
    placement = H3Placement(component_deployment, component_meshes, process_group)
    if process_group.device.type != "cuda" or torch.cuda.get_device_capability(
        process_group.device
    ) < (
        10,
        0,
    ):
        raise unsupported_setup("MiniMax H3 requires an SM100-class CUDA device")
    # Each state slot can have two unresolved outputs. Keep one additional
    # pipeline position available so slot reuse cannot overtake publication.
    unresolved_window = 2
    max_state_slots = min(
        int(request.max_batch_operations),
        int(pipeline_depth) // (unresolved_window + 1),
    )
    if max_state_slots < 2:
        raise unsupported_setup(
            "MiniMax H3 requires capacity for two state slots with two unresolved outputs each"
        )
    precision_policy = H3LinearPrecisionPolicy.from_config(request.quantization_config)
    logger.info(
        "resolved MiniMax H3 precision attention=%s mlp=%s text=%s video_vae=%s",
        precision_policy.transformer_attention,
        precision_policy.transformer_mlp,
        precision_policy.text_encoder,
        precision_policy.video_vae,
    )
    model = MiniMaxH3Runner.from_pretrained(
        request.model_path,
        placement,
        max_state_slots=max_state_slots,
        max_text_rows=request.max_model_len,
        max_video_seconds=request.max_video_seconds,
        cache_dir=request.load.download_dir,
        revision=request.load.revision,
        precision_policy=precision_policy,
    )

    # The allocation performed by the runner is authoritative: free device
    # memory may reduce the usable slots below the requested operation bound.
    state_slots = int(model.dedicated_state_geometry.slot_count)
    max_operations = min(state_slots, int(request.max_batch_operations))
    deployment = replace(
        _deployment(request),
        output_rank=placement.output_rank,
        kv_token_capacity=None,
        attention_backend=None,
        max_batch_operations=max_operations,
        max_batch_tokens=max_operations,
        max_request_pool_size=state_slots,
        generation_device=None,
    )
    weights = WeightSet.from_module(model)
    logger.info(
        "loaded model architecture=%s weight_version=%d", model.architecture, weights.version
    )
    return LoadedWorkerModel(
        model=model,
        tokenizer=None,
        deployment=deployment,
        architecture=model.architecture,
        weights=weights,
        weight_sidecars=entry.sidecars,
    )


def _checkpoint_request(
    config: WorkerProcessArgs,
    plan: WorkerPlan,
    mesh: DeviceMesh,
) -> WorkerModelLoadRequest:
    """Build the rank-local checkpoint load request from process and mesh configuration."""

    model = config.model
    if model is None:
        raise RuntimeError("validated model worker is missing model configuration")
    return WorkerModelLoadRequest(
        model_path=model.path,
        device=config.placement.device,
        block_size=config.resources.block_size,
        max_batch_operations=config.resources.max_batch_operations,
        max_batch_tokens=config.resources.max_batch_tokens,
        kv_token_capacity=config.resources.kv_token_capacity,
        max_model_len=config.resources.max_model_len,
        max_video_seconds=config.resources.max_video_seconds,
        quantization_config=model.quantization_config,
        attention_backend=model.attention_backend,
        execution=config.execution,
        parallel=TensorParallel.from_mesh(mesh),
        process_rank=config.placement.rank,
        process_world_size=config.placement.world_size,
        scope=plan.model_scope,
        generation_device=config.placement.generation_device,
        load=config.load,
    )


def _stub_worker_model(config: WorkerProcessArgs, plan: WorkerPlan) -> LoadedWorkerModel:
    """Construct the configured stub model and its rank-local layer configuration."""

    from ..models.stub import StubModel, stub_deployment

    stub = StubModel()
    weights = WeightSet.from_module(stub)
    return LoadedWorkerModel(
        model=stub,
        tokenizer=None,
        deployment=replace(
            stub_deployment(
                config.resources.block_size,
                max_batch_operations=config.resources.max_batch_operations,
                max_batch_tokens=config.resources.max_batch_tokens,
            ),
            device=config.placement.device,
            model_scope=plan.model_scope.value,
            rank=config.placement.rank,
            world_size=config.placement.world_size,
            kv_token_capacity=config.resources.kv_token_capacity,
            model_dtype=config.execution.model_dtype,
            kv_cache_dtype=config.execution.kv_cache_dtype,
            kv_memory_fraction=config.execution.kv_memory_fraction,
            generation_device=config.placement.generation_device,
        ),
        architecture=stub.architecture,
        weights=weights,
        weight_sidecars=("config.json",),
    )


def _resolve_input_tokens(model: ExecutionModel, tokenizer: Any | None) -> None:
    """Resolve model-specific input token identities from tokenizer metadata."""

    processor = model.image_processor
    injection = getattr(processor, "feature_injection", None)
    if processor is None or injection is None:
        return
    updates: dict[str, int] = {}
    for token_field, id_field in (("start_token", "start_token_id"), ("end_token", "end_token_id")):
        token = getattr(injection, token_field)
        token_id = getattr(injection, id_field)
        if token_id is not None or token is None:
            continue
        if tokenizer is None:
            raise unsupported_setup(
                f"model input declaration requires tokenizer resolution for {token!r}"
            )
        resolved = tokenizer.convert_tokens_to_ids(token)
        if resolved is None or int(resolved) < 0:
            raise unsupported_setup(f"tokenizer does not define declared token {token!r}")
        updates[id_field] = int(resolved)
    if not updates:
        return
    resolved_injection = replace(
        injection,
        start_token_id=updates.get("start_token_id", injection.start_token_id),
        end_token_id=updates.get("end_token_id", injection.end_token_id),
    )
    model.image_processor = replace(processor, feature_injection=resolved_injection)


def _deployment(request: WorkerModelLoadRequest) -> WorkerDeployment:
    """Construct validated rank-local deployment metadata for model loading."""

    execution = request.execution
    return WorkerDeployment(
        device=request.device,
        model_scope=request.scope.value,
        rank=request.process_rank,
        world_size=request.process_world_size,
        block_size=request.block_size,
        kv_token_capacity=request.kv_token_capacity,
        attention_backend=request.attention_backend,
        model_dtype=execution.model_dtype,
        kv_cache_dtype=execution.kv_cache_dtype,
        kv_memory_fraction=execution.kv_memory_fraction,
        max_batch_operations=request.max_batch_operations,
        max_batch_tokens=request.max_batch_tokens,
        max_request_pool_size=DEFAULT_MAX_REQUEST_POOL_SIZE,
        generation_device=request.generation_device,
    )


def _require_supported_scope(entry: CatalogEntry, scope: ModelLoadScope) -> None:
    """Require the requested model scope to be advertised by the catalog entry."""

    if scope not in entry.scopes:
        raise unsupported_setup(
            f"{entry.architecture} does not support {scope.value!r} model materialization"
        )


def _check_model_interface(model: nn.Module) -> ExecutionModel:
    """Require a constructed module to implement the execution-model contract."""

    if not isinstance(model, ExecutionModel):
        raise unsupported_setup(f"{type(model).__name__} must implement ExecutionModel")
    return model
