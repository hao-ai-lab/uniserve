"""Model discovery and materialization for one worker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from ..foundation.errors import unsupported_setup
from ..loader import LoadConfig, LoadRequest, WeightSet, get_model_loader
from ..loader.source import read_model_config, resolve_model_root
from ..models.minimax_h3 import MiniMaxH3Model
from ..models.minimax_h3.precision import H3LinearPrecisionPolicy
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.mesh import DeviceMesh, TensorParallel
from .capacity import DEFAULT_MAX_REQUEST_POOL_SIZE
from .catalog import CatalogEntry, resolve_catalog_entry
from .config import WorkerProcessArgs
from .execution_config import ExecutionConfig
from .plan import ModelLoadScope, WorkerPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerModelLoadRequest:
    model_path: str
    device: str
    block_size: int
    max_batch_operations: int
    max_batch_tokens: int
    kv_token_capacity: int | None
    attention_backend: str | None
    execution: ExecutionConfig
    parallel: TensorParallel
    max_model_len: int = 8192
    max_video_seconds: float = 15.0
    quantization_config: dict[str, object] = field(default_factory=dict)
    scope: ModelLoadScope = ModelLoadScope.WHOLE
    generation_device: str | None = None
    load: LoadConfig = LoadConfig()


@dataclass(frozen=True)
class LoadedWorkerModel:
    model: ExecutionModel | MiniMaxH3Model
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
    media_spool: str | None = None,
) -> LoadedWorkerModel:
    model_root, repository_id = resolve_model_root(request.model_path, request.load)
    config = read_model_config(model_root)
    entry = resolve_catalog_entry(tuple(str(value) for value in config.get("architectures") or ()))
    _require_supported_scope(entry, request.scope)

    if entry.model_class is MiniMaxH3Model:
        return _load_h3_worker_model(
            request,
            entry,
            mesh=mesh,
            pipeline_depth=pipeline_depth,
            media_spool=media_spool,
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
        scope=request.scope,
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
) -> LoadedWorkerModel:
    if config.use_stub_model:
        return _stub_worker_model(config, plan)
    return load_worker_model(
        _checkpoint_request(config, plan, mesh),
        mesh=mesh,
        pipeline_depth=config.ipc.pipeline_depth,
        media_spool=config.media_spool,
    )


def _load_h3_worker_model(
    request: WorkerModelLoadRequest,
    entry: CatalogEntry,
    *,
    mesh: DeviceMesh | None,
    pipeline_depth: int | None,
    media_spool: str | None,
) -> LoadedWorkerModel:
    from pathlib import Path

    if mesh is None or pipeline_depth is None:
        raise unsupported_setup("MiniMax H3 loading requires the worker device mesh")
    if request.parallel.size != 4 or mesh.size("tp") != 4 or mesh.size("sp") != 4:
        raise unsupported_setup("MiniMax H3 requires one TP4/SP4 replica")
    if mesh.local_device.type != "cuda" or torch.cuda.get_device_capability(mesh.local_device) < (
        10,
        0,
    ):
        raise unsupported_setup("MiniMax H3 requires an SM100-class CUDA device")
    if not media_spool or not Path(media_spool).expanduser().is_absolute():
        raise unsupported_setup("MiniMax H3 requires an absolute shared media spool")
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
    model = MiniMaxH3Model.from_pretrained(
        request.model_path,
        mesh,
        max_state_slots=max_state_slots,
        max_text_rows=request.max_model_len,
        max_video_seconds=request.max_video_seconds,
        cache_dir=request.load.download_dir,
        revision=request.load.revision,
        precision_policy=precision_policy,
    )
    state_slots = int(model.states.slot_count)
    max_operations = min(state_slots, int(request.max_batch_operations))
    deployment = replace(
        _deployment(request),
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
        scope=plan.model_scope,
        generation_device=config.placement.generation_device,
        load=config.load,
    )


def _stub_worker_model(config: WorkerProcessArgs, plan: WorkerPlan) -> LoadedWorkerModel:
    from ..server.stub import StubModel, stub_deployment

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
            tp_rank=config.placement.tp_rank,
            tp_size=config.placement.tp_size,
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
    execution = request.execution
    return WorkerDeployment(
        device=request.device,
        model_scope=request.scope.value,
        tp_rank=request.parallel.rank,
        tp_size=request.parallel.size,
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
    if scope not in entry.scopes:
        raise unsupported_setup(
            f"{entry.architecture} does not support {scope.value!r} model materialization"
        )


def _check_model_interface(model: nn.Module) -> ExecutionModel:
    if not isinstance(model, ExecutionModel):
        raise unsupported_setup(f"{type(model).__name__} must implement ExecutionModel")
    return model
