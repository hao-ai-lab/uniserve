"""Model discovery and materialization for one assembled worker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from torch import nn

from ..foundation.errors import capability_mismatch
from ..loader import LoadConfig, LoadRequest, WeightSet, get_model_loader
from ..loader.source import read_model_config, resolve_model_root
from ..models.identity import ModelIdentity, architecture_identity
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.mesh import TensorParallelSpec
from .capacity import DEFAULT_MAX_BATCH_OPS, DEFAULT_MAX_REQUEST_POOL_SIZE
from .catalog import CatalogEntry, resolve_catalog_entry
from .execution_config import ExecutionConfig
from .plan import ModelLoadScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerModelLoadRequest:
    model_path: str
    device: str
    block_size: int
    max_batch_tokens: int
    kv_token_capacity: int | None
    attention_backend: str | None
    execution: ExecutionConfig
    parallel: TensorParallelSpec
    scope: ModelLoadScope = ModelLoadScope.WHOLE
    generation_kv_capacity_tokens: int | None = None
    generation_device: str | None = None
    load: LoadConfig = LoadConfig()


@dataclass(frozen=True)
class LoadedWorkerModel:
    model: ExecutionModel
    tokenizer: Any | None
    entry: CatalogEntry
    model_path: str
    scope: ModelLoadScope
    deployment: WorkerDeployment
    identity: ModelIdentity
    weights: WeightSet


def load_worker_model(request: WorkerModelLoadRequest) -> LoadedWorkerModel:
    model_root, repository_id = resolve_model_root(request.model_path, request.load)
    config = read_model_config(model_root)
    entry = resolve_catalog_entry(tuple(str(value) for value in config.get("architectures") or ()))
    _require_supported_scope(entry, request.scope)

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
    model = _check_model_conformance(loaded.model)

    _resolve_input_tokens(model, loaded.tokenizer)
    deployment = _deployment(request)
    identity = ModelIdentity(
        architecture=model.architecture,
        architecture_digest=architecture_identity(
            entry.architecture,
            loaded.architecture_config,
        ),
        weight_digest=loaded.weights.digest,
    )
    logger.info(
        "loaded model architecture=%s architecture_digest=%s weight_digest=%s",
        identity.architecture,
        identity.architecture_digest,
        identity.weight_digest,
    )
    return LoadedWorkerModel(
        model=model,
        tokenizer=loaded.tokenizer,
        entry=entry,
        model_path=str(model_root),
        scope=request.scope,
        deployment=deployment,
        identity=identity,
        weights=loaded.weights,
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
            raise capability_mismatch(
                f"model input declaration requires tokenizer resolution for {token!r}"
            )
        resolved = tokenizer.convert_tokens_to_ids(token)
        if resolved is None or int(resolved) < 0:
            raise capability_mismatch(f"tokenizer does not define declared token {token!r}")
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
        generation_kv_capacity_tokens=request.generation_kv_capacity_tokens,
        attention_backend=request.attention_backend,
        model_dtype=execution.model_dtype,
        kv_cache_dtype=execution.kv_cache_dtype,
        kv_memory_fraction=execution.kv_memory_fraction,
        max_batch_operations=DEFAULT_MAX_BATCH_OPS,
        max_batch_tokens=request.max_batch_tokens,
        max_request_pool_size=DEFAULT_MAX_REQUEST_POOL_SIZE,
        generation_device=request.generation_device,
    )


def _require_supported_scope(entry: CatalogEntry, scope: ModelLoadScope) -> None:
    if scope not in entry.scopes:
        raise capability_mismatch(
            f"{entry.architecture} does not support {scope.value!r} model materialization"
        )


def _check_model_conformance(model: nn.Module) -> ExecutionModel:
    if not isinstance(model, ExecutionModel):
        raise capability_mismatch(f"{type(model).__name__} must implement ExecutionModel")
    return model
