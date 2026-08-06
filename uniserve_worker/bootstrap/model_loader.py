"""Model discovery and materialization for one assembled worker."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from torch import nn

from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import ExecutionConfig
from ..foundation.sizing import DEFAULT_MAX_BATCH_OPS
from ..loader import Loader
from ..loader.paths import read_config, resolve_model_path
from ..nn.mesh import TensorParallelSpec
from ..runtime.compile import TorchCompileConfig, compile_model_pieces
from ..spec import DeploymentOverlay, ModelLoadScope, ModelSpec, resolved_digest
from .catalog import CatalogEntry, resolve_catalog_entry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerModelLoadRequest:
    model_path: str
    device: str
    block_size: int
    kv_token_capacity: int | None
    attention_backend: str | None
    execution: ExecutionConfig
    parallel: TensorParallelSpec
    scope: ModelLoadScope = ModelLoadScope.WHOLE
    generation_kv_capacity_tokens: int | None = None
    generation_device: str | None = None


@dataclass(frozen=True)
class LoadedWorkerModel:
    model: nn.Module
    tokenizer: Any | None
    entry: CatalogEntry
    model_path: str
    scope: ModelLoadScope
    spec: ModelSpec
    overlay: DeploymentOverlay
    weight_digest: str
    resolved_digest: str


def load_worker_model(request: WorkerModelLoadRequest) -> LoadedWorkerModel:
    model_path = resolve_model_path(request.model_path)
    config = read_config(model_path)
    entry = resolve_catalog_entry(tuple(str(value) for value in config.get("architectures") or ()))
    _require_supported_scope(entry, request.scope)

    loaded = Loader().load(
        entry,
        config,
        device=request.device,
        model_path=model_path,
        attention_backend=request.attention_backend,
        model_scope=request.scope.value,
        execution=request.execution,
        parallel=request.parallel,
    )
    model = loaded.model

    _compile_model(model, request.execution)
    _check_model_conformance(model)
    weight_digest = _checkpoint_weight_digest(model_path, config)
    spec = _resolve_model_spec(model, weight_digest=weight_digest)
    spec = _resolve_input_tokens(spec, loaded.tokenizer)
    setattr(model, "spec", spec)
    overlay = _deployment_overlay(request, entry)
    digest = resolved_digest(spec, overlay)
    logger.info(
        "resolved model spec architecture=%s revision=%s digest=%s",
        spec.architecture,
        spec.revision,
        digest,
    )
    return LoadedWorkerModel(
        model=model,
        tokenizer=loaded.tokenizer,
        entry=entry,
        model_path=model_path,
        scope=request.scope,
        spec=spec,
        overlay=overlay,
        weight_digest=weight_digest,
        resolved_digest=digest,
    )


def _resolve_model_spec(
    model: nn.Module,
    *,
    weight_digest: str,
) -> ModelSpec:
    spec = model.spec
    if not isinstance(spec, ModelSpec):
        raise capability_mismatch(f"{type(model).__name__}.spec must be a ModelSpec")
    return replace(spec, revision=weight_digest)


def _resolve_input_tokens(spec: ModelSpec, tokenizer: Any | None) -> ModelSpec:
    images = spec.inputs.images
    if images is None or images.feature_injection is None:
        return spec
    injection = images.feature_injection
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
        return spec
    resolved_injection = replace(
        injection,
        start_token_id=updates.get("start_token_id", injection.start_token_id),
        end_token_id=updates.get("end_token_id", injection.end_token_id),
    )
    resolved_images = replace(images, feature_injection=resolved_injection)
    return replace(spec, inputs=replace(spec.inputs, images=resolved_images))


def _checkpoint_weight_digest(model_path: str, config: Mapping[str, Any]) -> str:
    """Hash every checkpoint tensor file into one stable weight identity."""

    root = Path(model_path)
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in suffixes
            or path.name.endswith((".safetensors.index.json", ".bin.index.json"))
        )
    )
    digest = hashlib.sha256(b"uniserve-checkpoint-weights\0")
    if not files:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        digest.update(b"config-only\0")
        digest.update(canonical.encode("utf-8"))
        return digest.hexdigest()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as checkpoint:
            while chunk := checkpoint.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _deployment_overlay(
    request: WorkerModelLoadRequest,
    entry: CatalogEntry,
) -> DeploymentOverlay:
    execution = request.execution
    return DeploymentOverlay(
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
        resources=entry.resources,
        max_batch_operations=DEFAULT_MAX_BATCH_OPS,
        generation_device=request.generation_device,
    )


def _require_supported_scope(entry: CatalogEntry, scope: ModelLoadScope) -> None:
    if scope not in entry.scopes:
        raise capability_mismatch(
            f"{entry.architecture} does not support {scope.value!r} model materialization"
        )


def _check_model_conformance(model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise capability_mismatch(f"{type(model).__name__} must inherit torch.nn.Module")


def _compile_model(model: nn.Module, execution: ExecutionConfig) -> None:
    config = TorchCompileConfig.from_runtime_config(execution.torch_compile)
    report = compile_model_pieces(model, config=config)
    if report.compiled:
        logger.info(
            "enabled model-stack torch.compile pieces count=%s",
            report.compiled,
        )
