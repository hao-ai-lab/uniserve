"""Model discovery and materialization for one assembled worker."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

from ..contracts.model_family import ModelFamilyDescriptor, ModelLoadScope
from ..contracts.model_protocols import UniModel
from ..contracts.model_spec import DeploymentOverlay, ModelSpec, resolved_digest
from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import get_execution_config
from ..loader import get_loader_for_descriptor
from ..loader.paths import read_config, resolve_model_path
from ..runtime.compile import TorchCompileConfig, compile_model_pieces
from .catalog import MODEL_CATALOG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerModelLoadRequest:
    model_path: str
    device: str
    block_size: int
    kv_token_capacity: int | None
    attention_backend: str | None
    scope: ModelLoadScope = ModelLoadScope.WHOLE
    generation_kv_capacity_tokens: int | None = None
    load_format: str = "default"
    tp_rank: int = 0
    tp_size: int = 1


@dataclass(frozen=True)
class LoadedWorkerModel:
    model: UniModel
    descriptor: ModelFamilyDescriptor
    model_path: str
    scope: ModelLoadScope
    spec: ModelSpec
    overlay: DeploymentOverlay
    resolved_digest: str


def load_worker_model(request: WorkerModelLoadRequest) -> LoadedWorkerModel:
    model_path = resolve_model_path(request.model_path)
    descriptor = MODEL_CATALOG.resolve_descriptor(model_architecture_candidates(model_path))
    model_class = descriptor.model_class
    _require_supported_scope(model_class, request.scope)
    config = read_config(model_path)

    loader_override = None if request.load_format.lower() == "default" else request.load_format
    model = (
        get_loader_for_descriptor(
            descriptor,
            override=loader_override,
        )
        .load_model(
            cast(type[UniModel], model_class),
            config,
            device=request.device,
            model_path=model_path,
            block_size=request.block_size,
            kv_token_capacity=request.kv_token_capacity,
            attention_backend=request.attention_backend,
            gen_snapshot_kv_capacity=(request.generation_kv_capacity_tokens),
            tower_role=request.scope.tower_role,
        )
        .model
    )

    _configure_model_tokenizer(model, model_path)
    _compile_model(model)
    _check_model_conformance(model)
    spec = _resolve_model_spec(model, model_path=model_path, config=config)
    overlay = _deployment_overlay(request)
    digest = resolved_digest(spec, overlay)
    logger.info(
        "resolved model spec architecture=%s revision=%s digest=%s",
        spec.architecture,
        spec.revision,
        digest,
    )
    return LoadedWorkerModel(
        model=model,
        descriptor=descriptor,
        model_path=model_path,
        scope=request.scope,
        spec=spec,
        overlay=overlay,
        resolved_digest=digest,
    )


def _resolve_model_spec(
    model: UniModel,
    *,
    model_path: str,
    config: Mapping[str, Any],
) -> ModelSpec:
    spec = model.model_spec()
    if spec is None:
        raise capability_mismatch(f"{type(model).__name__} declares no model_spec()")
    declared = spec.op_kinds()
    supported = frozenset(str(kind) for kind in model.supported_ops)
    if declared != supported:
        raise capability_mismatch(
            f"{type(model).__name__} model_spec routes accept {sorted(declared)} "
            f"but the model supports {sorted(supported)}"
        )
    return replace(spec, revision=_checkpoint_revision(model_path, config))


def _checkpoint_revision(model_path: str, config: Mapping[str, Any]) -> str:
    """Cheap stable checkpoint identity for the resolved spec.

    The sha256 of the canonical ``config.json`` payload; a checkpoint that
    ships no config is identified by its directory name. Not a weights digest.
    """
    if config:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(model_path).name


def _deployment_overlay(request: WorkerModelLoadRequest) -> DeploymentOverlay:
    execution = get_execution_config()
    return DeploymentOverlay(
        device=request.device,
        model_scope=request.scope.value,
        tp_rank=request.tp_rank,
        tp_size=request.tp_size,
        block_size=request.block_size,
        kv_token_capacity=request.kv_token_capacity,
        generation_kv_capacity_tokens=request.generation_kv_capacity_tokens,
        attention_backend=request.attention_backend,
        model_dtype=execution.model_dtype,
        kv_cache_dtype=execution.kv_cache_dtype,
    )


def _require_supported_scope(
    model_class: type[UniModel],
    scope: ModelLoadScope,
) -> None:
    if scope is ModelLoadScope.WHOLE:
        return
    declared = frozenset(
        str(value)
        for value in getattr(
            model_class,
            "supported_model_load_scopes",
            (),
        )
    )
    if scope.value not in declared:
        raise capability_mismatch(
            f"{model_class.__name__} does not support {scope.value!r} model materialization"
        )


def _check_model_conformance(model: UniModel) -> None:
    if not isinstance(model, UniModel):
        raise capability_mismatch(f"{type(model).__name__} must inherit UniModel")


def _compile_model(model: UniModel) -> None:
    config = TorchCompileConfig.from_runtime_config()
    report = compile_model_pieces(model, config=config)
    if report.compiled:
        logger.info(
            "enabled model-stack torch.compile pieces count=%s",
            report.compiled,
        )


def _configure_model_tokenizer(model: UniModel, model_path: str) -> None:
    hook = getattr(model, "configure_tokenizer", None)
    if not callable(hook):
        return
    _invoke_optional_model_hook(
        hook,
        model_path=model_path,
        tokenizer_vocab_size=_read_tokenizer_vocab_size(model_path),
    )


def _invoke_optional_model_hook(function: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items() if key in signature.parameters}
    )
    return function(**supported)


def _read_tokenizer_vocab_size(model_path: str) -> int | None:
    root = Path(model_path)
    max_token_id = -1
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            vocabulary = (data.get("model") or {}).get("vocab") or {}
            if isinstance(vocabulary, dict):
                token_ids = [int(value) for value in vocabulary.values() if isinstance(value, int)]
                if token_ids:
                    max_token_id = max(max_token_id, max(token_ids))
            added_tokens = data.get("added_tokens") or []
            if isinstance(added_tokens, list):
                token_ids = [
                    int(token["id"])
                    for token in added_tokens
                    if isinstance(token, dict) and isinstance(token.get("id"), int)
                ]
                if token_ids:
                    max_token_id = max(max_token_id, max(token_ids))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.debug(
                "could not read vocabulary size from %s",
                tokenizer_json,
                exc_info=True,
            )

    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists():
        try:
            data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
            decoder = data.get("added_tokens_decoder") or {}
            if isinstance(decoder, dict):
                token_ids = [int(key) for key in decoder]
                if token_ids:
                    max_token_id = max(max_token_id, max(token_ids))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.debug(
                "could not read vocabulary size from %s",
                tokenizer_config,
                exc_info=True,
            )
    return max_token_id + 1 if max_token_id >= 0 else None


def model_architecture_candidates(model_path: str) -> list[str]:
    config = read_config(model_path)
    architectures = [str(value) for value in config.get("architectures") or []]
    model_type = config.get("model_type")
    if model_type is not None:
        architectures.append(str(model_type))
    if not architectures:
        architectures.append(Path(model_path).name)
    architectures.extend(MODEL_CATALOG.detect_architectures(model_path))
    return architectures
