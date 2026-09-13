"""Combine rank launch configuration and computation bindings for model loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ..execution.model_entry import ModelEntry
from ..loader import LoadedModel, LoadRequest
from ..loader.loader import _load_model
from ..loader.source import ModelSource
from .config import WorkerProcessArgs


def prepare_worker_model(config: WorkerProcessArgs) -> ModelSource | None:
    """Resolve model metadata and validate placement before process-group creation."""

    if config.use_stub_model:
        return None
    if config.model is None:
        raise RuntimeError("validated model worker is missing model configuration")
    source = ModelSource.resolve(config.model.path, config.load)
    source.validate(dict(config.components))
    return source


def load_worker_model(
    config: WorkerProcessArgs,
    bindings: Mapping[str, ModelEntry],
    *,
    source: ModelSource | None,
) -> LoadedModel:
    """Load the checkpoint or the explicitly enabled deterministic stub."""

    if config.use_stub_model:
        from ..models.stub import StubConfig, StubModel

        model = StubModel(StubConfig(tuple(bindings)))
        declarations = {component.name: component for component in model.components(model.config)}
        for name, binding in bindings.items():
            binding.calls = declarations[name].calls
        return LoadedModel(
            model=model,
            bindings=bindings,
            tokenizer=None,
            worker_config=replace(
                config.execution, attention_backend="torch_sdpa", encoder_cache_entries=1024
            ),
            sources=(),
            architecture_config={},
        )
    launch = config.model
    if launch is None or source is None:
        raise RuntimeError("validated model worker is missing model configuration")
    return _load_model(
        LoadRequest(
            model_path=launch.path,
            execution=config.execution,
            bindings=bindings,
            max_text_rows=launch.max_text_rows,
            max_video_seconds=launch.max_video_seconds,
            quantization_config=launch.quantization_config,
            load=config.load,
            pipeline_depth=config.ipc.pipeline_depth,
        ),
        source,
    )
