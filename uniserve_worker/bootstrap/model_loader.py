"""Combine rank launch configuration and computation bindings for model loading."""

from __future__ import annotations

from dataclasses import replace

from ..loader import LoadedModel, LoadRequest, load_model
from ..nn.mesh import EntryBindings
from .config import WorkerProcessArgs


def load_worker_model(config: WorkerProcessArgs, bindings: EntryBindings) -> LoadedModel:
    """Load the checkpoint or the explicitly enabled deterministic stub."""

    if config.use_stub_model:
        from ..models.stub import StubModel

        model = StubModel()
        return LoadedModel(
            model=model,
            tokenizer=None,
            worker_config=replace(config.execution, attention_backend="torch_sdpa"),
            sources=(),
            architecture_config={},
        )
    launch = config.model
    if launch is None:
        raise RuntimeError("validated model worker is missing model configuration")
    return load_model(
        LoadRequest(
            model_path=launch.path,
            execution=config.execution,
            bindings=bindings,
            max_text_rows=launch.max_text_rows,
            max_video_seconds=launch.max_video_seconds,
            quantization_config=launch.quantization_config,
            load=config.load,
            pipeline_depth=config.ipc.pipeline_depth,
        )
    )
