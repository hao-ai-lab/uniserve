"""Construction boundary for architecture-owned checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import ExecutionConfig
from ..nn.mesh import TensorParallelSpec


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: nn.Module
    tokenizer: Any | None
    device: str


class Loader:
    """Invoke the checkpoint constructor owned by the selected model class."""

    def load(
        self,
        entry: Any,
        config: Any,
        *,
        model_path: str,
        device: str,
        attention_backend: str | None,
        model_scope: str,
        execution: ExecutionConfig,
        parallel: TensorParallelSpec,
    ) -> LoadedModel:
        constructor = getattr(entry.model_class, "from_checkpoint", None)
        if not callable(constructor):
            raise capability_mismatch(
                f"{entry.model_class.__name__} must implement from_checkpoint"
            )
        result = constructor(
            config,
            model_path=model_path,
            device=device,
            attention_backend=attention_backend,
            model_scope=model_scope,
            execution=execution,
            parallel=parallel,
        )
        if (
            not isinstance(result, tuple)
            or len(result) != 3
            or not isinstance(result[0], nn.Module)
            or not isinstance(result[2], str)
        ):
            raise capability_mismatch(
                f"{entry.model_class.__name__}.from_checkpoint returned an invalid model result"
            )
        return LoadedModel(model=result[0], tokenizer=result[1], device=result[2])


__all__ = ["LoadedModel", "Loader"]
