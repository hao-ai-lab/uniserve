"""CPU/meta dummy loader for tests and bring-up without real weights."""
from __future__ import annotations

from typing import Any, Type

import torch

from ..contracts.model_protocols import UniModel
from .base import BaseModelLoader, LoadResult
from .registry import register_loader

__all__ = [
    'DummyModelLoader',
]


class DummyModelLoader(BaseModelLoader):
    def load_model(
        self,
        model_cls: Type[UniModel],
        config: Any,
        *,
        device: str = "cpu",
        model_path: str | None = None,
        **kwargs: Any,
    ) -> LoadResult:
        del model_path, kwargs
        model = model_cls(config=config)  # type: ignore[call-arg]
        if hasattr(model, "to"):
            model.to(device)  # type: ignore[attr-defined]
        for param in getattr(model, "parameters", lambda: [])():
            if param.is_floating_point():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                param.zero_()
        return LoadResult(model=model, tokenizer=None, device=device)


register_loader("dummy", DummyModelLoader())
