"""CPU/meta dummy loader for tests and bring-up without real weights."""
from __future__ import annotations

from typing import Any, Type, cast

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
        model = cast(Any, model_cls)(config=config)
        move_to = getattr(model, "to", None)
        if callable(move_to):
            move_to(device)
        for param in getattr(model, "parameters", lambda: [])():
            if param.is_floating_point():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                param.zero_()
        return LoadResult(model=model, tokenizer=None, device=device)


register_loader("dummy", DummyModelLoader())
