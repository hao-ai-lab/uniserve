from __future__ import annotations

from typing import Any

from ..registry import get_dataset
from ..types import BenchmarkPoint, Example
from . import _registry

_ = _registry


def load_examples(point: BenchmarkPoint) -> tuple[list[Example], Any | None]:
    spec = get_dataset(point.dataset)
    if spec.requires_path and not point.dataset_path:
        raise ValueError(f"dataset {point.dataset!r} requires dataset_path")
    tokenizer: Any | None = None
    if spec.requires_tokenizer:
        if not point.tokenizer and not point.model:
            raise ValueError(f"dataset {point.dataset!r} requires tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            point.tokenizer or point.model,
            trust_remote_code=True,
        )
    rows = spec.loader(point, tokenizer)
    if len(rows) != point.load.num_prompts:
        raise ValueError(
            f"dataset resolved {len(rows)} rows; benchmark contract requires "
            f"exactly {point.load.num_prompts}"
        )
    return rows, tokenizer
