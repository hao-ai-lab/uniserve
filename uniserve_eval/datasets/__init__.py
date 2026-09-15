"""Registers dataset adapters and resolves configured example sources."""

from __future__ import annotations

from typing import Any

from ..types import BenchmarkPoint, Example
from .base import Dataset
from .beans import BeansDataset
from .jsonl import JsonlDataset
from .minimax_h3 import MiniMaxH3Dataset
from .mjhq import MJHQDataset
from .pie_bench import PieBenchDataset
from .sharegpt import ShareGPTDataset
from .ueval import UEvalDataset

DATASETS: dict[str, type[Dataset]] = {
    "sharegpt": ShareGPTDataset,
    "mjhq": MJHQDataset,
    "beans": BeansDataset,
    "ueval": UEvalDataset,
    "pie-bench": PieBenchDataset,
    "jsonl": JsonlDataset,
    "minimax-h3": MiniMaxH3Dataset,
}


def get_dataset(name: str) -> type[Dataset]:
    """Return the dataset adapter registered under a configuration name."""
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS)) or "(none)"
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return DATASETS[name]


def list_datasets() -> tuple[type[Dataset], ...]:
    """Return registered dataset adapters in name order."""
    return tuple(DATASETS[name] for name in sorted(DATASETS))


def load_examples(point: BenchmarkPoint) -> tuple[list[Example], Any | None]:
    """Load the exact row count and optional tokenizer for a benchmark point."""
    dataset_cls = get_dataset(point.dataset)
    if dataset_cls.requires_path and not point.dataset_path:
        raise ValueError(f"dataset {point.dataset!r} requires dataset_path")
    tokenizer: Any | None = None
    if dataset_cls.requires_tokenizer:
        if not point.tokenizer and not point.model:
            raise ValueError(f"dataset {point.dataset!r} requires tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            point.tokenizer or point.model,
            trust_remote_code=True,
        )
    rows = dataset_cls(point).load(tokenizer)
    if len(rows) != point.load.num_prompts:
        raise ValueError(
            f"dataset resolved {len(rows)} rows; benchmark point requires "
            f"exactly {point.load.num_prompts}"
        )
    return rows, tokenizer


__all__ = [
    "DATASETS",
    "BeansDataset",
    "Dataset",
    "JsonlDataset",
    "MJHQDataset",
    "MiniMaxH3Dataset",
    "PieBenchDataset",
    "ShareGPTDataset",
    "UEvalDataset",
    "get_dataset",
    "list_datasets",
    "load_examples",
]
