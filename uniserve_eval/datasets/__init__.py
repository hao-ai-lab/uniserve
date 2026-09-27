"""Registers dataset adapters and resolves configured example sources.

A benchmark point's `dataset` key selects an adapter from `DATASETS`.
`config.load_config` checks the point's tokenizer and path settings against
the adapter when a profile is loaded, and `pipeline.run.run_point` calls
`load_examples` to build the point's rows before the measured window opens.
"""

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
from .systemone import SystemOneDataset
from .ueval import UEvalDataset

# Maps the `dataset` configuration name to its adapter class.
DATASETS: dict[str, type[Dataset]] = {
    "sharegpt": ShareGPTDataset,
    "mjhq": MJHQDataset,
    "beans": BeansDataset,
    "ueval": UEvalDataset,
    "pie-bench": PieBenchDataset,
    "jsonl": JsonlDataset,
    "minimax-h3": MiniMaxH3Dataset,
    "systemone": SystemOneDataset,
}


def get_dataset(name: str) -> type[Dataset]:
    """Return the dataset adapter registered under a configuration name.

    Raises:
        KeyError: If no adapter is registered under `name`.
    """
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS)) or "(none)"
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return DATASETS[name]


def load_examples(point: BenchmarkPoint) -> tuple[list[Example], Any | None]:
    """Load the exact row count and optional tokenizer for a benchmark point.

    `config.load_config` validates a point's path and tokenizer settings
    with `Dataset.check_point`; the checks here also apply to a
    `BenchmarkPoint` constructed directly. A tokenizer is loaded only for
    adapters that declare `requires_tokenizer`, from `point.tokenizer` or,
    when that is empty, `point.model`. Errors from importing `transformers`,
    loading the tokenizer, and the adapter's `load` propagate unchanged.

    Returns:
        The adapter's rows and the tokenizer, or `None` when the adapter
        does not require one.

    Raises:
        KeyError: If `point.dataset` names no registered adapter.
        ValueError: If a required path or tokenizer source is missing, or
            the adapter returns a row count other than
            `point.load.num_prompts`.
    """
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
    "SystemOneDataset",
    "UEvalDataset",
    "get_dataset",
    "load_examples",
]
