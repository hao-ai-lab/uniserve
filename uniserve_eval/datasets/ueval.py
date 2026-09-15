"""Loads deterministic UEval prompts for interleaved generation."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, ClassVar

from ..types import BenchmarkPoint, Example
from .base import Dataset

UEVAL_HF_REPO = "zlab-princeton/UEval"
_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "query", "text")


class UEvalDataset(Dataset):
    """Adapts local or hosted UEval rows to normalized prompts."""

    name: ClassVar[str] = "ueval"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Extract, shuffle, and select the declared number of prompts."""
        point = self.point
        raw = (
            _load_local(point.dataset_path)
            if point.dataset_path
            else _load_hf(point)
        )
        prompts = [
            prompt for entry in raw if (prompt := _extract_prompt(entry))
        ]
        random.Random(point.load.seed).shuffle(prompts)
        prompts = prompts[: point.load.num_prompts]
        return [
            Example(id=f"ueval-{index:06d}", prompt=prompt)
            for index, prompt in enumerate(prompts)
        ]


def _load_local(dataset_path: str) -> list[dict[str, Any]]:
    """Load object rows from a local Parquet, JSON array, or JSON Lines file."""
    path = Path(dataset_path)
    if path.suffix.lower() == ".parquet":
        load_dataset = getattr(
            importlib.import_module("datasets"), "load_dataset"
        )
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [dict(row) for row in dataset]
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_hf(point: BenchmarkPoint) -> list[dict[str, Any]]:
    """Flatten all splits from the configured UEval hub revision."""
    try:
        load_dataset = getattr(
            importlib.import_module("datasets"), "load_dataset"
        )
    except ImportError as error:
        raise ImportError(
            "loading UEval from the Hugging Face hub requires "
            "the 'datasets' package"
        ) from error
    dataset = load_dataset(UEVAL_HF_REPO, revision=point.dataset_revision)
    rows: list[dict[str, Any]] = []
    if hasattr(dataset, "keys"):
        for key in dataset.keys():
            rows.extend(dict(row) for row in dataset[key])
    else:
        rows.extend(dict(row) for row in dataset)
    return rows


def _extract_prompt(entry: dict[str, Any]) -> str:
    """Return the first non-empty supported prompt field."""
    for field in _PROMPT_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
