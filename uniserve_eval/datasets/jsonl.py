"""Loads evaluator examples from a local JSON Lines file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

_OPTIONAL = (
    "messages",
    "prompt_len",
    "output_len",
    "max_tokens",
    "input_image_b64",
    "input_image_mime",
    "width",
    "height",
    "steps",
    "seed",
    "aspect_ratio",
)


class JsonlDataset(Dataset):
    """Adapts schema-checked JSON objects to benchmark examples."""

    name: ClassVar[str] = "jsonl"
    requires_path: ClassVar[bool] = True

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Read normalized rows up to the benchmark's declared count."""

        point = self.point
        if not point.dataset_path:
            raise ValueError("dataset 'jsonl' requires dataset_path")
        rows: list[Example] = []
        with Path(point.dataset_path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"jsonl row {line_no} must be an object")
                if "id" not in raw or "prompt" not in raw:
                    raise ValueError(f"jsonl row {line_no} requires id and prompt")
                if ("width" in raw) != ("height" in raw):
                    raise ValueError(f"jsonl row {line_no} must provide width and height together")
                fields = {key: raw[key] for key in _OPTIONAL if key in raw}
                rows.append(Example(id=str(raw["id"]), prompt=str(raw["prompt"]), **fields))
                if len(rows) == point.load.num_prompts:
                    break
        return rows
