"""Loads evaluator examples from a local JSON Lines file.

Each non-blank line is one JSON object that becomes one ``Example``. Rows
require ``id`` and ``prompt``; the per-example overrides named in
``_OPTIONAL`` are copied through unchanged when present. Unlike the
shuffling adapters, this one keeps file order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

# Example fields a row may set. Values are passed to ``Example`` as-is;
# neither this adapter nor ``Example`` checks their types.
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
    "seconds",
)


class JsonlDataset(Dataset):
    """Adapts JSON Lines objects to benchmark examples in file order."""

    name: ClassVar[str] = "jsonl"
    requires_path: ClassVar[bool] = True

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Read rows in file order up to the benchmark's declared count.

        Reading stops once ``num_prompts`` rows are collected, so later lines
        are neither parsed nor validated. A shorter file yields fewer rows;
        ``load_examples`` rejects that count mismatch.

        Args:
            tokenizer: Unused; rows may declare ``prompt_len`` and
                ``output_len`` directly.

        Returns:
            Examples in file order, at most ``num_prompts`` of them.

        Raises:
            ValueError: If ``dataset_path`` is unset, a line is not valid
                JSON, a row is not an object, a row lacks ``id`` or
                ``prompt``, or a row sets only one of ``width`` and
                ``height``.
            OSError: If the file cannot be opened.
        """
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
                    raise ValueError(
                        f"jsonl row {line_no} requires id and prompt"
                    )
                if ("width" in raw) != ("height" in raw):
                    raise ValueError(
                        f"jsonl row {line_no} must provide width "
                        f"and height together"
                    )
                fields = {key: raw[key] for key in _OPTIONAL if key in raw}
                rows.append(
                    Example(
                        id=str(raw["id"]), prompt=str(raw["prompt"]), **fields
                    )
                )
                if len(rows) == point.load.num_prompts:
                    break
        return rows
