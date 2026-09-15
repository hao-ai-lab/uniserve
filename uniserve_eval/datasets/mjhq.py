"""Loads deterministic text-to-image prompts from MJHQ-30K metadata."""

from __future__ import annotations

import json
import os
import random
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


class MJHQDataset(Dataset):
    """Adapts MJHQ metadata entries to text-only generation examples."""

    name: ClassVar[str] = "mjhq"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Select seeded prompt entries from local or downloaded metadata."""
        point = self.point
        path = point.dataset_path or ""
        if not (path and os.path.isfile(path)):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=MJHQ_REPO_ID,
                filename=MJHQ_META_FILENAME,
                repo_type="dataset",
                revision=point.dataset_revision,
            )

        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)

        entries = [
            (key, value)
            for key, value in meta.items()
            if isinstance(value, dict) and value.get("prompt")
        ]
        random.Random(point.load.seed).shuffle(entries)

        rows: list[Example] = []
        for key, value in entries:
            if len(rows) >= point.load.num_prompts:
                break
            rows.append(Example(id=f"mjhq-{key}", prompt=str(value["prompt"])))
        return rows
