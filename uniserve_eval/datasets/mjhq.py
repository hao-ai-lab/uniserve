"""MJHQ-30K prompt loader."""

from __future__ import annotations

import json
import os
import random
from typing import Any

from ..types import BenchmarkPoint, Example

MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


def load_mjhq(point: BenchmarkPoint, _tokenizer: Any) -> list[Example]:
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
