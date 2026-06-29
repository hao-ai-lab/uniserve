"""MJHQ-30K prompt loader for text-to-image performance benchmarking.

Uses the prompts from ``playgroundai/MJHQ-30K``'s ``meta_data.json`` (a dict
keyed by image hash with ``{"category", "prompt"}``). Only the prompts are needed
for a speed benchmark -- the reference images (for FID) are out of scope since we
measure performance only.

``dataset_path`` may point to a local ``meta_data.json``; otherwise it is
auto-downloaded from the HF dataset repo.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


def load_mjhq(
    dataset_path: str | None,
    num_requests: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    path = dataset_path or ""
    if not (path and os.path.isfile(path)):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=MJHQ_REPO_ID,
            filename=MJHQ_META_FILENAME,
            repo_type="dataset",
        )

    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)

    entries = [
        (key, value)
        for key, value in meta.items()
        if isinstance(value, dict) and value.get("prompt")
    ]
    random.Random(seed).shuffle(entries)

    rows: list[dict[str, Any]] = []
    for key, value in entries:
        if len(rows) >= num_requests:
            break
        rows.append(
            {
                "id": f"mjhq-{key}",
                "task": "t2i",
                "prompt": str(value["prompt"]),
                "category": value.get("category"),
            }
        )
    if not rows:
        raise ValueError("MJHQ-30K source yielded no usable prompts")
    return rows
