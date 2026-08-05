"""Deterministic Beans image-to-text workload."""

from __future__ import annotations

import base64
import importlib
import io
import random
from typing import Any

BEANS_REPOSITORY = "AI-Lab-Makerere/beans"


def load_beans(
    num_requests: int,
    *,
    seed: int,
    revision: str | None,
    question: str,
) -> list[dict[str, Any]]:
    load_dataset = getattr(importlib.import_module("datasets"), "load_dataset")
    dataset = load_dataset(
        BEANS_REPOSITORY,
        split="train",
        revision=revision,
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = indices[:num_requests]
    if len(selected) != num_requests:
        raise ValueError(f"Beans resolved {len(selected)} images; expected {num_requests}")
    rows = []
    for position, source_index in enumerate(selected):
        image = dataset[source_index]["image"].convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        rows.append(
            {
                "id": f"beans-{position:03d}-{source_index}",
                "prompt": question,
                "input_image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "input_image_mime": "image/jpeg",
            }
        )
    return rows
