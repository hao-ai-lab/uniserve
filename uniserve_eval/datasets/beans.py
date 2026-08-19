"""Beans image-to-text loader."""

from __future__ import annotations

import base64
import importlib
import io
import random
from typing import Any

from ..types import DEFAULT_I2T_QUESTION, BenchmarkPoint, Example

BEANS_REPOSITORY = "AI-Lab-Makerere/beans"


def load_beans(point: BenchmarkPoint, _tokenizer: Any) -> list[Example]:
    load_dataset = getattr(importlib.import_module("datasets"), "load_dataset")
    dataset = load_dataset(
        BEANS_REPOSITORY,
        split="train",
        revision=point.dataset_revision,
    )
    indices = list(range(len(dataset)))
    random.Random(point.load.seed).shuffle(indices)
    selected = indices[: point.load.num_prompts]
    question = point.question or DEFAULT_I2T_QUESTION
    rows: list[Example] = []
    for position, source_index in enumerate(selected):
        image = dataset[source_index]["image"].convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        rows.append(
            Example(
                id=f"beans-{position:03d}-{source_index}",
                prompt=question,
                input_image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                input_image_mime="image/jpeg",
            )
        )
    return rows
