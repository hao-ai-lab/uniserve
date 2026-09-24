"""Loads Beans photographs as deterministic image-to-text examples.

Each example pairs one Beans image, re-encoded as a base64 JPEG, with the
point's `question` or `DEFAULT_I2T_QUESTION`.
"""

from __future__ import annotations

import base64
import importlib
import io
import random
from typing import Any, ClassVar

from ..types import DEFAULT_I2T_QUESTION, Example
from .base import Dataset

BEANS_REPOSITORY = "AI-Lab-Makerere/beans"


class BeansDataset(Dataset):
    """Adapts the Beans training split to embedded JPEG prompts.

    With `dataset_path` set, rows come from the Parquet data at that path;
    otherwise they come from the Hub repository `BEANS_REPOSITORY` at
    `dataset_revision`.
    """

    name: ClassVar[str] = "beans"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Select seeded rows and encode their RGB images as JPEG data.

        The source rows are shuffled with `random.Random(point.load.seed)`
        and the first `point.load.num_prompts` are taken, so the same seed
        and source select the same rows, and a larger `num_prompts` extends
        a smaller selection. A source with fewer rows yields fewer examples,
        which `datasets.load_examples` rejects. Example ids are
        `beans-<position>-<source index>` with a zero-padded position.
        """
        point = self.point

        # `datasets` belongs to the optional `datasets` extra of
        # uniserve-eval and is imported only when rows are loaded.
        load_dataset = getattr(
            importlib.import_module("datasets"), "load_dataset"
        )
        dataset = (
            load_dataset(
                "parquet", data_files=point.dataset_path, split="train"
            )
            if point.dataset_path
            else load_dataset(
                BEANS_REPOSITORY,
                split="train",
                revision=point.dataset_revision,
            )
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
                    input_image_b64=base64.b64encode(buffer.getvalue()).decode(
                        "ascii"
                    ),
                    input_image_mime="image/jpeg",
                )
            )
        return rows
