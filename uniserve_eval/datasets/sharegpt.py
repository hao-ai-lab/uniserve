"""Loads deterministic two-turn ShareGPT text-generation examples."""

from __future__ import annotations

import json
import os
import random
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


class ShareGPTDataset(Dataset):
    """Selects tokenized prompts and target lengths from ShareGPT conversations."""  # noqa: E501

    name: ClassVar[str] = "sharegpt"
    requires_tokenizer: ClassVar[bool] = True

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Return seeded, non-empty prompt and completion pairs."""
        if tokenizer is None:
            raise ValueError("dataset 'sharegpt' requires a tokenizer")
        point = self.point
        path = point.dataset_path or ""
        if not _is_file_valid_json(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=SHAREGPT_REPO_ID,
                filename=SHAREGPT_FILENAME,
                repo_type="dataset",
                revision=point.dataset_revision,
            )

        with open(path, encoding="utf-8") as handle:
            dataset = json.load(handle)

        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0][
                    "value"
                ],
                data.get("conversations", data.get("conversation", []))[1][
                    "value"
                ],
            )
            for data in dataset
        ]

        random.seed(point.load.seed)
        random.shuffle(dataset)

        rows: list[Example] = []
        for prompt, completion in dataset:
            if len(rows) == point.load.num_prompts:
                break
            prompt_len = len(tokenizer.encode(prompt))
            output_len = len(tokenizer.encode(completion))
            if prompt_len < 2 or output_len < 2:
                continue
            rows.append(
                Example(
                    id=f"sharegpt-{len(rows):06d}",
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )
            )
        return rows


def _is_file_valid_json(path: str) -> bool:
    """Report whether a path names a readable JSON document."""
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            json.load(handle)
        return True
    except (ValueError, OSError):
        return False
