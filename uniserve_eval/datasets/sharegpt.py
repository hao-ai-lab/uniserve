"""Loads deterministic two-turn ShareGPT text-generation examples.

Each conversation contributes its first turn as the prompt and its second
turn as the reference completion. Only the completion's token count is kept:
it becomes ``Example.output_len``, which the text task requests as
``max_completion_tokens``.
"""

from __future__ import annotations

import json
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
        """Return seeded, non-empty prompt and completion pairs.

        A set ``dataset_path`` must name the conversation file: a missing
        path, a directory, or a file that does not parse as JSON raises the
        error from opening or parsing it rather than substituting the hub
        copy for the intended input. Only an unset path downloads the file
        from the hub at ``dataset_revision``, and download errors propagate.
        Pairs whose prompt or completion encodes to fewer than two tokens
        are skipped. Token counts use ``tokenizer.encode`` defaults, so they
        include any special tokens the tokenizer adds.

        Raises:
            ValueError: If no tokenizer is supplied.
        """
        if tokenizer is None:
            raise ValueError("dataset 'sharegpt' requires a tokenizer")
        point = self.point
        path = point.dataset_path
        if not path:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=SHAREGPT_REPO_ID,
                filename=SHAREGPT_FILENAME,
                repo_type="dataset",
                revision=point.dataset_revision,
            )

        with open(path, encoding="utf-8") as handle:
            dataset = json.load(handle)

        # Rows name their turn list either ``conversations`` or
        # ``conversation``; rows with fewer than two turns are dropped.
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

        # A private generator seeded by the load seed keeps the selection
        # deterministic without reseeding the process-global ``random``
        # generator other code in the process may draw from.
        random.Random(point.load.seed).shuffle(dataset)

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
