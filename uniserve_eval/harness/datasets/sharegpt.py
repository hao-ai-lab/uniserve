"""ShareGPT dataset loader for LLM serving, faithful to ``refs/sglang``.

This mirrors ``refs/sglang/python/sglang/benchmark/datasets/sharegpt.py``
``sample_sharegpt_requests`` so the filtered request set (and therefore the
reported numbers) match SGLang given the same seed, tokenizer, and
``--num-prompts``:

* repo ``anon8231489123/ShareGPT_Vicuna_unfiltered`` /
  ``ShareGPT_V3_unfiltered_cleaned_split.json`` (auto-downloaded when no local
  path is given);
* keep conversations with >= 2 turns; prompt = turn[0], completion = turn[1];
* ``random.shuffle`` then take the first ``num_requests`` that pass filters;
* ``output_len = len(tokenizer.encode(completion))`` unless a fixed length is set;
* prune ``prompt_len < 2`` or ``output_len < 2``; optional context-len upper cap.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


def _is_file_valid_json(path: str | None) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            json.load(handle)
        return True
    except (ValueError, OSError):
        return False


def load_sharegpt(
    dataset_path: str | None,
    num_requests: int,
    tokenizer: Any,
    *,
    fixed_output_len: int | None = None,
    context_len: int | None = None,
    seed: int = 42,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    path = dataset_path or ""
    if not _is_file_valid_json(path) and path == "":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
            repo_type="dataset",
            revision=revision,
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
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Seed before shuffle so the sampled request set is reproducible and matches
    # SGLang, which seeds the global ``random`` module with ``--seed`` (default 42).
    random.seed(seed)
    random.shuffle(dataset)

    rows: list[dict[str, Any]] = []
    for prompt, completion in dataset:
        if len(rows) == num_requests:
            break
        prompt_len = len(tokenizer.encode(prompt))
        output_len = (
            len(tokenizer.encode(completion)) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 2 or output_len < 2:
            continue
        if context_len and prompt_len + output_len > context_len:
            continue
        rows.append(
            {
                "id": f"sharegpt-{len(rows):06d}",
                "task": "text",
                "prompt": prompt,
                "prompt_len": prompt_len,
                "output_len": output_len,
            }
        )

    if not rows:
        raise ValueError(f"ShareGPT source yielded no usable rows for num_prompts={num_requests}")
    return rows
