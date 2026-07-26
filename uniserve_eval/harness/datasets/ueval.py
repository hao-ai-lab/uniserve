"""UEval prompt loader for interleaved text+image generation benchmarking.

UEval (``zlab-princeton/UEval``) has 1,000
expert-curated prompts across 8 real-world domains that require both text and
images in the output. For a speed benchmark we only need the request prompt; the
rubric-based quality scoring is out of scope.

``dataset_path`` may point at a local ``.jsonl``/``.json`` of prompts (objects
with a ``prompt``/``question`` field); otherwise the HF dataset is loaded via
``datasets.load_dataset`` and all splits/domains are concatenated.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

UEVAL_HF_REPO = "zlab-princeton/UEval"
_PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "query", "text")


def load_ueval(
    dataset_path: str | None,
    num_requests: int,
    *,
    seed: int = 42,
    hf_repo: str = UEVAL_HF_REPO,
    split: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    raw = _load_local(dataset_path) if dataset_path else _load_hf(hf_repo, split, revision)

    prompts: list[str] = []
    for entry in raw:
        prompt = _extract_prompt(entry)
        if prompt:
            prompts.append(prompt)

    random.Random(seed).shuffle(prompts)
    prompts = prompts[:num_requests]
    if not prompts:
        raise ValueError(
            "UEval source yielded no usable prompts; pass --dataset-path to a "
            "JSONL of {'prompt': ...} rows if the HF schema differs"
        )
    return [
        {"id": f"ueval-{idx:06d}", "task": "interleave", "prompt": prompt}
        for idx, prompt in enumerate(prompts)
    ]


def _load_local(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_hf(
    hf_repo: str, split: str | None, revision: str | None
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "loading UEval from the Hugging Face hub requires the 'datasets' "
            "package; install it or pass --dataset-path to a local prompt file"
        ) from error

    dataset = load_dataset(hf_repo, split=split, revision=revision)
    rows: list[dict[str, Any]] = []
    if hasattr(dataset, "keys"):  # DatasetDict: concatenate every split/domain.
        for key in dataset.keys():
            rows.extend(dict(row) for row in dataset[key])
    else:
        rows.extend(dict(row) for row in dataset)
    return rows


def _extract_prompt(entry: dict[str, Any]) -> str:
    for field in _PROMPT_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
