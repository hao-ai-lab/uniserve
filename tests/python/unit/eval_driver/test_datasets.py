"""Dataset adapters select rows from the configured source."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from uniserve_eval.datasets.sharegpt import ShareGPTDataset
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    TaskName,
)

pytestmark = pytest.mark.unit

# Counts whitespace-separated words as tokens.
_TOKENIZER = SimpleNamespace(encode=str.split)


def _point(dataset: str, dataset_path: str, num_prompts: int) -> BenchmarkPoint:
    return BenchmarkPoint(
        name="point",
        server="server",
        task=TaskName.TEXT,
        model="model",
        dataset=dataset,
        dataset_path=dataset_path,
        tokenizer="model",
        metrics=(MetricDefinition(("output_throughput",), "higher"),),
        load=LoadConfig(num_prompts=num_prompts, seed=7),
    )


def _sharegpt_file(path: Path) -> Path:
    conversations = [
        {
            "conversations": [
                {"from": "human", "value": f"question number {index}"},
                {"from": "gpt", "value": f"answer number {index}"},
            ]
        }
        for index in range(8)
    ]
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


def test_sharegpt_leaves_the_global_random_generator_untouched(
    tmp_path: Path,
) -> None:
    dataset = _sharegpt_file(tmp_path / "sharegpt.json")
    point = _point("sharegpt", str(dataset), num_prompts=3)

    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    rows = ShareGPTDataset(point).load(_TOKENIZER)

    assert len(rows) == 3
    assert random.random() == expected
