"""Dataset adapters select rows from the configured source."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from uniserve_eval.datasets.base import Dataset
from uniserve_eval.datasets.mjhq import MJHQDataset
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


class _HubDownloadError(Exception):
    """Raised in place of downloading a dataset file from the hub."""


@pytest.fixture(autouse=True)
def no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the hub download so no test here can reach the network."""

    def download(**kwargs: object) -> str:
        raise _HubDownloadError(str(kwargs.get("repo_id")))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


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


@pytest.mark.parametrize("source", ["directory", "missing.json"])
@pytest.mark.parametrize(
    "dataset_cls", [MJHQDataset, ShareGPTDataset], ids=["mjhq", "sharegpt"]
)
def test_an_unusable_dataset_path_is_not_replaced_by_the_hub(
    tmp_path: Path, source: str, dataset_cls: type[Dataset]
) -> None:
    path = tmp_path if source == "directory" else tmp_path / source
    point = _point(dataset_cls.name, str(path), num_prompts=1)

    with pytest.raises(OSError):
        dataset_cls(point).load(_TOKENIZER)


def test_sharegpt_rejects_a_local_file_that_is_not_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sharegpt.json"
    path.write_text('[{"conversations": [', encoding="utf-8")
    point = _point("sharegpt", str(path), num_prompts=1)

    with pytest.raises(json.JSONDecodeError):
        ShareGPTDataset(point).load(_TOKENIZER)
